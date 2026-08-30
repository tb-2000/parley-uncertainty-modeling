"""
Analyse full exact reachable Gaussian moment states under all URC choices.

The Gaussian knowledge state is
    g = (bias_x, bias_y, var_x, var_y, cov_xy)

This script performs NO clustering and NO medoid projection.

It:
1. derives the same 10 map-specific Gaussian MSE thresholds as the current model,
2. starts from the concrete initial full state
       (x, y, xhat, yhat, gstate) = (0,0,0,0,ZERO_STATE),
3. explores the union of all URC choices c=1..10,
4. uses the two-behaviour reduction for reachability:
       UPDATE exists iff MSE(g) >= tau_1
       SKIP   exists iff MSE(g) <  tau_10
   because all c that update have the same reset successor and all c that skip
   have the same movement successor structure,
5. enumerates exact reachable Gaussian moment states, knowledge contexts,
   and full core states.

Important:
- "exact" here means exact with respect to the Gaussian moment model.
  The Gaussian representation itself is still an approximation of the full
  positional distribution.
- State keys are rounded only to make floating-point recurrence stable.
"""

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import dijkstra

from full_gaussian_representatives_bias import (
    ZERO_STATE,
    MSE_SCALE,
    _controller,
    _direction,
    _move,
    _predict_state,
    _robot_outcomes,
    _state_key,
    _mse,
    _thresholds,
)


DEFAULT_START = (0, 0)
DEFAULT_TARGET = (9, 9)
DEFAULT_P = 0.01
DEFAULT_MAX_STEPS = 10


def load_map(map_id, maps_dir="maps"):
    path = Path(maps_dir) / f"map_{map_id}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", newline="") as file:
        rows = list(csv.reader(file))

    transposed = list(zip(*rows))
    map_data = [row[::-1] for row in transposed]

    return map_data


def derive_thresholds(map_data, target, p, max_steps):
    """
    Reproduce the same Gaussian threshold derivation as the current model:
    median Gaussian MSE after prediction ages 1..max_steps.
    """
    map_size = len(map_data)
    n = map_size - 1
    controller = _controller(map_data, target)

    mse_by_age = defaultdict(list)

    for start_x in range(map_size):
        for start_y in range(map_size):
            # Keep threshold derivation identical to current Gaussian code:
            # obstacle cells are not used as threshold seeds.
            if int(map_data[start_x][start_y]) > 9:
                continue

            xhat, yhat = start_x, start_y
            state = ZERO_STATE

            for age in range(max_steps + 1):
                state = _state_key(state)
                mse_by_age[age].append(_mse(state))

                if age >= max_steps or (xhat, yhat) == target:
                    break

                action = _direction(controller, xhat, yhat)

                if action is None:
                    break

                state = _predict_state(
                    state,
                    xhat,
                    yhat,
                    action,
                    n,
                    p,
                )

                xhat, yhat = _move(
                    xhat,
                    yhat,
                    action,
                    n,
                )

    thresholds = _thresholds(
        mse_by_age,
        max_steps,
        scale=MSE_SCALE,
    )

    return thresholds, controller


def physical_successors(x, y, action, n, p):
    """
    Return distinct physical successors reachable with positive probability.
    Duplicate clipped outcomes at borders are merged naturally by set().
    """
    result = set()

    for probability, dx, dy in _robot_outcomes(action, p):
        if probability <= 0.0:
            continue

        nx = min(max(x + dx, 0), n)
        ny = min(max(y + dy, 0), n)

        result.add((nx, ny))

    return tuple(sorted(result))


def analyse_map(
    map_id,
    maps_dir="maps",
    start=DEFAULT_START,
    target=DEFAULT_TARGET,
    p=DEFAULT_P,
    max_steps=DEFAULT_MAX_STEPS,
    progress_every=10000,
):
    map_data = load_map(
        map_id,
        maps_dir=maps_dir,
    )

    size = len(map_data)
    n = size - 1

    thresholds, controller = derive_thresholds(
        map_data,
        target,
        p,
        max_steps,
    )

    if len(thresholds) != 10:
        raise ValueError(
            f"Expected 10 thresholds, got {len(thresholds)}."
        )

    tau1 = int(thresholds[0])
    tau10 = int(thresholds[-1])

    gaussian_ids = {}
    gaussian_states = []

    def get_gstate_id(state):
        key = _state_key(state)

        if key not in gaussian_ids:
            gaussian_ids[key] = len(gaussian_states)
            gaussian_states.append(key)

        return gaussian_ids[key]

    zero_id = get_gstate_id(ZERO_STATE)

    if zero_id != 0:
        raise AssertionError(
            "ZERO_STATE must have gstate ID 0."
        )

    x0, y0 = start

    initial = (
        x0,
        y0,
        x0,
        y0,
        ZERO_STATE,
    )

    queue = deque([initial])

    seen_full = {
        (
            x0,
            y0,
            x0,
            y0,
            _state_key(ZERO_STATE),
        )
    }

    knowledge_contexts = {
        (
            x0,
            y0,
            _state_key(ZERO_STATE),
        )
    }

    reset_positions = set()

    prediction_cache = {}
    physical_cache = {}

    processed = 0
    max_queue = 1

    while queue:
        x, y, xhat, yhat, gstate = queue.popleft()

        processed += 1

        if (
            progress_every
            and processed % progress_every == 0
        ):
            print(
                f"    processed={processed}, "
                f"queue={len(queue)}, "
                f"seen={len(seen_full)}, "
                f"gaussians={len(gaussian_states)}"
            )

        get_gstate_id(gstate)

        raw_mse = _mse(gstate)
        scaled_mse = int(
            round(raw_mse * MSE_SCALE)
        )

        # ------------------------------------------------------------
        # UPDATE branch
        # ------------------------------------------------------------
        # At least one c causes an update iff the first threshold is reached.
        if scaled_mse >= tau1:
            reset_positions.add((x, y))

            successor = (
                x,
                y,
                x,
                y,
                ZERO_STATE,
            )

            successor_key = (
                x,
                y,
                x,
                y,
                _state_key(ZERO_STATE),
            )

            knowledge_contexts.add(
                (
                    x,
                    y,
                    _state_key(ZERO_STATE),
                )
            )

            if successor_key not in seen_full:
                seen_full.add(successor_key)
                queue.append(successor)

        # ------------------------------------------------------------
        # SKIP branch
        # ------------------------------------------------------------
        # At least one c can skip iff the largest threshold has not yet
        # been reached.
        if scaled_mse < tau10:
            action = _direction(
                controller,
                xhat,
                yhat,
            )

            if action is not None:
                prediction_key = (
                    _state_key(gstate),
                    xhat,
                    yhat,
                    action,
                )

                if prediction_key not in prediction_cache:
                    next_gstate = _predict_state(
                        gstate,
                        xhat,
                        yhat,
                        action,
                        n,
                        p,
                    )

                    next_xhat, next_yhat = _move(
                        xhat,
                        yhat,
                        action,
                        n,
                    )

                    prediction_cache[prediction_key] = (
                        _state_key(next_gstate),
                        next_xhat,
                        next_yhat,
                    )

                (
                    next_gstate,
                    next_xhat,
                    next_yhat,
                ) = prediction_cache[prediction_key]

                get_gstate_id(next_gstate)

                knowledge_contexts.add(
                    (
                        next_xhat,
                        next_yhat,
                        next_gstate,
                    )
                )

                physical_key = (
                    x,
                    y,
                    action,
                )

                if physical_key not in physical_cache:
                    physical_cache[physical_key] = (
                        physical_successors(
                            x,
                            y,
                            action,
                            n,
                            p,
                        )
                    )

                for next_x, next_y in physical_cache[
                    physical_key
                ]:
                    successor = (
                        next_x,
                        next_y,
                        next_xhat,
                        next_yhat,
                        next_gstate,
                    )

                    successor_key = (
                        next_x,
                        next_y,
                        next_xhat,
                        next_yhat,
                        next_gstate,
                    )

                    if successor_key not in seen_full:
                        seen_full.add(successor_key)
                        queue.append(successor)

        max_queue = max(
            max_queue,
            len(queue),
        )

    result = {
        "map": map_id,
        "reachable_full_core_states": len(seen_full),
        "reachable_exact_gaussian_ids": len(gaussian_states),
        "reachable_knowledge_contexts": len(knowledge_contexts),
        "reachable_reset_positions": len(reset_positions),
        "tau_1": tau1,
        "tau_10": tau10,
        "max_queue": max_queue,
    }

    return result


def write_summary(results, output):
    output = Path(output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "map",
        "reachable_full_core_states",
        "reachable_exact_gaussian_ids",
        "reachable_knowledge_contexts",
        "reachable_reset_positions",
        "tau_1",
        "tau_10",
        "max_queue",
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def parse_maps(text):
    if text is None:
        return list(range(10, 100))

    result = []

    for part in text.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(
                range(
                    int(start),
                    int(end) + 1,
                )
            )
        else:
            result.append(int(part))

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--maps",
        default=None,
        help=(
            "Maps, e.g. 10,11,14 or 10-20. "
            "Default: 10-99"
        ),
    )

    parser.add_argument(
        "--maps-dir",
        default="maps",
    )

    parser.add_argument(
        "--start-x",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--start-y",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--target-x",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--target-y",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--p",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--output",
        default=(
            "full_exact_gaussian_reachability/"
            "all_maps_summary.csv"
        ),
    )

    args = parser.parse_args()

    maps = parse_maps(
        args.maps
    )

    results = []

    for map_id in maps:
        print(
            f"\nAnalysing map {map_id} ..."
        )

        result = analyse_map(
            map_id=map_id,
            maps_dir=args.maps_dir,
            start=(
                args.start_x,
                args.start_y,
            ),
            target=(
                args.target_x,
                args.target_y,
            ),
            p=args.p,
            max_steps=args.max_steps,
            progress_every=args.progress_every,
        )

        results.append(result)

        print(
            "  reachable full core states "
            "(x,y,xhat,yhat,gstate): "
            f"{result['reachable_full_core_states']}"
        )

        print(
            "  reachable exact Gaussian IDs: "
            f"{result['reachable_exact_gaussian_ids']}"
        )

        print(
            "  reachable knowledge contexts "
            "(xhat,yhat,gstate): "
            f"{result['reachable_knowledge_contexts']}"
        )

        print(
            "  reachable reset positions: "
            f"{result['reachable_reset_positions']}"
        )

        print(
            f"  tau_1={result['tau_1']}, "
            f"tau_10={result['tau_10']}"
        )

    write_summary(
        results,
        args.output,
    )

    print(
        f"\nFinished. Summary: {args.output}"
    )


if __name__ == "__main__":
    main()
