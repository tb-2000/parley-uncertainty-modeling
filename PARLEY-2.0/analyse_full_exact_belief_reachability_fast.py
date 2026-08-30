#!/usr/bin/env python3
"""
Optimized exact reachability from the real initial state.

Same reachable core-state set as analyse_full_exact_belief_reachability.py,
but exploits an important fact:

For reachability, c=1..10 produces at most TWO distinct behaviours in a state:
  1) UPDATE, if at least one threshold tau_c has been reached.
  2) SKIP+MOVE, if at least one threshold tau_c has not yet been reached.

All c values that update have exactly the same successor.
All c values that skip have exactly the same stochastic movement successors.

Therefore this script does not enumerate the ten c values separately.
It also caches belief and physical successors.

Core state:
    (x, y, xhat, yhat, exact_belief)

No explicit step variable.
"""

import argparse
import csv
import json
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import full_belief_representatives as belief_impl


DEFAULT_FIRST_MAP = 10
DEFAULT_LAST_MAP = 99
DEFAULT_MAX_STEPS = 10
DEFAULT_P = 0.01
DEFAULT_TARGET = (9, 9)
DEFAULT_START = (0, 0)
DEFAULT_MAPS_DIR = "maps"
DEFAULT_OUTPUT_DIR = "full_exact_belief_reachability_fast"

ROUND_DIGITS = 14
EPS = 1e-15
DIRECTIONS = belief_impl.DIRECTIONS


def load_map(path):
    rows = []
    with Path(path).open("r", newline="") as f:
        rows.extend(csv.reader(f))
    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


def parse_maps(value):
    if value is None:
        return list(range(DEFAULT_FIRST_MAP, DEFAULT_LAST_MAP + 1))

    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def vector_key(vector):
    return tuple(round(float(v), ROUND_DIGITS) for v in vector)


def scaled_gini(vector):
    return int(round(
        belief_impl._representative_gini(vector) * 10000
    ))


def analyse_map(map_id, map_data, target, start, p, max_steps):
    size = len(map_data)
    n = size - 1

    sx, sy = start
    if int(map_data[sx][sy]) > 9:
        raise ValueError(f"Start {start} is blocked on map {map_id}.")

    # Same threshold derivation as the existing model.
    _, gini_by_age, controller = belief_impl._generate_records(
        map_data, target, p, max_steps
    )
    thresholds = belief_impl._thresholds(gini_by_age, max_steps)
    tau1 = int(thresholds[0])
    tau10 = int(thresholds[-1])

    certain = belief_impl._relative_vector(
        {(0, 0): 1.0}, 0, 0, n
    )
    certain_key = vector_key(certain)

    vector_by_key = {certain_key: certain}
    belief_id_by_key = {certain_key: 0}
    belief_vectors = [certain]

    def ensure_vector(vector):
        k = vector_key(vector)
        if k not in vector_by_key:
            vector_by_key[k] = vector
        if k not in belief_id_by_key:
            belief_id_by_key[k] = len(belief_vectors)
            belief_vectors.append(vector)
        return k, belief_id_by_key[k]

    @lru_cache(maxsize=None)
    def belief_successor(xhat, yhat, vkey, action):
        vector = vector_by_key[vkey]
        relative = belief_impl._vector_to_relative(vector, n)

        absolute = defaultdict(float)
        for (dx, dy), probability in relative.items():
            if probability <= EPS:
                continue
            ax = xhat + dx
            ay = yhat + dy
            if not (0 <= ax <= n and 0 <= ay <= n):
                raise ValueError(
                    "Reachable exact belief has out-of-grid support: "
                    f"xhat={xhat}, yhat={yhat}, dx={dx}, dy={dy}"
                )
            absolute[(ax, ay)] += probability

        propagated = belief_impl._propagate_absolute(
            dict(absolute), action, n, p
        )
        nxhat, nyhat = belief_impl._move(
            xhat, yhat, action, n
        )
        successor = belief_impl._relative_vector(
            propagated, nxhat, nyhat, n
        )
        next_vkey, next_bid = ensure_vector(successor)

        return nxhat, nyhat, next_vkey, next_bid

    @lru_cache(maxsize=None)
    def physical_successors(x, y, action):
        result = defaultdict(float)

        for actual_action in DIRECTIONS:
            q = (
                1.0 - 3.0 * p
                if actual_action == action
                else p
            )
            nx, ny = belief_impl._move(
                x, y, actual_action, n
            )
            result[(nx, ny)] += q

        return tuple(
            (nx, ny, q)
            for (nx, ny), q in result.items()
            if q > EPS
        )

    # Concrete initial state.
    initial = (sx, sy, sx, sy, certain_key)
    queue = deque([initial])
    seen = {initial}

    states = {}
    transitions = []

    knowledge_contexts = set()
    xhat_yhat_positions = set()
    physical_positions = set()
    reset_positions = set()

    update_behavior_count = 0
    skip_behavior_count = 0
    stochastic_move_branch_count = 0
    support_violations = 0

    processed = 0

    while queue:
        x, y, xhat, yhat, vkey = queue.popleft()
        processed += 1

        vector = vector_by_key[vkey]
        bid = belief_id_by_key[vkey]
        uncertainty = scaled_gini(vector)

        knowledge_contexts.add((xhat, yhat, bid))
        xhat_yhat_positions.add((xhat, yhat))
        physical_positions.add((x, y))

        states[(x, y, xhat, yhat, vkey)] = {
            "x": x,
            "y": y,
            "xhat": xhat,
            "yhat": yhat,
            "belief_id": bid,
            "uncertainty": uncertainty,
        }

        # Sanity: concrete actual state must lie in the belief support.
        relative = belief_impl._vector_to_relative(vector, n)
        dx = x - xhat
        dy = y - yhat
        if relative.get((dx, dy), 0.0) <= EPS:
            support_violations += 1

        # ------------------------------------------------------------
        # Behaviour 1: UPDATE exists iff at least one tau_c is reached.
        # Since thresholds are monotone, this is equivalent to U >= tau1.
        # ------------------------------------------------------------
        if uncertainty >= tau1:
            nk = (x, y, x, y, certain_key)
            reset_positions.add((x, y))

            transitions.append({
                "x": x,
                "y": y,
                "xhat": xhat,
                "yhat": yhat,
                "belief_id": bid,
                "uncertainty": uncertainty,
                "branch": "update",
                "next_x": x,
                "next_y": y,
                "next_xhat": x,
                "next_yhat": y,
                "next_belief_id": 0,
                "probability": 1.0,
            })
            update_behavior_count += 1

            if nk not in seen:
                seen.add(nk)
                queue.append(nk)

        # ------------------------------------------------------------
        # Behaviour 2: SKIP exists iff at least one tau_c is NOT reached.
        # Since tau10 is largest, this is equivalent to U < tau10.
        # ------------------------------------------------------------
        if uncertainty < tau10:
            action = belief_impl._direction(
                controller, xhat, yhat
            )

            if action is not None:
                (
                    nxhat,
                    nyhat,
                    next_vkey,
                    next_bid,
                ) = belief_successor(
                    xhat, yhat, vkey, action
                )

                skip_behavior_count += 1

                for nx, ny, q in physical_successors(
                    x, y, action
                ):
                    nk = (
                        nx,
                        ny,
                        nxhat,
                        nyhat,
                        next_vkey,
                    )

                    transitions.append({
                        "x": x,
                        "y": y,
                        "xhat": xhat,
                        "yhat": yhat,
                        "belief_id": bid,
                        "uncertainty": uncertainty,
                        "branch": "skip_move",
                        "commanded_action": action,
                        "next_x": nx,
                        "next_y": ny,
                        "next_xhat": nxhat,
                        "next_yhat": nyhat,
                        "next_belief_id": next_bid,
                        "probability": q,
                    })
                    stochastic_move_branch_count += 1

                    if nk not in seen:
                        seen.add(nk)
                        queue.append(nk)

        # Compact progress output for unexpectedly large graphs.
        if processed % 10000 == 0:
            print(
                f"    processed={processed}, "
                f"queue={len(queue)}, "
                f"seen={len(seen)}, "
                f"beliefs={len(belief_vectors)}"
            )

    summary = {
        "map_id": map_id,
        "start_x": sx,
        "start_y": sy,
        "target_x": target[0],
        "target_y": target[1],
        "p": p,
        "max_steps_for_thresholds": max_steps,
        "thresholds": thresholds,
        "tau1": tau1,
        "tau10": tau10,

        "reachable_full_core_state_count": len(seen),
        "reachable_exact_belief_id_count": len(belief_vectors),
        "reachable_knowledge_context_count": len(knowledge_contexts),
        "reachable_xhat_yhat_count": len(xhat_yhat_positions),
        "reachable_physical_position_count": len(physical_positions),
        "reachable_reset_position_count": len(reset_positions),

        "update_behavior_count": update_behavior_count,
        "skip_behavior_count": skip_behavior_count,
        "stochastic_move_branch_count": stochastic_move_branch_count,
        "support_violations": support_violations,
    }

    return summary, states, transitions, reset_positions


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=None)
    parser.add_argument("--maps-dir", default=DEFAULT_MAPS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--p", type=float, default=DEFAULT_P)
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS
    )
    parser.add_argument(
        "--start-x", type=int, default=DEFAULT_START[0]
    )
    parser.add_argument(
        "--start-y", type=int, default=DEFAULT_START[1]
    )
    parser.add_argument(
        "--target-x", type=int, default=DEFAULT_TARGET[0]
    )
    parser.add_argument(
        "--target-y", type=int, default=DEFAULT_TARGET[1]
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    start = (args.start_x, args.start_y)
    target = (args.target_x, args.target_y)

    all_summaries = []

    for map_id in parse_maps(args.maps):
        map_path = Path(args.maps_dir) / f"map_{map_id}.csv"
        if not map_path.exists():
            print(f"Skipping map {map_id}: {map_path} not found")
            continue

        print(f"\nAnalysing map {map_id} ...")
        map_data = load_map(map_path)

        summary, states, transitions, reset_positions = analyse_map(
            map_id=map_id,
            map_data=map_data,
            target=target,
            start=start,
            p=args.p,
            max_steps=args.max_steps,
        )
        all_summaries.append(summary)

        map_dir = output_root / f"map_{map_id}"
        map_dir.mkdir(parents=True, exist_ok=True)

        with (map_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

        write_csv(
            map_dir / "reachable_states.csv",
            list(states.values()),
        )
        write_csv(
            map_dir / "transitions.csv",
            transitions,
        )
        write_csv(
            map_dir / "reachable_reset_positions.csv",
            [
                {"x": x, "y": y}
                for x, y in sorted(reset_positions)
            ],
        )

        print(
            "  reachable full core states "
            "(x,y,xhat,yhat,belief): "
            f"{summary['reachable_full_core_state_count']}"
        )
        print(
            "  reachable exact belief IDs: "
            f"{summary['reachable_exact_belief_id_count']}"
        )
        print(
            "  reachable knowledge contexts "
            "(xhat,yhat,belief): "
            f"{summary['reachable_knowledge_context_count']}"
        )
        print(
            "  reachable reset positions: "
            f"{summary['reachable_reset_position_count']}"
        )
        print(
            "  support violations: "
            f"{summary['support_violations']}"
        )

    write_csv(
        output_root / "all_maps_summary.csv",
        all_summaries,
    )


if __name__ == "__main__":
    main()
