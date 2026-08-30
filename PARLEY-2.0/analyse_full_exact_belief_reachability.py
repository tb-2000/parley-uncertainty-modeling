#!/usr/bin/env python3
"""
Exact full reachability analysis for the PARLEY positional-belief model.

Unlike analyse_exact_reachable_beliefs.py, this script does NOT seed every
free cell as an independent reset location. It starts from the concrete robot
initial state (default: x=y=xhat=yhat=0, exact certainty) and explores the
complete reachable graph under ALL URC choices c=1..10.

State core:
    (x, y, xhat, yhat, belief_id)

For each reachable state and each possible URC choice c:
    if U_B(belief) >= tau_c:
        perfect localization/update:
            xhat' = x
            yhat' = y
            belief' = certainty
            x',y' = x,y
    else:
        skip localization and execute the MAPE action determined by xhat,yhat.
        The physical robot branches probabilistically over the four actual
        motion outcomes, while the positional belief is predicted
        deterministically.

Important:
- No explicit step/age variable is used.
- Belief vectors are exact reachable relative beliefs, deduplicated only by
  floating-point canonicalization (rounding to 14 digits).
- All c=1..10 are explored nondeterministically.
- Re-visiting the same core state closes the graph, so the analysis naturally
  includes arbitrarily many future update/move cycles without a fixed horizon.
- The ten thresholds are still derived offline from max_steps=10 exactly as in
  full_belief_representatives.py.

The result is the number of genuinely reachable physical+knowledge core states
from the actual initial state, rather than the union obtained by seeding every
possible localization position.

Examples:
    python analyse_full_exact_belief_reachability.py --maps 10
    python analyse_full_exact_belief_reachability.py --maps 10,14,23
    python analyse_full_exact_belief_reachability.py --maps 10-99

Optional:
    --start-x 0 --start-y 0
"""

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import full_belief_representatives as belief_impl


DIRECTIONS = belief_impl.DIRECTIONS
DEFAULT_FIRST_MAP = 10
DEFAULT_LAST_MAP = 99
DEFAULT_MAX_STEPS = 10
DEFAULT_P = 0.01
DEFAULT_TARGET = (9, 9)
DEFAULT_START = (0, 0)
DEFAULT_MAPS_DIR = "maps"
DEFAULT_OUTPUT_DIR = "full_exact_belief_reachability"
ROUND_DIGITS = 14
EPS = 1e-15


def load_map(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        rows.extend(csv.reader(f))
    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


def parse_maps(value):
    if value is None:
        return list(range(DEFAULT_FIRST_MAP, DEFAULT_LAST_MAP + 1))

    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(item))
    return sorted(set(result))


def vector_key(vector):
    return tuple(round(float(v), ROUND_DIGITS) for v in vector)


def scaled_gini(vector):
    return int(round(
        belief_impl._representative_gini(vector) * 10000
    ))


def threshold_reached(uncertainty, c, thresholds):
    return uncertainty >= int(thresholds[c - 1])


def certainty_vector(n):
    return belief_impl._relative_vector(
        {(0, 0): 1.0}, 0, 0, n
    )


def absolute_from_relative(vector, xhat, yhat, n):
    relative = belief_impl._vector_to_relative(vector, n)
    absolute = defaultdict(float)

    for (dx, dy), probability in relative.items():
        if probability <= EPS:
            continue
        x = xhat + dx
        y = yhat + dy
        if not (0 <= x <= n and 0 <= y <= n):
            raise ValueError(
                "Reachable exact belief contains an out-of-grid state: "
                f"xhat={xhat}, yhat={yhat}, dx={dx}, dy={dy}"
            )
        absolute[(x, y)] += probability

    return dict(absolute)


def exact_belief_successor(vector, xhat, yhat, action, n, p):
    absolute = absolute_from_relative(vector, xhat, yhat, n)
    propagated = belief_impl._propagate_absolute(
        absolute, action, n, p
    )
    nxhat, nyhat = belief_impl._move(
        xhat, yhat, action, n
    )
    successor = belief_impl._relative_vector(
        propagated, nxhat, nyhat, n
    )
    return successor, nxhat, nyhat


def physical_successors(x, y, commanded_action, n, p):
    """
    Return distinct physical successors and summed probabilities.
    Boundary clipping can make several actual directions end in the same cell.
    """
    result = defaultdict(float)

    for actual_action in DIRECTIONS:
        probability = (
            1.0 - 3.0 * p
            if actual_action == commanded_action
            else p
        )
        nx, ny = belief_impl._move(
            x, y, actual_action, n
        )
        result[(nx, ny)] += probability

    return {
        position: probability
        for position, probability in result.items()
        if probability > EPS
    }


def state_key(x, y, xhat, yhat, vector):
    return (
        int(x),
        int(y),
        int(xhat),
        int(yhat),
        vector_key(vector),
    )


def analyse_map(
    map_id,
    map_data,
    target,
    start,
    p,
    max_steps,
):
    size = len(map_data)
    n = size - 1

    sx, sy = start
    if not (0 <= sx <= n and 0 <= sy <= n):
        raise ValueError(f"Start {start} is outside the map.")
    if int(map_data[sx][sy]) > 9:
        raise ValueError(
            f"Start {start} is not a free cell on map {map_id}."
        )

    # Reuse exactly the current threshold derivation.
    _, gini_by_age, controller = belief_impl._generate_records(
        map_data, target, p, max_steps
    )
    thresholds = belief_impl._thresholds(
        gini_by_age, max_steps
    )

    certain = certainty_vector(n)

    belief_id_by_key = {}
    belief_vectors = []

    def get_belief_id(vector):
        key = vector_key(vector)
        if key not in belief_id_by_key:
            belief_id_by_key[key] = len(belief_vectors)
            belief_vectors.append(vector)
        return belief_id_by_key[key]

    certainty_id = get_belief_id(certain)

    # The queue contains concrete physical+knowledge states.
    initial_key = state_key(
        sx, sy, sx, sy, certain
    )

    queue = deque([initial_key])
    seen = {initial_key}

    # Store vector separately by its canonical key.
    vector_by_key = {
        vector_key(certain): certain
    }

    state_records = {}
    transitions = []

    reachable_knowledge_contexts = set()
    reachable_reset_positions = set()
    reachable_xhat_yhat = set()
    reachable_physical_positions = set()

    update_edges = 0
    skip_choice_edges = 0
    stochastic_move_edges = 0
    target_states = 0
    no_action_states = 0

    while queue:
        x, y, xhat, yhat, vkey = queue.popleft()
        vector = vector_by_key[vkey]
        bid = get_belief_id(vector)
        uncertainty = scaled_gini(vector)

        reachable_knowledge_contexts.add(
            (xhat, yhat, bid)
        )
        reachable_xhat_yhat.add((xhat, yhat))
        reachable_physical_positions.add((x, y))

        record = {
            "x": x,
            "y": y,
            "xhat": xhat,
            "yhat": yhat,
            "belief_id": bid,
            "uncertainty": uncertainty,
            "is_target_estimate": int(
                (xhat, yhat) == target
            ),
            "is_target_actual": int(
                (x, y) == target
            ),
        }
        state_records[(x, y, xhat, yhat, vkey)] = record

        # If the MAPE controller has no action at the estimate, no physical
        # movement can be generated from this state. We still retain the state.
        action = belief_impl._direction(
            controller, xhat, yhat
        )

        if (xhat, yhat) == target:
            target_states += 1
        if action is None:
            no_action_states += 1

        for c in range(1, max_steps + 1):
            tau = int(thresholds[c - 1])

            if threshold_reached(
                uncertainty, c, thresholds
            ):
                # Perfect localization. Physical position is unchanged.
                nx = x
                ny = y
                nxhat = x
                nyhat = y
                nvector = certain
                nbid = certainty_id

                reachable_reset_positions.add((x, y))

                nk = state_key(
                    nx, ny, nxhat, nyhat, nvector
                )
                vector_by_key[vector_key(nvector)] = nvector

                transitions.append({
                    "x": x,
                    "y": y,
                    "xhat": xhat,
                    "yhat": yhat,
                    "belief_id": bid,
                    "c": c,
                    "threshold": tau,
                    "uncertainty": uncertainty,
                    "branch": "update",
                    "commanded_action": "",
                    "probability": 1.0,
                    "next_x": nx,
                    "next_y": ny,
                    "next_xhat": nxhat,
                    "next_yhat": nyhat,
                    "next_belief_id": nbid,
                })
                update_edges += 1

                if nk not in seen:
                    seen.add(nk)
                    queue.append(nk)

                continue

            # c chose a threshold not yet reached -> skip localization and move.
            if action is None:
                # No controller move. Keep no synthetic self-loop here because
                # the actual PRISM MAPE model may handle target/no-action states
                # differently. The state is counted as reachable.
                continue

            next_vector, nxhat, nyhat = exact_belief_successor(
                vector, xhat, yhat, action, n, p
            )
            next_vkey = vector_key(next_vector)
            vector_by_key[next_vkey] = next_vector
            next_bid = get_belief_id(next_vector)

            physical = physical_successors(
                x, y, action, n, p
            )

            skip_choice_edges += 1

            for (nx, ny), probability in physical.items():
                nk = state_key(
                    nx,
                    ny,
                    nxhat,
                    nyhat,
                    next_vector,
                )

                transitions.append({
                    "x": x,
                    "y": y,
                    "xhat": xhat,
                    "yhat": yhat,
                    "belief_id": bid,
                    "c": c,
                    "threshold": tau,
                    "uncertainty": uncertainty,
                    "branch": "skip_move",
                    "commanded_action": action,
                    "probability": probability,
                    "next_x": nx,
                    "next_y": ny,
                    "next_xhat": nxhat,
                    "next_yhat": nyhat,
                    "next_belief_id": next_bid,
                })
                stochastic_move_edges += 1

                if nk not in seen:
                    seen.add(nk)
                    queue.append(nk)

    # Sanity check: every reached concrete physical position must have positive
    # probability in its corresponding exact belief.
    support_violations = 0
    for key, record in state_records.items():
        x = record["x"]
        y = record["y"]
        xhat = record["xhat"]
        yhat = record["yhat"]
        vector = vector_by_key[key[4]]
        absolute = absolute_from_relative(
            vector, xhat, yhat, n
        )
        if absolute.get((x, y), 0.0) <= EPS:
            support_violations += 1

    summary = {
        "map_id": map_id,
        "start_x": sx,
        "start_y": sy,
        "target_x": target[0],
        "target_y": target[1],
        "p": p,
        "max_steps_for_thresholds": max_steps,
        "thresholds": thresholds,

        "reachable_full_core_state_count": len(seen),
        "reachable_exact_belief_id_count": len(belief_vectors),
        "reachable_knowledge_context_count": len(
            reachable_knowledge_contexts
        ),
        "reachable_xhat_yhat_count": len(
            reachable_xhat_yhat
        ),
        "reachable_physical_position_count": len(
            reachable_physical_positions
        ),
        "reachable_reset_position_count": len(
            reachable_reset_positions
        ),

        "update_transition_count": update_edges,
        "skip_choice_count": skip_choice_edges,
        "stochastic_move_branch_count": stochastic_move_edges,

        "states_with_target_estimate": target_states,
        "states_with_no_controller_action": no_action_states,
        "support_violations": support_violations,
    }

    return (
        summary,
        state_records,
        transitions,
        belief_vectors,
        reachable_reset_positions,
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open(
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
    parser.add_argument(
        "--maps",
        default=None,
        help="e.g. 10 or 10,14,23 or 10-99",
    )
    parser.add_argument(
        "--maps-dir",
        default=DEFAULT_MAPS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--p",
        type=float,
        default=DEFAULT_P,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
    )
    parser.add_argument(
        "--start-x",
        type=int,
        default=DEFAULT_START[0],
    )
    parser.add_argument(
        "--start-y",
        type=int,
        default=DEFAULT_START[1],
    )
    parser.add_argument(
        "--target-x",
        type=int,
        default=DEFAULT_TARGET[0],
    )
    parser.add_argument(
        "--target-y",
        type=int,
        default=DEFAULT_TARGET[1],
    )
    args = parser.parse_args()

    maps = parse_maps(args.maps)
    start = (args.start_x, args.start_y)
    target = (args.target_x, args.target_y)

    output_root = Path(args.output_dir)
    output_root.mkdir(
        parents=True, exist_ok=True
    )

    all_rows = []

    for map_id in maps:
        path = (
            Path(args.maps_dir)
            / f"map_{map_id}.csv"
        )
        if not path.exists():
            print(
                f"Skipping map {map_id}: {path} not found"
            )
            continue

        print(f"\nAnalysing map {map_id} ...")
        map_data = load_map(path)

        (
            summary,
            states,
            transitions,
            belief_vectors,
            reset_positions,
        ) = analyse_map(
            map_id=map_id,
            map_data=map_data,
            target=target,
            start=start,
            p=args.p,
            max_steps=args.max_steps,
        )

        all_rows.append(summary)

        map_dir = output_root / f"map_{map_id}"
        map_dir.mkdir(
            parents=True, exist_ok=True
        )

        with (
            map_dir / "summary.json"
        ).open("w", encoding="utf-8") as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

        state_rows = []
        for key, record in states.items():
            state_rows.append(dict(record))

        write_csv(
            map_dir / "reachable_states.csv",
            state_rows,
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

        belief_rows = []
        for bid, vector in enumerate(belief_vectors):
            belief_rows.append({
                "belief_id": bid,
                "uncertainty": scaled_gini(vector),
                "support_size_at_origin_relative": sum(
                    1 for v in vector if v > EPS
                ),
            })
        write_csv(
            map_dir / "beliefs.csv",
            belief_rows,
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
            "  reachable xhat,yhat positions: "
            f"{summary['reachable_xhat_yhat_count']}"
        )
        print(
            "  actually reached reset/localization positions: "
            f"{summary['reachable_reset_position_count']}"
        )
        print(
            "  update transitions: "
            f"{summary['update_transition_count']}"
        )
        print(
            "  stochastic movement branches: "
            f"{summary['stochastic_move_branch_count']}"
        )
        print(
            "  support violations: "
            f"{summary['support_violations']}"
        )

    write_csv(
        output_root / "all_maps_summary.csv",
        all_rows,
    )

    if all_rows:
        def mean(field):
            return sum(
                float(row[field])
                for row in all_rows
            ) / len(all_rows)

        aggregate = {
            "maps": len(all_rows),
            "mean_reachable_full_core_state_count": mean(
                "reachable_full_core_state_count"
            ),
            "mean_reachable_exact_belief_id_count": mean(
                "reachable_exact_belief_id_count"
            ),
            "mean_reachable_knowledge_context_count": mean(
                "reachable_knowledge_context_count"
            ),
            "mean_reachable_xhat_yhat_count": mean(
                "reachable_xhat_yhat_count"
            ),
            "mean_reachable_reset_position_count": mean(
                "reachable_reset_position_count"
            ),
        }

        with (
            output_root / "aggregate.json"
        ).open("w", encoding="utf-8") as f:
            json.dump(
                aggregate,
                f,
                indent=2,
            )

        print("\nAggregate:")
        print(
            "  mean reachable full core states: "
            f"{aggregate['mean_reachable_full_core_state_count']:.2f}"
        )
        print(
            "  mean reachable exact belief IDs: "
            f"{aggregate['mean_reachable_exact_belief_id_count']:.2f}"
        )
        print(
            "  mean reachable knowledge contexts: "
            f"{aggregate['mean_reachable_knowledge_context_count']:.2f}"
        )


if __name__ == "__main__":
    main()
