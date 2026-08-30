#!/usr/bin/env python3
"""
Behavior-preserving minimisation of exact reachable Belief knowledge states.

Purpose
-------
Before forcing the exact reachable belief set into K=100 medoids, determine
how much it can be reduced *without changing the URC-relevant deterministic
behaviour*.

The script:
  1. generates the exact URC-closed reachable knowledge contexts
       (xhat, yhat, exact belief)
  2. assigns each context its current URC level 0..10
  3. builds the exact deterministic skip transition
  4. performs Moore-machine style partition refinement.

Two contexts are merged only if they have:
  - the same xhat,yhat,
  - the same current URC level,
  - the same terminal/nonterminal status,
  - and recursively equivalent exact skip successors.

Thus the quotient is exact w.r.t.:
  - current URC threshold decisions,
  - deterministic future URC-level evolution under the controller,
  - and estimated position.

No L1 distance, no medoids, no nearest-neighbour projection are used.

This is NOT forced to K=100.  The resulting number of equivalence classes is
the natural lossless size.  If it is already near 100, use this quotient.
If it is much larger, then any K=100 version is necessarily approximate and
should be treated as a separate approximation step.

Expected repository layout:
    maps/map_<id>.csv
    full_belief_representatives.py

Examples:
    python analyse_behavioral_belief_minimization.py --maps 10
    python analyse_behavioral_belief_minimization.py --maps 10,14,23
    python analyse_behavioral_belief_minimization.py --maps 10-99
"""

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import full_belief_representatives as belief_impl


DEFAULT_FIRST_MAP = 10
DEFAULT_LAST_MAP = 99
DEFAULT_MAX_STEPS = 10
DEFAULT_P = 0.01
DEFAULT_TARGET = (9, 9)
DEFAULT_MAPS_DIR = "maps"
DEFAULT_OUTPUT_DIR = "behavioral_belief_minimization"

ROUND_DIGITS = 14


def load_map(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        rows.extend(csv.reader(f))
    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


def vector_key(vector):
    return tuple(round(float(v), ROUND_DIGITS) for v in vector)


def scaled_gini(vector):
    return int(round(
        belief_impl._representative_gini(vector) * 10000
    ))


def urc_level(uncertainty, thresholds):
    """
    Level 0: below tau_1
    Level j: highest threshold tau_j reached, j in 1..10.
    """
    level = 0
    for i, threshold in enumerate(thresholds, start=1):
        if uncertainty >= int(threshold):
            level = i
        else:
            break
    return level


def absolute_from_relative(vector, xhat, yhat, n):
    relative = belief_impl._vector_to_relative(vector, n)
    absolute = defaultdict(float)

    for (dx, dy), probability in relative.items():
        x = xhat + dx
        y = yhat + dy

        # For exact reachable contexts this should never be necessary.
        if not (0 <= x <= n and 0 <= y <= n):
            raise ValueError(
                "Exact reachable context produced an out-of-grid offset: "
                f"xhat={xhat}, yhat={yhat}, dx={dx}, dy={dy}"
            )

        absolute[(x, y)] += probability

    return dict(absolute)


def exact_successor(vector, xhat, yhat, action, n, p):
    absolute = absolute_from_relative(
        vector, xhat, yhat, n
    )
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


def free_cells(map_data):
    size = len(map_data)
    return [
        (x, y)
        for x in range(size)
        for y in range(size)
        if int(map_data[x][y]) <= 9
    ]


def build_exact_urc_closed_graph(
    map_data,
    target,
    p,
    max_steps,
):
    """
    Returns:
      contexts: list of dicts
      successor: dict context_id -> successor_context_id
      thresholds
    """
    size = len(map_data)
    n = size - 1

    _, gini_by_age, controller = belief_impl._generate_records(
        map_data, target, p, max_steps
    )
    thresholds = belief_impl._thresholds(
        gini_by_age, max_steps
    )
    tau10 = int(thresholds[-1])

    contexts = []
    context_id_by_key = {}
    queue = deque()

    def ensure_context(xhat, yhat, vector, depth):
        key = (xhat, yhat, vector_key(vector))
        if key in context_id_by_key:
            cid = context_id_by_key[key]
            if depth < contexts[cid]["min_depth"]:
                contexts[cid]["min_depth"] = depth
            return cid, False

        cid = len(contexts)
        context_id_by_key[key] = cid
        uncertainty = scaled_gini(vector)

        contexts.append({
            "context_id": cid,
            "xhat": xhat,
            "yhat": yhat,
            "vector": vector,
            "vector_key": key[2],
            "uncertainty": uncertainty,
            "urc_level": urc_level(
                uncertainty, thresholds
            ),
            "min_depth": depth,
            "terminal_reason": "",
            "action": "",
        })
        queue.append(cid)
        return cid, True

    # Perfect localisation can reset certainty at every free position.
    for x, y in free_cells(map_data):
        absolute = {(x, y): 1.0}
        vector = belief_impl._relative_vector(
            absolute, x, y, n
        )
        ensure_context(x, y, vector, 0)

    successor = {}

    while queue:
        cid = queue.popleft()
        record = contexts[cid]

        xhat = record["xhat"]
        yhat = record["yhat"]
        vector = record["vector"]
        uncertainty = record["uncertainty"]

        if uncertainty >= tau10:
            record["terminal_reason"] = "tau10"
            continue

        if (xhat, yhat) == target:
            record["terminal_reason"] = "target"
            continue

        action = belief_impl._direction(
            controller, xhat, yhat
        )
        if action is None:
            record["terminal_reason"] = "no_action"
            continue

        record["action"] = action

        next_vector, nxhat, nyhat = exact_successor(
            vector, xhat, yhat, action, n, p
        )
        next_cid, _ = ensure_context(
            nxhat,
            nyhat,
            next_vector,
            record["min_depth"] + 1,
        )
        successor[cid] = next_cid

    return contexts, successor, thresholds


def initial_partition(contexts):
    """
    Initial observable signature.

    xhat,yhat are included deliberately.  This guarantees that merged states
    never hide a different estimated position/controller context.
    """
    groups = defaultdict(list)

    for record in contexts:
        terminal = bool(record["terminal_reason"])
        signature = (
            record["xhat"],
            record["yhat"],
            record["urc_level"],
            terminal,
            record["terminal_reason"],
            record["action"],
        )
        groups[signature].append(record["context_id"])

    blocks = list(groups.values())
    block_of = {}

    for block_id, members in enumerate(blocks):
        for cid in members:
            block_of[cid] = block_id

    return blocks, block_of


def refine_partition(contexts, successor):
    """
    Moore-style deterministic partition refinement.

    Nonterminal states remain equivalent only when their exact skip successors
    are in the same current equivalence block.
    """
    blocks, block_of = initial_partition(contexts)
    iteration = 0

    while True:
        iteration += 1
        groups = defaultdict(list)

        for record in contexts:
            cid = record["context_id"]
            terminal = bool(record["terminal_reason"])

            if terminal:
                successor_block = -1
            else:
                successor_block = block_of[successor[cid]]

            signature = (
                record["xhat"],
                record["yhat"],
                record["urc_level"],
                terminal,
                record["terminal_reason"],
                record["action"],
                successor_block,
            )
            groups[signature].append(cid)

        new_blocks = list(groups.values())
        new_block_of = {}

        for block_id, members in enumerate(new_blocks):
            for cid in members:
                new_block_of[cid] = block_id

        # Stable iff every context keeps the same equivalence relation.
        old_pairs = sorted(
            sorted(block) for block in blocks
        )
        new_pairs = sorted(
            sorted(block) for block in new_blocks
        )

        blocks = new_blocks
        block_of = new_block_of

        if old_pairs == new_pairs:
            break

    return blocks, block_of, iteration


def unique_exact_beliefs(contexts):
    return len({
        record["vector_key"]
        for record in contexts
    })


def analyse_map(
    map_id,
    map_data,
    target,
    p,
    max_steps,
):
    contexts, successor, thresholds = (
        build_exact_urc_closed_graph(
            map_data=map_data,
            target=target,
            p=p,
            max_steps=max_steps,
        )
    )

    blocks, block_of, refinement_iterations = (
        refine_partition(contexts, successor)
    )

    # Build quotient transitions.
    quotient_transitions = {}
    for block_id, members in enumerate(blocks):
        representative = contexts[members[0]]

        if representative["terminal_reason"]:
            continue

        succ_blocks = {
            block_of[successor[cid]]
            for cid in members
        }

        if len(succ_blocks) != 1:
            raise AssertionError(
                "Partition refinement produced a non-deterministic quotient."
            )

        quotient_transitions[block_id] = next(
            iter(succ_blocks)
        )

    exact_beliefs = unique_exact_beliefs(contexts)
    exact_contexts = len(contexts)
    quotient_states = len(blocks)

    merged_contexts = exact_contexts - quotient_states
    compression = (
        quotient_states / exact_contexts
        if exact_contexts else 0.0
    )

    summary = {
        "map_id": map_id,
        "exact_belief_count": exact_beliefs,
        "exact_knowledge_context_count": exact_contexts,
        "behavioral_equivalence_class_count": quotient_states,
        "merged_context_count": merged_contexts,
        "remaining_fraction": compression,
        "reduction_percent": (
            (1.0 - compression) * 100.0
        ),
        "factor_vs_k100": (
            quotient_states / 100.0
        ),
        "refinement_iterations": refinement_iterations,
        "thresholds": thresholds,
    }

    return (
        summary,
        contexts,
        successor,
        blocks,
        block_of,
        quotient_transitions,
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


def parse_maps(value):
    if value is None:
        return list(
            range(DEFAULT_FIRST_MAP, DEFAULT_LAST_MAP + 1)
        )

    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" in item:
            start, end = item.split("-", 1)
            result.extend(
                range(int(start), int(end) + 1)
            )
        else:
            result.append(int(item))

    return sorted(set(result))


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

    output_root = Path(args.output_dir)
    output_root.mkdir(
        parents=True, exist_ok=True
    )

    target = (args.target_x, args.target_y)
    all_rows = []

    for map_id in parse_maps(args.maps):
        path = (
            Path(args.maps_dir)
            / f"map_{map_id}.csv"
        )

        if not path.exists():
            print(
                f"Skipping map {map_id}: "
                f"{path} not found"
            )
            continue

        print(f"\nAnalysing map {map_id} ...")
        map_data = load_map(path)

        (
            summary,
            contexts,
            successor,
            blocks,
            block_of,
            quotient_transitions,
        ) = analyse_map(
            map_id,
            map_data,
            target,
            args.p,
            args.max_steps,
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

        context_rows = []
        for record in contexts:
            cid = record["context_id"]
            context_rows.append({
                "context_id": cid,
                "xhat": record["xhat"],
                "yhat": record["yhat"],
                "uncertainty": record[
                    "uncertainty"
                ],
                "urc_level": record[
                    "urc_level"
                ],
                "action": record["action"],
                "terminal_reason": record[
                    "terminal_reason"
                ],
                "min_depth": record["min_depth"],
                "successor_context_id": (
                    successor.get(cid, "")
                ),
                "equivalence_class": (
                    block_of[cid]
                ),
            })

        write_csv(
            map_dir / "contexts.csv",
            context_rows,
        )

        class_rows = []
        for block_id, members in enumerate(blocks):
            first = contexts[members[0]]
            class_rows.append({
                "class_id": block_id,
                "member_count": len(members),
                "xhat": first["xhat"],
                "yhat": first["yhat"],
                "urc_level": first["urc_level"],
                "action": first["action"],
                "terminal_reason": first[
                    "terminal_reason"
                ],
                "successor_class": (
                    quotient_transitions.get(
                        block_id, ""
                    )
                ),
            })

        write_csv(
            map_dir / "equivalence_classes.csv",
            class_rows,
        )

        print(
            "  exact reachable belief IDs: "
            f"{summary['exact_belief_count']}"
        )
        print(
            "  exact reachable knowledge contexts: "
            f"{summary['exact_knowledge_context_count']}"
        )
        print(
            "  lossless behavioral classes: "
            f"{summary['behavioral_equivalence_class_count']}"
        )
        print(
            "  lossless reduction: "
            f"{summary['reduction_percent']:.2f}%"
        )
        print(
            "  natural lossless size / K=100: "
            f"{summary['factor_vs_k100']:.2f}x"
        )
        print(
            "  refinement iterations: "
            f"{summary['refinement_iterations']}"
        )

    write_csv(
        output_root / "all_maps_summary.csv",
        all_rows,
    )

    if all_rows:
        mean_classes = sum(
            row["behavioral_equivalence_class_count"]
            for row in all_rows
        ) / len(all_rows)

        mean_contexts = sum(
            row["exact_knowledge_context_count"]
            for row in all_rows
        ) / len(all_rows)

        mean_reduction = sum(
            row["reduction_percent"]
            for row in all_rows
        ) / len(all_rows)

        aggregate = {
            "maps": len(all_rows),
            "mean_exact_knowledge_context_count": (
                mean_contexts
            ),
            "mean_behavioral_equivalence_class_count": (
                mean_classes
            ),
            "mean_lossless_reduction_percent": (
                mean_reduction
            ),
            "mean_factor_vs_k100": (
                mean_classes / 100.0
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
            "  mean exact knowledge contexts: "
            f"{mean_contexts:.2f}"
        )
        print(
            "  mean lossless behavioral classes: "
            f"{mean_classes:.2f}"
        )
        print(
            "  mean lossless reduction: "
            f"{mean_reduction:.2f}%"
        )


if __name__ == "__main__":
    main()
