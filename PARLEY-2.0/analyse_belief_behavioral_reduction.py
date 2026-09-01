"""
Analyse behavioral minimisation of exact reachable belief contexts.

Example:
    python analyse_belief_behavioral_reduction_v2.py \
        --maps-dir maps --start-map 10 --end-map 99 \
        --target-x 9 --target-y 9

Outputs:
    belief_behavioral_reduction/
        all_maps_summary.csv
        aggregate_summary.json
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from exact_reachable_belief_model_sourcecopy_v2 import (
    build_exact_belief_model,
)
from belief_behavioral_minimization import minimize_belief_model


def read_map(path):
    rows = []
    with open(path, "r", newline="") as handle:
        rows.extend(csv.reader(handle))

    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


def analyse_map(map_id, maps_dir, target, p, max_steps):
    map_data = read_map(
        maps_dir / f"map_{map_id}.csv"
    )

    exact = build_exact_belief_model(
        map_id=map_id,
        map_data=map_data,
        target=target,
        p=p,
        max_steps=max_steps,
    )

    reduced = minimize_belief_model(
        exact,
        map_size=len(map_data),
    )

    exact_transitions = len(exact["transitions"])
    reduced_transitions = len(reduced["transitions"])
    exact_contexts = exact["context_count"]
    reduced_contexts = reduced["context_count"]

    return {
        "map_id": map_id,
        "exact_contexts": exact_contexts,
        "initial_behavioral_classes":
            reduced["minimization"]["initial_class_count"],
        "behavioral_classes": reduced_contexts,
        "removed_contexts": exact_contexts - reduced_contexts,
        "reduction_percent": (
            100.0 * (exact_contexts - reduced_contexts) / exact_contexts
            if exact_contexts else 0.0
        ),
        "refinement_iterations":
            reduced["minimization"]["refinement_iterations"],
        "distinct_relative_beliefs": exact["belief_count"],
        "exact_transitions": exact_transitions,
        "reduced_transitions": reduced_transitions,
        "transition_reduction_percent": (
            100.0
            * (exact_transitions - reduced_transitions)
            / exact_transitions
            if exact_transitions else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument("--start-map", type=int, default=10)
    parser.add_argument("--end-map", type=int, default=99)
    parser.add_argument("--target-x", type=int, default=9)
    parser.add_argument("--target-y", type=int, default=9)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="belief_behavioral_reduction",
    )
    args = parser.parse_args()

    maps_dir = Path(args.maps_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for map_id in range(args.start_map, args.end_map + 1):
        map_path = maps_dir / f"map_{map_id}.csv"
        if not map_path.exists():
            print(f"Skipping map {map_id}: file not found")
            continue

        result = analyse_map(
            map_id,
            maps_dir,
            (args.target_x, args.target_y),
            args.p,
            args.max_steps,
        )
        results.append(result)

        print(
            f"Map {map_id:>2}: "
            f"exact={result['exact_contexts']:>5}, "
            f"behavioral={result['behavioral_classes']:>5}, "
            f"reduction={result['reduction_percent']:6.2f}%, "
            f"transitions={result['exact_transitions']:>5}"
            f"->{result['reduced_transitions']:>5}, "
            f"iterations={result['refinement_iterations']}"
        )

    if not results:
        raise SystemExit("No maps analysed.")

    csv_path = output_dir / "all_maps_summary.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    aggregate = {
        "maps_analysed": len(results),
        "mean_exact_contexts": mean(
            r["exact_contexts"] for r in results
        ),
        "mean_behavioral_classes": mean(
            r["behavioral_classes"] for r in results
        ),
        "mean_reduction_percent": mean(
            r["reduction_percent"] for r in results
        ),
        "median_reduction_percent": median(
            r["reduction_percent"] for r in results
        ),
        "min_reduction_percent": min(
            r["reduction_percent"] for r in results
        ),
        "max_reduction_percent": max(
            r["reduction_percent"] for r in results
        ),
        "mean_transition_reduction_percent": mean(
            r["transition_reduction_percent"] for r in results
        ),
    }

    json_path = output_dir / "aggregate_summary.json"
    json_path.write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )

    print("\nFinished.")
    print(f"Per-map CSV: {csv_path}")
    print(f"Aggregate JSON: {json_path}")
    print(
        f"Mean state reduction: "
        f"{aggregate['mean_reduction_percent']:.2f}%"
    )
    print(
        f"Mean transition reduction: "
        f"{aggregate['mean_transition_reduction_percent']:.2f}%"
    )


if __name__ == "__main__":
    main()
