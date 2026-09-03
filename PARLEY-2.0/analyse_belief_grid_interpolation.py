import argparse
import csv
from pathlib import Path

from full_belief_grid_interpolation import (
    analyse_grid,
    build_local_grids,
    generate_exact_occurrences,
    read_map_data,
)


DEFAULT_GRID_VALUES = (1, 2, 3, 5, 8, 10)


def analyse_map(
    map_id,
    maps_dir,
    target,
    p,
    max_steps,
    grid_per_position,
    neighbours,
):
    map_data = read_map_data(
        Path(maps_dir) / f"map_{map_id}.csv"
    )

    occurrences, thresholds, _ = generate_exact_occurrences(
        map_data,
        target,
        p=p,
        max_steps=max_steps,
    )

    grids = build_local_grids(
        occurrences,
        grid_per_position=grid_per_position,
    )

    summary = analyse_grid(
        occurrences,
        thresholds,
        grids,
        p=p,
        neighbours=neighbours,
    )

    return {
        "map_id": map_id,
        "grid_per_position": grid_per_position,
        "neighbours": neighbours,
        **summary,
    }


def write_csv(rows, output):
    if not rows:
        return

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep local reachable-belief grid sizes and report current "
            "and successor URC mismatch."
        )
    )
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument("--first-map", type=int, default=10)
    parser.add_argument("--last-map", type=int, default=99)
    parser.add_argument(
        "--maps",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit map IDs. Overrides first/last-map.",
    )
    parser.add_argument(
        "--grid-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_GRID_VALUES),
    )
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument("--target-x", type=int, default=9)
    parser.add_argument("--target-y", type=int, default=9)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--output",
        default="belief_grid_interpolation_summary.csv",
    )
    args = parser.parse_args()

    map_ids = (
        args.maps
        if args.maps
        else list(range(args.first_map, args.last_map + 1))
    )

    rows = []

    for map_id in map_ids:
        map_path = Path(args.maps_dir) / f"map_{map_id}.csv"
        if not map_path.exists():
            print(f"Skipping map {map_id}: {map_path} not found")
            continue

        for grid_per_position in args.grid_values:
            print(
                f"map={map_id}, "
                f"grid_per_position={grid_per_position}, "
                f"neighbours={args.neighbours}"
            )

            row = analyse_map(
                map_id=map_id,
                maps_dir=args.maps_dir,
                target=(args.target_x, args.target_y),
                p=args.p,
                max_steps=args.max_steps,
                grid_per_position=grid_per_position,
                neighbours=args.neighbours,
            )
            rows.append(row)

            print(
                "  current URC mismatch = "
                f"{100.0 * row['current_urc_mismatch_rate']:.3f}%"
            )
            print(
                "  successor URC mismatch = "
                f"{100.0 * row['successor_urc_mismatch_rate']:.3f}%"
            )
            print(
                "  weighted successor URC mismatch = "
                f"{100.0 * row['successor_component_weighted_urc_mismatch_rate']:.3f}%"
            )
            print(
                "  grid state contexts = "
                f"{row['grid_state_contexts']}"
            )

    write_csv(rows, args.output)
    print(f"\nFinished. Results: {args.output}")


if __name__ == "__main__":
    main()
