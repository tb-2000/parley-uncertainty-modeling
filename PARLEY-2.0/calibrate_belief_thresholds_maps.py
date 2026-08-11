import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import dijkstra

from top4_belief_states import (
    DIRECTIONS as BELIEF_DIRECTIONS,
    build_belief_automaton,
)


# Must match prism_model_generator_belief_threshold.py
MAPE_DIRECTIONS = ["west", "east", "south", "north"]

DIR_TO_COUNT_INDEX = {
    "east": 0,
    "west": 1,
    "north": 2,
    "south": 3,
}


def gini_uncertainty(signature):
    """
    Integer-scaled Gini impurity of

        (b1, b2, b3, b4, other)

    with percentages summing to 100:

        G = 10000 - sum(b_i^2)

    Larger values mean greater uncertainty.
    """
    return 10000 - sum(value * value for value in signature)


def read_map(filename):
    """
    Read a map exactly like the PRISM model generator.

    Returns:
        map_data: transposed/reversed representation used by dijkstra.py
        map_size
        obstacles: set of (x,y)
    """
    rows = []
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        rows.extend(reader)

    map_size = len(rows)

    if map_size == 0:
        raise ValueError(f"Empty map: {filename}")

    transposed = list(zip(*rows))
    map_data = [row[::-1] for row in transposed]

    obstacles = set()
    for x in range(map_size):
        for y in range(map_size):
            if int(map_data[x][y]) > 9:
                obstacles.add((x, y))

    return map_data, map_size, obstacles


def next_position(x, y, direction, n):
    """
    Deterministic MAP estimate update, identical to the Knowledge module.
    """
    if direction == "east":
        return min(x + 1, n), y
    if direction == "west":
        return max(x - 1, 0), y
    if direction == "north":
        return x, min(y + 1, n)
    if direction == "south":
        return x, max(y - 1, 0)

    raise ValueError(f"Unknown direction: {direction}")


def controller_for_map(map_data, target):
    """
    Build the same Dijkstra MAPE policy used by the model generator.

    The generator computes:
        _d = dijkstra.compute_directions(map_data, target)
        d = list(zip(*_d))

    and then reads direction = d[y][x].
    """
    raw = dijkstra.compute_directions(map_data, target)
    return list(zip(*raw))


def get_direction(controller, x, y):
    """
    Return the commanded movement at (x,y), or None when no movement
    is encoded (e.g. target/unreachable cell).
    """
    value = int(controller[y][x])

    if value < 0 or value >= len(MAPE_DIRECTIONS):
        return None

    return MAPE_DIRECTIONS[value]


def increment_counts(counts, direction):
    counts = list(counts)
    counts[DIR_TO_COUNT_INDEX[direction]] += 1
    return tuple(counts)


def collect_map_beliefs(
    map_id,
    map_path,
    target,
    count_to_state,
    states,
    max_steps,
    start_mode="all_free",
    fixed_start=(0, 0),
):
    """
    Collect belief states actually induced by the Dijkstra MAPE policy.

    start_mode="all_free":
        Start once from every non-obstacle position. This is recommended
        because the URC later contains a position-specific decision_x_y.

    start_mode="fixed":
        Use only fixed_start, useful for reproducing one concrete mission.

    A run is stopped when:
      * target is reached,
      * the MAPE controller has no valid command,
      * a controller loop is detected,
      * max_steps movements have been collected.

    Returns a list of occurrence rows. Each occurrence is one reachable
    abstract belief after 1..max_steps commanded MAPE movements.
    """
    map_data, map_size, obstacles = read_map(map_path)
    n = map_size - 1

    tx, ty = target
    if not (0 <= tx <= n and 0 <= ty <= n):
        raise ValueError(
            f"Target {target} outside map {map_id} with size {map_size}"
        )

    controller = controller_for_map(map_data, target)

    if start_mode == "fixed":
        starts = [fixed_start]
    else:
        starts = [
            (x, y)
            for x in range(map_size)
            for y in range(map_size)
            if (x, y) not in obstacles
        ]

    occurrences = []

    for start_x, start_y in starts:
        if (start_x, start_y) in obstacles:
            continue

        x, y = start_x, start_y
        counts = (0, 0, 0, 0)
        visited_controller_states = set()

        for age in range(1, max_steps + 1):
            if (x, y) == target:
                break

            # Defensive loop detection for malformed/unreachable policies.
            key = (x, y)
            if key in visited_controller_states:
                break
            visited_controller_states.add(key)

            direction = get_direction(controller, x, y)
            if direction is None:
                break

            counts = increment_counts(counts, direction)

            if counts not in count_to_state:
                raise KeyError(
                    f"No abstract belief for counts={counts}; "
                    f"increase max_steps or check belief generation."
                )

            state_id = count_to_state[counts]
            signature = states[state_id]["signature"]
            uncertainty = gini_uncertainty(signature)

            occurrences.append(
                {
                    "map_id": map_id,
                    "start_x": start_x,
                    "start_y": start_y,
                    "age": age,
                    "xhat": x,
                    "yhat": y,
                    "direction": direction,
                    "cnt_e": counts[0],
                    "cnt_w": counts[1],
                    "cnt_n": counts[2],
                    "cnt_s": counts[3],
                    "belief_state": state_id,
                    "b1": signature[0],
                    "b2": signature[1],
                    "b3": signature[2],
                    "b4": signature[3],
                    "other": signature[4],
                    "gini_uncertainty": uncertainty,
                }
            )

            x, y = next_position(x, y, direction, n)

    return occurrences


def percentile(sorted_values, q):
    if not sorted_values:
        raise ValueError("Cannot calculate percentile of empty values.")

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower

    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def choose_quantile_thresholds(values, number_of_thresholds=10):
    """
    Select quantile-based thresholds from uncertainty values that actually
    occur under the MAPE controllers of the calibration maps.

    Duplicates are removed. If quantiles collapse, distinct observed
    uncertainty values are added to reach number_of_thresholds when possible.
    """
    if not values:
        raise ValueError("No uncertainty samples were collected.")

    sorted_values = sorted(values)

    thresholds = []
    for i in range(1, number_of_thresholds + 1):
        q = i / number_of_thresholds
        thresholds.append(round(percentile(sorted_values, q)))

    thresholds = sorted(set(thresholds))

    distinct = sorted(set(sorted_values))

    if len(thresholds) < number_of_thresholds:
        # Add values spread across the observed range rather than simply
        # taking the smallest remaining values.
        for i in range(number_of_thresholds):
            if len(thresholds) >= number_of_thresholds:
                break

            q = i / max(1, number_of_thresholds - 1)
            idx = round(q * (len(distinct) - 1))
            candidate = distinct[idx]

            if candidate not in thresholds:
                thresholds.append(candidate)

        # Final fallback if many values still coincide.
        for candidate in distinct:
            if len(thresholds) >= number_of_thresholds:
                break
            if candidate not in thresholds:
                thresholds.append(candidate)

    thresholds = sorted(set(thresholds))

    if len(thresholds) > number_of_thresholds:
        selected = []
        for i in range(number_of_thresholds):
            q = i / max(1, number_of_thresholds - 1)
            idx = round(q * (len(thresholds) - 1))
            selected.append(thresholds[idx])
        thresholds = sorted(set(selected))

    return thresholds


def collect_all_maps(
    maps_dir,
    first_map,
    last_map,
    target,
    max_steps,
    p,
    start_mode,
    fixed_start,
):
    states, count_to_state, _ = build_belief_automaton(
        max_steps=max_steps,
        p=p,
    )

    all_occurrences = []
    loaded_maps = []
    missing_maps = []

    for map_id in range(first_map, last_map + 1):
        map_path = Path(maps_dir) / f"map_{map_id}.csv"

        if not map_path.exists():
            missing_maps.append(map_id)
            continue

        occurrences = collect_map_beliefs(
            map_id=map_id,
            map_path=map_path,
            target=target,
            count_to_state=count_to_state,
            states=states,
            max_steps=max_steps,
            start_mode=start_mode,
            fixed_start=fixed_start,
        )

        all_occurrences.extend(occurrences)
        loaded_maps.append(map_id)

    return states, all_occurrences, loaded_maps, missing_maps


def write_occurrences_csv(path, occurrences):
    fieldnames = [
        "map_id",
        "start_x",
        "start_y",
        "age",
        "xhat",
        "yhat",
        "direction",
        "cnt_e",
        "cnt_w",
        "cnt_n",
        "cnt_s",
        "belief_state",
        "b1",
        "b2",
        "b3",
        "b4",
        "other",
        "gini_uncertainty",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(occurrences)


def write_state_frequency_csv(path, occurrences):
    state_counter = Counter(
        row["belief_state"]
        for row in occurrences
    )

    uncertainty_counter = Counter(
        row["gini_uncertainty"]
        for row in occurrences
    )

    by_state = {}
    for row in occurrences:
        state_id = row["belief_state"]
        if state_id not in by_state:
            by_state[state_id] = row

    with open(path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "belief_state",
            "b1",
            "b2",
            "b3",
            "b4",
            "other",
            "gini_uncertainty",
            "state_occurrences",
            "same_gini_occurrences",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for state_id in sorted(by_state):
            row = by_state[state_id]
            uncertainty = row["gini_uncertainty"]

            writer.writerow(
                {
                    "belief_state": state_id,
                    "b1": row["b1"],
                    "b2": row["b2"],
                    "b3": row["b3"],
                    "b4": row["b4"],
                    "other": row["other"],
                    "gini_uncertainty": uncertainty,
                    "state_occurrences": state_counter[state_id],
                    "same_gini_occurrences": uncertainty_counter[uncertainty],
                }
            )


def print_summary(
    occurrences,
    thresholds,
    loaded_maps,
    missing_maps,
    target,
    max_steps,
    p,
    start_mode,
):
    states_seen = {
        row["belief_state"]
        for row in occurrences
    }
    uncertainty_values = [
        row["gini_uncertainty"]
        for row in occurrences
    ]

    print()
    print("MAPE-based Top-4 belief threshold calibration")
    print("==============================================")
    print(f"maps loaded: {len(loaded_maps)}")
    if loaded_maps:
        print(f"map range loaded: {min(loaded_maps)}..{max(loaded_maps)}")
    print(f"missing maps: {missing_maps if missing_maps else 'none'}")
    print(f"target: {target}")
    print(f"p: {p}")
    print(f"max belief steps: {max_steps}")
    print(f"start mode: {start_mode}")
    print(f"belief occurrences collected: {len(occurrences)}")
    print(f"distinct belief states observed: {len(states_seen)}")
    print(
        "distinct Gini values observed: "
        f"{len(set(uncertainty_values))}"
    )

    if uncertainty_values:
        print(f"minimum observed Gini: {min(uncertainty_values)}")
        print(f"maximum observed Gini: {max(uncertainty_values)}")

    print()
    print("Recommended thresholds:")
    print(thresholds)
    print()
    print(f"BELIEF_THRESHOLDS = {thresholds}")
    print()

    for i, threshold in enumerate(thresholds, start=1):
        print(
            f"decision={i:2d} -> "
            f"max_belief_uncertainty={threshold}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate Gini-based belief uncertainty thresholds from "
            "the Dijkstra MAPE controllers of maps 10-99."
        )
    )

    parser.add_argument(
        "--maps-dir",
        default="maps",
        help="Directory containing map_<id>.csv files.",
    )
    parser.add_argument("--first-map", type=int, default=10)
    parser.add_argument("--last-map", type=int, default=99)

    parser.add_argument(
        "--target-x",
        type=int,
        default=9,
        help="Target x-coordinate used for Dijkstra.",
    )
    parser.add_argument(
        "--target-y",
        type=int,
        default=9,
        help="Target y-coordinate used for Dijkstra.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum movements since the last perfect update.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.01,
        help="Probability of each wrong movement direction.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        default=10,
        help="Number of empirical thresholds to derive.",
    )

    parser.add_argument(
        "--start-mode",
        choices=["all_free", "fixed"],
        default="all_free",
        help=(
            "'all_free' calibrates from every free position on every map; "
            "'fixed' uses only --start-x/--start-y."
        ),
    )
    parser.add_argument("--start-x", type=int, default=0)
    parser.add_argument("--start-y", type=int, default=0)

    parser.add_argument(
        "--occurrences-csv",
        default="belief_mape_occurrences.csv",
    )
    parser.add_argument(
        "--states-csv",
        default="belief_mape_state_frequencies.csv",
    )

    args = parser.parse_args()

    states, occurrences, loaded_maps, missing_maps = collect_all_maps(
        maps_dir=args.maps_dir,
        first_map=args.first_map,
        last_map=args.last_map,
        target=(args.target_x, args.target_y),
        max_steps=args.max_steps,
        p=args.p,
        start_mode=args.start_mode,
        fixed_start=(args.start_x, args.start_y),
    )

    if not occurrences:
        raise RuntimeError(
            "No MAPE belief occurrences were collected. "
            "Check --maps-dir, map range, target and dijkstra import."
        )

    uncertainty_samples = [
        row["gini_uncertainty"]
        for row in occurrences
    ]

    thresholds = choose_quantile_thresholds(
        uncertainty_samples,
        number_of_thresholds=args.thresholds,
    )

    write_occurrences_csv(
        args.occurrences_csv,
        occurrences,
    )
    write_state_frequency_csv(
        args.states_csv,
        occurrences,
    )

    print_summary(
        occurrences=occurrences,
        thresholds=thresholds,
        loaded_maps=loaded_maps,
        missing_maps=missing_maps,
        target=(args.target_x, args.target_y),
        max_steps=args.max_steps,
        p=args.p,
        start_mode=args.start_mode,
    )

    print()
    print(f"Occurrence table: {args.occurrences_csv}")
    print(f"State frequency table: {args.states_csv}")


if __name__ == "__main__":
    main()
