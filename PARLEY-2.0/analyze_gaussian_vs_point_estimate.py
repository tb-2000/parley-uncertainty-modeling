import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import dijkstra


DIRECTIONS = ("west", "east", "south", "north")
MOVE = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}

MSE_SCALE = 1_000_000
ZERO_STATE = (0.0, 0.0, 0.0, 0.0, 0.0)


def _controller(map_data, target):
    return list(zip(*dijkstra.compute_directions(map_data, target)))


def _direction(controller, x, y):
    value = int(controller[y][x])
    return DIRECTIONS[value] if 0 <= value < 4 else None


def _clip(v, n):
    return min(max(v, 0), n)


def _move(x, y, action, n):
    dx, dy = MOVE[action]
    return _clip(x + dx, n), _clip(y + dy, n)


def _robot_outcomes(action, p):
    intended = 1.0 - 3.0 * p
    if action == "east":
        return ((intended, 1, 0), (p, 0, 1), (p, 0, -1), (p, -1, 0))
    if action == "west":
        return ((p, 1, 0), (p, 0, 1), (p, 0, -1), (intended, -1, 0))
    if action == "north":
        return ((p, 1, 0), (intended, 0, 1), (p, 0, -1), (p, -1, 0))
    if action == "south":
        return ((p, 1, 0), (p, 0, 1), (intended, 0, -1), (p, -1, 0))
    raise ValueError(f"Unknown action: {action}")


def _motion_moments(x, y, action, n, p):
    samples = []
    for probability, dx, dy in _robot_outcomes(action, p):
        nx = _clip(x + dx, n)
        ny = _clip(y + dy, n)
        samples.append((probability, float(nx - x), float(ny - y)))

    mean_dx = sum(prob * dx for prob, dx, _ in samples)
    mean_dy = sum(prob * dy for prob, _, dy in samples)
    var_x = sum(prob * (dx - mean_dx) ** 2 for prob, dx, _ in samples)
    var_y = sum(prob * (dy - mean_dy) ** 2 for prob, _, dy in samples)
    cov_xy = sum(
        prob * (dx - mean_dx) * (dy - mean_dy)
        for prob, dx, dy in samples
    )
    return (mean_dx, mean_dy), (var_x, var_y, cov_xy)


def _state_key(state, digits=14):
    out = []
    for value in state:
        rounded = round(float(value), digits)
        out.append(0.0 if abs(rounded) < 10 ** (-digits) else rounded)
    return tuple(out)


def _mse(state):
    bx, by, var_x, var_y, _ = state
    return bx * bx + by * by + var_x + var_y


def _predict_state(state, xhat, yhat, action, n, p):
    bx, by, var_x, var_y, cov_xy = state
    mean_move, q = _motion_moments(xhat, yhat, action, n, p)

    next_xhat, next_yhat = _move(xhat, yhat, action, n)
    estimate_dx = float(next_xhat - xhat)
    estimate_dy = float(next_yhat - yhat)

    return _state_key((
        bx + mean_move[0] - estimate_dx,
        by + mean_move[1] - estimate_dy,
        var_x + q[0],
        var_y + q[1],
        cov_xy + q[2],
    ))


def _load_map(path):
    with path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


def _gaussian_level(mse_value, thresholds_raw):
    reached = 0
    scaled = int(round(mse_value * MSE_SCALE))
    for level, threshold in enumerate(thresholds_raw, start=1):
        if scaled >= int(threshold):
            reached = level
        else:
            break
    return reached


def _summary(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "cv": None,
            "range": None,
        }

    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    variance = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(variance)
    cv = std / mean if abs(mean) > 1e-15 else 0.0

    return {
        "count": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": mean,
        "median": median,
        "std": std,
        "cv": cv,
        "range": vals[-1] - vals[0],
    }


def _generate_histories(map_data, target, p, max_steps):
    n = len(map_data) - 1
    controller = _controller(map_data, target)
    histories = []

    for start_x in range(len(map_data)):
        for start_y in range(len(map_data)):
            if int(map_data[start_x][start_y]) > 9:
                continue

            xhat, yhat = start_x, start_y
            state = ZERO_STATE
            trajectory = []

            for age in range(max_steps + 1):
                trajectory.append({
                    "age": age,
                    "xhat": xhat,
                    "yhat": yhat,
                    "state": state,
                    "mse": _mse(state),
                })

                if age >= max_steps or (xhat, yhat) == target:
                    break

                action = _direction(controller, xhat, yhat)
                if action is None:
                    break

                state = _predict_state(state, xhat, yhat, action, n, p)
                xhat, yhat = _move(xhat, yhat, action, n)

            histories.append({
                "start": [start_x, start_y],
                "trajectory": trajectory,
            })

    return histories


def analyse_map(map_id, maps_dir, gaussian_dir, target, p, max_steps):
    map_path = Path(maps_dir) / f"map_{map_id}.csv"
    model_path = Path(gaussian_dir) / f"map_{map_id}.json"

    if not map_path.exists():
        raise FileNotFoundError(map_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    map_data = _load_map(map_path)
    with model_path.open("r", encoding="utf-8") as f:
        model = json.load(f)

    thresholds = [int(v) for v in model["thresholds"]]
    histories = _generate_histories(map_data, target, p, max_steps)

    # 1) MSE variation at fixed age
    mse_by_age = defaultdict(list)

    # 2) fixed (age, xhat, yhat): does Gaussian contain extra info?
    conditional = defaultdict(list)

    for history in histories:
        for row in history["trajectory"]:
            mse_by_age[row["age"]].append(row["mse"])
            conditional[(row["age"], row["xhat"], row["yhat"])].append(row["mse"])

    age_stats = {
        str(age): _summary(mse_by_age[age])
        for age in sorted(mse_by_age)
    }

    conditional_groups = []
    groups_with_multiple_mse = 0
    groups_with_multiple_levels = 0

    for (age, xhat, yhat), values in sorted(conditional.items()):
        stats = _summary(values)
        levels = sorted({_gaussian_level(v, thresholds) for v in values})

        if len({round(v, 14) for v in values}) > 1:
            groups_with_multiple_mse += 1
        if len(levels) > 1:
            groups_with_multiple_levels += 1

        conditional_groups.append({
            "age": age,
            "xhat": xhat,
            "yhat": yhat,
            "count": len(values),
            "mse_min": stats["min"],
            "mse_median": stats["median"],
            "mse_max": stats["max"],
            "mse_range": stats["range"],
            "mse_cv": stats["cv"],
            "levels": levels,
            "multiple_levels": len(levels) > 1,
        })

    # 3) Gaussian threshold crossing time vs periodic c-step update
    threshold_crossing = {}
    total_periodic_cases = 0
    different_from_periodic = 0
    crossing_differences = []

    for c, threshold in enumerate(thresholds, start=1):
        threshold_value = threshold / MSE_SCALE
        crossing_times = []

        for history in histories:
            crossing = None
            for row in history["trajectory"]:
                if row["age"] == 0:
                    continue
                if row["mse"] >= threshold_value:
                    crossing = row["age"]
                    break

            # Only compare trajectories long enough that periodic c is meaningful.
            max_age = history["trajectory"][-1]["age"]
            if max_age < c:
                continue

            total_periodic_cases += 1
            crossing_times.append(crossing)

            if crossing != c:
                different_from_periodic += 1

            if crossing is not None:
                crossing_differences.append(crossing - c)

        finite = [x for x in crossing_times if x is not None]
        threshold_crossing[str(c)] = {
            "threshold_mse": threshold_value,
            "eligible_trajectories": len(crossing_times),
            "crossing_time_summary": _summary(finite),
            "never_crossed": sum(x is None for x in crossing_times),
            "same_as_periodic": sum(x == c for x in crossing_times),
            "different_from_periodic": sum(x != c for x in crossing_times),
            "different_from_periodic_rate": (
                sum(x != c for x in crossing_times) / len(crossing_times)
                if crossing_times else 0.0
            ),
        }

    conditional_multi = [g for g in conditional_groups if g["count"] > 1]
    conditional_level_multi = [g for g in conditional_multi if g["multiple_levels"]]

    return {
        "map_id": map_id,
        "histories": len(histories),
        "thresholds": thresholds,
        "age_stats": age_stats,
        "conditional_summary": {
            "groups_total": len(conditional_groups),
            "groups_with_multiple_occurrences": len(conditional_multi),
            "groups_with_multiple_mse_values": groups_with_multiple_mse,
            "groups_with_multiple_urc_levels": groups_with_multiple_levels,
            "multi_level_rate_among_multi_occurrence_groups": (
                len(conditional_level_multi) / len(conditional_multi)
                if conditional_multi else 0.0
            ),
        },
        "conditional_groups": conditional_groups,
        "periodic_comparison": {
            "cases": total_periodic_cases,
            "different": different_from_periodic,
            "different_rate": (
                different_from_periodic / total_periodic_cases
                if total_periodic_cases else 0.0
            ),
            "crossing_age_minus_c_summary": _summary(crossing_differences),
            "by_threshold": threshold_crossing,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyse whether Gaussian uncertainty adds update information beyond a 1..10 step counter."
    )
    parser.add_argument("--first-map", type=int, default=10)
    parser.add_argument("--last-map", type=int, default=99)
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument("--gaussian-dir", default="gaussian_models")
    parser.add_argument("--target-x", type=int, default=9)
    parser.add_argument("--target-y", type=int, default=9)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output-json", default="gaussian_vs_point_estimate_analysis.json")
    parser.add_argument("--output-age-csv", default="gaussian_vs_point_estimate_by_age.csv")
    parser.add_argument("--output-map-csv", default="gaussian_vs_point_estimate_by_map.csv")
    args = parser.parse_args()

    results = []

    for map_id in range(args.first_map, args.last_map + 1):
        try:
            result = analyse_map(
                map_id=map_id,
                maps_dir=args.maps_dir,
                gaussian_dir=args.gaussian_dir,
                target=(args.target_x, args.target_y),
                p=args.p,
                max_steps=args.max_steps,
            )
        except FileNotFoundError as exc:
            print(f"skip map {map_id}: missing {exc}")
            continue

        results.append(result)
        cond = result["conditional_summary"]
        periodic = result["periodic_comparison"]

        print(
            f"map {map_id}: "
            f"conditional multi-level={cond['groups_with_multiple_urc_levels']}/"
            f"{cond['groups_with_multiple_occurrences']} "
            f"({cond['multi_level_rate_among_multi_occurrence_groups']:.3%}), "
            f"Gaussian-vs-periodic disagreement={periodic['different_rate']:.3%}"
        )

    if not results:
        raise RuntimeError("No maps analysed.")

    # Aggregate by age across all maps
    all_mse_by_age = defaultdict(list)
    total_cond_multi = 0
    total_cond_multi_levels = 0
    total_periodic_cases = 0
    total_periodic_diff = 0

    for result in results:
        total_cond_multi += result["conditional_summary"]["groups_with_multiple_occurrences"]
        total_cond_multi_levels += result["conditional_summary"]["groups_with_multiple_urc_levels"]
        total_periodic_cases += result["periodic_comparison"]["cases"]
        total_periodic_diff += result["periodic_comparison"]["different"]

        for group in result["conditional_groups"]:
            # Reconstruct only summary-level age distribution is not enough,
            # so use each group's median weighted by count for a compact aggregate.
            if group["count"] > 0 and group["mse_median"] is not None:
                all_mse_by_age[group["age"]].extend(
                    [group["mse_median"]] * group["count"]
                )

    aggregate_age_stats = {
        str(age): _summary(values)
        for age, values in sorted(all_mse_by_age.items())
    }

    output = {
        "config": {
            "first_map": args.first_map,
            "last_map": args.last_map,
            "maps_dir": args.maps_dir,
            "gaussian_dir": args.gaussian_dir,
            "target": [args.target_x, args.target_y],
            "p": args.p,
            "max_steps": args.max_steps,
        },
        "aggregate": {
            "maps": len(results),
            "conditional_multi_occurrence_groups": total_cond_multi,
            "conditional_groups_with_multiple_urc_levels": total_cond_multi_levels,
            "conditional_multi_level_rate": (
                total_cond_multi_levels / total_cond_multi
                if total_cond_multi else 0.0
            ),
            "gaussian_vs_periodic_cases": total_periodic_cases,
            "gaussian_vs_periodic_different": total_periodic_diff,
            "gaussian_vs_periodic_disagreement_rate": (
                total_periodic_diff / total_periodic_cases
                if total_periodic_cases else 0.0
            ),
            "age_stats": aggregate_age_stats,
        },
        "maps": results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    with open(args.output_map_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "map_id",
            "histories",
            "conditional_multi_occurrence_groups",
            "conditional_groups_with_multiple_urc_levels",
            "conditional_multi_level_rate",
            "periodic_cases",
            "periodic_different",
            "periodic_disagreement_rate",
        ])
        for result in results:
            cond = result["conditional_summary"]
            periodic = result["periodic_comparison"]
            writer.writerow([
                result["map_id"],
                result["histories"],
                cond["groups_with_multiple_occurrences"],
                cond["groups_with_multiple_urc_levels"],
                cond["multi_level_rate_among_multi_occurrence_groups"],
                periodic["cases"],
                periodic["different"],
                periodic["different_rate"],
            ])

    with open(args.output_age_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "age", "count", "min_mse", "median_mse", "mean_mse",
            "max_mse", "range_mse", "std_mse", "cv_mse"
        ])
        for age, stats in sorted(
            ((int(age), stats) for age, stats in aggregate_age_stats.items())
        ):
            writer.writerow([
                age,
                stats["count"],
                stats["min"],
                stats["median"],
                stats["mean"],
                stats["max"],
                stats["range"],
                stats["std"],
                stats["cv"],
            ])

    print("\nAggregate:")
    print(
        "  fixed-(age,xhat,yhat) groups with multiple URC levels: "
        f"{total_cond_multi_levels}/{total_cond_multi} "
        f"({output['aggregate']['conditional_multi_level_rate']:.3%})"
    )
    print(
        "  Gaussian threshold crossing != periodic c-step update: "
        f"{total_periodic_diff}/{total_periodic_cases} "
        f"({output['aggregate']['gaussian_vs_periodic_disagreement_rate']:.3%})"
    )
    print(f"\nWrote {args.output_json}")
    print(f"Wrote {args.output_map_csv}")
    print(f"Wrote {args.output_age_csv}")


if __name__ == "__main__":
    main()
