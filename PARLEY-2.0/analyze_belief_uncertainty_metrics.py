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


def _propagate_absolute(belief, action, n, p):
    result = defaultdict(float)
    for (x, y), prior in belief.items():
        for actual in DIRECTIONS:
            q = 1.0 - 3.0 * p if actual == action else p
            nx, ny = _move(x, y, actual, n)
            result[(nx, ny)] += prior * q
    return dict(result)


def _gini(belief):
    return 1.0 - sum(prob * prob for prob in belief.values())


def _entropy(belief):
    return -sum(prob * math.log(prob) for prob in belief.values() if prob > 0.0)


def _map_uncertainty(belief):
    return 1.0 - max(belief.values())


def _spatial_mse(belief, xhat, yhat):
    return sum(
        prob * ((x - xhat) ** 2 + (y - yhat) ** 2)
        for (x, y), prob in belief.items()
    )


def _spatial_mae(belief, xhat, yhat):
    return sum(
        prob * math.sqrt((x - xhat) ** 2 + (y - yhat) ** 2)
        for (x, y), prob in belief.items()
    )


METRICS = {
    "gini": _gini,
    "entropy": _entropy,
    "map_uncertainty": _map_uncertainty,
}


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

    vals = sorted(float(v) for v in values)
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    variance = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(variance)
    cv = std / abs(mean) if abs(mean) > 1e-15 else 0.0

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


def _pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-20 or vy <= 1e-20:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _load_map(path):
    with path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


def _generate_records(map_data, target, p, max_steps):
    n = len(map_data) - 1
    controller = _controller(map_data, target)
    records = []

    for sx in range(len(map_data)):
        for sy in range(len(map_data)):
            if int(map_data[sx][sy]) > 9:
                continue

            belief = {(sx, sy): 1.0}
            xhat, yhat = sx, sy

            for age in range(max_steps + 1):
                metrics = {
                    "gini": _gini(belief),
                    "entropy": _entropy(belief),
                    "map_uncertainty": _map_uncertainty(belief),
                    "spatial_mse": _spatial_mse(belief, xhat, yhat),
                    "spatial_mae": _spatial_mae(belief, xhat, yhat),
                }

                records.append({
                    "age": age,
                    "xhat": xhat,
                    "yhat": yhat,
                    **metrics,
                })

                if age >= max_steps or (xhat, yhat) == target:
                    break

                action = _direction(controller, xhat, yhat)
                if action is None:
                    break

                belief = _propagate_absolute(belief, action, n, p)
                xhat, yhat = _move(xhat, yhat, action, n)

    return records


def analyse_map(map_id, maps_dir, target, p, max_steps):
    map_path = Path(maps_dir) / f"map_{map_id}.csv"
    if not map_path.exists():
        raise FileNotFoundError(map_path)

    map_data = _load_map(map_path)
    records = _generate_records(map_data, target, p, max_steps)

    metric_names = [
        "gini",
        "entropy",
        "map_uncertainty",
        "spatial_mse",
        "spatial_mae",
    ]

    ages = [r["age"] for r in records]

    metric_results = {}
    for metric in metric_names:
        values = [r[metric] for r in records]
        corr = _pearson(ages, values)

        by_age = defaultdict(list)
        conditional = defaultdict(list)

        for r in records:
            by_age[r["age"]].append(r[metric])
            conditional[(r["age"], r["xhat"], r["yhat"])].append(r[metric])

        age_stats = {
            str(age): _summary(vals)
            for age, vals in sorted(by_age.items())
        }

        conditional_groups = []
        multi_occurrence = 0
        varying_groups = 0
        cvs = []
        ranges = []

        for (age, xhat, yhat), vals in sorted(conditional.items()):
            stats = _summary(vals)
            unique_vals = {round(v, 14) for v in vals}
            if len(vals) > 1:
                multi_occurrence += 1
                cvs.append(stats["cv"])
                ranges.append(stats["range"])
                if len(unique_vals) > 1:
                    varying_groups += 1

            conditional_groups.append({
                "age": age,
                "xhat": xhat,
                "yhat": yhat,
                "count": len(vals),
                "min": stats["min"],
                "median": stats["median"],
                "max": stats["max"],
                "range": stats["range"],
                "cv": stats["cv"],
                "varies": len(unique_vals) > 1,
            })

        metric_results[metric] = {
            "corr_age": corr,
            "overall": _summary(values),
            "by_age": age_stats,
            "conditional_summary": {
                "groups_total": len(conditional_groups),
                "groups_with_multiple_occurrences": multi_occurrence,
                "groups_with_variation": varying_groups,
                "variation_rate_among_multi_occurrence_groups": (
                    varying_groups / multi_occurrence if multi_occurrence else 0.0
                ),
                "median_conditional_cv": _summary(cvs)["median"] if cvs else 0.0,
                "mean_conditional_cv": _summary(cvs)["mean"] if cvs else 0.0,
                "median_conditional_range": _summary(ranges)["median"] if ranges else 0.0,
                "max_conditional_range": _summary(ranges)["max"] if ranges else 0.0,
            },
            "conditional_groups": conditional_groups,
        }

    return {
        "map_id": map_id,
        "records": len(records),
        "metrics": metric_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare belief uncertainty metrics for whether they contain "
            "information beyond age and estimated position."
        )
    )
    parser.add_argument("--first-map", type=int, default=10)
    parser.add_argument("--last-map", type=int, default=99)
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument("--target-x", type=int, default=9)
    parser.add_argument("--target-y", type=int, default=9)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--output-json",
        default="belief_uncertainty_metric_analysis.json",
    )
    parser.add_argument(
        "--output-summary-csv",
        default="belief_uncertainty_metric_summary.csv",
    )
    parser.add_argument(
        "--output-age-csv",
        default="belief_uncertainty_metric_by_age.csv",
    )
    args = parser.parse_args()

    results = []
    for map_id in range(args.first_map, args.last_map + 1):
        try:
            result = analyse_map(
                map_id=map_id,
                maps_dir=args.maps_dir,
                target=(args.target_x, args.target_y),
                p=args.p,
                max_steps=args.max_steps,
            )
        except FileNotFoundError as exc:
            print(f"skip map {map_id}: missing {exc}")
            continue

        results.append(result)

        parts = []
        for metric in (
            "gini",
            "entropy",
            "map_uncertainty",
            "spatial_mse",
            "spatial_mae",
        ):
            info = result["metrics"][metric]
            cond = info["conditional_summary"]
            parts.append(
                f"{metric}: corr(age)={info['corr_age']:.4f}, "
                f"condCV={cond['median_conditional_cv']:.4f}, "
                f"vary={cond['variation_rate_among_multi_occurrence_groups']:.1%}"
            )
        print(f"map {map_id}: " + " | ".join(parts))

    if not results:
        raise RuntimeError("No maps analysed.")

    metric_names = [
        "gini",
        "entropy",
        "map_uncertainty",
        "spatial_mse",
        "spatial_mae",
    ]

    aggregate = {}
    for metric in metric_names:
        corrs = []
        med_cvs = []
        mean_cvs = []
        med_ranges = []
        max_ranges = []
        varying = 0
        multi = 0

        for result in results:
            info = result["metrics"][metric]
            cond = info["conditional_summary"]
            if info["corr_age"] is not None:
                corrs.append(info["corr_age"])
            med_cvs.append(cond["median_conditional_cv"])
            mean_cvs.append(cond["mean_conditional_cv"])
            med_ranges.append(cond["median_conditional_range"])
            max_ranges.append(cond["max_conditional_range"])
            varying += cond["groups_with_variation"]
            multi += cond["groups_with_multiple_occurrences"]

        aggregate[metric] = {
            "maps": len(results),
            "corr_age_across_maps": _summary(corrs),
            "median_conditional_cv_across_maps": _summary(med_cvs),
            "mean_conditional_cv_across_maps": _summary(mean_cvs),
            "median_conditional_range_across_maps": _summary(med_ranges),
            "max_conditional_range_across_maps": _summary(max_ranges),
            "conditional_groups_with_variation": varying,
            "conditional_multi_occurrence_groups": multi,
            "conditional_variation_rate": varying / multi if multi else 0.0,
        }

    output = {
        "config": {
            "first_map": args.first_map,
            "last_map": args.last_map,
            "maps_dir": args.maps_dir,
            "target": [args.target_x, args.target_y],
            "p": args.p,
            "max_steps": args.max_steps,
        },
        "aggregate": aggregate,
        "maps": results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    with open(args.output_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric",
            "mean_corr_age",
            "median_corr_age",
            "mean_map_median_conditional_cv",
            "median_map_median_conditional_cv",
            "conditional_variation_rate",
            "max_conditional_range_seen",
        ])
        for metric in metric_names:
            agg = aggregate[metric]
            writer.writerow([
                metric,
                agg["corr_age_across_maps"]["mean"],
                agg["corr_age_across_maps"]["median"],
                agg["median_conditional_cv_across_maps"]["mean"],
                agg["median_conditional_cv_across_maps"]["median"],
                agg["conditional_variation_rate"],
                agg["max_conditional_range_across_maps"]["max"],
            ])

    with open(args.output_age_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "map_id",
            "metric",
            "age",
            "count",
            "min",
            "median",
            "mean",
            "max",
            "range",
            "std",
            "cv",
        ])
        for result in results:
            for metric in metric_names:
                for age_str, stats in result["metrics"][metric]["by_age"].items():
                    writer.writerow([
                        result["map_id"],
                        metric,
                        int(age_str),
                        stats["count"],
                        stats["min"],
                        stats["median"],
                        stats["mean"],
                        stats["max"],
                        stats["range"],
                        stats["std"],
                        stats["cv"],
                    ])

    print("\n" + "=" * 100)
    print("Aggregate comparison")
    print("=" * 100)
    print(
        f"{'metric':<18} {'corr(age)':>12} {'cond CV':>12} "
        f"{'vary groups':>14} {'max range':>12}"
    )

    for metric in metric_names:
        agg = aggregate[metric]
        print(
            f"{metric:<18} "
            f"{agg['corr_age_across_maps']['mean']:>12.4f} "
            f"{agg['median_conditional_cv_across_maps']['mean']:>12.4f} "
            f"{agg['conditional_variation_rate']:>13.2%} "
            f"{agg['max_conditional_range_across_maps']['max']:>12.6f}"
        )

    print(f"\nWrote {args.output_json}")
    print(f"Wrote {args.output_summary_csv}")
    print(f"Wrote {args.output_age_csv}")


if __name__ == "__main__":
    main()
