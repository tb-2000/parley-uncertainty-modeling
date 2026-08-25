#!/usr/bin/env python3
"""
compare_gaussian_metrics_full.py

Vergleicht die beiden Distanzmetriken der neuen Gaussian-K=100-Pipeline:

    1) Frobenius
    2) Bures-Wasserstein (geschlossene 2x2-Formel)

Die Analyse orientiert sich an full_gaussian_representatives.py:
- gleiche erreichbare Gaussian-Zustaende unter dem Dijkstra-MAPE-Controller
- gleicher max_steps-Horizont
- gleiche gewichtete k-Medoids-Logik
- Sigma=0 ist immer Repraesentant / gstate 0
- gleiche repraesentantenbasierte Transition:
      R_i + Q(xhat,yhat,a) -> nearest representative

Standard:
    Maps 10..99
    K = 100
    max_steps = 10

Ausgaben:
    gaussian_metric_comparison_full/
        global_summary.csv
        per_map.csv
        comparison.json

Gemessen werden u.a.:
- state_count
- weighted clustering error in BOTH metrics
- max clustering error in BOTH metrics
- mean/max transition projection error in BOTH metrics
- unique gstate edges
- self-loop fraction
- number of reachable raw Sigma candidates

Wichtig:
Die "native" Fehler einer Metrik sind nicht direkt miteinander vergleichbar,
weil Frobenius und Bures unterschiedliche Skalen haben. Deshalb wird jede
gefundene Repraesentantenmenge in BEIDEN Metriken ausgewertet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import full_gaussian_representatives as fg


METRICS = (
    "frobenius",
    "bures_wasserstein",
)


def _evaluate_assignment(
    candidates,
    representatives,
    assignment_metric,
    eval_metric,
):
    """
    Assign each raw Sigma to nearest representative according to assignment_metric,
    but measure the resulting error in eval_metric.
    """
    total_weight = 0
    weighted_sum = 0.0
    weighted_sq = 0.0
    max_error = 0.0

    for item in candidates:
        sigma = item["sigma"]
        weight = int(item["weight"])

        state_id = fg._nearest(
            sigma,
            representatives,
            assignment_metric,
        )

        representative = representatives[
            state_id
        ]

        error = fg._distance(
            sigma,
            representative,
            eval_metric,
        )

        total_weight += weight
        weighted_sum += weight * error
        weighted_sq += weight * error * error
        max_error = max(
            max_error,
            error,
        )

    if total_weight == 0:
        return {
            "weighted_mean": 0.0,
            "weighted_rmse": 0.0,
            "max": 0.0,
        }

    return {
        "weighted_mean": (
            weighted_sum / total_weight
        ),
        "weighted_rmse": math.sqrt(
            weighted_sq / total_weight
        ),
        "max": max_error,
    }


def _transition_rows(
    representatives,
    map_data,
    target,
    p,
    metric,
):
    """
    Build exactly the same representative-state transition system as the
    Gaussian model builder, but retain projection errors for analysis.
    """
    controller = fg._controller(
        map_data,
        target,
    )
    n = len(map_data) - 1

    rows = []

    for x in range(
        len(map_data)
    ):
        for y in range(
            len(map_data)
        ):
            action = fg._direction(
                controller,
                x,
                y,
            )

            if action is None:
                continue

            for (
                state_id,
                sigma,
            ) in enumerate(
                representatives
            ):
                q = fg._motion_covariance(
                    x,
                    y,
                    action,
                    n,
                    p,
                )

                successor_sigma = (
                    fg._sigma_key(
                        fg._add_sigma(
                            sigma,
                            q,
                        )
                    )
                )

                next_state = fg._nearest(
                    successor_sigma,
                    representatives,
                    metric,
                )

                rows.append(
                    {
                        "xhat": x,
                        "yhat": y,
                        "action": action,
                        "gstate": state_id,
                        "gstate_next": next_state,
                        "successor_sigma": successor_sigma,
                    }
                )

    return rows


def _evaluate_transition_projection(
    rows,
    representatives,
    eval_metric,
):
    if not rows:
        return {
            "mean": 0.0,
            "rmse": 0.0,
            "max": 0.0,
        }

    errors = []

    for row in rows:
        representative = representatives[
            row["gstate_next"]
        ]

        error = fg._distance(
            row["successor_sigma"],
            representative,
            eval_metric,
        )

        errors.append(
            error
        )

    return {
        "mean": (
            sum(errors)
            / len(errors)
        ),
        "rmse": math.sqrt(
            sum(e * e for e in errors)
            / len(errors)
        ),
        "max": max(errors),
    }


def _transition_structure(rows):
    edges = {
        (
            row["gstate"],
            row["gstate_next"],
        )
        for row in rows
    }

    self_loops = sum(
        1
        for row in rows
        if (
            row["gstate"]
            == row["gstate_next"]
        )
    )

    return {
        "transition_rows": len(rows),
        "unique_gstate_edges": len(
            edges
        ),
        "self_loop_fraction": (
            self_loops / len(rows)
            if rows
            else 0.0
        ),
    }


def _analyse_metric(
    metric,
    candidates,
    trace_by_age,
    map_data,
    target,
    p,
    k,
    max_steps,
):
    representatives = fg._cluster(
        candidates,
        min(
            k,
            len(candidates),
        ),
        metric=metric,
        max_iter=100,
    )

    state_count = len(
        representatives
    )

    # Verify reset state.
    if representatives[0] != (
        0.0,
        0.0,
        0.0,
    ):
        raise ValueError(
            f"{metric}: representative 0 is not Sigma=0"
        )

    frobenius_cluster = (
        _evaluate_assignment(
            candidates,
            representatives,
            assignment_metric=metric,
            eval_metric="frobenius",
        )
    )

    bures_cluster = (
        _evaluate_assignment(
            candidates,
            representatives,
            assignment_metric=metric,
            eval_metric="bures_wasserstein",
        )
    )

    rows = _transition_rows(
        representatives,
        map_data,
        target,
        p,
        metric,
    )

    frobenius_transition = (
        _evaluate_transition_projection(
            rows,
            representatives,
            "frobenius",
        )
    )

    bures_transition = (
        _evaluate_transition_projection(
            rows,
            representatives,
            "bures_wasserstein",
        )
    )

    structure = (
        _transition_structure(
            rows
        )
    )

    thresholds = fg._thresholds(
        trace_by_age,
        max_steps,
        scale=fg.TRACE_SCALE,
    )

    return {
        "metric": metric,
        "state_count": state_count,
        "thresholds": thresholds,
        "cluster_error_frobenius": (
            frobenius_cluster
        ),
        "cluster_error_bures": (
            bures_cluster
        ),
        "transition_error_frobenius": (
            frobenius_transition
        ),
        "transition_error_bures": (
            bures_transition
        ),
        "transition_structure": structure,
    }


def _load_map(path):
    rows = []

    with path.open(
        "r",
        newline="",
    ) as file:
        rows.extend(
            csv.reader(file)
        )

    if not rows:
        raise ValueError(
            f"Empty map: {path}"
        )

    transposed = list(
        zip(*rows)
    )

    return [
        row[::-1]
        for row in transposed
    ]


def analyse_map(
    map_id,
    map_path,
    target,
    p,
    k,
    max_steps,
):
    map_data = _load_map(
        map_path
    )

    (
        candidates,
        trace_by_age,
        _controller,
    ) = fg._generate_records(
        map_data,
        target,
        p,
        max_steps,
    )

    result = {
        "map": map_id,
        "raw_sigma_candidates": len(
            candidates
        ),
        "metrics": {},
    }

    for metric in METRICS:
        print(
            f"map {map_id}: clustering with {metric}..."
        )

        result[
            "metrics"
        ][metric] = _analyse_metric(
            metric=metric,
            candidates=candidates,
            trace_by_age=trace_by_age,
            map_data=map_data,
            target=target,
            p=p,
            k=k,
            max_steps=max_steps,
        )

    return result


def _flatten_map_result(
    result,
):
    rows = []

    for (
        metric,
        data,
    ) in result[
        "metrics"
    ].items():
        row = {
            "map": result["map"],
            "metric": metric,
            "raw_sigma_candidates": result[
                "raw_sigma_candidates"
            ],
            "state_count": data[
                "state_count"
            ],

            "cluster_frobenius_weighted_mean":
                data[
                    "cluster_error_frobenius"
                ][
                    "weighted_mean"
                ],
            "cluster_frobenius_weighted_rmse":
                data[
                    "cluster_error_frobenius"
                ][
                    "weighted_rmse"
                ],
            "cluster_frobenius_max":
                data[
                    "cluster_error_frobenius"
                ][
                    "max"
                ],

            "cluster_bures_weighted_mean":
                data[
                    "cluster_error_bures"
                ][
                    "weighted_mean"
                ],
            "cluster_bures_weighted_rmse":
                data[
                    "cluster_error_bures"
                ][
                    "weighted_rmse"
                ],
            "cluster_bures_max":
                data[
                    "cluster_error_bures"
                ][
                    "max"
                ],

            "transition_frobenius_mean":
                data[
                    "transition_error_frobenius"
                ][
                    "mean"
                ],
            "transition_frobenius_rmse":
                data[
                    "transition_error_frobenius"
                ][
                    "rmse"
                ],
            "transition_frobenius_max":
                data[
                    "transition_error_frobenius"
                ][
                    "max"
                ],

            "transition_bures_mean":
                data[
                    "transition_error_bures"
                ][
                    "mean"
                ],
            "transition_bures_rmse":
                data[
                    "transition_error_bures"
                ][
                    "rmse"
                ],
            "transition_bures_max":
                data[
                    "transition_error_bures"
                ][
                    "max"
                ],

            "unique_gstate_edges":
                data[
                    "transition_structure"
                ][
                    "unique_gstate_edges"
                ],
            "self_loop_fraction":
                data[
                    "transition_structure"
                ][
                    "self_loop_fraction"
                ],
            "transition_rows":
                data[
                    "transition_structure"
                ][
                    "transition_rows"
                ],
        }

        rows.append(
            row
        )

    return rows


def _aggregate(
    per_map_rows,
):
    summary = []

    for metric in METRICS:
        rows = [
            row
            for row in per_map_rows
            if row["metric"] == metric
        ]

        summary.append(
            {
                "metric": metric,
                "maps": len(rows),
                "mean_raw_sigma_candidates": mean(
                    row[
                        "raw_sigma_candidates"
                    ]
                    for row in rows
                ),
                "mean_state_count": mean(
                    row[
                        "state_count"
                    ]
                    for row in rows
                ),
                "max_state_count": max(
                    row[
                        "state_count"
                    ]
                    for row in rows
                ),

                "mean_cluster_frobenius":
                    mean(
                        row[
                            "cluster_frobenius_weighted_mean"
                        ]
                        for row in rows
                    ),
                "mean_cluster_bures":
                    mean(
                        row[
                            "cluster_bures_weighted_mean"
                        ]
                        for row in rows
                    ),

                "mean_transition_frobenius":
                    mean(
                        row[
                            "transition_frobenius_mean"
                        ]
                        for row in rows
                    ),
                "mean_transition_bures":
                    mean(
                        row[
                            "transition_bures_mean"
                        ]
                        for row in rows
                    ),

                "mean_unique_gstate_edges":
                    mean(
                        row[
                            "unique_gstate_edges"
                        ]
                        for row in rows
                    ),
                "mean_self_loop_fraction":
                    mean(
                        row[
                            "self_loop_fraction"
                        ]
                        for row in rows
                    ),

                "max_cluster_frobenius":
                    max(
                        row[
                            "cluster_frobenius_max"
                        ]
                        for row in rows
                    ),
                "max_cluster_bures":
                    max(
                        row[
                            "cluster_bures_max"
                        ]
                        for row in rows
                    ),
                "max_transition_frobenius":
                    max(
                        row[
                            "transition_frobenius_max"
                        ]
                        for row in rows
                    ),
                "max_transition_bures":
                    max(
                        row[
                            "transition_bures_max"
                        ]
                        for row in rows
                    ),
            }
        )

    return summary


def _write_csv(
    path,
    rows,
):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("maps"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "gaussian_metric_comparison_full"
        ),
    )

    parser.add_argument(
        "--start-map",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--end-map",
        type=int,
        default=99,
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
        "--k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = (
        args.target_x,
        args.target_y,
    )

    complete = []
    per_map_rows = []

    for map_id in range(
        args.start_map,
        args.end_map + 1,
    ):
        map_path = (
            args.maps_dir
            / f"map_{map_id}.csv"
        )

        if not map_path.exists():
            print(
                f"skip map {map_id}: missing {map_path}"
            )
            continue

        result = analyse_map(
            map_id=map_id,
            map_path=map_path,
            target=target,
            p=args.p,
            k=args.k,
            max_steps=args.max_steps,
        )

        complete.append(
            result
        )

        rows = _flatten_map_result(
            result
        )

        per_map_rows.extend(
            rows
        )

        frob = result[
            "metrics"
        ][
            "frobenius"
        ]

        bures = result[
            "metrics"
        ][
            "bures_wasserstein"
        ]

        print(
            f"map {map_id}: "
            f"F-cluster(F)={frob['cluster_error_frobenius']['weighted_mean']:.6f}, "
            f"F-cluster(BW)={frob['cluster_error_bures']['weighted_mean']:.6f} | "
            f"BW-cluster(F)={bures['cluster_error_frobenius']['weighted_mean']:.6f}, "
            f"BW-cluster(BW)={bures['cluster_error_bures']['weighted_mean']:.6f}"
        )

    if not per_map_rows:
        raise RuntimeError(
            "No maps analysed."
        )

    global_summary = _aggregate(
        per_map_rows
    )

    _write_csv(
        args.output_dir
        / "per_map.csv",
        per_map_rows,
    )

    _write_csv(
        args.output_dir
        / "global_summary.csv",
        global_summary,
    )

    output = {
        "settings": {
            "maps": [
                args.start_map,
                args.end_map,
            ],
            "target": list(target),
            "p": args.p,
            "K": args.k,
            "max_steps": args.max_steps,
            "metrics": list(METRICS),
        },
        "global_summary": global_summary,
        "per_map": complete,
    }

    with (
        args.output_dir
        / "comparison.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print("GLOBAL SUMMARY")

    for row in global_summary:
        print(
            f"{row['metric']:>18} | "
            f"cluster F={row['mean_cluster_frobenius']:.6f} | "
            f"cluster BW={row['mean_cluster_bures']:.6f} | "
            f"transition F={row['mean_transition_frobenius']:.6f} | "
            f"transition BW={row['mean_transition_bures']:.6f} | "
            f"mean edges={row['mean_unique_gstate_edges']:.1f}"
        )

    print("=" * 80)
    print(
        f"Results written to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
