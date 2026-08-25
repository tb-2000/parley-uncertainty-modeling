#!/usr/bin/env python3
"""
full_gaussian_representatives_frobenius_knee.py

Bestimmt einen geeigneten State-Budget-Wert K fuer das diskretisierte
Gaussian Knowledge Model mit FROBENIUS-DISTANZ.

Das Skript ist bewusst analog zu:
    full_belief_representatives_geometric_knee_per_map.py

Vorgehen:
  1. Erzeuge pro Map alle unter dem Dijkstra-MAPE-Controller erreichbaren
     Raw-Kovarianzmatrizen Sigma bis max_steps.
  2. Teste mehrere K-Werte mit derselben gewichteten k-Medoids-Logik wie
     full_gaussian_representatives.py.
  3. Berechne pro K:
       - gewichteten mittleren Frobenius-Quantisierungsfehler
       - maximalen Frobenius-Fehler
       - Clustering-Objective
       - Konvergenzdiagnostik
  4. Bestimme pro Map einen geometrischen Knee Point.
  5. Aggregiere dieselben K-Werte ueber alle Maps und bestimme den
     GLOBALEN geometrischen Knee Point.

Geometrischer Knee Point:
  - K und mean Frobenius error werden unabhaengig auf [0,1] normiert.
  - Die Gerade zwischen erstem und letztem Punkt wird gebildet.
  - Der innere Punkt mit groesstem senkrechtem Abstand zur Geraden ist
    der Knee Point.

Standard-K-Werte:
    25, 50, 75, 100, 125, 150, 175, 200

Standard:
    maps 10..99
    target (9,9)
    p=0.01
    max_steps=10

Ausgaben:
    gaussian_frobenius_knee/
        map_<id>_k_evaluation.csv
        knee_by_map.csv
        aggregated_k_evaluation.csv
        summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import full_gaussian_representatives as fg


METRIC = "frobenius"


def _read_map_data(map_path):
    rows = []

    with open(
        map_path,
        "r",
        newline="",
    ) as file:
        rows.extend(
            csv.reader(file)
        )

    if not rows:
        raise ValueError(
            f"Empty map: {map_path}"
        )

    transposed = list(
        zip(*rows)
    )

    return [
        row[::-1]
        for row in transposed
    ]


def _distance_matrix(sigmas):
    count = len(sigmas)

    matrix = [
        [0.0] * count
        for _ in range(count)
    ]

    for i in range(count):
        for j in range(
            i + 1,
            count,
        ):
            distance = fg._frobenius(
                sigmas[i],
                sigmas[j],
            )

            matrix[i][j] = distance
            matrix[j][i] = distance

    return matrix


def _cluster_with_diagnostics(
    candidates,
    k,
    max_iter=100,
    objective_tolerance=1e-9,
):
    """
    Same weighted k-medoids idea as full_gaussian_representatives.py,
    but returns diagnostics needed for K evaluation.
    """
    sigmas = [
        item["sigma"]
        for item in candidates
    ]

    weights = [
        item["weight"]
        for item in candidates
    ]

    if not sigmas:
        raise ValueError(
            "No Gaussian candidates."
        )

    if k < 2:
        raise ValueError(
            "K must be >= 2."
        )

    if k > len(sigmas):
        raise ValueError(
            f"K={k} > candidate count={len(sigmas)}"
        )

    zero_sigma = (
        0.0,
        0.0,
        0.0,
    )

    try:
        zero_index = sigmas.index(
            zero_sigma
        )
    except ValueError as exc:
        raise ValueError(
            "Sigma=0 reset state missing."
        ) from exc

    matrix = _distance_matrix(
        sigmas
    )

    if len(sigmas) <= k:
        medoids = list(
            range(len(sigmas))
        )
        objective = 0.0
        iterations = 0
        converged = True
    else:
        # Farthest-first initialisation, reset state fixed.
        medoids = [
            zero_index
        ]

        nearest = [
            matrix[i][zero_index]
            for i in range(
                len(sigmas)
            )
        ]

        while len(medoids) < k:
            next_medoid = max(
                (
                    i
                    for i in range(
                        len(sigmas)
                    )
                    if i not in medoids
                ),
                key=lambda i: (
                    nearest[i],
                    -i,
                ),
            )

            medoids.append(
                next_medoid
            )

            for i in range(
                len(sigmas)
            ):
                nearest[i] = min(
                    nearest[i],
                    matrix[i][next_medoid],
                )

        def assign(
            current_medoids,
        ):
            clusters = {
                medoid: []
                for medoid in current_medoids
            }

            assignments = []

            for i in range(
                len(sigmas)
            ):
                medoid = min(
                    current_medoids,
                    key=lambda candidate: (
                        matrix[i][candidate],
                        candidate,
                    ),
                )

                assignments.append(
                    medoid
                )

                clusters[
                    medoid
                ].append(
                    i
                )

            return (
                assignments,
                clusters,
            )

        def weighted_objective(
            assignments,
        ):
            return sum(
                weights[i]
                * matrix[
                    i
                ][
                    assignments[i]
                ]
                for i in range(
                    len(sigmas)
                )
            )

        previous_objective = None
        objective = None
        converged = False
        iterations = 0

        for iteration in range(
            1,
            max_iter + 1,
        ):
            iterations = iteration

            (
                assignments,
                clusters,
            ) = assign(
                medoids
            )

            current_objective = (
                weighted_objective(
                    assignments
                )
            )

            refined = []

            for medoid in medoids:
                members = clusters[
                    medoid
                ]

                if medoid == zero_index:
                    refined.append(
                        medoid
                    )
                    continue

                best = min(
                    members,
                    key=lambda candidate: (
                        sum(
                            weights[j]
                            * matrix[
                                candidate
                            ][
                                j
                            ]
                            for j in members
                        ),
                        candidate,
                    ),
                )

                refined.append(
                    best
                )

            if (
                set(refined)
                == set(medoids)
            ):
                medoids = refined
                objective = current_objective
                converged = True
                break

            (
                new_assignments,
                _,
            ) = assign(
                refined
            )

            new_objective = (
                weighted_objective(
                    new_assignments
                )
            )

            if (
                previous_objective
                is not None
            ):
                denominator = max(
                    abs(
                        previous_objective
                    ),
                    1e-15,
                )

                relative_improvement = (
                    previous_objective
                    - new_objective
                ) / denominator

                if (
                    relative_improvement >= 0.0
                    and relative_improvement
                    < objective_tolerance
                ):
                    medoids = refined
                    objective = new_objective
                    converged = True
                    break

            medoids = refined
            previous_objective = (
                new_objective
            )
            objective = new_objective

        if objective is None:
            (
                assignments,
                _,
            ) = assign(
                medoids
            )

            objective = (
                weighted_objective(
                    assignments
                )
            )

    # Put Sigma=0 first.
    medoids = [
        zero_index
    ] + sorted(
        medoid
        for medoid in medoids
        if medoid != zero_index
    )

    return {
        "medoid_indices": medoids,
        "representatives": [
            sigmas[index]
            for index in medoids
        ],
        "distance_matrix": matrix,
        "objective": objective,
        "iterations": iterations,
        "converged": converged,
    }


def _mean_assignment_error(
    candidates,
    medoid_indices,
    distance_matrix,
):
    weights = [
        item["weight"]
        for item in candidates
    ]

    weighted_sum = 0.0
    total_weight = 0.0
    maximum = 0.0

    for i in range(
        len(candidates)
    ):
        distance = min(
            distance_matrix[
                i
            ][
                medoid
            ]
            for medoid in medoid_indices
        )

        weighted_sum += (
            weights[i]
            * distance
        )

        total_weight += (
            weights[i]
        )

        maximum = max(
            maximum,
            distance,
        )

    mean_error = (
        weighted_sum
        / total_weight
        if total_weight
        else 0.0
    )

    return (
        mean_error,
        maximum,
    )


def _weighted_silhouette_score(
    candidates,
    medoid_indices,
    distance_matrix,
):
    """
    Optional diagnostic analogous to the belief Knee script.
    """
    if (
        len(medoid_indices) <= 1
        or len(candidates) <= 1
    ):
        return float("nan")

    weights = [
        item["weight"]
        for item in candidates
    ]

    medoid_to_cluster = {
        medoid: cluster_id
        for (
            cluster_id,
            medoid,
        ) in enumerate(
            medoid_indices
        )
    }

    assignments = []
    clusters = {
        cluster_id: []
        for cluster_id in range(
            len(medoid_indices)
        )
    }

    for i in range(
        len(candidates)
    ):
        medoid = min(
            medoid_indices,
            key=lambda candidate: (
                distance_matrix[
                    i
                ][
                    candidate
                ],
                candidate,
            ),
        )

        cluster_id = (
            medoid_to_cluster[
                medoid
            ]
        )

        assignments.append(
            cluster_id
        )
        clusters[
            cluster_id
        ].append(
            i
        )

    silhouette_values = []

    for (
        i,
        own_cluster,
    ) in enumerate(
        assignments
    ):
        own_members = clusters[
            own_cluster
        ]

        if len(
            own_members
        ) <= 1:
            silhouette_values.append(
                0.0
            )
            continue

        own_weight = sum(
            weights[j]
            for j in own_members
            if j != i
        )

        if own_weight <= 0:
            a_i = 0.0
        else:
            a_i = sum(
                weights[j]
                * distance_matrix[
                    i
                ][
                    j
                ]
                for j in own_members
                if j != i
            ) / own_weight

        b_i = None

        for (
            cluster_id,
            members,
        ) in clusters.items():
            if (
                cluster_id == own_cluster
                or not members
            ):
                continue

            cluster_weight = sum(
                weights[j]
                for j in members
            )

            mean_distance = sum(
                weights[j]
                * distance_matrix[
                    i
                ][
                    j
                ]
                for j in members
            ) / cluster_weight

            if (
                b_i is None
                or mean_distance < b_i
            ):
                b_i = mean_distance

        if b_i is None:
            silhouette = 0.0
        else:
            denominator = max(
                a_i,
                b_i,
            )

            silhouette = (
                0.0
                if denominator <= 1e-15
                else (
                    b_i - a_i
                ) / denominator
            )

        silhouette_values.append(
            silhouette
        )

    total_weight = sum(
        weights
    )

    return sum(
        weights[i]
        * silhouette_values[i]
        for i in range(
            len(candidates)
        )
    ) / total_weight


def evaluate_k_values(
    candidates,
    k_values,
    max_iter=100,
    objective_tolerance=1e-9,
):
    results = []

    for k in sorted(
        set(k_values)
    ):
        if (
            k < 2
            or k > len(candidates)
        ):
            continue

        clustered = (
            _cluster_with_diagnostics(
                candidates,
                k=k,
                max_iter=max_iter,
                objective_tolerance=objective_tolerance,
            )
        )

        (
            mean_error,
            max_error,
        ) = _mean_assignment_error(
            candidates,
            clustered[
                "medoid_indices"
            ],
            clustered[
                "distance_matrix"
            ],
        )

        silhouette = (
            _weighted_silhouette_score(
                candidates,
                clustered[
                    "medoid_indices"
                ],
                clustered[
                    "distance_matrix"
                ],
            )
        )

        results.append(
            {
                "k": k,
                "mean_frobenius_error":
                    mean_error,
                "max_frobenius_error":
                    max_error,
                "silhouette":
                    silhouette,
                "objective":
                    clustered[
                        "objective"
                    ],
                "iterations":
                    clustered[
                        "iterations"
                    ],
                "converged":
                    clustered[
                        "converged"
                    ],
            }
        )

    if not results:
        raise ValueError(
            "No valid K values."
        )

    return results


def _geometric_knee_point(
    points,
):
    """
    Same geometric knee construction as the uploaded belief script,
    now on mean Frobenius quantisation error.
    """
    ordered = sorted(
        points,
        key=lambda row: row["k"],
    )

    if not ordered:
        return (
            None,
            {},
        )

    if len(ordered) == 1:
        only = dict(
            ordered[0]
        )
        only[
            "geometric_distance"
        ] = 0.0

        return (
            only,
            {
                ordered[0][
                    "k"
                ]: 0.0
            },
        )

    k_values = [
        row["k"]
        for row in ordered
    ]

    errors = [
        row[
            "mean_frobenius_error"
        ]
        for row in ordered
    ]

    k_min = min(
        k_values
    )
    k_max = max(
        k_values
    )
    e_min = min(
        errors
    )
    e_max = max(
        errors
    )

    def normalise(
        value,
        low,
        high,
    ):
        if abs(
            high - low
        ) <= 1e-15:
            return 0.0

        return (
            value - low
        ) / (
            high - low
        )

    normalised = [
        (
            normalise(
                row["k"],
                k_min,
                k_max,
            ),
            normalise(
                row[
                    "mean_frobenius_error"
                ],
                e_min,
                e_max,
            ),
        )
        for row in ordered
    ]

    x1, y1 = (
        normalised[0]
    )
    x2, y2 = (
        normalised[-1]
    )

    denominator = math.sqrt(
        (y2 - y1) ** 2
        + (x2 - x1) ** 2
    )

    distances_by_k = {}

    for (
        index,
        row,
    ) in enumerate(
        ordered
    ):
        x0, y0 = (
            normalised[
                index
            ]
        )

        if denominator <= 1e-15:
            distance = 0.0
        else:
            distance = abs(
                (y2 - y1) * x0
                - (x2 - x1) * y0
                + x2 * y1
                - y2 * x1
            ) / denominator

        distances_by_k[
            row["k"]
        ] = distance

    if len(
        ordered
    ) < 3:
        best_index = 0
    else:
        best_index = max(
            range(
                1,
                len(ordered) - 1,
            ),
            key=lambda index: (
                distances_by_k[
                    ordered[index]["k"]
                ],
                -ordered[index]["k"],
            ),
        )

    knee = dict(
        ordered[
            best_index
        ]
    )

    knee[
        "geometric_distance"
    ] = distances_by_k[
        knee["k"]
    ]

    return (
        knee,
        distances_by_k,
    )


def _aggregate_k_results(
    all_map_results,
):
    if not all_map_results:
        return []

    map_count = len(
        all_map_results
    )

    by_k = {}

    for (
        _map_id,
        results,
    ) in all_map_results.items():
        for row in results:
            by_k.setdefault(
                row["k"],
                [],
            ).append(
                row
            )

    aggregated = []

    for k in sorted(
        by_k
    ):
        rows = by_k[
            k
        ]

        # Compare same maps at every K.
        if len(
            rows
        ) != map_count:
            continue

        aggregated.append(
            {
                "k": k,
                "maps": len(
                    rows
                ),
                "mean_frobenius_error":
                    sum(
                        row[
                            "mean_frobenius_error"
                        ]
                        for row in rows
                    ) / len(rows),
                "mean_max_frobenius_error":
                    sum(
                        row[
                            "max_frobenius_error"
                        ]
                        for row in rows
                    ) / len(rows),
                "mean_silhouette":
                    sum(
                        row[
                            "silhouette"
                        ]
                        for row in rows
                    ) / len(rows),
                "mean_objective":
                    sum(
                        row[
                            "objective"
                        ]
                        for row in rows
                    ) / len(rows),
            }
        )

    return aggregated


def _write_csv(
    path,
    rows,
):
    if not rows:
        return

    with open(
        path,
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


def evaluate_knee_all_maps(
    first_map=10,
    last_map=99,
    maps_dir="maps",
    target=(9, 9),
    p=0.01,
    max_steps=10,
    k_values=None,
    output_dir="gaussian_frobenius_knee",
    max_iter=100,
    objective_tolerance=1e-9,
):
    if k_values is None:
        k_values = [
            25,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
        ]

    output_root = Path(
        output_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_map_results = {}
    summary_rows = []

    regular_k_set = set(
        k_values
    )

    for map_id in range(
        first_map,
        last_map + 1,
    ):
        map_path = (
            Path(maps_dir)
            / f"map_{map_id}.csv"
        )

        if not map_path.exists():
            print(
                f"skip map {map_id}: "
                f"{map_path} missing"
            )
            continue

        map_data = (
            _read_map_data(
                map_path
            )
        )

        (
            candidates,
            _,
            _,
        ) = fg._generate_records(
            map_data,
            target,
            p,
            max_steps,
        )

        valid_k_values = [
            k
            for k in k_values
            if (
                2
                <= k
                <= len(candidates)
            )
        ]

        # Full raw set is diagnostic only, like in belief script.
        if (
            len(candidates) >= 2
            and len(candidates)
            not in valid_k_values
        ):
            valid_k_values.append(
                len(candidates)
            )

        results = evaluate_k_values(
            candidates,
            valid_k_values,
            max_iter=max_iter,
            objective_tolerance=objective_tolerance,
        )

        per_map_knee_candidates = [
            row
            for row in results
            if row["k"] in regular_k_set
        ]

        (
            per_map_knee,
            per_map_distances,
        ) = _geometric_knee_point(
            per_map_knee_candidates
        )

        all_map_results[
            map_id
        ] = results

        detail_rows = []

        previous_error = None

        for row in sorted(
            results,
            key=lambda item: item["k"],
        ):
            improvement = (
                ""
                if previous_error is None
                else (
                    previous_error
                    - row[
                        "mean_frobenius_error"
                    ]
                )
            )

            detail_rows.append(
                {
                    "map_id": map_id,
                    "unique_raw_sigma":
                        len(candidates),
                    **row,
                    "frobenius_improvement_from_previous_k":
                        improvement,
                    "is_geometric_knee":
                        int(
                            per_map_knee
                            is not None
                            and row["k"]
                            == per_map_knee["k"]
                        ),
                    "geometric_knee_distance":
                        per_map_distances.get(
                            row["k"],
                            "",
                        ),
                }
            )

            previous_error = row[
                "mean_frobenius_error"
            ]

        _write_csv(
            output_root
            / f"map_{map_id}_k_evaluation.csv",
            detail_rows,
        )

        summary_rows.append(
            {
                "map_id": map_id,
                "unique_raw_sigma":
                    len(candidates),
                "geometric_knee_k":
                    (
                        per_map_knee["k"]
                        if per_map_knee
                        is not None
                        else ""
                    ),
                "mean_frobenius_error_at_knee":
                    (
                        per_map_knee[
                            "mean_frobenius_error"
                        ]
                        if per_map_knee
                        is not None
                        else ""
                    ),
                "max_frobenius_error_at_knee":
                    (
                        per_map_knee[
                            "max_frobenius_error"
                        ]
                        if per_map_knee
                        is not None
                        else ""
                    ),
                "silhouette_at_knee":
                    (
                        per_map_knee[
                            "silhouette"
                        ]
                        if per_map_knee
                        is not None
                        else ""
                    ),
                "geometric_knee_distance":
                    (
                        per_map_knee[
                            "geometric_distance"
                        ]
                        if per_map_knee
                        is not None
                        else ""
                    ),
            }
        )

        print(
            f"map {map_id}: "
            f"{len(candidates)} raw Sigma, "
            f"knee K="
            f"{per_map_knee['k'] if per_map_knee else 'n/a'}, "
            f"mean F="
            f"{per_map_knee['mean_frobenius_error']:.6f}"
            if per_map_knee
            else ""
        )

    _write_csv(
        output_root
        / "knee_by_map.csv",
        summary_rows,
    )

    aggregated = (
        _aggregate_k_results(
            all_map_results
        )
    )

    (
        global_knee,
        global_distances,
    ) = _geometric_knee_point(
        aggregated
    )

    aggregated_rows = []

    previous_error = None

    for row in aggregated:
        improvement = (
            ""
            if previous_error is None
            else (
                previous_error
                - row[
                    "mean_frobenius_error"
                ]
            )
        )

        aggregated_rows.append(
            {
                **row,
                "frobenius_improvement_from_previous_k":
                    improvement,
                "is_geometric_knee":
                    int(
                        global_knee
                        is not None
                        and row["k"]
                        == global_knee["k"]
                    ),
                "geometric_distance":
                    global_distances.get(
                        row["k"],
                        "",
                    ),
            }
        )

        previous_error = row[
            "mean_frobenius_error"
        ]

    _write_csv(
        output_root
        / "aggregated_k_evaluation.csv",
        aggregated_rows,
    )

    summary = {
        "settings": {
            "first_map": first_map,
            "last_map": last_map,
            "target": list(
                target
            ),
            "p": p,
            "max_steps": max_steps,
            "metric": METRIC,
            "k_values": k_values,
        },
        "global_geometric_knee":
            global_knee,
        "per_map_knees":
            summary_rows,
        "aggregated_curve":
            aggregated_rows,
    }

    with open(
        output_root
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 72)

    if global_knee is not None:
        print(
            "Aggregated Gaussian Frobenius geometric knee:"
        )
        print(
            f"  K={global_knee['k']}"
        )
        print(
            f"  mean Frobenius error="
            f"{global_knee['mean_frobenius_error']:.6f}"
        )
        print(
            f"  mean max Frobenius error="
            f"{global_knee['mean_max_frobenius_error']:.6f}"
        )
        print(
            f"  geometric distance="
            f"{global_knee['geometric_distance']:.6f}"
        )

    print("=" * 72)
    print(
        f"Results written to: "
        f"{output_root}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--first-map",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--last-map",
        type=int,
        default=99,
    )
    parser.add_argument(
        "--maps-dir",
        default="maps",
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
        "--output-dir",
        default="gaussian_frobenius_knee",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[
            25,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
        ],
    )

    return parser.parse_args()


def main():
    args = parse_args()

    evaluate_knee_all_maps(
        first_map=args.first_map,
        last_map=args.last_map,
        maps_dir=args.maps_dir,
        target=(
            args.target_x,
            args.target_y,
        ),
        p=args.p,
        max_steps=args.max_steps,
        k_values=args.k_values,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
