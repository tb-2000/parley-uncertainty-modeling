import csv
import json
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


def _gini_uncertainty(belief):
    """
    Gini uncertainty of a positional belief:
        U_G(b) = 1 - sum_s b(s)^2

    0 means complete certainty.
    Larger values mean the probability mass is more dispersed.
    """
    return 1.0 - sum(
        probability * probability
        for probability in belief.values()
    )


def _relative_vector(belief, xhat, yhat, n):
    # Full positional belief relative to xhat,yhat. Offsets range -N..N.
    width = 2 * n + 1
    vector = [0.0] * (width * width)

    for (x, y), probability in belief.items():
        dx = x - xhat
        dy = y - yhat
        index = (dy + n) * width + (dx + n)
        vector[index] += probability

    return tuple(vector)


def _vector_to_relative(vector, n):
    width = 2 * n + 1
    result = {}
    for index, probability in enumerate(vector):
        if probability <= 1e-15:
            continue
        oy, ox = divmod(index, width)
        result[(ox - n, oy - n)] = probability
    return result


def _l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def _generate_records(map_data, target, p, max_steps):
    map_size = len(map_data)
    n = map_size - 1
    controller = _controller(map_data, target)

    records = []
    gini_by_age = defaultdict(list)

    for sx in range(map_size):
        for sy in range(map_size):
            if int(map_data[sx][sy]) > 9:
                continue

            belief = {(sx, sy): 1.0}
            xhat, yhat = sx, sy

            for age in range(max_steps + 1):
                vector = _relative_vector(belief, xhat, yhat, n)
                records.append({
                    "vector": vector,
                    "xhat": xhat,
                    "yhat": yhat,
                    "age": age,
                })

                gini_by_age[age].append(
                    _gini_uncertainty(belief)
                )

                if age >= max_steps or (xhat, yhat) == target:
                    break

                action = _direction(controller, xhat, yhat)
                if action is None:
                    break

                belief = _propagate_absolute(
                    belief, action, n, p
                )
                xhat, yhat = _move(
                    xhat, yhat, action, n
                )

    # Deduplicate identical relative beliefs while retaining frequency.
    unique = {}
    for record in records:
        key = tuple(round(v, 14) for v in record["vector"])
        if key not in unique:
            unique[key] = {
                "vector": record["vector"],
                "weight": 1,
            }
        else:
            unique[key]["weight"] += 1

    candidates = list(unique.values())
    return candidates, gini_by_age, controller


def _distance_matrix(vectors):
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _l1(vectors[i], vectors[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def _cluster(
    candidates,
    k,
    max_iter=100,
    objective_tolerance=1e-9,
):
    """
    Weighted k-medoids-style quantisation with farthest-first initialisation.

    Convergence criteria:
      1. the medoid set no longer changes, or
      2. the relative improvement in the weighted clustering objective
         falls below objective_tolerance.

    max_iter is only a safety limit.
    """
    vectors = [item["vector"] for item in candidates]
    weights = [item["weight"] for item in candidates]

    if len(vectors) <= k:
        medoids = list(range(len(vectors)))
        matrix = _distance_matrix(vectors)
        objective = 0.0
        iterations = 0
        converged = True
    else:
        matrix = _distance_matrix(vectors)

        # Always preserve the exact certain belief as state 0.
        zero_index = min(
            range(len(vectors)),
            key=lambda i: _l1(
                vectors[i],
                tuple(
                    1.0 if j == len(vectors[i]) // 2 else 0.0
                    for j in range(len(vectors[i]))
                ),
            ),
        )

        # Farthest-first initialisation.
        medoids = [zero_index]
        nearest = [
            matrix[i][zero_index]
            for i in range(len(vectors))
        ]

        while len(medoids) < k:
            nxt = max(
                (
                    i
                    for i in range(len(vectors))
                    if i not in medoids
                ),
                key=lambda i: (
                    nearest[i],
                    -i,
                ),
            )
            medoids.append(nxt)

            for i in range(len(vectors)):
                nearest[i] = min(
                    nearest[i],
                    matrix[i][nxt],
                )

        def assign(current_medoids):
            clusters = {
                medoid: []
                for medoid in current_medoids
            }

            assignments = []

            for i in range(len(vectors)):
                medoid = min(
                    current_medoids,
                    key=lambda candidate: (
                        matrix[i][candidate],
                        candidate,
                    ),
                )
                assignments.append(medoid)
                clusters[medoid].append(i)

            return assignments, clusters

        def weighted_objective(assignments):
            return sum(
                weights[i] * matrix[i][assignments[i]]
                for i in range(len(vectors))
            )

        previous_objective = None
        objective = None
        converged = False
        iterations = 0

        for iteration in range(1, max_iter + 1):
            iterations = iteration

            assignments, clusters = assign(medoids)
            current_objective = weighted_objective(assignments)

            refined = []

            for medoid in medoids:
                members = clusters[medoid]

                # Preserve the exact certain belief.
                if medoid == zero_index:
                    refined.append(medoid)
                    continue

                best = min(
                    members,
                    key=lambda candidate: (
                        sum(
                            weights[j]
                            * matrix[candidate][j]
                            for j in members
                        ),
                        candidate,
                    ),
                )
                refined.append(best)

            refined_set = set(refined)
            medoid_set = set(medoids)

            if refined_set == medoid_set:
                medoids = refined
                objective = current_objective
                converged = True
                break

            # Re-evaluate objective after the medoid update.
            new_assignments, _ = assign(refined)
            new_objective = weighted_objective(
                new_assignments
            )

            if previous_objective is not None:
                denominator = max(
                    abs(previous_objective),
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
            previous_objective = new_objective
            objective = new_objective

        if objective is None:
            assignments, _ = assign(medoids)
            objective = weighted_objective(
                assignments
            )

    # Put the certain representative first.
    certain = max(
        medoids,
        key=lambda i: max(vectors[i]),
    )

    medoids = [
        certain
    ] + sorted(
        medoid
        for medoid in medoids
        if medoid != certain
    )

    representatives = [
        vectors[i]
        for i in medoids
    ]

    return {
        "representatives": representatives,
        "medoid_indices": medoids,
        "objective": objective,
        "iterations": iterations,
        "converged": converged,
        "distance_matrix": matrix,
    }


def _nearest(vector, representatives):
    return min(
        range(len(representatives)),
        key=lambda i: (
            _l1(vector, representatives[i]),
            i,
        ),
    )


def _assign_to_representatives(
    candidates,
    medoid_indices,
    distance_matrix,
):
    """
    Assign every unique belief candidate to its nearest medoid index.
    Returns cluster id (0..K-1) for each candidate.
    """
    medoid_to_cluster = {
        medoid_index: cluster_id
        for cluster_id, medoid_index in enumerate(
            medoid_indices
        )
    }

    assignments = []

    for i in range(len(candidates)):
        medoid_index = min(
            medoid_indices,
            key=lambda m: (
                distance_matrix[i][m],
                m,
            ),
        )
        assignments.append(
            medoid_to_cluster[medoid_index]
        )

    return assignments


def _weighted_silhouette_score(
    candidates,
    assignments,
    distance_matrix,
    k,
):
    """
    Weighted silhouette coefficient using the occurrence frequencies that
    were retained during relative-belief deduplication.

    For each belief i:
      a(i) = weighted mean distance to beliefs in its own cluster
      b(i) = minimum weighted mean distance to another cluster
      s(i) = (b(i)-a(i)) / max(a(i), b(i))

    The final score is the occurrence-weighted mean of s(i).
    """
    if k <= 1 or len(candidates) <= 1:
        return float("nan")

    weights = [
        item["weight"]
        for item in candidates
    ]

    clusters = {
        cluster_id: []
        for cluster_id in range(k)
    }

    for i, cluster_id in enumerate(assignments):
        clusters[cluster_id].append(i)

    silhouette_values = []

    for i, own_cluster in enumerate(assignments):
        own_members = clusters[own_cluster]

        # Standard convention: singleton cluster gets silhouette 0.
        if len(own_members) <= 1:
            silhouette_values.append(0.0)
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
                weights[j] * distance_matrix[i][j]
                for j in own_members
                if j != i
            ) / own_weight

        b_i = None

        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster or not members:
                continue

            cluster_weight = sum(
                weights[j]
                for j in members
            )

            mean_distance = sum(
                weights[j] * distance_matrix[i][j]
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
            denominator = max(a_i, b_i)

            if denominator <= 1e-15:
                silhouette = 0.0
            else:
                silhouette = (
                    b_i - a_i
                ) / denominator

        silhouette_values.append(
            silhouette
        )

    total_weight = sum(weights)

    return sum(
        weights[i] * silhouette_values[i]
        for i in range(len(candidates))
    ) / total_weight


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

    for i in range(len(candidates)):
        distance = min(
            distance_matrix[i][m]
            for m in medoid_indices
        )

        weighted_sum += (
            weights[i] * distance
        )
        total_weight += weights[i]
        maximum = max(
            maximum,
            distance,
        )

    mean = (
        weighted_sum / total_weight
        if total_weight
        else 0.0
    )

    return mean, maximum



def _select_elbow_k(
    results,
    improvement_threshold=1.0 / 500.0,
):
    """
    Select an elbow from the mean L1 quantisation-error curve.

    The candidate K values are sorted increasingly. For each step, compute

        improvement = previous_mean_L1 - current_mean_L1

    The elbow is the first K for which the improvement drops below the
    configured absolute threshold. This captures the point where adding
    further representatives yields only a small reduction in mean L1 error.

    If no step falls below the threshold, the largest tested K is returned.

    Note:
    K spacings can differ. Therefore the raw improvement and the improvement
    per added representative are both returned for diagnostics, while the
    selection itself intentionally uses the absolute improvement requested
    for the PARLEY experiment.
    """
    ordered = sorted(
        results,
        key=lambda row: row["k"],
    )

    if len(ordered) == 1:
        return {
            "k": ordered[0]["k"],
            "previous_k": None,
            "mean_l1_error": ordered[0]["mean_l1_error"],
            "improvement": None,
            "improvement_per_added_state": None,
            "threshold": improvement_threshold,
            "threshold_met": False,
        }

    for previous, current in zip(
        ordered[:-1],
        ordered[1:],
    ):
        improvement = (
            previous["mean_l1_error"]
            - current["mean_l1_error"]
        )

        delta_k = (
            current["k"]
            - previous["k"]
        )

        improvement_per_added_state = (
            improvement / delta_k
            if delta_k > 0
            else float("nan")
        )

        if improvement <= improvement_threshold:
            return {
                "k": current["k"],
                "previous_k": previous["k"],
                "mean_l1_error": current["mean_l1_error"],
                "improvement": improvement,
                "improvement_per_added_state": (
                    improvement_per_added_state
                ),
                "threshold": improvement_threshold,
                "threshold_met": True,
            }

    # No elbow found within the tested range.
    previous = ordered[-2]
    current = ordered[-1]
    improvement = (
        previous["mean_l1_error"]
        - current["mean_l1_error"]
    )
    delta_k = current["k"] - previous["k"]

    return {
        "k": current["k"],
        "previous_k": previous["k"],
        "mean_l1_error": current["mean_l1_error"],
        "improvement": improvement,
        "improvement_per_added_state": (
            improvement / delta_k
            if delta_k > 0
            else float("nan")
        ),
        "threshold": improvement_threshold,
        "threshold_met": False,
    }


def evaluate_k_values(
    candidates,
    k_values,
    max_iter=100,
    objective_tolerance=1e-9,
):
    """
    Evaluate multiple K values for one map.

    Best K is selected by highest weighted silhouette score.
    Ties are resolved in favour of the smaller K.
    """
    results = []

    for k in sorted(set(k_values)):
        if k < 2 or k > len(candidates):
            continue

        clustered = _cluster(
            candidates,
            k=k,
            max_iter=max_iter,
            objective_tolerance=objective_tolerance,
        )

        assignments = _assign_to_representatives(
            candidates,
            clustered["medoid_indices"],
            clustered["distance_matrix"],
        )

        silhouette = _weighted_silhouette_score(
            candidates,
            assignments,
            clustered["distance_matrix"],
            k=len(clustered["medoid_indices"]),
        )

        mean_error, max_error = (
            _mean_assignment_error(
                candidates,
                clustered["medoid_indices"],
                clustered["distance_matrix"],
            )
        )

        results.append(
            {
                "k": k,
                "silhouette": silhouette,
                "mean_l1_error": mean_error,
                "max_l1_error": max_error,
                "objective": clustered["objective"],
                "iterations": clustered["iterations"],
                "converged": clustered["converged"],
            }
        )

    if not results:
        raise ValueError(
            "No valid K values for silhouette evaluation."
        )

    best = max(
        results,
        key=lambda row: (
            row["silhouette"],
            -row["k"],
        ),
    )

    return results, best


def _representative_gini(vector):
    return 1.0 - sum(
        probability * probability
        for probability in vector
    )


def _thresholds(gini_by_age, max_steps):
    """
    Map c=1..10 to typical full-belief Gini uncertainty after c predictions.
    Median is robust against map-edge/path outliers. Values are scaled
    to integers 0..10000 for PRISM 4.7.
    """
    result = []
    previous = 0

    for age in range(1, max_steps + 1):
        values = sorted(gini_by_age.get(age, []))
        if not values:
            value = previous / 10000.0
        else:
            middle = len(values) // 2
            if len(values) % 2:
                value = values[middle]
            else:
                value = (values[middle - 1] + values[middle]) / 2.0

        scaled = int(round(value * 10000))
        # Keep c=1..10 ordered even if borders make entropy locally decrease.
        scaled = max(previous, scaled)
        result.append(scaled)
        previous = scaled

    return result


def _transition_from_representative(
    vector,
    xhat,
    yhat,
    action,
    n,
    p,
    representatives,
):
    """
    Reconstruct an absolute belief around xhat,yhat, propagate it using the
    same clipped Robot dynamics, re-centre around the new nominal estimate,
    then project to the nearest representative.
    """
    relative = _vector_to_relative(vector, n)
    absolute = defaultdict(float)

    for (dx, dy), probability in relative.items():
        # Unreachable state/position combinations can produce out-of-grid
        # offsets. Clamp them so the transition is still defined.
        ax = _clip(xhat + dx, n)
        ay = _clip(yhat + dy, n)
        absolute[(ax, ay)] += probability

    propagated = _propagate_absolute(
        absolute, action, n, p
    )
    nxhat, nyhat = _move(
        xhat, yhat, action, n
    )
    successor_vector = _relative_vector(
        propagated, nxhat, nyhat, n
    )

    return _nearest(
        successor_vector, representatives
    )


def build_belief_model(
    map_id,
    map_data,
    target,
    p=0.01,
    k=100,
    max_steps=10,
    cache_dir="belief_models",
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"map_{map_id}.json"

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as file:
            cached = json.load(file)
        if (
            cached.get("k") == k
            and cached.get("max_steps") == max_steps
            and abs(cached.get("p", -1) - p) < 1e-15
        ):
            return cached

    candidates, gini_by_age, controller = _generate_records(
        map_data, target, p, max_steps
    )

    clustered = _cluster(
        candidates,
        min(k, len(candidates)),
        max_iter=100,
        objective_tolerance=1e-9,
    )

    representatives = clustered[
        "representatives"
    ]

    # If fewer than K unique reachable beliefs exist, keep the real count.
    state_count = len(representatives)
    n = len(map_data) - 1

    uncertainties = [
        int(round(
            _representative_gini(vector)
            * 10000
        ))
        for vector in representatives
    ]

    thresholds = _thresholds(
        gini_by_age, max_steps
    )

    transitions = {}

    for x in range(len(map_data)):
        for y in range(len(map_data)):
            action = _direction(controller, x, y)
            if action is None:
                continue

            for state_id, vector in enumerate(representatives):
                next_state = _transition_from_representative(
                    vector,
                    x,
                    y,
                    action,
                    n,
                    p,
                    representatives,
                )
                transitions[
                    f"{x},{y},{state_id}"
                ] = {
                    "action": action,
                    "next_state": next_state,
                }

    model = {
        "map_id": map_id,
        "k": k,
        "state_count": state_count,
        "max_steps": max_steps,
        "p": p,
        "thresholds": thresholds,
        "uncertainties": uncertainties,
        "transitions": transitions,
        "clustering_iterations": clustered[
            "iterations"
        ],
        "clustering_converged": clustered[
            "converged"
        ],
        "clustering_objective": clustered[
            "objective"
        ],
    }

    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(model, file)

    return model



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

    transposed = list(zip(*rows))
    return [
        row[::-1]
        for row in transposed
    ]



def _geometric_knee_point(points):
    """
    Determine the geometric knee point of an error curve and compute
    the geometric distance for every tested K.

    Procedure:
      1. sort by K;
      2. normalise K and mean L1 error independently to [0,1];
      3. compute the perpendicular distance of every point to the
         straight line connecting the first and last point;
      4. choose the interior point with maximum distance as the knee.

    Returns:
        knee:
            dictionary for the selected knee point
        distances_by_k:
            mapping K -> geometric distance
    """
    ordered = sorted(
        points,
        key=lambda row: row["k"],
    )

    if not ordered:
        return None, {}

    if len(ordered) == 1:
        only = dict(ordered[0])
        only["geometric_distance"] = 0.0
        return only, {
            ordered[0]["k"]: 0.0
        }

    k_values = [
        row["k"]
        for row in ordered
    ]
    errors = [
        row["mean_l1_error"]
        for row in ordered
    ]

    k_min = min(k_values)
    k_max = max(k_values)
    e_min = min(errors)
    e_max = max(errors)

    def normalise(value, low, high):
        if abs(high - low) <= 1e-15:
            return 0.0
        return (value - low) / (high - low)

    normalised = [
        (
            normalise(row["k"], k_min, k_max),
            normalise(
                row["mean_l1_error"],
                e_min,
                e_max,
            ),
        )
        for row in ordered
    ]

    x1, y1 = normalised[0]
    x2, y2 = normalised[-1]

    denominator = (
        (y2 - y1) ** 2
        + (x2 - x1) ** 2
    ) ** 0.5

    distances_by_k = {}

    for index, row in enumerate(ordered):
        x0, y0 = normalised[index]

        if denominator <= 1e-15:
            distance = 0.0
        else:
            distance = abs(
                (y2 - y1) * x0
                - (x2 - x1) * y0
                + x2 * y1
                - y2 * x1
            ) / denominator

        distances_by_k[row["k"]] = distance

    # Knee must be an interior point. Endpoints define the reference line.
    if len(ordered) < 3:
        best_index = 0
    else:
        best_index = max(
            range(1, len(ordered) - 1),
            key=lambda index: (
                distances_by_k[ordered[index]["k"]],
                -ordered[index]["k"],
            ),
        )

    knee = dict(ordered[best_index])
    knee["geometric_distance"] = (
        distances_by_k[knee["k"]]
    )

    return knee, distances_by_k


def _aggregate_k_results(
    all_map_results,
):
    """
    Aggregate mean/max L1 and silhouette values across maps for each K.

    Only K values available for every processed map are used so that the
    aggregated curve compares the same set of maps at every point.
    """
    if not all_map_results:
        return []

    map_count = len(all_map_results)

    by_k = {}

    for map_id, results in all_map_results.items():
        for row in results:
            k = row["k"]
            by_k.setdefault(k, []).append(row)

    aggregated = []

    for k in sorted(by_k):
        rows = by_k[k]

        if len(rows) != map_count:
            continue

        aggregated.append(
            {
                "k": k,
                "maps": len(rows),
                "mean_l1_error": sum(
                    row["mean_l1_error"]
                    for row in rows
                ) / len(rows),
                "mean_max_l1_error": sum(
                    row["max_l1_error"]
                    for row in rows
                ) / len(rows),
                "mean_silhouette": sum(
                    row["silhouette"]
                    for row in rows
                ) / len(rows),
            }
        )

    return aggregated


def _write_aggregated_k_csv(
    path,
    aggregated_results,
    knee,
    distances_by_k,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "k",
            "maps",
            "mean_l1_error",
            "mean_max_l1_error",
            "mean_silhouette",
            "is_geometric_knee",
            "geometric_distance",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in aggregated_results:
            writer.writerow(
                {
                    **row,
                    "is_geometric_knee": int(
                        knee is not None
                        and row["k"] == knee["k"]
                    ),
                    "geometric_distance": (
                        distances_by_k.get(
                            row["k"],
                            "",
                        )
                    ),
                }
            )


def evaluate_best_k_all_maps(
    first_map=10,
    last_map=99,
    maps_dir="maps",
    target=(9, 9),
    p=0.01,
    max_steps=10,
    k_values=None,
    output_dir="belief_k_evaluation",
    max_iter=100,
    objective_tolerance=1e-9,
):
    """
    Evaluate candidate K values for every map and write:
      - one detailed CSV per map;
      - one summary CSV with the silhouette-best K for each map.

    Silhouette is used only to rank K values. Quantisation errors and
    convergence diagnostics are written alongside it for interpretation.
    """
    if k_values is None:
        k_values = [
            25,
            50,
            75,
            100,
            125,
            150,
            200,
            250,
            300,
        ]

    output_root = Path(output_dir)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    all_map_results = {}

    for map_id in range(
        first_map,
        last_map + 1,
    ):
        map_path = (
            Path(maps_dir)
            / f"map_{map_id}.csv"
        )

        map_data = _read_map_data(
            map_path
        )

        candidates, _, _ = _generate_records(
            map_data,
            target,
            p,
            max_steps,
        )

        valid_k_values = [
            k
            for k in k_values
            if 2 <= k <= len(candidates)
        ]

        # Also allow the full unique-belief set as a diagnostic endpoint,
        # but do not force it if it would duplicate an existing K.
        if (
            len(candidates) >= 2
            and len(candidates) not in valid_k_values
        ):
            valid_k_values.append(
                len(candidates)
            )

        results, best = evaluate_k_values(
            candidates,
            valid_k_values,
            max_iter=max_iter,
            objective_tolerance=objective_tolerance,
        )

        regular_k_set = set(k_values)
        per_map_knee_candidates = [
            row
            for row in results
            if row["k"] in regular_k_set
        ]
        (
            per_map_knee,
            per_map_knee_distances,
        ) = _geometric_knee_point(
            per_map_knee_candidates
        )

        all_map_results[map_id] = results

        detail_path = (
            output_root
            / f"map_{map_id}_k_evaluation.csv"
        )

        with open(
            detail_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            fieldnames = [
                "map_id",
                "unique_relative_beliefs",
                "k",
                "silhouette",
                "mean_l1_error",
                "max_l1_error",
                "objective",
                "iterations",
                "converged",
                "best_by_silhouette",
                "l1_improvement_from_previous_k",
                "is_geometric_knee",
                "geometric_knee_distance",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            ordered_results = sorted(
                results,
                key=lambda item: item["k"],
            )

            previous_mean_error = None

            for row in ordered_results:
                if previous_mean_error is None:
                    l1_improvement = ""
                else:
                    l1_improvement = (
                        previous_mean_error
                        - row["mean_l1_error"]
                    )

                writer.writerow(
                    {
                        "map_id": map_id,
                        "unique_relative_beliefs": len(candidates),
                        **row,
                        "best_by_silhouette": int(
                            row["k"] == best["k"]
                        ),
                        "l1_improvement_from_previous_k": (
                            l1_improvement
                        ),
                        "is_geometric_knee": int(
                            per_map_knee is not None
                            and row["k"] == per_map_knee["k"]
                        ),
                        "geometric_knee_distance": (
                            per_map_knee_distances.get(
                                row["k"],
                                "",
                            )
                        ),
                    }
                )

                previous_mean_error = row[
                    "mean_l1_error"
                ]

        summary_rows.append(
            {
                "map_id": map_id,
                "unique_relative_beliefs": len(candidates),
                "best_k_by_silhouette": best["k"],
                "best_silhouette": best["silhouette"],
                "mean_l1_error_at_best_k": best[
                    "mean_l1_error"
                ],
                "max_l1_error_at_best_k": best[
                    "max_l1_error"
                ],
                "iterations_at_best_k": best[
                    "iterations"
                ],
                "converged_at_best_k": int(
                    best["converged"]
                ),
                "geometric_knee_k": (
                    per_map_knee["k"]
                    if per_map_knee is not None
                    else ""
                ),
                "mean_l1_error_at_geometric_knee": (
                    per_map_knee["mean_l1_error"]
                    if per_map_knee is not None
                    else ""
                ),
                "max_l1_error_at_geometric_knee": (
                    next(
                        (
                            row["max_l1_error"]
                            for row in results
                            if (
                                per_map_knee is not None
                                and row["k"] == per_map_knee["k"]
                            )
                        ),
                        "",
                    )
                ),
                "silhouette_at_geometric_knee": (
                    next(
                        (
                            row["silhouette"]
                            for row in results
                            if (
                                per_map_knee is not None
                                and row["k"] == per_map_knee["k"]
                            )
                        ),
                        "",
                    )
                ),
                "geometric_knee_distance": (
                    per_map_knee["geometric_distance"]
                    if per_map_knee is not None
                    else ""
                ),
            }
        )


        print(
            f"map {map_id}: "
            f"{len(candidates)} unique relative beliefs, "
            f"best silhouette K={best['k']}, "
            f"silhouette={best['silhouette']:.6f}, "
            f"geometric knee K="
            f"{per_map_knee['k'] if per_map_knee is not None else 'n/a'}, "
            f"iterations={best['iterations']}, "
            f"converged={best['converged']}"
        )


    summary_path = (
        output_root
        / "best_k_by_map.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "map_id",
            "unique_relative_beliefs",
            "best_k_by_silhouette",
            "best_silhouette",
            "mean_l1_error_at_best_k",
            "max_l1_error_at_best_k",
            "iterations_at_best_k",
            "converged_at_best_k",
            "geometric_knee_k",
            "mean_l1_error_at_geometric_knee",
            "max_l1_error_at_geometric_knee",
            "silhouette_at_geometric_knee",
            "geometric_knee_distance",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregated_results = _aggregate_k_results(
        all_map_results
    )

    knee, aggregated_distances = _geometric_knee_point(
        aggregated_results
    )

    aggregated_path = (
        output_root
        / "aggregated_k_evaluation.csv"
    )

    _write_aggregated_k_csv(
        aggregated_path,
        aggregated_results,
        knee,
        aggregated_distances,
    )

    print(
        f"Written: {summary_path}"
    )
    print(
        f"Written: {aggregated_path}"
    )

    if knee is not None:
        print()
        print(
            "Aggregated geometric knee:"
        )
        print(
            f"  K={knee['k']}"
        )
        print(
            f"  mean L1={knee['mean_l1_error']:.6f}"
        )
        print(
            f"  mean max L1={knee['mean_max_l1_error']:.6f}"
        )
        print(
            f"  geometric distance="
            f"{knee['geometric_distance']:.6f}"
        )


def precompute_maps(
    first_map=10,
    last_map=99,
    maps_dir="maps",
    target=(9, 9),
    p=0.01,
    k=100,
    max_steps=10,
    cache_dir="belief_models",
):
    for map_id in range(first_map, last_map + 1):
        rows = []
        with open(
            Path(maps_dir) / f"map_{map_id}.csv",
            "r",
            newline="",
        ) as file:
            rows.extend(csv.reader(file))

        transposed = list(zip(*rows))
        map_data = [row[::-1] for row in transposed]

        model = build_belief_model(
            map_id=map_id,
            map_data=map_data,
            target=target,
            p=p,
            k=k,
            max_steps=max_steps,
            cache_dir=cache_dir,
        )

        print(
            f"map {map_id}: "
            f"{model['state_count']} representatives, "
            f"thresholds={model['thresholds']}"
        )


if __name__ == "__main__":
    evaluate_best_k_all_maps()
