import argparse
import csv
import json
import math
from collections import defaultdict

import dijkstra


DIRECTIONS = ("west", "east", "south", "north")

MOVE = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}


# ---------------------------------------------------------------------------
# Map / MAPE controller
# ---------------------------------------------------------------------------

def read_map(filename):
    rows = []

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        rows.extend(reader)

    if not rows:
        raise ValueError(f"Empty map: {filename}")

    map_size = len(rows)

    transposed = list(zip(*rows))
    map_data = [row[::-1] for row in transposed]

    obstacles = set()

    for x in range(map_size):
        for y in range(map_size):
            if int(map_data[x][y]) > 9:
                obstacles.add((x, y))

    return map_data, map_size, obstacles


def build_controller(map_data, target):
    raw = dijkstra.compute_directions(map_data, target)
    return list(zip(*raw))


def controller_direction(controller, x, y):
    value = int(controller[y][x])

    if 0 <= value < 4:
        return DIRECTIONS[value]

    return None


def nominal_next_position(x, y, action, n):
    dx, dy = MOVE[action]

    return (
        min(max(x + dx, 0), n),
        min(max(y + dy, 0), n),
    )


# ---------------------------------------------------------------------------
# Full 10x10 positional belief
# ---------------------------------------------------------------------------

def point_belief(x, y):
    return {(x, y): 1.0}


def propagate_full_belief(
    belief,
    commanded_action,
    n,
    p=0.01,
):
    """
    Exact prediction step over all absolute grid positions.

    Intended movement: 1-3p
    Each other cardinal movement: p

    Grid borders follow the same clipping semantics as the PRISM Robot model.
    """
    result = defaultdict(float)

    for (x, y), prior in belief.items():
        for actual_action in DIRECTIONS:
            probability = (
                1.0 - 3.0 * p
                if actual_action == commanded_action
                else p
            )

            nx, ny = nominal_next_position(
                x,
                y,
                actual_action,
                n,
            )

            result[(nx, ny)] += prior * probability

    # Remove tiny floating-point noise.
    return {
        position: probability
        for position, probability in result.items()
        if probability > 1e-15
    }


def belief_to_vector(belief, map_size):
    vector = []

    for y in range(map_size):
        for x in range(map_size):
            vector.append(
                belief.get((x, y), 0.0)
            )

    return tuple(vector)


def vector_to_sparse(vector, map_size):
    result = {}

    for index, probability in enumerate(vector):
        if probability <= 1e-15:
            continue

        y, x = divmod(index, map_size)
        result[(x, y)] = probability

    return result


# ---------------------------------------------------------------------------
# Reachable belief generation
# ---------------------------------------------------------------------------

def generate_reachable_beliefs(
    map_data,
    map_size,
    obstacles,
    target,
    max_steps=10,
    p=0.01,
):
    """
    Generate all beliefs relevant to the fixed MAPE controller.

    After a perfect update, xhat,yhat equals the ground truth. We therefore
    start once from every free grid position. From each such position, the
    MAPE controller determines a unique commanded-action sequence.

    Each generated record stores:
      - full 100-dimensional positional belief
      - current nominal estimate xhat,yhat
      - age since last update
      - next commanded action
      - exact full successor belief
    """
    controller = build_controller(
        map_data,
        target,
    )

    records = []
    vector_to_record_id = {}

    def add_record(
        belief,
        xhat,
        yhat,
        age,
        action,
        successor_vector,
    ):
        vector = belief_to_vector(
            belief,
            map_size,
        )

        key = (
            tuple(round(value, 14) for value in vector),
            xhat,
            yhat,
            age,
        )

        if key in vector_to_record_id:
            record_id = vector_to_record_id[key]
            records[record_id]["occurrences"] += 1
            return record_id

        record_id = len(records)
        vector_to_record_id[key] = record_id

        records.append(
            {
                "id": record_id,
                "vector": vector,
                "xhat": xhat,
                "yhat": yhat,
                "age": age,
                "action": action,
                "successor_vector": successor_vector,
                "occurrences": 1,
            }
        )

        return record_id

    for start_x in range(map_size):
        for start_y in range(map_size):
            if (start_x, start_y) in obstacles:
                continue

            belief = point_belief(
                start_x,
                start_y,
            )

            xhat = start_x
            yhat = start_y

            for age in range(max_steps + 1):
                action = None
                successor_vector = None

                if (
                    age < max_steps
                    and (xhat, yhat) != target
                ):
                    action = controller_direction(
                        controller,
                        xhat,
                        yhat,
                    )

                if action is not None:
                    successor_belief = propagate_full_belief(
                        belief,
                        action,
                        map_size - 1,
                        p=p,
                    )

                    successor_vector = belief_to_vector(
                        successor_belief,
                        map_size,
                    )

                add_record(
                    belief,
                    xhat,
                    yhat,
                    age,
                    action,
                    successor_vector,
                )

                if action is None:
                    break

                belief = successor_belief

                xhat, yhat = nominal_next_position(
                    xhat,
                    yhat,
                    action,
                    map_size - 1,
                )

    return records


# ---------------------------------------------------------------------------
# Distances / medoid quantisation
# ---------------------------------------------------------------------------

def l1_distance(a, b):
    return sum(
        abs(x - y)
        for x, y in zip(a, b)
    )


def build_distance_matrix(vectors):
    n = len(vectors)

    matrix = [
        [0.0] * n
        for _ in range(n)
    ]

    for i in range(n):
        for j in range(i + 1, n):
            distance = l1_distance(
                vectors[i],
                vectors[j],
            )

            matrix[i][j] = distance
            matrix[j][i] = distance

    return matrix


def farthest_first_initialisation(
    distance_matrix,
    k,
):
    """
    Deterministic k-center-like initialisation.
    """
    n = len(distance_matrix)

    if k > n:
        raise ValueError(
            f"K={k} > number of beliefs={n}"
        )

    medoids = [0]
    medoid_set = {0}

    nearest = [
        distance_matrix[i][0]
        for i in range(n)
    ]

    while len(medoids) < k:
        candidate = max(
            (
                i
                for i in range(n)
                if i not in medoid_set
            ),
            key=lambda i: (
                nearest[i],
                -i,
            ),
        )

        medoids.append(candidate)
        medoid_set.add(candidate)

        for i in range(n):
            nearest[i] = min(
                nearest[i],
                distance_matrix[i][candidate],
            )

    return medoids


def assign_to_medoids(
    distance_matrix,
    medoids,
    weights=None,
):
    n = len(distance_matrix)

    assignment = []
    clusters = {
        medoid: []
        for medoid in medoids
    }

    for i in range(n):
        medoid = min(
            medoids,
            key=lambda m: (
                distance_matrix[i][m],
                m,
            ),
        )

        assignment.append(medoid)
        clusters[medoid].append(i)

    return assignment, clusters


def refine_medoids(
    distance_matrix,
    medoids,
    clusters,
    weights,
):
    refined = []

    for medoid in medoids:
        members = clusters[medoid]

        best = min(
            members,
            key=lambda candidate: (
                sum(
                    weights[member]
                    * distance_matrix[
                        candidate
                    ][member]
                    for member in members
                ),
                candidate,
            ),
        )

        refined.append(best)

    return refined


def quantise_vectors(
    vectors,
    weights,
    k,
    max_iterations=8,
):
    distance_matrix = build_distance_matrix(
        vectors
    )

    medoids = farthest_first_initialisation(
        distance_matrix,
        k,
    )

    for _ in range(max_iterations):
        assignment, clusters = assign_to_medoids(
            distance_matrix,
            medoids,
        )

        refined = refine_medoids(
            distance_matrix,
            medoids,
            clusters,
            weights,
        )

        if set(refined) == set(medoids):
            medoids = refined
            break

        medoids = refined

    assignment, clusters = assign_to_medoids(
        distance_matrix,
        medoids,
    )

    medoids = sorted(set(medoids))

    medoid_to_state = {
        medoid: state_id
        for state_id, medoid in enumerate(medoids)
    }

    # Re-assign against final medoid set.
    assignment, clusters = assign_to_medoids(
        distance_matrix,
        medoids,
    )

    candidate_to_state = {
        i: medoid_to_state[assignment[i]]
        for i in range(len(vectors))
    }

    weighted_distance_sum = 0.0
    total_weight = 0.0
    max_distance = 0.0

    for i, assigned_medoid in enumerate(assignment):
        distance = distance_matrix[
            i
        ][assigned_medoid]

        weighted_distance_sum += (
            weights[i] * distance
        )
        total_weight += weights[i]
        max_distance = max(
            max_distance,
            distance,
        )

    return {
        "medoid_indices": medoids,
        "candidate_to_state": candidate_to_state,
        "mean_error": (
            weighted_distance_sum
            / total_weight
            if total_weight
            else 0.0
        ),
        "max_error": max_distance,
    }


# ---------------------------------------------------------------------------
# Transition quantisation error
# ---------------------------------------------------------------------------

def nearest_medoid_for_vector(
    vector,
    medoid_vectors,
):
    best_state = None
    best_distance = None

    for state_id, representative in enumerate(
        medoid_vectors
    ):
        distance = l1_distance(
            vector,
            representative,
        )

        if (
            best_distance is None
            or distance < best_distance - 1e-15
            or (
                abs(distance - best_distance) <= 1e-15
                and state_id < best_state
            )
        ):
            best_state = state_id
            best_distance = distance

    return best_state, best_distance


def evaluate_k(
    records,
    k,
):
    vectors = [
        record["vector"]
        for record in records
    ]

    weights = [
        record["occurrences"]
        for record in records
    ]

    quantised = quantise_vectors(
        vectors,
        weights,
        k=k,
    )

    medoid_vectors = [
        vectors[index]
        for index in quantised[
            "medoid_indices"
        ]
    ]

    transition_errors = []

    for record in records:
        successor = record[
            "successor_vector"
        ]

        if successor is None:
            continue

        _, distance = nearest_medoid_for_vector(
            successor,
            medoid_vectors,
        )

        transition_errors.append(
            distance
        )

    return {
        "k": k,
        "representation_mean": quantised[
            "mean_error"
        ],
        "representation_max": quantised[
            "max_error"
        ],
        "transition_mean": (
            sum(transition_errors)
            / len(transition_errors)
            if transition_errors
            else 0.0
        ),
        "transition_max": (
            max(transition_errors)
            if transition_errors
            else 0.0
        ),
        "medoid_indices": quantised[
            "medoid_indices"
        ],
    }


# ---------------------------------------------------------------------------
# K selection
# ---------------------------------------------------------------------------

def select_k_by_error_tolerance(
    results,
    transition_tolerance=0.05,
    representation_tolerance=0.02,
):
    """
    Primary recommendation:
    choose the SMALLEST K satisfying explicit approximation tolerances.

    This is easier to justify scientifically than arbitrary weighted scores.
    """
    acceptable = [
        result
        for result in sorted(
            results,
            key=lambda row: row["k"],
        )
        if (
            result["transition_mean"]
            <= transition_tolerance
            and result["representation_mean"]
            <= representation_tolerance
        )
    ]

    if acceptable:
        return acceptable[0], True

    # No K meets both targets: return the largest tested model and flag it.
    return max(
        results,
        key=lambda row: row["k"],
    ), False


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def write_k_evaluation(
    path,
    results,
    selected_k,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "k",
                "representation_mean_l1",
                "representation_max_l1",
                "transition_mean_l1",
                "transition_max_l1",
                "selected",
            ]
        )

        for result in sorted(
            results,
            key=lambda row: row["k"],
        ):
            writer.writerow(
                [
                    result["k"],
                    result[
                        "representation_mean"
                    ],
                    result[
                        "representation_max"
                    ],
                    result[
                        "transition_mean"
                    ],
                    result[
                        "transition_max"
                    ],
                    int(
                        result["k"]
                        == selected_k
                    ),
                ]
            )


def write_selected_representatives(
    path,
    records,
    selected_result,
    map_size,
):
    vectors = [
        record["vector"]
        for record in records
    ]

    medoid_indices = selected_result[
        "medoid_indices"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        header = [
            "belief_state",
            "source_record",
        ]

        for y in range(map_size):
            for x in range(map_size):
                header.append(
                    f"p_{x}_{y}"
                )

        writer.writerow(header)

        for belief_state, record_index in enumerate(
            medoid_indices
        ):
            writer.writerow(
                [
                    belief_state,
                    record_index,
                    *vectors[record_index],
                ]
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build full positional beliefs for one PARLEY robot map and "
            "quantise the reachable belief space to a finite representative set."
        )
    )

    parser.add_argument(
        "--map-id",
        type=int,
        default=10,
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
        "--max-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--p",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--transition-tolerance",
        type=float,
        default=0.05,
        help=(
            "Maximum desired mean L1 error for predicted beliefs "
            "after quantisation."
        ),
    )

    parser.add_argument(
        "--representation-tolerance",
        type=float,
        default=0.02,
        help=(
            "Maximum desired mean L1 error for reachable-belief representation."
        ),
    )

    args = parser.parse_args()

    map_path = (
        f"{args.maps_dir}/map_{args.map_id}.csv"
    )

    map_data, map_size, obstacles = (
        read_map(map_path)
    )

    records = generate_reachable_beliefs(
        map_data=map_data,
        map_size=map_size,
        obstacles=obstacles,
        target=(
            args.target_x,
            args.target_y,
        ),
        max_steps=args.max_steps,
        p=args.p,
    )

    number_of_beliefs = len(records)

    print(
        f"Reachable full beliefs: {number_of_beliefs}"
    )

    base_k_values = [
        10,
        20,
        30,
        50,
        75,
        100,
        150,
        200,
        300,
        400,
        500,
        750,
        1000,
    ]

    k_values = sorted(
        {
            k
            for k in base_k_values
            if k <= number_of_beliefs
        }
        | {number_of_beliefs}
    )

    results = []

    for k in k_values:
        print(
            f"Evaluating K={k} ..."
        )

        result = evaluate_k(
            records,
            k=k,
        )

        results.append(result)

        print(
            "  representation mean="
            f'{result["representation_mean"]:.6f}, '
            "transition mean="
            f'{result["transition_mean"]:.6f}'
        )

    selected, tolerance_met = (
        select_k_by_error_tolerance(
            results,
            transition_tolerance=(
                args.transition_tolerance
            ),
            representation_tolerance=(
                args.representation_tolerance
            ),
        )
    )

    print()
    print(
        f"Selected K={selected['k']}"
    )

    if tolerance_met:
        print(
            "Selection reason: smallest K satisfying both error tolerances."
        )
    else:
        print(
            "WARNING: no tested K satisfied both requested tolerances; "
            "largest tested K selected."
        )

    write_k_evaluation(
        "full_belief_k_evaluation.csv",
        results,
        selected_k=selected["k"],
    )

    write_selected_representatives(
        "full_belief_representatives.csv",
        records,
        selected,
        map_size,
    )

    print()
    print(
        "Written: full_belief_k_evaluation.csv"
    )
    print(
        "Written: full_belief_representatives.csv"
    )


if __name__ == "__main__":
    main()
