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
):
    """
    Weighted k-medoids-style quantisation with farthest-first initialisation.

    The refinement now runs until the medoid set no longer changes.
    max_iter is only a safety limit.
    """
    vectors = [item["vector"] for item in candidates]
    weights = [item["weight"] for item in candidates]

    if len(vectors) <= k:
        medoids = list(range(len(vectors)))
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

        # Refine until convergence.
        converged = False

        for _ in range(max_iter):
            clusters = {
                medoid: []
                for medoid in medoids
            }

            for i in range(len(vectors)):
                medoid = min(
                    medoids,
                    key=lambda candidate: (
                        matrix[i][candidate],
                        candidate,
                    ),
                )
                clusters[medoid].append(i)

            refined = []

            for medoid in medoids:
                members = clusters[medoid]

                # Keep the exact certainty belief fixed.
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

            if set(refined) == set(medoids):
                medoids = refined
                converged = True
                break

            medoids = refined

        if not converged:
            print(
                f"Warning: k-medoids did not converge "
                f"within max_iter={max_iter}"
            )

    # Put certainty representative first.
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

    return [
        vectors[i]
        for i in medoids
    ]


def _nearest(vector, representatives):
    return min(
        range(len(representatives)),
        key=lambda i: (
            _l1(vector, representatives[i]),
            i,
        ),
    )


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

    # Always rebuild the belief model so existing map_X.json files
    # are overwritten with the newly converged medoids.


    candidates, gini_by_age, controller = _generate_records(
        map_data, target, p, max_steps
    )

    representatives = _cluster(
        candidates,
        min(k, len(candidates)),
        max_iter=100,
    )

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
    }

    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(model, file)

    return model


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
    precompute_maps()
