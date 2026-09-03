import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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


def _relative_vector(belief, xhat, yhat, n):
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


def _propagate_relative_vector(vector, xhat, yhat, action, n, p):
    relative = _vector_to_relative(vector, n)
    absolute = defaultdict(float)

    for (dx, dy), probability in relative.items():
        ax = _clip(xhat + dx, n)
        ay = _clip(yhat + dy, n)
        absolute[(ax, ay)] += probability

    propagated = _propagate_absolute(absolute, action, n, p)
    nxhat, nyhat = _move(xhat, yhat, action, n)

    return (
        _relative_vector(propagated, nxhat, nyhat, n),
        nxhat,
        nyhat,
    )


def _gini_vector(vector):
    return 1.0 - sum(float(v) * float(v) for v in vector)


def _scaled_gini(vector):
    return int(round(_gini_vector(vector) * 10000))


def _urc_level(uncertainty, thresholds):
    level = 0
    for index, threshold in enumerate(thresholds, start=1):
        if uncertainty >= threshold:
            level = index
        else:
            break
    return level


def _thresholds(gini_by_age, max_steps):
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

        scaled = max(previous, int(round(value * 10000)))
        result.append(scaled)
        previous = scaled

    return result


def generate_exact_occurrences(map_data, target, p=0.01, max_steps=10):
    """
    Enumerate the same reachable MAPE belief trajectories used for threshold
    calibration, but retain full vectors and exact one-step successors.
    """
    map_size = len(map_data)
    n = map_size - 1
    controller = _controller(map_data, target)

    occurrences = []
    gini_by_age = defaultdict(list)

    for sx in range(map_size):
        for sy in range(map_size):
            if int(map_data[sx][sy]) > 9:
                continue

            belief = {(sx, sy): 1.0}
            xhat, yhat = sx, sy

            for age in range(max_steps + 1):
                vector = _relative_vector(belief, xhat, yhat, n)
                gini_by_age[age].append(_gini_vector(vector))

                action = None
                successor_vector = None
                nxhat = None
                nyhat = None

                if age < max_steps and (xhat, yhat) != target:
                    action = _direction(controller, xhat, yhat)
                    if action is not None:
                        successor_belief = _propagate_absolute(
                            belief, action, n, p
                        )
                        nxhat, nyhat = _move(xhat, yhat, action, n)
                        successor_vector = _relative_vector(
                            successor_belief, nxhat, nyhat, n
                        )

                occurrences.append({
                    "start_x": sx,
                    "start_y": sy,
                    "age": age,
                    "xhat": xhat,
                    "yhat": yhat,
                    "vector": vector,
                    "action": action,
                    "successor_vector": successor_vector,
                    "next_xhat": nxhat,
                    "next_yhat": nyhat,
                })

                if action is None:
                    break

                belief = successor_belief
                xhat, yhat = nxhat, nyhat

    thresholds = _thresholds(gini_by_age, max_steps)

    for occ in occurrences:
        occ["uncertainty"] = _scaled_gini(occ["vector"])
        occ["urc_level"] = _urc_level(
            occ["uncertainty"], thresholds
        )
        if occ["successor_vector"] is not None:
            occ["successor_uncertainty"] = _scaled_gini(
                occ["successor_vector"]
            )
            occ["successor_urc_level"] = _urc_level(
                occ["successor_uncertainty"], thresholds
            )
        else:
            occ["successor_uncertainty"] = None
            occ["successor_urc_level"] = None

    return occurrences, thresholds, controller


def _l1(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def _certain_vector(length):
    result = np.zeros(length, dtype=float)
    result[length // 2] = 1.0
    return tuple(result.tolist())


def _deduplicate_local(occurrences):
    """
    Group reachable exact beliefs by current estimated position. Occurrence
    counts are retained so farthest-first tie-breaking can prefer frequent
    beliefs.
    """
    local = defaultdict(dict)

    for occ in occurrences:
        pos = (occ["xhat"], occ["yhat"])
        key = tuple(round(v, 14) for v in occ["vector"])

        if key not in local[pos]:
            local[pos][key] = {
                "vector": occ["vector"],
                "weight": 1,
            }
        else:
            local[pos][key]["weight"] += 1

    return {
        pos: list(entries.values())
        for pos, entries in local.items()
    }


def _select_grid_points(candidates, k):
    """
    Position-local farthest-first grid selection.

    All grid points are actual reachable beliefs. The certain belief is kept
    as local state 0 whenever it occurs (it does for every traversable reset
    position in this model).
    """
    if not candidates:
        return []

    vectors = [item["vector"] for item in candidates]
    weights = [item["weight"] for item in candidates]
    k = min(k, len(vectors))

    certain = _certain_vector(len(vectors[0]))
    zero_index = min(
        range(len(vectors)),
        key=lambda i: (_l1(vectors[i], certain), -weights[i], i),
    )

    if k == len(vectors):
        selected = list(range(len(vectors)))
        selected.remove(zero_index)
        selected = [zero_index] + selected
        return [vectors[i] for i in selected]

    selected = [zero_index]
    nearest = [
        _l1(vector, vectors[zero_index])
        for vector in vectors
    ]

    while len(selected) < k:
        nxt = max(
            (i for i in range(len(vectors)) if i not in selected),
            key=lambda i: (nearest[i], weights[i], -i),
        )
        selected.append(nxt)

        for i, vector in enumerate(vectors):
            nearest[i] = min(
                nearest[i],
                _l1(vector, vectors[nxt]),
            )

    return [vectors[i] for i in selected]


def build_local_grids(occurrences, grid_per_position):
    local_candidates = _deduplicate_local(occurrences)
    grids = {}

    for pos, candidates in local_candidates.items():
        grids[pos] = _select_grid_points(
            candidates,
            grid_per_position,
        )

    return grids


def _interpolate(vector, grid, neighbours=5, tol=1e-10):
    """
    Convex interpolation on the nearest local grid points:

        min ||G lambda - b||_2^2
        s.t. lambda_i >= 0, sum(lambda_i) = 1.

    Returns local grid-state IDs and interpolation weights.
    """
    if not grid:
        raise ValueError("Cannot interpolate into an empty grid.")

    b = np.asarray(vector, dtype=float)
    matrix_all = np.asarray(grid, dtype=float)

    l1_distances = np.abs(matrix_all - b).sum(axis=1)
    exact = int(np.argmin(l1_distances))

    if l1_distances[exact] <= tol:
        return [(exact, 1.0)]

    m = min(max(1, neighbours), len(grid))
    ids = np.argsort(l1_distances)[:m]
    G = matrix_all[ids].T

    inverse = 1.0 / np.maximum(l1_distances[ids], 1e-12)
    x0 = inverse / inverse.sum()

    def objective(weights):
        diff = G @ weights - b
        return float(diff @ diff)

    constraints = [{
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
    }]
    bounds = [(0.0, 1.0)] * len(ids)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 300},
    )

    if result.success:
        weights = np.maximum(result.x, 0.0)
    else:
        # Safe numerical fallback. Still convex, but not least-squares optimal.
        weights = x0

    total = float(weights.sum())
    if total <= 0:
        return [(exact, 1.0)]
    weights /= total

    answer = [
        (int(state_id), float(weight))
        for state_id, weight in zip(ids, weights)
        if weight > 1e-9
    ]

    # Renormalise after dropping tiny coefficients.
    total = sum(weight for _, weight in answer)
    return [
        (state_id, weight / total)
        for state_id, weight in answer
    ]


def _reconstruct(interpolation, grid):
    result = np.zeros(len(grid[0]), dtype=float)
    for state_id, weight in interpolation:
        result += weight * np.asarray(grid[state_id], dtype=float)
    result[result < 0] = 0.0
    total = float(result.sum())
    if total > 0:
        result /= total
    return tuple(result.tolist())


def build_grid_transitions(
    grids,
    map_data,
    controller,
    p=0.01,
    neighbours=5,
):
    """
    Construct a finite position-local grid model.

    The weights are interpolation coefficients. They can be emitted as PRISM
    probabilistic branches for an experimental abstraction, but should be
    interpreted as numerical abstraction weights, not physical robot-motion
    probabilities.
    """
    n = len(map_data) - 1
    transitions = {}

    for (xhat, yhat), grid in sorted(grids.items()):
        action = _direction(controller, xhat, yhat)
        if action is None:
            continue

        nxhat, nyhat = _move(xhat, yhat, action, n)
        successor_grid = grids.get((nxhat, nyhat))
        if not successor_grid:
            continue

        for state_id, vector in enumerate(grid):
            successor_vector, _, _ = _propagate_relative_vector(
                vector, xhat, yhat, action, n, p
            )
            interpolation = _interpolate(
                successor_vector,
                successor_grid,
                neighbours=neighbours,
            )

            transitions[f"{xhat},{yhat},{state_id}"] = {
                "action": action,
                "next_xhat": nxhat,
                "next_yhat": nyhat,
                "successors": [
                    {
                        "state": successor_id,
                        "weight": weight,
                    }
                    for successor_id, weight in interpolation
                ],
            }

    return transitions


def analyse_grid(
    occurrences,
    thresholds,
    grids,
    p=0.01,
    neighbours=5,
):
    """
    Compare exact reachable beliefs with the grid/interpolation abstraction.

    Two URC views are reported:
      reconstructed_*:
        classify the convexly reconstructed belief itself.
      component_weighted_*:
        if interpolation coefficients are interpreted as PRISM branches,
        report the probability mass landing in a grid state with the wrong
        URC level.

    successor_* repeats the comparison after one exact MAPE prediction from
    the reconstructed current belief and interpolation in the successor grid.
    """
    n = int((len(occurrences[0]["vector"]) ** 0.5 - 1) / 2)

    current_total = 0
    transition_total = 0

    current_reconstructed_mismatches = 0
    successor_reconstructed_mismatches = 0

    current_component_weighted_mismatch = 0.0
    successor_component_weighted_mismatch = 0.0

    current_l1_sum = 0.0
    successor_l1_sum = 0.0

    current_level_abs_error = 0.0
    successor_level_abs_error = 0.0

    current_support_sum = 0
    successor_support_sum = 0

    worst_current_l1 = 0.0
    worst_successor_l1 = 0.0

    for occ in occurrences:
        pos = (occ["xhat"], occ["yhat"])
        grid = grids[pos]

        interpolation = _interpolate(
            occ["vector"],
            grid,
            neighbours=neighbours,
        )
        reconstructed = _reconstruct(interpolation, grid)

        exact_level = occ["urc_level"]
        reconstructed_level = _urc_level(
            _scaled_gini(reconstructed),
            thresholds,
        )

        current_total += 1
        current_reconstructed_mismatches += int(
            reconstructed_level != exact_level
        )
        current_level_abs_error += abs(
            reconstructed_level - exact_level
        )
        current_support_sum += len(interpolation)

        weighted_wrong = 0.0
        for state_id, weight in interpolation:
            grid_level = _urc_level(
                _scaled_gini(grid[state_id]),
                thresholds,
            )
            if grid_level != exact_level:
                weighted_wrong += weight
        current_component_weighted_mismatch += weighted_wrong

        current_l1 = _l1(occ["vector"], reconstructed)
        current_l1_sum += current_l1
        worst_current_l1 = max(worst_current_l1, current_l1)

        if occ["action"] is None:
            continue

        successor_exact = occ["successor_vector"]
        next_pos = (occ["next_xhat"], occ["next_yhat"])
        next_grid = grids[next_pos]

        # Because prediction is linear, propagating the reconstructed convex
        # belief is equivalent to propagating its weighted grid components.
        successor_from_reconstruction, _, _ = _propagate_relative_vector(
            reconstructed,
            occ["xhat"],
            occ["yhat"],
            occ["action"],
            n,
            p,
        )

        successor_interpolation = _interpolate(
            successor_from_reconstruction,
            next_grid,
            neighbours=neighbours,
        )
        successor_reconstructed = _reconstruct(
            successor_interpolation,
            next_grid,
        )

        exact_successor_level = occ["successor_urc_level"]
        approx_successor_level = _urc_level(
            _scaled_gini(successor_reconstructed),
            thresholds,
        )

        transition_total += 1
        successor_reconstructed_mismatches += int(
            approx_successor_level != exact_successor_level
        )
        successor_level_abs_error += abs(
            approx_successor_level - exact_successor_level
        )
        successor_support_sum += len(successor_interpolation)

        successor_weighted_wrong = 0.0
        for state_id, weight in successor_interpolation:
            grid_level = _urc_level(
                _scaled_gini(next_grid[state_id]),
                thresholds,
            )
            if grid_level != exact_successor_level:
                successor_weighted_wrong += weight
        successor_component_weighted_mismatch += successor_weighted_wrong

        successor_l1 = _l1(
            successor_exact,
            successor_reconstructed,
        )
        successor_l1_sum += successor_l1
        worst_successor_l1 = max(
            worst_successor_l1,
            successor_l1,
        )

    local_sizes = [len(grid) for grid in grids.values()]

    return {
        "exact_occurrences": current_total,
        "exact_transitions": transition_total,
        "grid_positions": len(grids),
        "grid_state_contexts": sum(local_sizes),
        "mean_grid_states_per_position": (
            sum(local_sizes) / len(local_sizes)
            if local_sizes else 0.0
        ),
        "max_grid_states_per_position": max(local_sizes, default=0),

        "current_urc_mismatch_rate": (
            current_reconstructed_mismatches / current_total
            if current_total else 0.0
        ),
        "current_component_weighted_urc_mismatch_rate": (
            current_component_weighted_mismatch / current_total
            if current_total else 0.0
        ),
        "current_mean_abs_urc_level_error": (
            current_level_abs_error / current_total
            if current_total else 0.0
        ),

        "successor_urc_mismatch_rate": (
            successor_reconstructed_mismatches / transition_total
            if transition_total else 0.0
        ),
        "successor_component_weighted_urc_mismatch_rate": (
            successor_component_weighted_mismatch / transition_total
            if transition_total else 0.0
        ),
        "successor_mean_abs_urc_level_error": (
            successor_level_abs_error / transition_total
            if transition_total else 0.0
        ),

        "mean_current_l1_reconstruction_error": (
            current_l1_sum / current_total
            if current_total else 0.0
        ),
        "max_current_l1_reconstruction_error": worst_current_l1,
        "mean_successor_l1_reconstruction_error": (
            successor_l1_sum / transition_total
            if transition_total else 0.0
        ),
        "max_successor_l1_reconstruction_error": worst_successor_l1,

        "mean_current_interpolation_support": (
            current_support_sum / current_total
            if current_total else 0.0
        ),
        "mean_successor_interpolation_support": (
            successor_support_sum / transition_total
            if transition_total else 0.0
        ),
    }


def build_grid_belief_model(
    map_id,
    map_data,
    target,
    p=0.01,
    grid_per_position=5,
    neighbours=5,
    max_steps=10,
    cache_dir="belief_grid_models",
):
    occurrences, thresholds, controller = generate_exact_occurrences(
        map_data,
        target,
        p=p,
        max_steps=max_steps,
    )

    grids = build_local_grids(
        occurrences,
        grid_per_position=grid_per_position,
    )

    transitions = build_grid_transitions(
        grids,
        map_data,
        controller,
        p=p,
        neighbours=neighbours,
    )

    analysis = analyse_grid(
        occurrences,
        thresholds,
        grids,
        p=p,
        neighbours=neighbours,
    )

    uncertainties = {
        f"{x},{y}": [
            _scaled_gini(vector)
            for vector in grid
        ]
        for (x, y), grid in grids.items()
    }

    model = {
        "map_id": map_id,
        "p": p,
        "max_steps": max_steps,
        "grid_per_position": grid_per_position,
        "neighbours": neighbours,
        "thresholds": thresholds,
        "max_local_states": max(
            (len(grid) for grid in grids.values()),
            default=1,
        ),
        "local_state_counts": {
            f"{x},{y}": len(grid)
            for (x, y), grid in grids.items()
        },
        "uncertainties": uncertainties,
        "transitions": transitions,
        "analysis": analysis,
    }

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"map_{map_id}.json"
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)

    return model


def read_map_data(map_path):
    rows = []
    with open(map_path, "r", newline="") as file:
        rows.extend(csv.reader(file))
    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


def main():
    parser = argparse.ArgumentParser(
        description="Build/test a reachable-belief grid interpolation model."
    )
    parser.add_argument("--map-id", type=int, required=True)
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument("--target-x", type=int, default=9)
    parser.add_argument("--target-y", type=int, default=9)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--grid-per-position", type=int, default=5)
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--cache-dir", default="belief_grid_models")
    args = parser.parse_args()

    map_data = read_map_data(
        Path(args.maps_dir) / f"map_{args.map_id}.csv"
    )

    model = build_grid_belief_model(
        map_id=args.map_id,
        map_data=map_data,
        target=(args.target_x, args.target_y),
        p=args.p,
        grid_per_position=args.grid_per_position,
        neighbours=args.neighbours,
        max_steps=args.max_steps,
        cache_dir=args.cache_dir,
    )

    print(json.dumps(model["analysis"], indent=2))


if __name__ == "__main__":
    main()
