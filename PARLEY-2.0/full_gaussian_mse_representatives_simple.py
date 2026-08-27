#!/usr/bin/env python3
"""
Final simplified Gaussian builder for PARLEY.

Architecture adapted from the original full_gaussian_representatives.py:

    Gaussian state:
        G = (mu_x, mu_y, var_x, var_y, cov_xy)

    Direct Gaussian prediction:
        mu'    = mu + E[D - d_hat]
        Sigma' = Sigma + Cov(D)

    Uncertainty:
        MSE = trace(Sigma) + ||mu||^2

    Clustering:
        weighted k-medoids, K=100

    Distance:
        w2_mse1_level1

    Representative transition:
        representative Gaussian
          -> direct Gaussian prediction
          -> nearest representative

No full belief vector is propagated for representative transitions.
"""

from __future__ import annotations

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

K_DEFAULT = 100
MAX_STEPS_DEFAULT = 10
P_DEFAULT = 0.01
LAMBDA_MSE = 1.0
LAMBDA_LEVEL = 1.0


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
        return (
            (intended, 1, 0),
            (p, 0, 1),
            (p, 0, -1),
            (p, -1, 0),
        )
    if action == "west":
        return (
            (p, 1, 0),
            (p, 0, 1),
            (p, 0, -1),
            (intended, -1, 0),
        )
    if action == "north":
        return (
            (p, 1, 0),
            (intended, 0, 1),
            (p, 0, -1),
            (p, -1, 0),
        )
    if action == "south":
        return (
            (p, 1, 0),
            (p, 0, 1),
            (intended, 0, -1),
            (p, -1, 0),
        )
    raise ValueError(f"Unknown action: {action}")


def _motion_error_moments(xhat, yhat, action, n, p):
    """
    Moments of eta = D - d_hat.

    D is the stochastic Robot displacement.
    d_hat is the deterministic Knowledge displacement.
    Grid clipping is approximated at the current estimated position.
    """
    dx_hat, dy_hat = MOVE[action]

    nx_hat = _clip(xhat + dx_hat, n)
    ny_hat = _clip(yhat + dy_hat, n)

    deterministic_dx = float(nx_hat - xhat)
    deterministic_dy = float(ny_hat - yhat)

    samples = []

    for probability, dx, dy in _robot_outcomes(action, p):
        nx = _clip(xhat + dx, n)
        ny = _clip(yhat + dy, n)

        actual_dx = float(nx - xhat)
        actual_dy = float(ny - yhat)

        samples.append(
            (
                probability,
                actual_dx - deterministic_dx,
                actual_dy - deterministic_dy,
            )
        )

    mean_x = sum(probability * dx for probability, dx, _ in samples)
    mean_y = sum(probability * dy for probability, _, dy in samples)

    var_x = sum(
        probability * (dx - mean_x) ** 2
        for probability, dx, _ in samples
    )
    var_y = sum(
        probability * (dy - mean_y) ** 2
        for probability, _, dy in samples
    )
    cov_xy = sum(
        probability * (dx - mean_x) * (dy - mean_y)
        for probability, dx, dy in samples
    )

    return float(mean_x), float(mean_y), float(var_x), float(var_y), float(cov_xy)


def _gaussian_key(state, digits=14):
    def clean(value):
        rounded = round(float(value), digits)
        return 0.0 if abs(rounded) < 10 ** (-digits) else rounded

    return tuple(clean(value) for value in state)


def _trace(state):
    return state[2] + state[3]


def _bias2(state):
    return state[0] ** 2 + state[1] ** 2


def _mse(state):
    return _trace(state) + _bias2(state)


def _predict_gaussian(state, xhat, yhat, action, n, p):
    mean_x, mean_y, qx, qy, qc = _motion_error_moments(
        xhat, yhat, action, n, p
    )

    mu_x, mu_y, var_x, var_y, cov_xy = state

    return _gaussian_key(
        (
            mu_x + mean_x,
            mu_y + mean_y,
            var_x + qx,
            var_y + qy,
            cov_xy + qc,
        )
    )


def _sigma(state):
    return state[2], state[3], state[4]


def _bures_wasserstein_squared_sigma(a, b):
    ax, ay, ac = a
    bx, by, bc = b

    trace_a = ax + ay
    trace_b = bx + by
    trace_ab = ax * bx + ay * by + 2.0 * ac * bc

    det_a = ax * ay - ac * ac
    det_b = bx * by - bc * bc

    if det_a < 0.0 and abs(det_a) < 1e-12:
        det_a = 0.0
    if det_b < 0.0 and abs(det_b) < 1e-12:
        det_b = 0.0

    if det_a < 0.0 or det_b < 0.0:
        raise ValueError("Non-PSD covariance in Bures-Wasserstein distance.")

    inner = trace_ab + 2.0 * math.sqrt(det_a * det_b)

    if inner < 0.0 and abs(inner) < 1e-12:
        inner = 0.0
    if inner < 0.0:
        raise ValueError("Negative Bures inner term.")

    squared = trace_a + trace_b - 2.0 * math.sqrt(inner)

    if squared < 0.0 and abs(squared) < 1e-10:
        squared = 0.0
    if squared < 0.0:
        raise ValueError("Negative Bures-Wasserstein squared distance.")

    return squared


def _gaussian_wasserstein_squared(a, b):
    dmu_x = a[0] - b[0]
    dmu_y = a[1] - b[1]

    return (
        dmu_x * dmu_x
        + dmu_y * dmu_y
        + _bures_wasserstein_squared_sigma(_sigma(a), _sigma(b))
    )


def _generate_records(map_data, target, p, max_steps):
    map_size = len(map_data)
    n = map_size - 1
    controller = _controller(map_data, target)

    records = []
    mse_by_age = defaultdict(list)

    for start_x in range(map_size):
        for start_y in range(map_size):
            if int(map_data[start_x][start_y]) > 9:
                continue

            xhat = start_x
            yhat = start_y
            state = (0.0, 0.0, 0.0, 0.0, 0.0)

            for age in range(max_steps + 1):
                state = _gaussian_key(state)

                records.append(
                    {
                        "state": state,
                        "xhat": xhat,
                        "yhat": yhat,
                        "age": age,
                    }
                )
                mse_by_age[age].append(_mse(state))

                if age >= max_steps or (xhat, yhat) == target:
                    break

                action = _direction(controller, xhat, yhat)
                if action is None:
                    break

                state = _predict_gaussian(
                    state,
                    xhat,
                    yhat,
                    action,
                    n,
                    p,
                )
                xhat, yhat = _move(xhat, yhat, action, n)

    unique = {}

    for record in records:
        state = record["state"]

        if state not in unique:
            unique[state] = {
                "state": state,
                "weight": 1,
            }
        else:
            unique[state]["weight"] += 1

    candidates = list(unique.values())

    zero = (0.0, 0.0, 0.0, 0.0, 0.0)

    candidates.sort(
        key=lambda item: (
            0 if item["state"] == zero else 1,
            _mse(item["state"]),
            item["state"],
        )
    )

    return records, candidates, mse_by_age, controller


def _thresholds(mse_by_age, max_steps):
    result = []
    previous = 0.0

    for age in range(1, max_steps + 1):
        values = sorted(mse_by_age.get(age, []))

        if not values:
            value = previous
        else:
            middle = len(values) // 2
            if len(values) % 2:
                value = values[middle]
            else:
                value = (values[middle - 1] + values[middle]) / 2.0

        value = max(previous, value)
        result.append(value)
        previous = value

    return result


def _level_from_mse(mse_value, thresholds):
    level = 0

    for index, threshold in enumerate(thresholds, start=1):
        if mse_value >= threshold - 1e-15:
            level = index
        else:
            break

    return level


def _distance_matrix(candidates, thresholds):
    states = [item["state"] for item in candidates]
    mse_values = [_mse(state) for state in states]
    mse_max = max(max(mse_values), 1e-15)

    levels = [
        _level_from_mse(mse, thresholds)
        for mse in mse_values
    ]

    count = len(states)
    matrix = [[0.0] * count for _ in range(count)]

    for i in range(count):
        for j in range(i + 1, count):
            dmse = (mse_values[i] - mse_values[j]) / mse_max
            dlevel = (levels[i] - levels[j]) / 10.0

            squared = (
                _gaussian_wasserstein_squared(states[i], states[j])
                + LAMBDA_MSE * dmse * dmse
                + LAMBDA_LEVEL * dlevel * dlevel
            )

            value = math.sqrt(max(0.0, squared))
            matrix[i][j] = value
            matrix[j][i] = value

    return matrix, mse_max


def _cluster(candidates, k, matrix, max_iter=100):
    states = [item["state"] for item in candidates]
    weights = [item["weight"] for item in candidates]

    zero = (0.0, 0.0, 0.0, 0.0, 0.0)
    zero_index = states.index(zero)

    if len(states) <= k:
        medoids = list(range(len(states)))
    else:
        medoids = [zero_index]
        nearest = [
            matrix[i][zero_index]
            for i in range(len(states))
        ]

        while len(medoids) < k:
            next_medoid = max(
                (
                    i
                    for i in range(len(states))
                    if i not in medoids
                ),
                key=lambda i: (nearest[i], -i),
            )
            medoids.append(next_medoid)

            for i in range(len(states)):
                nearest[i] = min(
                    nearest[i],
                    matrix[i][next_medoid],
                )

        def objective(current_medoids):
            return sum(
                weights[i]
                * min(matrix[i][medoid] for medoid in current_medoids)
                for i in range(len(states))
            )

        best_medoids = list(medoids)
        best_objective = objective(medoids)
        seen = {}

        for iteration in range(max_iter):
            key = tuple(sorted(medoids))

            if key in seen:
                medoids = best_medoids
                break

            seen[key] = iteration

            clusters = {
                medoid: []
                for medoid in medoids
            }

            for i in range(len(states)):
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

                if medoid == zero_index:
                    refined.append(medoid)
                    continue

                best = min(
                    members,
                    key=lambda candidate: (
                        sum(
                            weights[member]
                            * matrix[candidate][member]
                            for member in members
                        ),
                        candidate,
                    ),
                )
                refined.append(best)

            refined_objective = objective(refined)

            if refined_objective < best_objective:
                best_objective = refined_objective
                best_medoids = list(refined)

            if set(refined) == set(medoids):
                medoids = refined
                best_medoids = list(refined)
                break

            medoids = refined

        medoids = best_medoids

    medoids = [zero_index] + sorted(
        medoid
        for medoid in medoids
        if medoid != zero_index
    )

    return medoids


def _nearest_state(state, candidates, medoids, thresholds, mse_max):
    zero = (0.0, 0.0, 0.0, 0.0, 0.0)

    if _gaussian_key(state) == zero:
        return 0

    state_mse = _mse(state)
    state_level = _level_from_mse(state_mse, thresholds)

    best_block = 1
    best_value = float("inf")

    for block in range(1, len(medoids)):
        representative = candidates[medoids[block]]["state"]

        rep_mse = _mse(representative)
        rep_level = _level_from_mse(rep_mse, thresholds)

        dmse = (state_mse - rep_mse) / mse_max
        dlevel = (state_level - rep_level) / 10.0

        value = (
            _gaussian_wasserstein_squared(
                state,
                representative,
            )
            + LAMBDA_MSE * dmse * dmse
            + LAMBDA_LEVEL * dlevel * dlevel
        )

        if value < best_value:
            best_value = value
            best_block = block

    return best_block


def _build_transitions(
    map_data,
    controller,
    candidates,
    medoids,
    thresholds,
    mse_max,
    p,
):
    n = len(map_data) - 1
    transitions = {}

    for x in range(len(map_data)):
        for y in range(len(map_data)):
            action = _direction(controller, x, y)

            if action is None:
                continue

            for state_id, candidate_index in enumerate(medoids):
                state = candidates[candidate_index]["state"]

                successor_state = _predict_gaussian(
                    state,
                    x,
                    y,
                    action,
                    n,
                    p,
                )

                next_state = _nearest_state(
                    successor_state,
                    candidates,
                    medoids,
                    thresholds,
                    mse_max,
                )

                transitions[f"{x},{y},{state_id}"] = {
                    "action": action,
                    "next_state": next_state,
                }

    return transitions


def build_gaussian_model(
    map_id,
    map_data,
    target=(9, 9),
    p=P_DEFAULT,
    k=K_DEFAULT,
    max_steps=MAX_STEPS_DEFAULT,
    cache_dir="gaussian_simple_models",
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records, candidates, mse_by_age, controller = _generate_records(
        map_data,
        target,
        p,
        max_steps,
    )

    thresholds = _thresholds(
        mse_by_age,
        max_steps,
    )

    matrix, mse_max = _distance_matrix(
        candidates,
        thresholds,
    )

    medoids = _cluster(
        candidates,
        min(k, len(candidates)),
        matrix,
        max_iter=100,
    )

    representatives = []

    for state_id, candidate_index in enumerate(medoids):
        state = candidates[candidate_index]["state"]
        mse = _mse(state)

        representatives.append(
            {
                "state_id": state_id,
                "candidate_index": candidate_index,
                "mu_x": state[0],
                "mu_y": state[1],
                "var_x": state[2],
                "var_y": state[3],
                "cov_xy": state[4],
                "trace": _trace(state),
                "bias2": _bias2(state),
                "mse": mse,
                "uncertainty_level": _level_from_mse(
                    mse,
                    thresholds,
                ),
                "weight": candidates[candidate_index]["weight"],
            }
        )

    transitions = _build_transitions(
        map_data,
        controller,
        candidates,
        medoids,
        thresholds,
        mse_max,
        p,
    )

    model = {
        "schema_version": 1,
        "model_type": "gaussian_mse_direct_simple",
        "map_id": map_id,
        "k": k,
        "state_count": len(medoids),
        "max_steps": max_steps,
        "p": p,
        "metric": "w2_mse1_level1",
        "lambda_mse": LAMBDA_MSE,
        "lambda_level": LAMBDA_LEVEL,
        "uncertainty_metric": "mse",
        "transition_semantics": (
            "direct Gaussian moment prediction: "
            "mu'=mu+E[D-d_hat], "
            "Sigma'=Sigma+Cov(D), "
            "then nearest representative"
        ),
        "thresholds": thresholds,
        "uncertainties": [
            representative["mse"]
            for representative in representatives
        ],
        "representatives": representatives,
        "transitions": transitions,
        "raw_history_count": len(records),
        "unique_gaussian_count": len(candidates),
    }

    zero = model["representatives"][0]

    for field in (
        "mu_x",
        "mu_y",
        "var_x",
        "var_y",
        "cov_xy",
    ):
        if abs(float(zero[field])) > 1e-15:
            raise ValueError(
                "Representative 0 is not exact certainty."
            )

    cache_path = cache_dir / f"map_{map_id}.json"

    with cache_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model,
            file,
            indent=2,
        )

    return model


def _load_map(path):
    rows = []

    with path.open(
        "r",
        newline="",
    ) as file:
        rows.extend(csv.reader(file))

    transposed = list(zip(*rows))

    return [
        list(row[::-1])
        for row in transposed
    ]


def precompute_maps(
    first_map=10,
    last_map=99,
    maps_dir="maps",
    target=(9, 9),
    p=P_DEFAULT,
    k=K_DEFAULT,
    max_steps=MAX_STEPS_DEFAULT,
    cache_dir="gaussian_simple_models",
):
    for map_id in range(first_map, last_map + 1):
        path = Path(maps_dir) / f"map_{map_id}.csv"

        if not path.exists():
            print(
                f"skip map {map_id}: "
                f"{path} missing"
            )
            continue

        map_data = _load_map(path)

        model = build_gaussian_model(
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
            f"{model['state_count']} states, "
            f"unique Gaussian states="
            f"{model['unique_gaussian_count']}, "
            f"metric={model['metric']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build simplified direct Gaussian "
            "PARLEY knowledge models."
        )
    )

    parser.add_argument(
        "--maps-dir",
        default="maps",
    )
    parser.add_argument(
        "--output-dir",
        default="gaussian_simple_models",
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
        default=P_DEFAULT,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K_DEFAULT,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS_DEFAULT,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    precompute_maps(
        first_map=args.start_map,
        last_map=args.end_map,
        maps_dir=args.maps_dir,
        target=(
            args.target_x,
            args.target_y,
        ),
        p=args.p,
        k=args.k,
        max_steps=args.max_steps,
        cache_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
