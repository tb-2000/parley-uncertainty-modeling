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

# Scale used to store MSE values as PRISM-compatible integers.
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
    """Return E[delta X] and covariance Q of the clipped stochastic move."""
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
    return (float(mean_dx), float(mean_dy)), (
        float(var_x), float(var_y), float(cov_xy)
    )


def _state_key(state, digits=14):
    def clean(value):
        rounded = round(float(value), digits)
        return 0.0 if abs(rounded) < 10 ** (-digits) else rounded
    return tuple(clean(v) for v in state)


def _split_state(state):
    bx, by, var_x, var_y, cov_xy = state
    return (bx, by), (var_x, var_y, cov_xy)


def _trace(sigma):
    return sigma[0] + sigma[1]


def _mse(state):
    """E[||X-xhat||^2] = ||bias||^2 + trace(Sigma)."""
    bias, sigma = _split_state(state)
    return bias[0] ** 2 + bias[1] ** 2 + _trace(sigma)


def _predict_state(state, xhat, yhat, action, n, p):
    """
    Gaussian prediction in error coordinates.

      b' = b + E[delta X] - delta xhat
      Sigma' = Sigma + Q
    """
    (bx, by), (var_x, var_y, cov_xy) = _split_state(state)
    mean_move, q = _motion_moments(xhat, yhat, action, n, p)

    next_xhat, next_yhat = _move(xhat, yhat, action, n)
    estimate_dx = float(next_xhat - xhat)
    estimate_dy = float(next_yhat - yhat)

    successor = (
        bx + mean_move[0] - estimate_dx,
        by + mean_move[1] - estimate_dy,
        var_x + q[0],
        var_y + q[1],
        cov_xy + q[2],
    )
    return _state_key(successor)


def _bures_wasserstein_squared(a, b):
    """Squared Bures-Wasserstein distance between 2x2 PSD covariances."""
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
        raise ValueError(f"Non-PSD covariance: det(A)={det_a}, det(B)={det_b}")

    inner = trace_ab + 2.0 * math.sqrt(det_a * det_b)
    if inner < 0.0 and abs(inner) < 1e-12:
        inner = 0.0
    if inner < 0.0:
        raise ValueError(f"Negative Bures inner term: {inner}")

    squared = trace_a + trace_b - 2.0 * math.sqrt(inner)
    if squared < 0.0 and abs(squared) < 1e-10:
        squared = 0.0
    if squared < 0.0:
        raise ValueError(f"Negative Bures-Wasserstein squared distance: {squared}")
    return squared


def _wasserstein(a, b):
    """2-Wasserstein distance between N(bias_a,Sigma_a) and N(bias_b,Sigma_b)."""
    bias_a, sigma_a = _split_state(a)
    bias_b, sigma_b = _split_state(b)
    mean_sq = (bias_a[0] - bias_b[0]) ** 2 + (bias_a[1] - bias_b[1]) ** 2
    return math.sqrt(mean_sq + _bures_wasserstein_squared(sigma_a, sigma_b))


def _distance(a, b, metric):
    if metric != "wasserstein":
        raise ValueError("metric must be 'wasserstein'")
    return _wasserstein(a, b)


def _generate_records(map_data, target, p, max_steps):
    """Generate reachable raw Gaussian states under the fixed MAPE policy."""
    map_size = len(map_data)
    n = map_size - 1
    controller = _controller(map_data, target)
    records = []
    mse_by_age = defaultdict(list)

    for start_x in range(map_size):
        for start_y in range(map_size):
            if int(map_data[start_x][start_y]) > 9:
                continue

            xhat, yhat = start_x, start_y
            state = ZERO_STATE

            for age in range(max_steps + 1):
                state = _state_key(state)
                records.append({"state": state, "xhat": xhat, "yhat": yhat, "age": age})
                mse_by_age[age].append(_mse(state))

                if age >= max_steps or (xhat, yhat) == target:
                    break
                action = _direction(controller, xhat, yhat)
                if action is None:
                    break

                state = _predict_state(state, xhat, yhat, action, n, p)
                xhat, yhat = _move(xhat, yhat, action, n)

    unique = {}
    for record in records:
        state = record["state"]
        if state not in unique:
            unique[state] = {"state": state, "weight": 1}
        else:
            unique[state]["weight"] += 1

    candidates = list(unique.values())
    candidates.sort(
        key=lambda item: (
            0 if item["state"] == ZERO_STATE else 1,
            _mse(item["state"]),
            item["state"],
        )
    )
    return candidates, mse_by_age, controller


def _distance_matrix(states, metric):
    count = len(states)
    matrix = [[0.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            value = _distance(states[i], states[j], metric)
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def _cluster(candidates, k, metric="wasserstein", max_iter=100):
    """Weighted k-medoids in Gaussian 2-Wasserstein geometry."""
    states = [item["state"] for item in candidates]
    weights = [item["weight"] for item in candidates]
    if not states:
        raise ValueError("No reachable Gaussian states.")

    try:
        zero_index = states.index(ZERO_STATE)
    except ValueError as exc:
        raise ValueError("Exact reset Gaussian state is missing.") from exc

    if len(states) <= k:
        medoids = list(range(len(states)))
    else:
        matrix = _distance_matrix(states, metric)
        medoids = [zero_index]
        nearest = [matrix[i][zero_index] for i in range(len(states))]

        while len(medoids) < k:
            next_medoid = max(
                (i for i in range(len(states)) if i not in medoids),
                key=lambda i: (nearest[i], -i),
            )
            medoids.append(next_medoid)
            for i in range(len(states)):
                nearest[i] = min(nearest[i], matrix[i][next_medoid])

        def objective(current):
            return sum(
                weights[i] * min(matrix[i][medoid] for medoid in current)
                for i in range(len(states))
            )

        best_medoids = list(medoids)
        best_objective = objective(medoids)
        seen = set()

        for _ in range(max_iter):
            key = tuple(sorted(medoids))
            if key in seen:
                break
            seen.add(key)

            clusters = {medoid: [] for medoid in medoids}
            for i in range(len(states)):
                medoid = min(medoids, key=lambda candidate: (matrix[i][candidate], candidate))
                clusters[medoid].append(i)

            refined = []
            for medoid in medoids:
                members = clusters[medoid]
                if zero_index in members and medoid == zero_index:
                    refined.append(zero_index)
                    continue
                best = min(
                    members,
                    key=lambda candidate: (
                        sum(weights[m] * matrix[candidate][m] for m in members),
                        candidate,
                    ),
                )
                refined.append(best)

            current_objective = objective(refined)
            if current_objective < best_objective:
                best_objective = current_objective
                best_medoids = list(refined)
            if set(refined) == set(medoids):
                medoids = refined
                break
            medoids = refined

        medoids = best_medoids
        print(f"Gaussian k-medoids finished; metric={metric}, objective={best_objective}")

    medoids = [zero_index] + sorted(m for m in medoids if m != zero_index)
    return [states[index] for index in medoids]


def _nearest(state, representatives, metric):
    """Project a predicted Gaussian onto the nearest non-reset representative."""
    state = _state_key(state)
    if state == ZERO_STATE:
        return 0
    if len(representatives) <= 1:
        raise ValueError("Need at least one non-zero Gaussian representative.")
    return min(
        range(1, len(representatives)),
        key=lambda index: (_distance(state, representatives[index], metric), index),
    )


def _thresholds(mse_by_age, max_steps, scale=MSE_SCALE):
    """c=1..max_steps -> median Gaussian MSE after c predictions."""
    result = []
    previous = 0
    for age in range(1, max_steps + 1):
        values = sorted(mse_by_age.get(age, []))
        if not values:
            scaled = previous
        else:
            middle = len(values) // 2
            value = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
            scaled = int(round(value * scale))
        scaled = max(previous, scaled)
        result.append(scaled)
        previous = scaled
    return result


def _transition_from_representative(state, xhat, yhat, action, n, p, representatives, metric):
    successor = _predict_state(state, xhat, yhat, action, n, p)
    return _nearest(successor, representatives, metric)


def build_gaussian_model(
    map_id,
    map_data,
    target,
    p=0.01,
    k=100,
    max_steps=10,
    metric="wasserstein",
    cache_dir="gaussian_models",
):
    """Build one map-specific finite Gaussian (bias + covariance) knowledge model."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"map_{map_id}.json"

    candidates, mse_by_age, controller = _generate_records(map_data, target, p, max_steps)
    representatives = _cluster(candidates, min(k, len(candidates)), metric=metric, max_iter=100)
    state_count = len(representatives)
    n = len(map_data) - 1

    uncertainties = [int(round(_mse(state) * MSE_SCALE)) for state in representatives]
    thresholds = _thresholds(mse_by_age, max_steps, scale=MSE_SCALE)

    transitions = {}
    for x in range(len(map_data)):
        for y in range(len(map_data)):
            action = _direction(controller, x, y)
            if action is None:
                continue
            for state_id, state in enumerate(representatives):
                next_state = _transition_from_representative(
                    state, x, y, action, n, p, representatives, metric
                )
                transitions[f"{x},{y},{state_id}"] = {
                    "action": action,
                    "next_state": next_state,
                }

    model = {
        "map_id": map_id,
        "k": k,
        "state_count": state_count,
        "max_steps": max_steps,
        "p": p,
        "metric": metric,
        "uncertainty_metric": "mse",
        "mse_scale": MSE_SCALE,
        "thresholds": thresholds,
        "uncertainties": uncertainties,
        "representatives": [],
        "transitions": transitions,
    }

    for state_id, state in enumerate(representatives):
        (bias_x, bias_y), sigma = _split_state(state)
        model["representatives"].append({
            "state_id": state_id,
            "bias_x": bias_x,
            "bias_y": bias_y,
            "var_x": sigma[0],
            "var_y": sigma[1],
            "cov_xy": sigma[2],
            "trace": _trace(sigma),
            "bias_squared": bias_x ** 2 + bias_y ** 2,
            "mse": _mse(state),
        })

    zero = model["representatives"][0]
    for field in ("bias_x", "bias_y", "var_x", "var_y", "cov_xy"):
        if abs(float(zero[field])) > 1e-15:
            raise ValueError("Gaussian state 0 is not exact bias=0, Sigma=0.")

    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)
    return model


def precompute_maps(
    first_map=10,
    last_map=99,
    maps_dir="maps",
    target=(9, 9),
    p=0.01,
    k=100,
    max_steps=10,
    metric="wasserstein",
    cache_dir="gaussian_models",
):
    for map_id in range(first_map, last_map + 1):
        path = Path(maps_dir) / f"map_{map_id}.csv"
        if not path.exists():
            print(f"skip map {map_id}: {path} missing")
            continue

        with path.open("r", newline="") as file:
            rows = list(csv.reader(file))
        transposed = list(zip(*rows))
        map_data = [row[::-1] for row in transposed]

        model = build_gaussian_model(
            map_id=map_id,
            map_data=map_data,
            target=target,
            p=p,
            k=k,
            max_steps=max_steps,
            metric=metric,
            cache_dir=cache_dir,
        )
        print(
            f"map {map_id}: {model['state_count']} Gaussian representatives, "
            f"metric={metric}, uncertainty=mse, thresholds={model['thresholds']}"
        )


if __name__ == "__main__":
    precompute_maps()
