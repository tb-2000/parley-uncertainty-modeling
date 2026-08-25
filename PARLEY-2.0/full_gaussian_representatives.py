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

TRACE_SCALE = 1_000_000


# ---------------------------------------------------------------------------
# Map / MAPE controller
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Gaussian covariance dynamics
# ---------------------------------------------------------------------------

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


def _motion_covariance(x, y, action, n, p):
    """
    Covariance Q(x,y,a) of the stochastic displacement used by the PRISM Robot.

    Returned tuple:
        (var_x, var_y, cov_xy)
    """
    samples = []

    for probability, dx, dy in _robot_outcomes(action, p):
        nx = _clip(x + dx, n)
        ny = _clip(y + dy, n)

        actual_dx = float(nx - x)
        actual_dy = float(ny - y)

        samples.append(
            (probability, actual_dx, actual_dy)
        )

    mean_dx = sum(
        probability * dx
        for probability, dx, _ in samples
    )
    mean_dy = sum(
        probability * dy
        for probability, _, dy in samples
    )

    var_x = sum(
        probability * (dx - mean_dx) ** 2
        for probability, dx, _ in samples
    )
    var_y = sum(
        probability * (dy - mean_dy) ** 2
        for probability, _, dy in samples
    )
    cov_xy = sum(
        probability
        * (dx - mean_dx)
        * (dy - mean_dy)
        for probability, dx, dy in samples
    )

    return (
        float(var_x),
        float(var_y),
        float(cov_xy),
    )


def _add_sigma(a, b):
    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    )


def _sigma_key(sigma, digits=14):
    def clean(value):
        rounded = round(float(value), digits)
        return 0.0 if abs(rounded) < 10 ** (-digits) else rounded

    return tuple(clean(value) for value in sigma)


def _trace(sigma):
    return sigma[0] + sigma[1]


# ---------------------------------------------------------------------------
# Distances on covariance matrices
# ---------------------------------------------------------------------------

def _frobenius(a, b):

    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dc = a[2] - b[2]

    return math.sqrt(
        dx * dx
        + dy * dy
        + 2.0 * dc * dc
    )


def _bures_wasserstein(a, b):
    """
    Closed-form Bures-Wasserstein distance for 2x2 PSD covariance matrices.

    For
        A = [[ax, ac],
             [ac, ay]]
        B = [[bx, bc],
             [bc, by]]

    we use

        d_BW^2(A,B)
          = tr(A) + tr(B)
            - 2 * sqrt(
                tr(A B)
                + 2 * sqrt(det(A) det(B))
              )

    This avoids eigenvalue decompositions / matrix square roots and is
    considerably faster for the 2x2 covariance matrices used here.

    Sigma=0 is valid.
    """
    ax, ay, ac = a
    bx, by, bc = b

    trace_a = ax + ay
    trace_b = bx + by

    # tr(A B) for symmetric 2x2 matrices.
    trace_ab = (
        ax * bx
        + ay * by
        + 2.0 * ac * bc
    )

    # PSD determinants. Clamp tiny negative values caused by floating-point
    # round-off to zero.
    det_a = ax * ay - ac * ac
    det_b = bx * by - bc * bc

    if det_a < 0.0 and abs(det_a) < 1e-12:
        det_a = 0.0
    if det_b < 0.0 and abs(det_b) < 1e-12:
        det_b = 0.0

    if det_a < 0.0 or det_b < 0.0:
        raise ValueError(
            f"Non-PSD covariance in Bures-Wasserstein distance: "
            f"det(A)={det_a}, det(B)={det_b}"
        )

    inner = (
        trace_ab
        + 2.0 * math.sqrt(
            det_a * det_b
        )
    )

    if inner < 0.0 and abs(inner) < 1e-12:
        inner = 0.0

    if inner < 0.0:
        raise ValueError(
            f"Negative Bures inner term: {inner}"
        )

    squared = (
        trace_a
        + trace_b
        - 2.0 * math.sqrt(inner)
    )

    if squared < 0.0 and abs(squared) < 1e-10:
        squared = 0.0

    if squared < 0.0:
        raise ValueError(
            f"Negative Bures-Wasserstein squared distance: {squared}"
        )

    return math.sqrt(squared)


def _distance(a, b, metric):
    if metric == "frobenius":
        return _frobenius(a, b)

    if metric == "bures_wasserstein":
        return _bures_wasserstein(a, b)

    raise ValueError(
        "metric must be 'frobenius' or 'bures_wasserstein'"
    )


# ---------------------------------------------------------------------------
# Reachable Gaussian-state generation
# ---------------------------------------------------------------------------

def _generate_records(
    map_data,
    target,
    p,
    max_steps,
):
    """
    Generate raw covariance states induced by the fixed MAPE policy.

    As in the belief model:
      * start once from every free grid position,
      * perfect update means Sigma=0,
      * follow the unique MAPE action sequence,
      * stop at target/no-action/max_steps.

    Returns:
      candidates:
          unique raw Sigma states with occurrence weights
      trace_by_age:
          raw trace(Sigma) samples after age=0..max_steps predictions
      controller:
          map-specific Dijkstra MAPE controller
    """
    map_size = len(map_data)
    n = map_size - 1
    controller = _controller(
        map_data,
        target,
    )

    records = []
    trace_by_age = defaultdict(list)

    for start_x in range(map_size):
        for start_y in range(map_size):
            if int(map_data[start_x][start_y]) > 9:
                continue

            xhat = start_x
            yhat = start_y
            sigma = (0.0, 0.0, 0.0)

            for age in range(max_steps + 1):
                sigma = _sigma_key(sigma)

                records.append(
                    {
                        "sigma": sigma,
                        "xhat": xhat,
                        "yhat": yhat,
                        "age": age,
                    }
                )

                trace_by_age[age].append(
                    _trace(sigma)
                )

                if (
                    age >= max_steps
                    or (xhat, yhat) == target
                ):
                    break

                action = _direction(
                    controller,
                    xhat,
                    yhat,
                )

                if action is None:
                    break

                q = _motion_covariance(
                    xhat,
                    yhat,
                    action,
                    n,
                    p,
                )

                sigma = _add_sigma(
                    sigma,
                    q,
                )

                xhat, yhat = _move(
                    xhat,
                    yhat,
                    action,
                    n,
                )

    # Deduplicate identical covariance matrices and preserve their frequencies.
    unique = {}

    for record in records:
        sigma = record["sigma"]

        if sigma not in unique:
            unique[sigma] = {
                "sigma": sigma,
                "weight": 1,
            }
        else:
            unique[sigma]["weight"] += 1

    candidates = list(
        unique.values()
    )

    # Stable ordering; exact certainty first.
    candidates.sort(
        key=lambda item: (
            0
            if item["sigma"] == (0.0, 0.0, 0.0)
            else 1,
            _trace(item["sigma"]),
            item["sigma"],
        )
    )

    return (
        candidates,
        trace_by_age,
        controller,
    )


# ---------------------------------------------------------------------------
# Weighted k-medoids quantisation, parallel to belief model
# ---------------------------------------------------------------------------

def _distance_matrix(sigmas, metric):
    count = len(sigmas)
    matrix = [
        [0.0] * count
        for _ in range(count)
    ]

    for i in range(count):
        for j in range(i + 1, count):
            value = _distance(
                sigmas[i],
                sigmas[j],
                metric,
            )
            matrix[i][j] = value
            matrix[j][i] = value

    return matrix


def _cluster(
    candidates,
    k,
    metric="bures_wasserstein",
    max_iter=100,
):
    """
    Weighted k-medoids with farthest-first initialisation.

    The exact covariance Sigma=0 is permanently kept as representative 0.
    This mirrors the certainty-state handling in full_belief_representatives.py.
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
            "No reachable Gaussian covariance states."
        )

    zero_sigma = (0.0, 0.0, 0.0)

    try:
        zero_index = sigmas.index(
            zero_sigma
        )
    except ValueError as exc:
        raise ValueError(
            "Exact reset covariance Sigma=0 is missing."
        ) from exc

    if len(sigmas) <= k:
        medoids = list(
            range(len(sigmas))
        )
    else:
        matrix = _distance_matrix(
            sigmas,
            metric,
        )

        # Farthest-first initialisation with reset state fixed.
        medoids = [zero_index]
        nearest = [
            matrix[i][zero_index]
            for i in range(len(sigmas))
        ]

        while len(medoids) < k:
            next_medoid = max(
                (
                    i
                    for i in range(len(sigmas))
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

            for i in range(len(sigmas)):
                nearest[i] = min(
                    nearest[i],
                    matrix[i][next_medoid],
                )

        def objective(current_medoids):
            return sum(
                weights[i]
                * min(
                    matrix[i][medoid]
                    for medoid in current_medoids
                )
                for i in range(len(sigmas))
            )

        best_medoids = list(
            medoids
        )
        best_objective = objective(
            medoids
        )

        seen = {}
        converged = False
        cycle_detected = False

        for iteration in range(max_iter):
            key = tuple(
                sorted(medoids)
            )

            if key in seen:
                print(
                    f"Gaussian medoid cycle detected: "
                    f"iteration {seen[key]} -> {iteration}"
                )
                cycle_detected = True
                medoids = best_medoids
                break

            seen[key] = iteration

            clusters = {
                medoid: []
                for medoid in medoids
            }

            for i in range(len(sigmas)):
                medoid = min(
                    medoids,
                    key=lambda candidate: (
                        matrix[i][candidate],
                        candidate,
                    ),
                )
                clusters[medoid].append(
                    i
                )

            refined = []

            for medoid in medoids:
                members = clusters[
                    medoid
                ]

                # Preserve exact reset state.
                if medoid == zero_index:
                    refined.append(
                        medoid
                    )
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

                refined.append(
                    best
                )

            refined_objective = objective(
                refined
            )

            if refined_objective < best_objective:
                best_objective = refined_objective
                best_medoids = list(
                    refined
                )

            if set(refined) == set(medoids):
                medoids = refined
                converged = True
                break

            medoids = refined

        medoids = best_medoids

        if converged:
            print(
                f"Gaussian k-medoids converged; "
                f"metric={metric}, objective={best_objective}"
            )
        elif cycle_detected:
            print(
                f"Gaussian k-medoids stopped on cycle; "
                f"metric={metric}, best objective={best_objective}"
            )
        else:
            print(
                f"Warning: Gaussian k-medoids reached max_iter={max_iter}; "
                f"metric={metric}, best objective={best_objective}"
            )

    # State 0 must be exact Sigma=0.
    medoids = [
        zero_index
    ] + sorted(
        medoid
        for medoid in medoids
        if medoid != zero_index
    )

    return [
        sigmas[index]
        for index in medoids
    ]


def _nearest(
    sigma,
    representatives,
    metric,
):
    return min(
        range(len(representatives)),
        key=lambda index: (
            _distance(
                sigma,
                representatives[index],
                metric,
            ),
            index,
        ),
    )


# ---------------------------------------------------------------------------
# Gaussian uncertainty thresholds
# ---------------------------------------------------------------------------

def _thresholds(
    trace_by_age,
    max_steps,
    scale=TRACE_SCALE,
):
    """
    Parallel to belief thresholds:
      c=1..10 -> median trace(Sigma) after c predictions.

    Monotonicity is enforced because the URC decision values 1..10 should
    represent nondecreasing accepted uncertainty.
    """
    result = []
    previous = 0

    for age in range(
        1,
        max_steps + 1,
    ):
        values = sorted(
            trace_by_age.get(
                age,
                [],
            )
        )

        if not values:
            scaled = previous
        else:
            middle = len(values) // 2

            if len(values) % 2:
                value = values[middle]
            else:
                value = (
                    values[middle - 1]
                    + values[middle]
                ) / 2.0

            scaled = int(
                round(
                    value * scale
                )
            )

        scaled = max(
            previous,
            scaled,
        )

        result.append(
            scaled
        )
        previous = scaled

    return result


# ---------------------------------------------------------------------------
# Representative-state transition
# ---------------------------------------------------------------------------

def _transition_from_representative(
    sigma,
    xhat,
    yhat,
    action,
    n,
    p,
    representatives,
    metric,
):
    """
    Gaussian analogue of belief representative propagation:

        representative Sigma
          -> Sigma + Q(xhat,yhat,action)
          -> nearest representative

    This makes the abstract finite-state transition deterministic by
    construction; no post-hoc Markov refinement is required.
    """
    q = _motion_covariance(
        xhat,
        yhat,
        action,
        n,
        p,
    )

    successor_sigma = _sigma_key(
        _add_sigma(
            sigma,
            q,
        )
    )

    return _nearest(
        successor_sigma,
        representatives,
        metric,
    )


# ---------------------------------------------------------------------------
# Public model builder
# ---------------------------------------------------------------------------

def build_gaussian_model(
    map_id,
    map_data,
    target,
    p=0.01,
    k=100,
    max_steps=10,
    metric="bures_wasserstein",
    cache_dir="gaussian_models",
):
    """
    Build one map-specific finite Gaussian knowledge model.

    Returned interface intentionally mirrors build_belief_model():
        state_count
        thresholds
        uncertainties
        transitions

    Additional Gaussian metadata:
        representatives
        metric
        trace_scale
    """
    cache_dir = Path(
        cache_dir
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        cache_dir
        / f"map_{map_id}.json"
    )

    (
        candidates,
        trace_by_age,
        controller,
    ) = _generate_records(
        map_data,
        target,
        p,
        max_steps,
    )

    representatives = _cluster(
        candidates,
        min(k, len(candidates)),
        metric=metric,
        max_iter=100,
    )

    state_count = len(
        representatives
    )
    n = len(map_data) - 1

    # Scalar Gaussian uncertainty used by URC:
    # trace(Sigma) = var_x + var_y.
    uncertainties = [
        int(
            round(
                _trace(sigma)
                * TRACE_SCALE
            )
        )
        for sigma in representatives
    ]

    thresholds = _thresholds(
        trace_by_age,
        max_steps,
        scale=TRACE_SCALE,
    )

    transitions = {}

    for x in range(
        len(map_data)
    ):
        for y in range(
            len(map_data)
        ):
            action = _direction(
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
                next_state = (
                    _transition_from_representative(
                        sigma,
                        x,
                        y,
                        action,
                        n,
                        p,
                        representatives,
                        metric,
                    )
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
        "metric": metric,
        "trace_scale": TRACE_SCALE,
        "thresholds": thresholds,
        "uncertainties": uncertainties,
        "representatives": [
            {
                "state_id": state_id,
                "var_x": sigma[0],
                "var_y": sigma[1],
                "cov_xy": sigma[2],
                "trace": _trace(sigma),
            }
            for state_id, sigma in enumerate(
                representatives
            )
        ],
        "transitions": transitions,
    }

    # Strong reset invariant.
    zero = model[
        "representatives"
    ][0]

    if (
        abs(zero["var_x"]) > 1e-15
        or abs(zero["var_y"]) > 1e-15
        or abs(zero["cov_xy"]) > 1e-15
    ):
        raise ValueError(
            "Gaussian state 0 is not exact Sigma=0."
        )

    with open(
        cache_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model,
            file,
            indent=2,
        )

    return model


def precompute_maps(
    first_map=10,
    last_map=99,
    maps_dir="maps",
    target=(9, 9),
    p=0.01,
    k=100,
    max_steps=10,
    metric="bures_wasserstein",
    cache_dir="gaussian_models",
):
    for map_id in range(
        first_map,
        last_map + 1,
    ):
        path = (
            Path(maps_dir)
            / f"map_{map_id}.csv"
        )

        if not path.exists():
            print(
                f"skip map {map_id}: {path} missing"
            )
            continue

        rows = []

        with open(
            path,
            "r",
            newline="",
        ) as file:
            rows.extend(
                csv.reader(file)
            )

        transposed = list(
            zip(*rows)
        )
        map_data = [
            row[::-1]
            for row in transposed
        ]

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
            f"map {map_id}: "
            f"{model['state_count']} Gaussian representatives, "
            f"metric={metric}, "
            f"thresholds={model['thresholds']}"
        )


if __name__ == "__main__":
    precompute_maps()
