#!/usr/bin/env python3
"""
Exact budget-neutral refinement for the improved Gaussian PARLEY model.

This version deliberately matches the successful distance-analysis/model-builder
semantics:

INITIALISATION
--------------
1. Enumerate exact reachable positional beliefs.
2. Moment-match each history to G=(mu,Sigma).
3. Deduplicate to UNIQUE Gaussian states.
4. Use occurrence count as candidate weight.
5. Weighted K=100 k-medoids with distance w2_mse1_level1.

Distance:
    d^2 =
        W2^2
        + 1 * ((MSE_i - MSE_j) / MSE_max)^2
        + 1 * ((level_i - level_j) / 10)^2

TRANSITION SEMANTICS
--------------------
For every representative and every (xhat,yhat,action), use exactly the same
semantics as the Gaussian model generator:

    representative's stored relative source belief
      -> re-anchor at current (xhat,yhat), with grid clipping
      -> exact PARLEY Robot prediction
      -> moment-match successor to (mu',Sigma')
      -> project to nearest current representative

REFINEMENT
----------
K stays exactly 100.

The algorithm:
  - ranks source clusters by exact-history successor-level mismatch,
  - proposes alternative medoids from the same cluster,
  - recomputes the full K=100 assignment and exact representative transition
    relation,
  - accepts a replacement only if global successor-level mismatch decreases.

This is slower than the approximate refinement, but it optimizes exactly the
same transition semantics that will later appear in the generated PRISM model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dijkstra


DIRECTIONS = ("west", "east", "south", "north")
MOVE = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}

K_FIXED = 100
LAMBDA_MSE = 1.0
LAMBDA_LEVEL = 1.0

DEFAULT_MAX_ROUNDS = 15
DEFAULT_BLOCKS_TO_TRY = 8
DEFAULT_CANDIDATES_PER_BLOCK = 5


# ---------------------------------------------------------------------------
# PARLEY dynamics
# ---------------------------------------------------------------------------

def _clip(v: int, n: int) -> int:
    return min(max(v, 0), n)


def _move(
    x: int,
    y: int,
    action: str,
    n: int,
) -> Tuple[int, int]:
    dx, dy = MOVE[action]
    return _clip(x + dx, n), _clip(y + dy, n)


def _propagate_absolute(
    belief: Dict[Tuple[int, int], float],
    action: str,
    n: int,
    p: float,
) -> Dict[Tuple[int, int], float]:
    result = defaultdict(float)

    for (x, y), prior in belief.items():
        for actual in DIRECTIONS:
            q = 1.0 - 3.0 * p if actual == action else p
            nx, ny = _move(x, y, actual, n)
            result[(nx, ny)] += prior * q

    return dict(result)


def _read_map(path: Path):
    rows = []

    with path.open("r", newline="") as file:
        rows.extend(csv.reader(file))

    transposed = list(zip(*rows))
    return [list(row[::-1]) for row in transposed]


def _controller(map_data, target):
    return list(zip(*dijkstra.compute_directions(map_data, target)))


def _direction(controller, x, y):
    value = int(controller[y][x])
    return DIRECTIONS[value] if 0 <= value < 4 else None


# ---------------------------------------------------------------------------
# Relative belief representation
# ---------------------------------------------------------------------------

def _relative_vector(
    belief,
    xhat: int,
    yhat: int,
    n: int,
):
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


# ---------------------------------------------------------------------------
# Gaussian moments / metrics
# ---------------------------------------------------------------------------

def _gaussian_key(state, digits=14):
    def clean(v):
        r = round(float(v), digits)
        return 0.0 if abs(r) < 10 ** (-digits) else r

    return tuple(clean(v) for v in state)


def _moments_from_belief(belief, xhat, yhat):
    mu_x = sum(
        (x - xhat) * probability
        for (x, y), probability in belief.items()
    )
    mu_y = sum(
        (y - yhat) * probability
        for (x, y), probability in belief.items()
    )

    var_x = 0.0
    var_y = 0.0
    cov_xy = 0.0

    for (x, y), probability in belief.items():
        ex = (x - xhat) - mu_x
        ey = (y - yhat) - mu_y
        var_x += probability * ex * ex
        var_y += probability * ey * ey
        cov_xy += probability * ex * ey

    return _gaussian_key(
        (mu_x, mu_y, var_x, var_y, cov_xy)
    )


def _sigma(state):
    return state[2], state[3], state[4]


def _trace(state):
    return state[2] + state[3]


def _bias2(state):
    return state[0] ** 2 + state[1] ** 2


def _mse(state):
    return _trace(state) + _bias2(state)


def _bures_sq(a, b):
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
        raise ValueError("Non-PSD covariance encountered.")

    inner = trace_ab + 2.0 * math.sqrt(det_a * det_b)

    if inner < 0.0 and abs(inner) < 1e-12:
        inner = 0.0

    result = (
        trace_a
        + trace_b
        - 2.0 * math.sqrt(inner)
    )

    if result < 0.0 and abs(result) < 1e-10:
        result = 0.0

    return result


def _w2_sq(a, b):
    dmx = a[0] - b[0]
    dmy = a[1] - b[1]

    return (
        dmx * dmx
        + dmy * dmy
        + _bures_sq(
            _sigma(a),
            _sigma(b),
        )
    )


# ---------------------------------------------------------------------------
# Exact histories / unique weighted candidates
# ---------------------------------------------------------------------------

def _generate_records(
    map_data,
    target,
    p,
    max_steps,
):
    size = len(map_data)
    n = size - 1
    ctrl = _controller(map_data, target)

    records = []
    mse_by_age = defaultdict(list)

    for sx in range(size):
        for sy in range(size):
            if int(map_data[sx][sy]) > 9:
                continue

            belief = {(sx, sy): 1.0}
            xhat, yhat = sx, sy

            for age in range(max_steps + 1):
                state = _moments_from_belief(
                    belief,
                    xhat,
                    yhat,
                )

                records.append({
                    "start_x": sx,
                    "start_y": sy,
                    "age": age,
                    "xhat": xhat,
                    "yhat": yhat,
                    "belief": belief,
                    "state": state,
                    "vector": _relative_vector(
                        belief,
                        xhat,
                        yhat,
                        n,
                    ),
                })

                mse_by_age[age].append(
                    _mse(state)
                )

                if age >= max_steps or (xhat, yhat) == target:
                    break

                action = _direction(
                    ctrl,
                    xhat,
                    yhat,
                )

                if action is None:
                    break

                belief = _propagate_absolute(
                    belief,
                    action,
                    n,
                    p,
                )

                xhat, yhat = _move(
                    xhat,
                    yhat,
                    action,
                    n,
                )

    unique = {}

    for record in records:
        key = record["state"]

        if key not in unique:
            unique[key] = {
                "state": key,
                "vector": record["vector"],
                "weight": 1,
            }
        else:
            unique[key]["weight"] += 1

    candidates = list(unique.values())

    zero = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    candidates.sort(
        key=lambda item: (
            0 if item["state"] == zero else 1,
            _mse(item["state"]),
            item["state"],
        )
    )

    return records, candidates, mse_by_age, ctrl


def _thresholds(mse_by_age, max_steps):
    result = []
    previous = 0.0

    for age in range(1, max_steps + 1):
        values = sorted(
            mse_by_age.get(age, [])
        )

        if not values:
            value = previous
        else:
            mid = len(values) // 2

            if len(values) % 2:
                value = values[mid]
            else:
                value = (
                    values[mid - 1]
                    + values[mid]
                ) / 2.0

        value = max(previous, value)
        result.append(value)
        previous = value

    return result


def _level_from_mse(value, thresholds):
    level = 0

    for index, threshold in enumerate(
        thresholds,
        start=1,
    ):
        if value >= threshold - 1e-15:
            level = index
        else:
            break

    return level


# ---------------------------------------------------------------------------
# Candidate-level distance and weighted k-medoids
# ---------------------------------------------------------------------------

def _build_candidate_distance_matrix(
    candidates,
    thresholds,
):
    states = [
        item["state"]
        for item in candidates
    ]

    mse_max = max(
        _mse(state)
        for state in states
    )
    denom = max(mse_max, 1e-15)

    levels = [
        _level_from_mse(
            _mse(state),
            thresholds,
        )
        for state in states
    ]

    count = len(states)
    matrix = [
        [0.0] * count
        for _ in range(count)
    ]

    for i in range(count):
        for j in range(i + 1, count):
            dmse = (
                _mse(states[i])
                - _mse(states[j])
            ) / denom

            dlevel = (
                levels[i]
                - levels[j]
            ) / 10.0

            squared = (
                _w2_sq(
                    states[i],
                    states[j],
                )
                + LAMBDA_MSE
                * dmse
                * dmse
                + LAMBDA_LEVEL
                * dlevel
                * dlevel
            )

            value = math.sqrt(
                max(0.0, squared)
            )

            matrix[i][j] = value
            matrix[j][i] = value

    return matrix, levels


def _weighted_kmedoids(
    candidates,
    matrix,
):
    states = [
        item["state"]
        for item in candidates
    ]
    weights = [
        item["weight"]
        for item in candidates
    ]

    zero = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    zero_index = states.index(zero)

    if len(states) <= K_FIXED:
        return list(range(len(states)))

    # EXACT same farthest-first + weighted medoid refinement structure as
    # the distance-analysis/model-builder path.
    medoids = [zero_index]
    nearest = [
        matrix[i][zero_index]
        for i in range(len(states))
    ]

    while len(medoids) < K_FIXED:
        next_medoid = max(
            (
                i
                for i in range(len(states))
                if i not in medoids
            ),
            key=lambda i: (
                nearest[i],
                -i,
            ),
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
            * min(
                matrix[i][m]
                for m in current_medoids
            )
            for i in range(len(states))
        )

    best_medoids = list(medoids)
    best_objective = objective(medoids)
    seen = {}

    for iteration in range(100):
        key = tuple(sorted(medoids))

        if key in seen:
            break

        seen[key] = iteration

        clusters = {
            medoid: []
            for medoid in medoids
        }

        for i in range(len(states)):
            medoid = min(
                medoids,
                key=lambda m: (
                    matrix[i][m],
                    m,
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

        refined_objective = objective(
            refined
        )

        if refined_objective < best_objective:
            best_objective = refined_objective
            best_medoids = list(refined)

        if set(refined) == set(medoids):
            best_medoids = list(refined)
            break

        medoids = refined

    medoids = [zero_index] + sorted(
        m
        for m in best_medoids
        if m != zero_index
    )

    return medoids


def _assign_candidates(
    candidates,
    medoids,
    matrix,
):
    zero = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    assignment = {}

    for i, candidate in enumerate(
        candidates
    ):
        if candidate["state"] == zero:
            block = 0
        else:
            block = min(
                range(1, len(medoids)),
                key=lambda b: (
                    matrix[i][medoids[b]],
                    b,
                ),
            )

        assignment[i] = block

    return assignment


def _candidate_index_for_state(
    state,
    candidate_index_by_state,
):
    return candidate_index_by_state[
        _gaussian_key(state)
    ]


def _nearest_rep_for_state(
    state,
    candidates,
    medoids,
    thresholds,
):
    zero = (0.0, 0.0, 0.0, 0.0, 0.0)
    if _gaussian_key(state) == zero:
        return 0

    denom = _CURRENT_MSE_DENOM
    state_mse = _mse(state)
    state_level = _level_from_mse(state_mse, thresholds)

    best_block = 1
    best_distance = float("inf")

    for block in range(1, len(medoids)):
        rep_index = medoids[block]
        rep_state = candidates[rep_index]["state"]
        rep_mse = _CURRENT_CANDIDATE_MSE[rep_index]
        rep_level = _CURRENT_CANDIDATE_LEVEL[rep_index]

        dmse = (state_mse - rep_mse) / denom
        dlevel = (state_level - rep_level) / 10.0
        squared = (
            _w2_sq(state, rep_state)
            + LAMBDA_MSE * dmse * dmse
            + LAMBDA_LEVEL * dlevel * dlevel
        )

        if squared < best_distance:
            best_distance = squared
            best_block = block

    return best_block


# ---------------------------------------------------------------------------
# Exact representative-transition semantics
# ---------------------------------------------------------------------------

def _representative_transition(
    representative,
    xhat,
    yhat,
    action,
    n,
    p,
    candidates,
    medoids,
    thresholds,
):
    relative = _vector_to_relative(
        representative["vector"],
        n,
    )

    absolute = defaultdict(float)

    for (dx, dy), probability in relative.items():
        ax = _clip(xhat + dx, n)
        ay = _clip(yhat + dy, n)
        absolute[(ax, ay)] += probability

    propagated = _propagate_absolute(
        absolute,
        action,
        n,
        p,
    )

    nxhat, nyhat = _move(
        xhat,
        yhat,
        action,
        n,
    )

    successor_state = _moments_from_belief(
        propagated,
        nxhat,
        nyhat,
    )

    next_block = _nearest_rep_for_state(
        successor_state,
        candidates,
        medoids,
        thresholds,
    )

    return next_block


def _build_successor_state_cache(
    map_data,
    controller,
    candidates,
    p,
):
    """Cache the exact successor Gaussian before nearest-representative projection.

    This part depends only on candidate representative + (xhat,yhat,action),
    not on the current medoid set, so it is computed once per map.
    """
    n = len(map_data) - 1
    cache = {}

    for x in range(len(map_data)):
        for y in range(len(map_data)):
            action = _direction(controller, x, y)
            if action is None:
                continue

            for candidate_index, representative in enumerate(candidates):
                relative = _vector_to_relative(representative["vector"], n)
                absolute = defaultdict(float)

                for (dx, dy), probability in relative.items():
                    ax = _clip(x + dx, n)
                    ay = _clip(y + dy, n)
                    absolute[(ax, ay)] += probability

                propagated = _propagate_absolute(absolute, action, n, p)
                nxhat, nyhat = _move(x, y, action, n)
                successor_state = _moments_from_belief(
                    propagated, nxhat, nyhat
                )
                cache[(x, y, candidate_index)] = successor_state

    return cache


def _build_transition_table(
    map_data,
    controller,
    candidates,
    medoids,
    thresholds,
    p,
):
    transitions = {}

    for x in range(len(map_data)):
        for y in range(len(map_data)):
            action = _direction(controller, x, y)
            if action is None:
                continue

            for block, medoid_index in enumerate(medoids):
                successor_state = _CURRENT_SUCCESSOR_STATE_CACHE[
                    (x, y, medoid_index)
                ]
                next_block = _nearest_rep_for_state(
                    successor_state, candidates, medoids, thresholds
                )
                transitions[(x, y, block)] = {
                    "action": action,
                    "next_block": next_block,
                }

    return transitions


# ---------------------------------------------------------------------------
# Exact-history evaluation
# ---------------------------------------------------------------------------

def _evaluate_partition(
    records,
    candidates,
    medoids,
    thresholds,
    transitions,
    candidate_index_by_state,
):
    assignment = _assign_candidates(
        candidates,
        medoids,
        _CURRENT_DISTANCE_MATRIX,
    )

    total = 0
    mismatches = 0
    block_total = defaultdict(int)
    block_mismatch = defaultdict(int)

    for record in records:
        if record["age"] >= _CURRENT_MAX_STEPS:
            continue

        xhat = record["xhat"]
        yhat = record["yhat"]

        action = _direction(
            _CURRENT_CONTROLLER,
            xhat,
            yhat,
        )

        if action is None:
            continue

        current_candidate = (
            candidate_index_by_state[
                record["state"]
            ]
        )
        source_block = assignment[
            current_candidate
        ]

        transition = transitions[
            (
                xhat,
                yhat,
                source_block,
            )
        ]

        abstract_next_block = (
            transition["next_block"]
        )

        exact_successor_belief = (
            _propagate_absolute(
                record["belief"],
                action,
                _CURRENT_N,
                _CURRENT_P,
            )
        )

        nxhat, nyhat = _move(
            xhat,
            yhat,
            action,
            _CURRENT_N,
        )

        exact_successor_state = (
            _moments_from_belief(
                exact_successor_belief,
                nxhat,
                nyhat,
            )
        )

        exact_level = _level_from_mse(
            _mse(
                exact_successor_state
            ),
            thresholds,
        )

        rep_state = candidates[
            medoids[
                abstract_next_block
            ]
        ]["state"]

        abstract_level = (
            _level_from_mse(
                _mse(rep_state),
                thresholds,
            )
        )

        total += 1
        block_total[
            source_block
        ] += 1

        if exact_level != abstract_level:
            mismatches += 1
            block_mismatch[
                source_block
            ] += 1

    block_scores = {
        block: (
            block_mismatch[block]
            / block_total[block]
        )
        for block in block_total
    }

    return {
        "successor_level_mismatch_fraction":
            mismatches / total
            if total
            else 0.0,
        "mismatches":
            mismatches,
        "total":
            total,
        "block_scores":
            block_scores,
        "assignment":
            assignment,
    }


# ---------------------------------------------------------------------------
# Medoid replacement refinement
# ---------------------------------------------------------------------------

def _members_of_block(
    assignment,
    block,
):
    return [
        candidate_index
        for candidate_index, assigned
        in assignment.items()
        if assigned == block
    ]


def _replacement_candidates(
    block,
    current_medoid_index,
    assignment,
    candidates,
    matrix,
    max_candidates,
):
    members = _members_of_block(
        assignment,
        block,
    )

    if block == 0:
        return []

    ranked = sorted(
        (
            candidate
            for candidate in members
            if candidate
            != current_medoid_index
        ),
        key=lambda candidate: (
            sum(
                candidates[member][
                    "weight"
                ]
                * matrix[
                    candidate
                ][
                    member
                ]
                for member in members
            ),
            candidate,
        ),
    )

    return ranked[:max_candidates]


def refine(
    map_data,
    records,
    candidates,
    thresholds,
    controller,
    p,
    matrix,
    medoids,
    max_rounds,
    blocks_to_try,
    candidates_per_block,
):
    candidate_index_by_state = {
        item["state"]: index
        for index, item in enumerate(
            candidates
        )
    }

    history = []

    transitions = _build_transition_table(
        map_data,
        controller,
        candidates,
        medoids,
        thresholds,
        p,
    )

    evaluation = _evaluate_partition(
        records,
        candidates,
        medoids,
        thresholds,
        transitions,
        candidate_index_by_state,
    )

    for round_index in range(
        max_rounds + 1
    ):
        error = evaluation[
            "successor_level_mismatch_fraction"
        ]

        history.append({
            "round": round_index,
            "successor_level_mismatch_fraction":
                error,
        })

        print(
            f"    round={round_index}, "
            f"successor mismatch={error:.2%}"
        )

        if round_index >= max_rounds:
            break

        candidate_blocks = sorted(
            evaluation["block_scores"],
            key=lambda block: (
                -evaluation[
                    "block_scores"
                ][block],
                block,
            ),
        )[:blocks_to_try]

        accepted = False

        for block in candidate_blocks:
            if block == 0:
                continue

            current_medoid = medoids[
                block
            ]

            replacements = (
                _replacement_candidates(
                    block,
                    current_medoid,
                    evaluation[
                        "assignment"
                    ],
                    candidates,
                    matrix,
                    candidates_per_block,
                )
            )

            for replacement in replacements:
                if replacement in medoids:
                    continue

                trial_medoids = list(
                    medoids
                )
                trial_medoids[
                    block
                ] = replacement

                trial_transitions = (
                    _build_transition_table(
                        map_data,
                        controller,
                        candidates,
                        trial_medoids,
                        thresholds,
                        p,
                    )
                )

                trial_evaluation = (
                    _evaluate_partition(
                        records,
                        candidates,
                        trial_medoids,
                        thresholds,
                        trial_transitions,
                        candidate_index_by_state,
                    )
                )

                trial_error = (
                    trial_evaluation[
                        "successor_level_mismatch_fraction"
                    ]
                )

                if (
                    trial_error
                    < error - 1e-12
                ):
                    print(
                        f"      accepted block "
                        f"{block}: "
                        f"{error:.2%} -> "
                        f"{trial_error:.2%}"
                    )

                    medoids = trial_medoids
                    transitions = (
                        trial_transitions
                    )
                    evaluation = (
                        trial_evaluation
                    )
                    accepted = True
                    break

            if accepted:
                break

        if not accepted:
            print(
                "      no improving exact-semantic "
                "medoid replacement found; stopping."
            )
            break

    return (
        medoids,
        transitions,
        history,
        evaluation,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_outputs(
    map_id,
    output_dir,
    map_data,
    candidates,
    medoids,
    thresholds,
    transitions,
    history,
    evaluation,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    representatives = []

    for block, medoid_index in enumerate(
        medoids
    ):
        state = candidates[
            medoid_index
        ]["state"]

        representatives.append({
            "state_id": block,
            "gstate": block,
            "candidate_index":
                medoid_index,
            "mu_x": state[0],
            "mu_y": state[1],
            "var_x": state[2],
            "var_y": state[3],
            "cov_xy": state[4],
            "trace": _trace(state),
            "bias2": _bias2(state),
            "mse": _mse(state),
            "uncertainty_level":
                _level_from_mse(
                    _mse(state),
                    thresholds,
                ),
            "weight":
                candidates[
                    medoid_index
                ]["weight"],
        })

    lookup = []

    for (
        x,
        y,
        block,
    ), transition in sorted(
        transitions.items()
    ):
        action = transition[
            "action"
        ]
        nx, ny = _move(
            x,
            y,
            action,
            len(map_data) - 1,
        )

        next_block = transition[
            "next_block"
        ]

        lookup.append({
            "xhat": x,
            "yhat": y,
            "gstate": block,
            "action": action,
            "xhat_next": nx,
            "yhat_next": ny,
            "gstate_next":
                next_block,
            "source_level":
                representatives[
                    block
                ][
                    "uncertainty_level"
                ],
            "successor_level":
                representatives[
                    next_block
                ][
                    "uncertainty_level"
                ],
        })

    transition_dict = {}

    for (
        x,
        y,
        block,
    ), transition in sorted(
        transitions.items()
    ):
        transition_dict[
            f"{x},{y},{block}"
        ] = {
            "action": transition["action"],
            "next_state": int(
                transition["next_block"]
            ),
        }

    data = {
        "schema_version": 1,
        "model_type": "gaussian_mse_refined",
        "map_id": map_id,
        "state_count":
            len(medoids),
        "k": K_FIXED,
        "metric":
            "w2_mse1_level1",
        "distance":
            "w2_mse1_level1",
        "lambda_mse":
            LAMBDA_MSE,
        "lambda_level":
            LAMBDA_LEVEL,
        "uncertainty_metric":
            "mse",
        "threshold_units": "raw_mse",
        "initialization":
            "weighted_unique_gaussian_kmedoids",
        "transition_semantics":
            (
                "representative relative belief -> "
                "re-anchor at xhat,yhat -> exact robot "
                "prediction -> moment matching -> nearest "
                "representative"
            ),
        "thresholds":
            list(thresholds),
        "uncertainties": [
            float(
                _mse(
                    candidates[
                        medoid_index
                    ]["state"]
                )
            )
            for medoid_index
            in medoids
        ],
        "transitions":
            transition_dict,
        "successor_level_mismatch_fraction":
            evaluation[
                "successor_level_mismatch_fraction"
            ],
        "refinement_history":
            history,
        "representatives":
            representatives,
        "lookup":
            lookup,
    }

    json_path = (
        output_dir / f"map_{map_id}.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )

    csv_path = (
        output_dir / f"map_{map_id}_transitions.csv"
    )

    fieldnames = [
        "xhat",
        "yhat",
        "gstate",
        "action",
        "xhat_next",
        "yhat_next",
        "gstate_next",
        "source_level",
        "successor_level",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(lookup)

    return {
        "map": map_id,
        "states": len(medoids),
        "successor_level_mismatch":
            evaluation[
                "successor_level_mismatch_fraction"
            ],
        "refinement_rounds":
            max(
                0,
                len(history) - 1,
            ),
        "lookup_transitions":
            len(lookup),
    }


# Globals used only to avoid passing immutable evaluation context repeatedly.
_CURRENT_DISTANCE_MATRIX = None
_CURRENT_MAX_STEPS = None
_CURRENT_CONTROLLER = None
_CURRENT_N = None
_CURRENT_P = None
_CURRENT_SUCCESSOR_STATE_CACHE = None
_CURRENT_CANDIDATE_MSE = None
_CURRENT_CANDIDATE_LEVEL = None
_CURRENT_MSE_DENOM = None


def analyse_map(
    map_id,
    map_path,
    output_dir,
    target,
    p,
    max_steps,
    max_rounds,
    blocks_to_try,
    candidates_per_block,
):
    global _CURRENT_DISTANCE_MATRIX
    global _CURRENT_MAX_STEPS
    global _CURRENT_CONTROLLER
    global _CURRENT_N
    global _CURRENT_P
    global _CURRENT_SUCCESSOR_STATE_CACHE
    global _CURRENT_CANDIDATE_MSE
    global _CURRENT_CANDIDATE_LEVEL
    global _CURRENT_MSE_DENOM

    print(
        f"[map {map_id}] loading map ..."
    )
    map_data = _read_map(
        map_path
    )

    print(
        f"[map {map_id}] enumerating exact histories ..."
    )
    (
        records,
        candidates,
        mse_by_age,
        controller,
    ) = _generate_records(
        map_data,
        target,
        p,
        max_steps,
    )

    print(
        f"[map {map_id}] histories="
        f"{len(records)}, "
        f"unique Gaussian states="
        f"{len(candidates)}"
    )

    thresholds = _thresholds(
        mse_by_age,
        max_steps,
    )

    print(
        f"[map {map_id}] building "
        f"w2_mse1_level1 distance matrix ..."
    )

    matrix, _ = (
        _build_candidate_distance_matrix(
            candidates,
            thresholds,
        )
    )

    print(
        f"[map {map_id}] weighted "
        f"K=100 k-medoids ..."
    )

    medoids = _weighted_kmedoids(
        candidates,
        matrix,
    )

    if len(medoids) != min(
        K_FIXED,
        len(candidates),
    ):
        raise RuntimeError(
            "Unexpected medoid count."
        )

    _CURRENT_DISTANCE_MATRIX = matrix
    _CURRENT_MAX_STEPS = max_steps
    _CURRENT_CONTROLLER = controller
    _CURRENT_N = len(map_data) - 1
    _CURRENT_P = p
    _CURRENT_CANDIDATE_MSE = [_mse(item["state"]) for item in candidates]
    _CURRENT_CANDIDATE_LEVEL = [
        _level_from_mse(value, thresholds)
        for value in _CURRENT_CANDIDATE_MSE
    ]
    _CURRENT_MSE_DENOM = max(max(_CURRENT_CANDIDATE_MSE), 1e-15)

    print(
        f"[map {map_id}] caching exact representative successor Gaussians ..."
    )
    _CURRENT_SUCCESSOR_STATE_CACHE = _build_successor_state_cache(
        map_data, controller, candidates, p
    )
    print(
        f"[map {map_id}] cached successor Gaussians="
        f"{len(_CURRENT_SUCCESSOR_STATE_CACHE)}"
    )

    candidate_index_by_state = {
        item["state"]: index
        for index, item in enumerate(
            candidates
        )
    }

    print(
        f"[map {map_id}] building exact "
        f"representative transitions ..."
    )

    transitions = _build_transition_table(
        map_data,
        controller,
        candidates,
        medoids,
        thresholds,
        p,
    )

    initial_eval = _evaluate_partition(
        records,
        candidates,
        medoids,
        thresholds,
        transitions,
        candidate_index_by_state,
    )

    print(
        f"[map {map_id}] initial successor mismatch="
        f"{initial_eval['successor_level_mismatch_fraction']:.2%}"
    )

    print(
        f"[map {map_id}] exact-semantic "
        f"medoid replacement refinement ..."
    )

    (
        medoids,
        transitions,
        history,
        final_eval,
    ) = refine(
        map_data,
        records,
        candidates,
        thresholds,
        controller,
        p,
        matrix,
        medoids,
        max_rounds,
        blocks_to_try,
        candidates_per_block,
    )

    return _write_outputs(
        map_id,
        output_dir,
        map_data,
        candidates,
        medoids,
        thresholds,
        transitions,
        history,
        final_eval,
    )


def _write_summary(
    output_dir,
    rows,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "gaussian_refined_exact_k100_summary.csv"
    )

    fieldnames = [
        "map",
        "states",
        "successor_level_mismatch",
        "refinement_rounds",
        "lookup_transitions",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    with (
        output_dir
        / "gaussian_refined_exact_k100_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "analysed_maps":
                    len(rows),
                "maps":
                    rows,
            },
            file,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate final refined K=100 Gaussian MSE representatives."
        )
    )

    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("maps"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gaussian_refined_models"),
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
        "--max-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
    )
    parser.add_argument(
        "--blocks-to-try",
        type=int,
        default=DEFAULT_BLOCKS_TO_TRY,
    )
    parser.add_argument(
        "--candidates-per-block",
        type=int,
        default=DEFAULT_CANDIDATES_PER_BLOCK,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    target = (
        args.target_x,
        args.target_y,
    )

    rows = []

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
                f"[skip] map {map_id}: "
                f"{map_path} missing"
            )
            continue

        result = analyse_map(
            map_id,
            map_path,
            args.output_dir,
            target,
            args.p,
            args.max_steps,
            args.max_rounds,
            args.blocks_to_try,
            args.candidates_per_block,
        )

        rows.append(
            result
        )

        print(
            f"[map {map_id}] FINAL "
            f"states={result['states']}, "
            f"successor mismatch="
            f"{result['successor_level_mismatch']:.2%}, "
            f"rounds="
            f"{result['refinement_rounds']}"
        )

    _write_summary(
        args.output_dir,
        rows,
    )


if __name__ == "__main__":
    main()
