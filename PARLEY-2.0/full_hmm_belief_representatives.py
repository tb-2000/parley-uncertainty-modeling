"""
full_hmm_belief_representatives.py

Build finite representative HMM-belief states from the HMM abstraction created
by full_hmm_abstraction.py.

Input
-----
    hmm_models/map_<id>.json

Expected input components:
    - 361 exact hidden error states S_t = Q_e(X_t - Xhat_t)
    - sparse position/action-dependent transition model A
    - emission model B
    - initial distribution pi

Default PARLEY comparison semantics
-----------------------------------
The current sensor model uses perfect localization (sigma_obs=0). To remain
comparable with the existing Point-Estimate and Gaussian models, there is NO
observation after every move.

Between localization events:
    beta_{t+1} = beta_t A_t

At [update]:
    beta <- pi

The generic HMM correction function using B is nevertheless implemented for a
later noisy-observation experiment:
    beta_j <- B[j,o] * beta_pred_j / normalizer

Belief abstraction
------------------
The raw HMM belief beta is a 361-dimensional probability vector. Storing all
entries directly in PRISM is impractical, so reachable raw beliefs are
generated offline and clustered into K representative beliefs.

K values tested by default:
    50, 75, 100, 125, 150, 175, 200

Distance
--------
Hidden states live on the 2-D error grid. Plain L1/L2 distance between belief
vectors ignores this geometry. Exact 2-D Wasserstein distance for every pair
of beliefs would make a K-sweep unnecessarily expensive.

This script therefore uses an efficient, reproducible approximation:
    quantile-embedded sliced Wasserstein-2 (SW2)

For fixed projection directions theta_l:
    - project every error point e_i onto theta_l
    - compute the inverse CDF of each discrete belief on fixed quantile levels
    - concatenate the projected quantiles

Euclidean distance in the resulting embedding approximates sliced W2:
    SW2^2(beta,gamma)
      ~= mean_l mean_q
          |F^{-1}_{theta_l#beta}(q)-F^{-1}_{theta_l#gamma}(q)|^2

Representatives are actual reachable beliefs (weighted k-medoids-style
alternating assignment/update), not synthetic centroids.

Outputs
-------
For each map:
    hmm_belief_models/map_<id>/
        k_050.json
        k_075.json
        ...
        k_200.json
        k_sweep_summary.csv
        k_sweep_summary.json

Each K-model contains:
    - representative belief vectors (stored sparsely)
    - representative HMM-MSE
    - ten monotone HMM-MSE thresholds (ages 1..10)
    - reachable position/representative transitions
    - approximation metrics

The script intentionally does NOT generate PRISM yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_K_VALUES = (50, 75, 100, 125, 150, 175, 200)
DEFAULT_MAX_STEPS = 10
DEFAULT_PROJECTIONS = 16
DEFAULT_QUANTILES = 64
DEFAULT_RANDOM_SEED = 7

BELIEF_ROUND_DIGITS = 14
PROB_EPS = 1e-15


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RawRecord:
    occurrence_id: int
    start_xhat: int
    start_yhat: int
    age: int
    xhat: int
    yhat: int
    action: Optional[str]
    belief_uid: int
    successor_belief_uid: Optional[int]


# ---------------------------------------------------------------------------
# Input loading / validation
# ---------------------------------------------------------------------------

def load_hmm_model(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)

    required = {
        "map_id",
        "grid_size",
        "n",
        "A",
        "B",
        "pi",
        "hidden_states",
        "quantization",
        "sensor_model",
    }
    missing = sorted(required - set(model))
    if missing:
        raise ValueError(f"{path}: missing required fields: {missing}")

    state_count = int(model["quantization"]["state_count"])
    if state_count != len(model["hidden_states"]):
        raise ValueError(
            f"{path}: quantization state_count={state_count}, "
            f"but hidden_states has {len(model['hidden_states'])} entries."
        )

    state_ids = [int(s["state_id"]) for s in model["hidden_states"]]
    if sorted(state_ids) != list(range(state_count)):
        raise ValueError(
            f"{path}: hidden state IDs must be exactly 0..{state_count - 1}."
        )

    return model


def build_initial_belief(model: dict) -> np.ndarray:
    n_states = len(model["hidden_states"])
    pi = np.zeros(n_states, dtype=np.float64)

    pi_info = model["pi"]
    if pi_info.get("type") == "delta":
        sid = int(pi_info["state"])
        pi[sid] = float(pi_info.get("probability", 1.0))
    elif "probabilities" in pi_info:
        values = np.asarray(pi_info["probabilities"], dtype=np.float64)
        if values.shape != (n_states,):
            raise ValueError("pi probability vector has wrong size.")
        pi[:] = values
    else:
        raise ValueError(
            "Unsupported pi format. Expected delta or probabilities."
        )

    total = float(pi.sum())
    if total <= 0.0:
        raise ValueError("Initial distribution pi has zero total mass.")
    pi /= total
    return pi


# ---------------------------------------------------------------------------
# Sparse HMM filtering
# ---------------------------------------------------------------------------

def build_transition_index(model: dict):
    """
    Convert JSON keys:
        "xhat,yhat,action,state_i" -> [{next_state, probability}, ...]
    into a dictionary indexed by (xhat,yhat,action,state_i).

    Also infer the MAPE-selected action at every estimate position from A.
    """
    transition_index = {}
    actions_by_position = defaultdict(set)

    for key, row in model["A"].items():
        xhat_s, yhat_s, action, state_s = key.split(",")
        xhat = int(xhat_s)
        yhat = int(yhat_s)
        state_i = int(state_s)

        sparse_row = tuple(
            (int(entry["next_state"]), float(entry["probability"]))
            for entry in row
            if float(entry["probability"]) > 0.0
        )

        total = sum(prob for _, prob in sparse_row)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"A row {key} sums to {total}.")

        transition_index[(xhat, yhat, action, state_i)] = sparse_row
        actions_by_position[(xhat, yhat)].add(action)

    policy = {}
    for pos, actions in actions_by_position.items():
        if len(actions) != 1:
            raise ValueError(
                f"Expected policy_only HMM with one action at {pos}, "
                f"got {sorted(actions)}."
            )
        policy[pos] = next(iter(actions))

    return transition_index, policy


def predict_belief(
    beta: np.ndarray,
    xhat: int,
    yhat: int,
    action: str,
    transition_index: dict,
) -> np.ndarray:
    """
    HMM prediction:
        beta_pred[j] = sum_i beta[i] A^{xhat,yhat,action}_{ij}

    Only nonzero belief entries are traversed.
    """
    result = np.zeros_like(beta)

    active = np.flatnonzero(beta > PROB_EPS)
    for state_i in active:
        mass = float(beta[state_i])
        key = (xhat, yhat, action, int(state_i))

        row = transition_index.get(key)
        if row is None:
            raise KeyError(
                "Missing A row for a currently active hidden state: "
                f"xhat={xhat}, yhat={yhat}, action={action}, "
                f"state={state_i}, beta={mass:.6g}."
            )

        for state_j, probability in row:
            result[state_j] += mass * probability

    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("Prediction produced zero probability mass.")
    result /= total

    return result


def build_emission_index(model: dict):
    """
    B[state_i] -> {observation_o: probability}
    """
    result = {}
    for state_s, row in model["B"].items():
        state_i = int(state_s)
        result[state_i] = {
            int(entry["observation"]): float(entry["probability"])
            for entry in row
            if float(entry["probability"]) > 0.0
        }
    return result


def correct_belief(
    beta_pred: np.ndarray,
    observation: int,
    emission_index: dict,
) -> np.ndarray:
    """
    Generic HMM correction:
        beta[j] proportional to B[j,o] * beta_pred[j]

    This function is not used in the default fair-comparison trajectory
    generation because there is no observation between localization events.
    """
    posterior = np.zeros_like(beta_pred)

    active = np.flatnonzero(beta_pred > PROB_EPS)
    for state_j in active:
        likelihood = emission_index.get(int(state_j), {}).get(
            int(observation), 0.0
        )
        if likelihood > 0.0:
            posterior[state_j] = beta_pred[state_j] * likelihood

    normalizer = float(posterior.sum())
    if normalizer <= 0.0:
        raise ValueError(
            f"Observation {observation} has zero likelihood under prediction."
        )

    posterior /= normalizer
    return posterior


def filter_step(
    beta: np.ndarray,
    xhat: int,
    yhat: int,
    action: str,
    transition_index: dict,
    emission_index: Optional[dict] = None,
    observation: Optional[int] = None,
) -> np.ndarray:
    """
    One complete HMM filtering step.

    No observation:
        beta' = beta A

    Observation o:
        beta_pred = beta A
        beta' proportional to B[:,o] * beta_pred
    """
    beta_pred = predict_belief(
        beta, xhat, yhat, action, transition_index
    )

    if observation is None:
        return beta_pred

    if emission_index is None:
        raise ValueError("emission_index required when observation is given.")

    return correct_belief(beta_pred, observation, emission_index)


# ---------------------------------------------------------------------------
# Estimated-position dynamics
# ---------------------------------------------------------------------------

MOVE = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}


def clip(v: int, n: int) -> int:
    return min(max(v, 0), n)


def move_estimate(
    xhat: int,
    yhat: int,
    action: str,
    n: int,
) -> Tuple[int, int]:
    dx, dy = MOVE[action]
    return clip(xhat + dx, n), clip(yhat + dy, n)


# ---------------------------------------------------------------------------
# Reachable raw-belief generation
# ---------------------------------------------------------------------------

def belief_key(beta: np.ndarray) -> bytes:
    """
    Stable deduplication key. Probabilities generated from p=0.01 are very
    structured, but rounding protects against insignificant FP differences.
    """
    rounded = np.round(beta, BELIEF_ROUND_DIGITS)
    rounded[np.abs(rounded) < 10 ** (-BELIEF_ROUND_DIGITS)] = 0.0
    return rounded.tobytes()


def generate_reachable_beliefs(
    model: dict,
    max_steps: int,
):
    """
    Start a fresh post-localization trajectory from every estimate position
    for which the MAPE policy contains an action.

    For each start:
        beta_0 = pi
        beta_{t+1} = beta_t A_t
        Xhat_{t+1} = deterministic MAPE estimate move

    No B correction is performed between moves under the default
    perfect-event-localization semantics.

    Returns:
        unique_beliefs: list[np.ndarray]
        unique_weights: occurrence count per unique belief
        records: every trajectory occurrence, with exact raw successor links
    """
    n = int(model["n"])
    transition_index, policy = build_transition_index(model)
    pi = build_initial_belief(model)

    unique_beliefs: List[np.ndarray] = []
    unique_weights: List[int] = []
    uid_by_key: Dict[bytes, int] = {}

    def register(beta: np.ndarray) -> int:
        key = belief_key(beta)
        uid = uid_by_key.get(key)
        if uid is None:
            uid = len(unique_beliefs)
            uid_by_key[key] = uid
            unique_beliefs.append(beta.copy())
            unique_weights.append(1)
        else:
            unique_weights[uid] += 1
        return uid

    temp_records = []

    for start_xhat in range(n + 1):
        for start_yhat in range(n + 1):
            if (start_xhat, start_yhat) not in policy:
                continue

            xhat = start_xhat
            yhat = start_yhat
            beta = pi.copy()

            for age in range(max_steps + 1):
                current_uid = register(beta)

                action = policy.get((xhat, yhat))
                successor_uid = None

                if age < max_steps and action is not None:
                    next_beta = predict_belief(
                        beta,
                        xhat,
                        yhat,
                        action,
                        transition_index,
                    )
                    successor_uid = register(next_beta)
                else:
                    next_beta = None

                temp_records.append(
                    (
                        start_xhat,
                        start_yhat,
                        age,
                        xhat,
                        yhat,
                        action,
                        current_uid,
                        successor_uid,
                    )
                )

                if next_beta is None:
                    break

                beta = next_beta
                xhat, yhat = move_estimate(xhat, yhat, action, n)

    # register() counts both current occurrences and temporary successor
    # lookups above. Recompute occurrence weights from actual records only.
    occurrence_counts = Counter(row[6] for row in temp_records)
    unique_weights = [
        int(occurrence_counts.get(uid, 0))
        for uid in range(len(unique_beliefs))
    ]

    records = [
        RawRecord(
            occurrence_id=i,
            start_xhat=row[0],
            start_yhat=row[1],
            age=row[2],
            xhat=row[3],
            yhat=row[4],
            action=row[5],
            belief_uid=row[6],
            successor_belief_uid=row[7],
        )
        for i, row in enumerate(temp_records)
    ]

    return (
        np.asarray(unique_beliefs, dtype=np.float64),
        np.asarray(unique_weights, dtype=np.float64),
        records,
        transition_index,
        policy,
    )


# ---------------------------------------------------------------------------
# Belief uncertainty and thresholds
# ---------------------------------------------------------------------------

def hidden_squared_errors(model: dict) -> np.ndarray:
    values = np.zeros(len(model["hidden_states"]), dtype=np.float64)
    for state in model["hidden_states"]:
        values[int(state["state_id"])] = float(state["squared_error"])
    return values


def belief_mse(
    beliefs: np.ndarray,
    squared_errors: np.ndarray,
) -> np.ndarray:
    """
    U_HMM(beta) = E_beta[||e||^2]
                = sum_i beta_i (e_x,i^2 + e_y,i^2)
    """
    return beliefs @ squared_errors


def monotone_age_thresholds(
    records: Sequence[RawRecord],
    unique_mse: np.ndarray,
    max_steps: int,
) -> List[float]:
    """
    T_k = median raw HMM-MSE among occurrences at age k, k=1..max_steps.
    Afterwards enforce T_1 <= ... <= T_max_steps.
    """
    thresholds = []
    previous = 0.0

    for age in range(1, max_steps + 1):
        values = [
            float(unique_mse[r.belief_uid])
            for r in records
            if r.age == age
        ]

        if values:
            threshold = float(np.median(np.asarray(values)))
        else:
            threshold = previous

        threshold = max(previous, threshold)
        thresholds.append(threshold)
        previous = threshold

    return thresholds


def uncertainty_level(value: float, thresholds: Sequence[float]) -> int:
    """
    0 = below threshold 1
    1..max_steps = highest reached threshold
    """
    level = 0
    for i, threshold in enumerate(thresholds, start=1):
        if value + 1e-15 >= threshold:
            level = i
        else:
            break
    return level


# ---------------------------------------------------------------------------
# Quantile-embedded sliced Wasserstein
# ---------------------------------------------------------------------------

def projection_directions(count: int) -> np.ndarray:
    """
    Deterministic half-circle directions. theta and -theta are redundant for
    sliced Wasserstein, so [0,pi) is sufficient.
    """
    angles = np.linspace(0.0, math.pi, count, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def build_sw2_embedding(
    beliefs: np.ndarray,
    error_points: np.ndarray,
    n_projections: int,
    n_quantiles: int,
) -> np.ndarray:
    """
    Approximate sliced-W2 using inverse-CDF samples of 1-D projections.

    Embedding scaling is chosen so Euclidean distance equals:
        sqrt(mean_projection mean_quantile squared quantile difference)
    """
    n_beliefs = beliefs.shape[0]
    directions = projection_directions(n_projections)

    # Midpoint quantiles avoid exactly 0 and 1.
    quantiles = (
        np.arange(n_quantiles, dtype=np.float64) + 0.5
    ) / n_quantiles

    embedding = np.empty(
        (n_beliefs, n_projections * n_quantiles),
        dtype=np.float64,
    )

    scale = 1.0 / math.sqrt(n_projections * n_quantiles)

    for d_idx, direction in enumerate(directions):
        projected = error_points @ direction
        order = np.argsort(projected, kind="mergesort")
        sorted_projected = projected[order]

        block_start = d_idx * n_quantiles
        block_end = block_start + n_quantiles

        for b_idx in range(n_beliefs):
            weights = beliefs[b_idx, order]
            cdf = np.cumsum(weights)
            cdf[-1] = 1.0

            indices = np.searchsorted(
                cdf,
                quantiles,
                side="left",
            )
            indices = np.minimum(indices, len(sorted_projected) - 1)

            embedding[
                b_idx, block_start:block_end
            ] = sorted_projected[indices] * scale

    return embedding


def pairwise_euclidean(embedding: np.ndarray) -> np.ndarray:
    """
    Dense pairwise distance matrix. Reachable-belief sets for max_steps=10
    are expected to be small enough (~<= 1000 occurrences before dedup).
    """
    sq = np.sum(embedding * embedding, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (embedding @ embedding.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, out=d2)


# ---------------------------------------------------------------------------
# Weighted k-medoids-style clustering
# ---------------------------------------------------------------------------

def weighted_kmedoids(
    distance_matrix: np.ndarray,
    weights: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic-seeded weighted k-medoids-style alternating optimization.

    Initialization:
        weighted k-medoids++ over pairwise distances

    Update:
        exact weighted medoid inside each assigned cluster

    Returns:
        medoid_indices
        assignment for every raw unique belief
        distance-to-assigned-medoid
    """
    n = distance_matrix.shape[0]

    if k >= n:
        medoids = np.arange(n, dtype=int)
        assignment = np.arange(n, dtype=int)
        return medoids, assignment, np.zeros(n, dtype=np.float64)

    rng = np.random.default_rng(seed)

    # First medoid: highest-weight point, deterministic tie break.
    first = int(np.argmax(weights))
    medoids = [first]

    nearest_sq = distance_matrix[:, first] ** 2

    while len(medoids) < k:
        probs = weights * nearest_sq
        probs[np.asarray(medoids, dtype=int)] = 0.0
        total = float(probs.sum())

        if total <= 0.0:
            remaining = [
                i for i in range(n)
                if i not in set(medoids)
            ]
            medoids.append(remaining[0])
        else:
            probs /= total
            candidate = int(rng.choice(n, p=probs))
            if candidate in medoids:
                # Extremely unlikely numerical corner case.
                for i in np.argsort(-probs):
                    if int(i) not in medoids:
                        candidate = int(i)
                        break
            medoids.append(candidate)

        nearest_sq = np.minimum(
            nearest_sq,
            distance_matrix[:, medoids[-1]] ** 2,
        )

    medoids = np.asarray(medoids, dtype=int)

    previous = None
    assignment = None

    for _ in range(max_iter):
        dist_to_medoids = distance_matrix[:, medoids]
        assignment = np.argmin(dist_to_medoids, axis=1)

        new_medoids = medoids.copy()

        for cluster_id in range(k):
            members = np.flatnonzero(assignment == cluster_id)

            if len(members) == 0:
                current_dist = np.min(dist_to_medoids, axis=1)
                score = weights * current_dist
                candidate_order = np.argsort(-score)

                for candidate in candidate_order:
                    if int(candidate) not in set(new_medoids.tolist()):
                        new_medoids[cluster_id] = int(candidate)
                        break
                continue

            sub_d = distance_matrix[np.ix_(members, members)]
            member_weights = weights[members]

            # Candidate medoid c minimizes sum_r w_r d(r,c).
            costs = member_weights @ sub_d
            best_local = int(np.argmin(costs))
            new_medoids[cluster_id] = int(members[best_local])

        # Remove accidental duplicates by replacing duplicates with far points.
        used = set()
        duplicates = []
        for idx, medoid in enumerate(new_medoids):
            medoid = int(medoid)
            if medoid in used:
                duplicates.append(idx)
            else:
                used.add(medoid)

        if duplicates:
            dist_to_used = np.min(
                distance_matrix[:, np.asarray(sorted(used), dtype=int)],
                axis=1,
            )
            candidates = np.argsort(-(weights * dist_to_used))

            for cluster_id in duplicates:
                for candidate in candidates:
                    candidate = int(candidate)
                    if candidate not in used:
                        new_medoids[cluster_id] = candidate
                        used.add(candidate)
                        break

        new_medoids = np.asarray(new_medoids, dtype=int)

        if previous is not None and np.array_equal(
            np.sort(new_medoids), np.sort(previous)
        ):
            medoids = new_medoids
            break

        previous = medoids
        medoids = new_medoids

    dist_to_medoids = distance_matrix[:, medoids]
    assignment = np.argmin(dist_to_medoids, axis=1)
    assigned_distance = dist_to_medoids[
        np.arange(n), assignment
    ]

    # Stable representative numbering:
    # representative 0 should be the exact reset belief pi if present.
    zero_raw = int(np.argmax(weights))  # overwritten below by zero-distance pi search
    # pi is normally the only belief with MSE 0; identify via distance pattern later.
    # We simply place the medoid with the smallest original index first for stable JSON.
    order = np.argsort(medoids)
    medoids = medoids[order]

    remap = np.empty(k, dtype=int)
    for new_id, old_cluster_id in enumerate(order):
        remap[old_cluster_id] = new_id
    assignment = remap[assignment]

    assigned_distance = distance_matrix[
        np.arange(n),
        medoids[assignment],
    ]

    return medoids, assignment, assigned_distance


# ---------------------------------------------------------------------------
# Transition abstraction and metrics
# ---------------------------------------------------------------------------

def build_representative_transitions(
    records: Sequence[RawRecord],
    assignment: np.ndarray,
):
    """
    For every actually reached:
        (xhat, yhat, action, representative)

    inspect all raw transitions represented by that abstract state.

    If all lead to the same next representative, the abstract transition is
    exact/deterministic.

    If multiple successors occur because distinct raw beliefs were merged,
    choose the modal successor and record the ambiguity. This is analogous to
    measuring transition mismatch in the Gaussian abstraction.
    """
    successor_counts = defaultdict(Counter)

    for record in records:
        if (
            record.action is None
            or record.successor_belief_uid is None
        ):
            continue

        current_rep = int(assignment[record.belief_uid])
        next_rep = int(assignment[record.successor_belief_uid])

        key = (
            record.xhat,
            record.yhat,
            record.action,
            current_rep,
        )
        successor_counts[key][next_rep] += 1

    transitions = {}
    total = 0
    mismatch = 0
    ambiguous_keys = 0

    for key, counts in successor_counts.items():
        total_here = sum(counts.values())
        best_next, best_count = min(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )

        total += total_here
        mismatch += total_here - best_count

        if len(counts) > 1:
            ambiguous_keys += 1

        xhat, yhat, action, rep = key
        json_key = f"{xhat},{yhat},{rep}"

        transitions[json_key] = {
            "action": action,
            "next_belief_state": int(best_next),
            "support_count": int(total_here),
            "agreement": float(best_count / total_here),
            "candidate_successors": [
                {
                    "next_belief_state": int(next_rep),
                    "count": int(count),
                    "fraction": float(count / total_here),
                }
                for next_rep, count in sorted(
                    counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        }

    mismatch_rate = (
        float(mismatch / total)
        if total > 0 else 0.0
    )

    return transitions, {
        "transition_occurrences": int(total),
        "transition_mismatches": int(mismatch),
        "transition_mismatch_rate": mismatch_rate,
        "ambiguous_transition_keys": int(ambiguous_keys),
        "transition_key_count": int(len(successor_counts)),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total > 0 else 0.0


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    cutoff = q * float(w.sum())
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def compute_metrics(
    unique_mse: np.ndarray,
    representative_mse: np.ndarray,
    assignment: np.ndarray,
    assigned_distance: np.ndarray,
    weights: np.ndarray,
    thresholds: Sequence[float],
    transition_metrics: dict,
):
    approx_mse = representative_mse[assignment]
    abs_error = np.abs(approx_mse - unique_mse)

    rel_error = np.zeros_like(abs_error)
    mask = unique_mse > 1e-12
    rel_error[mask] = abs_error[mask] / unique_mse[mask]

    raw_levels = np.asarray(
        [uncertainty_level(v, thresholds) for v in unique_mse],
        dtype=int,
    )
    rep_levels = np.asarray(
        [uncertainty_level(v, thresholds) for v in approx_mse],
        dtype=int,
    )
    disagreement = (raw_levels != rep_levels).astype(np.float64)

    return {
        "sw2_mean": weighted_mean(assigned_distance, weights),
        "sw2_p95": weighted_quantile(
            assigned_distance, weights, 0.95
        ),
        "sw2_max": float(np.max(assigned_distance)),
        "mse_abs_mean": weighted_mean(abs_error, weights),
        "mse_rel_mean": weighted_mean(rel_error, weights),
        "urc_level_disagreement": weighted_mean(
            disagreement, weights
        ),
        **transition_metrics,
    }


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def sparse_belief(beta: np.ndarray, eps: float = 1e-14):
    return [
        {
            "hidden_state": int(i),
            "probability": float(beta[i]),
        }
        for i in np.flatnonzero(beta > eps)
    ]


def error_points_from_model(model: dict) -> np.ndarray:
    points = np.zeros(
        (len(model["hidden_states"]), 2),
        dtype=np.float64,
    )
    for state in model["hidden_states"]:
        sid = int(state["state_id"])
        points[sid, 0] = float(state["error_x"])
        points[sid, 1] = float(state["error_y"])
    return points


def ensure_reset_representative_zero(
    model: dict,
    beliefs: np.ndarray,
    medoids: np.ndarray,
    assignment: np.ndarray,
):
    """
    Re-number representatives so the exact pi/reset belief is representative 0.

    If pi was not selected as a medoid (unlikely because it is heavily
    repeated), replace the medoid of pi's cluster with pi itself.
    """
    pi = build_initial_belief(model)
    distances_to_pi = np.max(np.abs(beliefs - pi[None, :]), axis=1)
    pi_uid = int(np.argmin(distances_to_pi))

    if distances_to_pi[pi_uid] > 1e-12:
        raise ValueError("Exact pi belief not found among reachable beliefs.")

    cluster = int(assignment[pi_uid])

    if int(medoids[cluster]) != pi_uid:
        medoids = medoids.copy()
        medoids[cluster] = pi_uid

    if cluster != 0:
        # Swap representative IDs cluster <-> 0.
        medoids = medoids.copy()
        medoids[[0, cluster]] = medoids[[cluster, 0]]

        new_assignment = assignment.copy()
        mask0 = assignment == 0
        maskc = assignment == cluster
        new_assignment[mask0] = cluster
        new_assignment[maskc] = 0
        assignment = new_assignment

    return medoids, assignment


# ---------------------------------------------------------------------------
# One K run
# ---------------------------------------------------------------------------

def run_k(
    model: dict,
    beliefs: np.ndarray,
    weights: np.ndarray,
    records: Sequence[RawRecord],
    distance_matrix: np.ndarray,
    unique_mse: np.ndarray,
    thresholds: Sequence[float],
    k: int,
    seed: int,
    max_steps: int,
    n_projections: int,
    n_quantiles: int,
):
    effective_k = min(int(k), len(beliefs))

    medoids, assignment, _ = weighted_kmedoids(
        distance_matrix=distance_matrix,
        weights=weights,
        k=effective_k,
        seed=seed + effective_k,
    )

    medoids, assignment = ensure_reset_representative_zero(
        model=model,
        beliefs=beliefs,
        medoids=medoids,
        assignment=assignment,
    )

    assigned_distance = distance_matrix[
        np.arange(len(beliefs)),
        medoids[assignment],
    ]

    representatives = beliefs[medoids]
    squared_errors = hidden_squared_errors(model)
    representative_mse = belief_mse(
        representatives,
        squared_errors,
    )

    transitions, transition_metrics = (
        build_representative_transitions(
            records=records,
            assignment=assignment,
        )
    )

    metrics = compute_metrics(
        unique_mse=unique_mse,
        representative_mse=representative_mse,
        assignment=assignment,
        assigned_distance=assigned_distance,
        weights=weights,
        thresholds=thresholds,
        transition_metrics=transition_metrics,
    )

    rep_occurrences = np.zeros(effective_k, dtype=int)
    for uid, count in enumerate(weights.astype(int)):
        rep_occurrences[int(assignment[uid])] += int(count)

    rep_payload = []
    for rep_id in range(effective_k):
        raw_uid = int(medoids[rep_id])
        beta = representatives[rep_id]

        rep_payload.append({
            "belief_state": int(rep_id),
            "raw_belief_uid": raw_uid,
            "occurrences": int(rep_occurrences[rep_id]),
            "mse": float(representative_mse[rep_id]),
            "support_size": int(np.count_nonzero(beta > 1e-14)),
            "belief": sparse_belief(beta),
        })

    payload = {
        "map_id": int(model["map_id"]),
        "hidden_state_count": int(len(model["hidden_states"])),
        "raw_unique_belief_count": int(len(beliefs)),
        "raw_occurrence_count": int(len(records)),
        "belief_state_count": int(effective_k),
        "requested_k": int(k),
        "max_steps": int(max_steps),
        "distance": {
            "name": "quantile_embedded_sliced_wasserstein_2",
            "projections": int(n_projections),
            "quantiles_per_projection": int(n_quantiles),
            "note": (
                "Euclidean distance in the quantile projection embedding "
                "approximates sliced W2; representatives are actual raw beliefs."
            ),
        },
        "uncertainty_metric": (
            "expected_squared_error = sum_i beta_i * ||e_i||^2"
        ),
        "thresholds": [float(v) for v in thresholds],
        "reset_belief_state": 0,
        "representatives": rep_payload,
        "belief_transitions": transitions,
        "metrics": metrics,
        "filtering_semantics": {
            "between_updates": "beta_next = beta * A[xhat,yhat,action]",
            "observation_between_moves": False,
            "on_update": "beta = pi",
            "B_available_for_future_noisy_observation_experiment": True,
        },
    }

    return payload


# ---------------------------------------------------------------------------
# Map-level K sweep
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = (
    "map_id",
    "requested_k",
    "effective_k",
    "raw_unique_beliefs",
    "raw_occurrences",
    "sw2_mean",
    "sw2_p95",
    "sw2_max",
    "mse_abs_mean",
    "mse_rel_mean",
    "urc_level_disagreement",
    "transition_mismatch_rate",
    "transition_mismatches",
    "transition_occurrences",
    "ambiguous_transition_keys",
    "transition_key_count",
)


def process_map(
    input_path: Path,
    output_root: Path,
    k_values: Sequence[int],
    max_steps: int,
    n_projections: int,
    n_quantiles: int,
    seed: int,
):
    model = load_hmm_model(input_path)

    print(f"\nMap {model['map_id']}: generating reachable HMM beliefs ...")

    (
        beliefs,
        weights,
        records,
        _transition_index,
        _policy,
    ) = generate_reachable_beliefs(
        model=model,
        max_steps=max_steps,
    )

    print(
        f"  raw occurrences={len(records)}, "
        f"unique beliefs={len(beliefs)}"
    )

    squared_errors = hidden_squared_errors(model)
    unique_mse = belief_mse(beliefs, squared_errors)

    thresholds = monotone_age_thresholds(
        records=records,
        unique_mse=unique_mse,
        max_steps=max_steps,
    )

    error_points = error_points_from_model(model)

    print(
        f"  building SW2 embedding: "
        f"{n_projections} projections x {n_quantiles} quantiles ..."
    )
    embedding = build_sw2_embedding(
        beliefs=beliefs,
        error_points=error_points,
        n_projections=n_projections,
        n_quantiles=n_quantiles,
    )

    print("  computing pairwise belief distances ...")
    distance_matrix = pairwise_euclidean(embedding)

    map_dir = output_root / f"map_{model['map_id']}"
    map_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for k in k_values:
        print(f"  K={k} ...")

        payload = run_k(
            model=model,
            beliefs=beliefs,
            weights=weights,
            records=records,
            distance_matrix=distance_matrix,
            unique_mse=unique_mse,
            thresholds=thresholds,
            k=k,
            seed=seed,
            max_steps=max_steps,
            n_projections=n_projections,
            n_quantiles=n_quantiles,
        )

        output_path = map_dir / f"k_{int(k):03d}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        m = payload["metrics"]

        summary = {
            "map_id": int(model["map_id"]),
            "requested_k": int(k),
            "effective_k": int(payload["belief_state_count"]),
            "raw_unique_beliefs": int(len(beliefs)),
            "raw_occurrences": int(len(records)),
            "sw2_mean": float(m["sw2_mean"]),
            "sw2_p95": float(m["sw2_p95"]),
            "sw2_max": float(m["sw2_max"]),
            "mse_abs_mean": float(m["mse_abs_mean"]),
            "mse_rel_mean": float(m["mse_rel_mean"]),
            "urc_level_disagreement": float(
                m["urc_level_disagreement"]
            ),
            "transition_mismatch_rate": float(
                m["transition_mismatch_rate"]
            ),
            "transition_mismatches": int(
                m["transition_mismatches"]
            ),
            "transition_occurrences": int(
                m["transition_occurrences"]
            ),
            "ambiguous_transition_keys": int(
                m["ambiguous_transition_keys"]
            ),
            "transition_key_count": int(
                m["transition_key_count"]
            ),
        }
        summaries.append(summary)

        print(
            "    "
            f"SW2 mean={summary['sw2_mean']:.6f}, "
            f"MSE rel={100*summary['mse_rel_mean']:.3f}%, "
            f"URC disagree={100*summary['urc_level_disagreement']:.3f}%, "
            f"transition mismatch="
            f"{100*summary['transition_mismatch_rate']:.3f}%"
        )

    csv_path = map_dir / "k_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    json_path = map_dir / "k_sweep_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "map_id": int(model["map_id"]),
                "input": str(input_path),
                "max_steps": int(max_steps),
                "k_values": [int(k) for k in k_values],
                "thresholds": thresholds,
                "distance": {
                    "name": "quantile_embedded_sliced_wasserstein_2",
                    "projections": int(n_projections),
                    "quantiles_per_projection": int(n_quantiles),
                },
                "summaries": summaries,
            },
            f,
            indent=2,
        )

    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")

    return summaries


# ---------------------------------------------------------------------------
# Multi-map driver
# ---------------------------------------------------------------------------

def parse_k_values(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("K values must be positive.")
        values.append(value)

    if not values:
        raise ValueError("At least one K value is required.")

    return values


def precompute_maps(
    first_map: int,
    last_map: int,
    hmm_dir: Path,
    output_dir: Path,
    k_values: Sequence[int],
    max_steps: int,
    n_projections: int,
    n_quantiles: int,
    seed: int,
):
    all_summaries = []

    for map_id in range(first_map, last_map + 1):
        input_path = hmm_dir / f"map_{map_id}.json"

        if not input_path.exists():
            print(f"skip map {map_id}: {input_path} missing")
            continue

        summaries = process_map(
            input_path=input_path,
            output_root=output_dir,
            k_values=k_values,
            max_steps=max_steps,
            n_projections=n_projections,
            n_quantiles=n_quantiles,
            seed=seed,
        )
        all_summaries.extend(summaries)

    if all_summaries:
        combined_path = output_dir / "all_maps_k_sweep_summary.csv"
        output_dir.mkdir(parents=True, exist_ok=True)

        with combined_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=SUMMARY_FIELDS,
            )
            writer.writeheader()
            writer.writerows(all_summaries)

        print(f"\nCombined summary: {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate reachable HMM beliefs, cluster them into representative "
            "belief states, and evaluate a K sweep."
        )
    )

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
        "--hmm-dir",
        type=Path,
        default=Path("hmm_models"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hmm_belief_models"),
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated K values, e.g. 50,75,100,125,150,175,200",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
    )
    parser.add_argument(
        "--projections",
        type=int,
        default=DEFAULT_PROJECTIONS,
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=DEFAULT_QUANTILES,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)

    precompute_maps(
        first_map=args.first_map,
        last_map=args.last_map,
        hmm_dir=args.hmm_dir,
        output_dir=args.output_dir,
        k_values=k_values,
        max_steps=args.max_steps,
        n_projections=args.projections,
        n_quantiles=args.quantiles,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
