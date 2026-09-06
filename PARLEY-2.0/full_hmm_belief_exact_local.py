"""
Exact reachable HMM-belief model for PARLEY with position-local state IDs.

Pipeline
--------
1. Load the exact HMM dynamics from hmm_models/map_<id>.json.
2. Use prediction ages 1..10 ONLY to calibrate the ten monotone HMM-MSE
   thresholds.
3. Enumerate ALL exact reachable HMM-belief contexts
       (xhat, yhat, beta)
   from every post-localization estimate position under the fixed MAPE policy.
4. Continue exact reachability until:
       * target / no MAPE action, or
       * HMM-MSE >= tau_10.
5. Encode every exact context in PRISM as:
       (xhat, yhat, hmm_state)
   where hmm_state is a POSITION-LOCAL ID of one exact HMM belief beta.

There is:
- no behavioral merging,
- no substate,
- no K,
- no medoids,
- no nearest-representative projection.

Important:
    hmm_state=7 is not globally unique.
The complete exact knowledge-state identity is:
    (xhat, yhat, hmm_state)

At every position:
    hmm_state=0 <-> beta = pi
so perfect localization can always reset with:
    xhat'=x & yhat'=y & hmm_state'=0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from full_hmm_belief_representatives import (
    PROB_EPS,
    RawRecord,
    belief_key,
    belief_mse,
    build_initial_belief,
    build_transition_index,
    hidden_squared_errors,
    load_hmm_model,
    monotone_age_thresholds,
    move_estimate,
    predict_belief,
    sparse_belief,
)

DEFAULT_MAX_STEPS = 10


def uncertainty_level(value: float, thresholds: Sequence[float]) -> int:
    """Highest HMM-MSE threshold reached: 0..10."""
    level = 0

    for i, threshold in enumerate(thresholds, start=1):
        if value + 1e-15 >= float(threshold):
            level = i
        else:
            break

    return level


def calibrate_thresholds(model: dict, max_steps: int = DEFAULT_MAX_STEPS):
    """
    Calibrate:
        tau_k = median HMM-MSE at prediction age k, k=1..10
    and enforce monotonicity.

    max_steps affects threshold calibration only, not runtime reachability.
    """
    n = int(model["n"])
    transition_index, policy = build_transition_index(model)
    pi = build_initial_belief(model)

    unique_beliefs: List[np.ndarray] = []
    uid_by_key: Dict[bytes, int] = {}
    records = []

    def register(beta: np.ndarray) -> int:
        key = belief_key(beta)

        if key not in uid_by_key:
            uid_by_key[key] = len(unique_beliefs)
            unique_beliefs.append(beta.copy())

        return uid_by_key[key]

    occurrence_id = 0

    for start_xhat in range(n + 1):
        for start_yhat in range(n + 1):
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

                records.append(
                    RawRecord(
                        occurrence_id=occurrence_id,
                        start_xhat=start_xhat,
                        start_yhat=start_yhat,
                        age=age,
                        xhat=xhat,
                        yhat=yhat,
                        action=action,
                        belief_uid=current_uid,
                        successor_belief_uid=successor_uid,
                    )
                )
                occurrence_id += 1

                if next_beta is None:
                    break

                beta = next_beta
                xhat, yhat = move_estimate(
                    xhat,
                    yhat,
                    action,
                    n,
                )

    beliefs = np.asarray(unique_beliefs, dtype=np.float64)
    squared_errors = hidden_squared_errors(model)
    unique_mse = belief_mse(beliefs, squared_errors)

    thresholds = monotone_age_thresholds(
        records=records,
        unique_mse=unique_mse,
        max_steps=max_steps,
    )

    if len(thresholds) != 10:
        raise ValueError(
            f"Expected exactly 10 HMM thresholds, got {len(thresholds)}."
        )

    return [float(v) for v in thresholds]


def build_exact_local_hmm_model(
    model: dict,
    target: Tuple[int, int],
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """
    Build all exact reachable HMM-belief contexts.

    PRISM state:
        (xhat, yhat, hmm_state)

    hmm_state is a local index inside one estimated position.
    State 0 is reserved for beta=pi at every position.
    """
    thresholds = calibrate_thresholds(
        model=model,
        max_steps=max_steps,
    )

    n = int(model["n"])
    transition_index, policy = build_transition_index(model)
    pi = build_initial_belief(model)
    squared_errors = hidden_squared_errors(model)
    tau10 = float(thresholds[-1])

    # Global distinct-beta table for diagnostics only.
    beliefs: List[np.ndarray] = []
    belief_uid_by_key: Dict[bytes, int] = {}

    def register_belief(beta: np.ndarray) -> int:
        key = belief_key(beta)

        if key not in belief_uid_by_key:
            belief_uid_by_key[key] = len(beliefs)
            beliefs.append(beta.copy())

        return belief_uid_by_key[key]

    pi_uid = register_belief(pi)

    # Local HMM state tables.
    # position_beliefs[(xhat,yhat)][local_id] = exact beta
    position_beliefs = defaultdict(list)
    local_id_by_context = {}
    queue = deque()

    def register_context(
        xhat: int,
        yhat: int,
        beta: np.ndarray,
    ):
        beta_key = belief_key(beta)
        context_key = (
            int(xhat),
            int(yhat),
            beta_key,
        )

        if context_key in local_id_by_context:
            return local_id_by_context[context_key], False

        position = (int(xhat), int(yhat))
        local_id = len(position_beliefs[position])

        local_id_by_context[context_key] = local_id
        position_beliefs[position].append(beta.copy())
        register_belief(beta)

        queue.append((
            int(xhat),
            int(yhat),
            beta.copy(),
        ))

        return local_id, True

    # Seed beta=pi first at every position. Therefore local ID 0 is always reset.
    for xhat in range(n + 1):
        for yhat in range(n + 1):
            local_id, _ = register_context(
                xhat,
                yhat,
                pi,
            )

            if local_id != 0:
                raise AssertionError(
                    f"Reset HMM belief at ({xhat},{yhat}) must have local ID 0."
                )

    transitions = {}

    while queue:
        xhat, yhat, beta = queue.popleft()
        state_id = local_id_by_context[
            (xhat, yhat, belief_key(beta))
        ]

        mse = float(np.dot(beta, squared_errors))
        level = uncertainty_level(mse, thresholds)

        # At level 10 every c in 1..10 localizes.
        if level >= 10 or mse + 1e-15 >= tau10:
            continue

        if (xhat, yhat) == tuple(target):
            continue

        action = policy.get((xhat, yhat))
        if action is None:
            continue

        next_beta = predict_belief(
            beta,
            xhat,
            yhat,
            action,
            transition_index,
        )

        next_xhat, next_yhat = move_estimate(
            xhat,
            yhat,
            action,
            n,
        )

        next_state_id, _ = register_context(
            next_xhat,
            next_yhat,
            next_beta,
        )

        transitions[f"{xhat},{yhat},{state_id}"] = {
            "action": action,
            "next_xhat": int(next_xhat),
            "next_yhat": int(next_yhat),
            "next_state": int(next_state_id),
        }

    contexts = []
    uncertainties = {}
    urc_levels = {}
    local_state_counts = {}

    for xhat in range(n + 1):
        for yhat in range(n + 1):
            states_here = position_beliefs[(xhat, yhat)]

            local_state_counts[f"{xhat},{yhat}"] = len(states_here)
            uncertainties[f"{xhat},{yhat}"] = []
            urc_levels[f"{xhat},{yhat}"] = []

            for state_id, beta in enumerate(states_here):
                mse = float(np.dot(beta, squared_errors))
                level = uncertainty_level(mse, thresholds)
                uid = register_belief(beta)

                uncertainties[f"{xhat},{yhat}"].append(mse)
                urc_levels[f"{xhat},{yhat}"].append(level)

                contexts.append({
                    "xhat": int(xhat),
                    "yhat": int(yhat),
                    "hmm_state": int(state_id),
                    "belief_uid": int(uid),
                    "mse": mse,
                    "urc_level": int(level),
                    "support_size": int(
                        np.count_nonzero(beta > PROB_EPS)
                    ),
                })

    max_local_states = max(
        local_state_counts.values(),
        default=1,
    )

    reachable_hidden_states = sorted({
        int(hidden_state)
        for beta in beliefs
        for hidden_state in np.flatnonzero(beta > PROB_EPS)
    })

    belief_payload = []

    for uid, beta in enumerate(beliefs):
        mse = float(np.dot(beta, squared_errors))

        belief_payload.append({
            "belief_uid": int(uid),
            "mse": mse,
            "urc_level": int(
                uncertainty_level(mse, thresholds)
            ),
            "support_size": int(
                np.count_nonzero(beta > PROB_EPS)
            ),
            "belief": sparse_belief(beta),
        })

    return {
        "mode": "exact_local_hmm_beliefs",
        "representation": "xhat_yhat_local_hmm_state",
        "map_id": int(model["map_id"]),
        "grid_size": int(model["grid_size"]),
        "n": int(model["n"]),
        "max_steps": int(max_steps),
        "max_steps_semantics": "threshold_calibration_only",

        "uncertainty_metric":
            "expected_squared_error = sum_i beta_i * ||e_i||^2",
        "thresholds": [float(v) for v in thresholds],

        "hidden_state_count_full": int(len(model["hidden_states"])),
        "reachable_hidden_state_count": int(
            len(reachable_hidden_states)
        ),
        "reachable_hidden_states": reachable_hidden_states,

        "exact_belief_count": int(len(beliefs)),
        "exact_context_count": int(len(contexts)),
        "context_count": int(len(contexts)),

        "max_local_states": int(max_local_states),
        "max_hmm_state": int(max_local_states - 1),
        "local_state_counts": local_state_counts,

        "contexts": contexts,
        "belief_transitions": transitions,
        "uncertainties": uncertainties,
        "urc_levels": urc_levels,

        "reset_local_state": 0,
        "exact_beliefs": belief_payload,

        "filtering_semantics": {
            "between_updates":
                "beta_next = beta * A[xhat,yhat,action]",
            "observation_between_moves": False,
            "on_update": "beta = pi",
            "reachability_stop":
                "target/no-action or HMM-MSE >= tau_10",
            "frontier": False,
        },
    }


def process_map(
    map_id: int,
    hmm_dir: Path,
    output_dir: Path,
    target: Tuple[int, int],
    max_steps: int,
):
    input_path = hmm_dir / f"map_{map_id}.json"

    if not input_path.exists():
        print(f"skip map {map_id}: {input_path} missing")
        return None

    model = load_hmm_model(input_path)

    result = build_exact_local_hmm_model(
        model=model,
        target=target,
        max_steps=max_steps,
    )

    map_dir = output_dir / f"map_{map_id}"
    map_dir.mkdir(parents=True, exist_ok=True)

    output_path = map_dir / "exact_local.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"map {map_id}: "
        f"hidden {result['reachable_hidden_state_count']}/"
        f"{result['hidden_state_count_full']}, "
        f"exact beliefs={result['exact_belief_count']}, "
        f"exact contexts={result['exact_context_count']}, "
        f"max local HMM states={result['max_local_states']}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build all exact reachable HMM-belief contexts and encode them "
            "with position-local hmm_state IDs."
        )
    )

    parser.add_argument("--first-map", type=int, default=10)
    parser.add_argument("--last-map", type=int, default=99)

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

    parser.add_argument("--target-x", type=int, default=9)
    parser.add_argument("--target-y", type=int, default=9)

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=(
            "Used only to calibrate the ten HMM-MSE thresholds. "
            "It does not truncate exact reachability."
        ),
    )

    args = parser.parse_args()
    target = (args.target_x, args.target_y)

    for map_id in range(
        args.first_map,
        args.last_map + 1,
    ):
        process_map(
            map_id=map_id,
            hmm_dir=args.hmm_dir,
            output_dir=args.output_dir,
            target=target,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
