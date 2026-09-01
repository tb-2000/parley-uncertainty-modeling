"""
Exact reachable + behavioral HMM-belief abstraction for PARLEY.

Pipeline
--------
1. Load the exact HMM dynamics produced by full_hmm_abstraction.py.
2. Use age 1..10 ONLY to calibrate ten monotone HMM-MSE thresholds.
3. Enumerate exact reachable HMM-belief contexts from every post-localization
   estimate position under the fixed MAPE policy.
4. Continue exact reachability until:
       * target / no MAPE action, or
       * HMM-MSE has reached tau_10.
   There is NO artificial max_steps frontier in the runtime automaton.
5. Compute deterministic behavioral equivalence by partition refinement.
6. Encode every quotient class structurally as:

       (xhat, yhat, hmm_state, substate)

   where hmm_state is the HMM-MSE threshold level 0..10.

The 361 hidden error states remain the exact basis of the HMM dynamics.
However, only hidden states that occur with positive probability in an exact
reachable HMM belief are reported as `reachable_hidden_states`; unreachable
hidden states never appear in the PRISM Knowledge abstraction.

No K, no medoids, no Wasserstein projection, and no nearest-representative
projection are used by this model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from full_hmm_belief_representatives import (
    BELIEF_ROUND_DIGITS,
    PROB_EPS,
    belief_key,
    belief_mse,
    build_initial_belief,
    build_transition_index,
    hidden_squared_errors,
    monotone_age_thresholds,
    move_estimate,
    predict_belief,
    sparse_belief,
    load_hmm_model,
    RawRecord,
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
    Reproduce the current HMM threshold semantics:
        tau_k = median HMM-MSE at prediction age k, k=1..10,
    made monotonically nondecreasing.

    This trajectory horizon is ONLY for threshold calibration.
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


def build_exact_reachable_contexts(
    model: dict,
    thresholds: Sequence[float],
    target: Tuple[int, int],
):
    """
    Exact union-reachability of HMM-belief contexts.

    A context is:
        (xhat, yhat, exact beta)

    From uncertainty levels 0..9 a movement successor is retained because some
    URC choice c can still skip localization. At level 10 every c in 1..10
    localizes, so no further movement successor is needed.

    All post-localization estimate positions are seeded with beta=pi. This
    supports the single perfect-localization PRISM update:
        xhat'=x, yhat'=y, hmm_state'=0, substate'=0.
    """
    n = int(model["n"])
    transition_index, policy = build_transition_index(model)
    pi = build_initial_belief(model)
    squared_errors = hidden_squared_errors(model)
    tau10 = float(thresholds[-1])

    beliefs: List[np.ndarray] = []
    belief_uid_by_key: Dict[bytes, int] = {}

    def register_belief(beta: np.ndarray) -> int:
        key = belief_key(beta)

        if key not in belief_uid_by_key:
            belief_uid_by_key[key] = len(beliefs)
            beliefs.append(beta.copy())

        return belief_uid_by_key[key]

    pi_uid = register_belief(pi)

    contexts = []
    context_id_by_key = {}
    queue = deque()

    def register_context(xhat: int, yhat: int, beta: np.ndarray) -> int:
        uid = register_belief(beta)
        key = (int(xhat), int(yhat), int(uid))

        if key not in context_id_by_key:
            context_id = len(contexts)
            context_id_by_key[key] = context_id
            mse = float(np.dot(beta, squared_errors))

            contexts.append({
                "context_id": context_id,
                "xhat": int(xhat),
                "yhat": int(yhat),
                "belief_uid": int(uid),
                "mse": mse,
                "urc_level": uncertainty_level(mse, thresholds),
            })
            queue.append((int(xhat), int(yhat), int(uid)))

        return context_id_by_key[key]

    reset_contexts = {}

    for xhat in range(n + 1):
        for yhat in range(n + 1):
            context_id = register_context(
                xhat,
                yhat,
                pi,
            )
            reset_contexts[f"{xhat},{yhat}"] = context_id

    transitions = {}

    while queue:
        xhat, yhat, uid = queue.popleft()
        context_id = context_id_by_key[(xhat, yhat, uid)]
        beta = beliefs[uid]

        mse = float(np.dot(beta, squared_errors))
        level = uncertainty_level(mse, thresholds)

        # Every controller threshold localizes at level 10.
        if level >= 10 or mse + 1e-15 >= tau10:
            continue

        # No further Knowledge movement required at the target.
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

        next_context = register_context(
            next_xhat,
            next_yhat,
            next_beta,
        )

        transitions[str(context_id)] = {
            "action": action,
            "next_context": int(next_context),
        }

    reachable_hidden_states = sorted({
        int(hidden_state)
        for beta in beliefs
        for hidden_state in np.flatnonzero(beta > PROB_EPS)
    })

    # Beliefs are diagnostics only; PRISM never stores them directly.
    belief_payload = []
    for uid, beta in enumerate(beliefs):
        mse = float(np.dot(beta, squared_errors))
        belief_payload.append({
            "belief_uid": int(uid),
            "mse": mse,
            "urc_level": uncertainty_level(mse, thresholds),
            "support_size": int(np.count_nonzero(beta > PROB_EPS)),
            "belief": sparse_belief(beta),
        })

    return {
        "contexts": contexts,
        "transitions": transitions,
        "beliefs": belief_payload,
        "belief_arrays": beliefs,
        "reset_contexts": reset_contexts,
        "reset_belief_uid": int(pi_uid),
        "reachable_hidden_states": reachable_hidden_states,
        "exact_context_count": len(contexts),
        "exact_belief_count": len(beliefs),
    }


def _canonical_partition(labels):
    blocks = defaultdict(list)

    for state, label in enumerate(labels):
        blocks[label].append(state)

    return sorted(tuple(v) for v in blocks.values())


def _labels_for(signatures):
    mapping = {}
    result = []

    for signature in signatures:
        if signature not in mapping:
            mapping[signature] = len(mapping)

        result.append(mapping[signature])

    return result


def minimize_behaviorally(
    exact: dict,
    thresholds: Sequence[float],
    map_size: int,
):
    """
    Deterministic behavioral quotient.

    Initial signature:
        (xhat, yhat, HMM-MSE threshold level, MAPE action/terminal)

    Refinement:
        signature_i+1 =
            (initial_signature, class_i(successor))

    until the partition is stable.
    """
    contexts = exact["contexts"]
    transitions = {
        int(source): {
            "action": item["action"],
            "next_context": int(item["next_context"]),
        }
        for source, item in exact["transitions"].items()
    }

    by_id = {
        int(context["context_id"]): context
        for context in contexts
    }

    state_count = len(contexts)

    if set(by_id) != set(range(state_count)):
        raise ValueError(
            "Exact HMM context IDs must be contiguous."
        )

    initial_signatures = []

    for state in range(state_count):
        context = by_id[state]
        transition = transitions.get(state)

        initial_signatures.append((
            int(context["xhat"]),
            int(context["yhat"]),
            int(context["urc_level"]),
            transition["action"] if transition is not None else None,
        ))

    labels = _labels_for(initial_signatures)
    initial_class_count = len(set(labels))
    iterations = 0

    while True:
        signatures = []

        for state in range(state_count):
            transition = transitions.get(state)

            successor_class = (
                labels[transition["next_context"]]
                if transition is not None
                else None
            )

            signatures.append((
                initial_signatures[state],
                successor_class,
            ))

        new_labels = _labels_for(signatures)
        iterations += 1

        if _canonical_partition(new_labels) == _canonical_partition(labels):
            labels = new_labels
            break

        labels = new_labels

    members_by_block = defaultdict(list)

    for exact_state, block in enumerate(labels):
        members_by_block[block].append(exact_state)

    # Put all certainty/reset classes first in stable row-major order.
    reset_blocks = {}

    for x in range(map_size):
        for y in range(map_size):
            exact_context = int(
                exact["reset_contexts"][f"{x},{y}"]
            )
            reset_blocks[(x, y)] = labels[exact_context]

    if len(set(reset_blocks.values())) != map_size * map_size:
        raise AssertionError(
            "Reset HMM contexts of different positions were merged."
        )

    ordered_blocks = []
    seen = set()

    for x in range(map_size):
        for y in range(map_size):
            block = reset_blocks[(x, y)]

            if block not in seen:
                ordered_blocks.append(block)
                seen.add(block)

    for block in sorted(members_by_block):
        if block not in seen:
            ordered_blocks.append(block)
            seen.add(block)

    class_id_of_block = {
        block: class_id
        for class_id, block in enumerate(ordered_blocks)
    }

    exact_to_class = [
        class_id_of_block[labels[state]]
        for state in range(state_count)
    ]

    class_members = defaultdict(list)

    for exact_state, class_id in enumerate(exact_to_class):
        class_members[class_id].append(exact_state)

    quotient_contexts = []
    quotient_transitions = {}

    for class_id in range(len(class_members)):
        members = class_members[class_id]
        representative = members[0]

        positions = {
            (
                int(by_id[s]["xhat"]),
                int(by_id[s]["yhat"]),
            )
            for s in members
        }

        levels = {
            int(by_id[s]["urc_level"])
            for s in members
        }

        actions = {
            transitions[s]["action"]
            if s in transitions
            else None
            for s in members
        }

        successors = {
            exact_to_class[transitions[s]["next_context"]]
            if s in transitions
            else None
            for s in members
        }

        if len(positions) != 1:
            raise AssertionError(
                f"Behavioral class {class_id} mixes positions."
            )

        if len(levels) != 1:
            raise AssertionError(
                f"Behavioral class {class_id} mixes HMM levels."
            )

        if len(actions) != 1:
            raise AssertionError(
                f"Behavioral class {class_id} mixes actions."
            )

        if len(successors) != 1:
            raise AssertionError(
                f"Behavioral class {class_id} mixes successor classes."
            )

        xhat, yhat = next(iter(positions))
        level = next(iter(levels))
        action = next(iter(actions))
        successor = next(iter(successors))

        quotient_contexts.append({
            "class_id": int(class_id),
            "xhat": int(xhat),
            "yhat": int(yhat),
            "hmm_state": int(level),
            "urc_level": int(level),
            "representative_mse": float(
                by_id[representative]["mse"]
            ),
            "representative_exact_context": int(representative),
            "member_count": int(len(members)),
        })

        if action is not None:
            quotient_transitions[str(class_id)] = {
                "action": action,
                "next_context": int(successor),
            }

    by_class = {
        int(item["class_id"]): item
        for item in quotient_contexts
    }

    reset_class = {}

    for x in range(map_size):
        for y in range(map_size):
            exact_context = int(
                exact["reset_contexts"][f"{x},{y}"]
            )
            class_id = exact_to_class[exact_context]
            reset_class[(x, y)] = class_id

            if int(by_class[class_id]["hmm_state"]) != 0:
                raise AssertionError(
                    f"Reset HMM context at ({x},{y}) is not level 0. "
                    "This normally means tau_1 <= 0."
                )

    # Local substate IDs within one (xhat,yhat,hmm_state).
    grouped = defaultdict(list)

    for item in quotient_contexts:
        grouped[(
            int(item["xhat"]),
            int(item["yhat"]),
            int(item["hmm_state"]),
        )].append(int(item["class_id"]))

    max_substate = 0
    structured_to_class = {}

    for key, class_ids in grouped.items():
        xhat, yhat, level = key
        class_ids = sorted(class_ids)
        ordered = []

        certainty = reset_class.get((xhat, yhat))

        if level == 0 and certainty in class_ids:
            ordered.append(certainty)

        ordered.extend(
            class_id
            for class_id in class_ids
            if class_id not in ordered
        )

        for substate, class_id in enumerate(ordered):
            by_class[class_id]["substate"] = int(substate)
            max_substate = max(max_substate, substate)

            structured_key = f"{xhat},{yhat},{level},{substate}"

            if structured_key in structured_to_class:
                raise AssertionError(
                    f"Duplicate structured HMM context {structured_key}."
                )

            structured_to_class[structured_key] = int(class_id)

    # Enrich transitions with source/target tuple used directly by PRISM.
    for source, transition in quotient_transitions.items():
        src = by_class[int(source)]
        dst = by_class[int(transition["next_context"])]

        transition["source"] = {
            "xhat": int(src["xhat"]),
            "yhat": int(src["yhat"]),
            "hmm_state": int(src["hmm_state"]),
            "substate": int(src["substate"]),
        }

        transition["target"] = {
            "xhat": int(dst["xhat"]),
            "yhat": int(dst["yhat"]),
            "hmm_state": int(dst["hmm_state"]),
            "substate": int(dst["substate"]),
        }

    reset_contexts = {}

    for x in range(map_size):
        for y in range(map_size):
            class_id = reset_class[(x, y)]
            item = by_class[class_id]

            if (
                int(item["hmm_state"]) != 0
                or int(item["substate"]) != 0
            ):
                raise AssertionError(
                    f"Reset HMM context at ({x},{y}) must be "
                    "(hmm_state=0, substate=0)."
                )

            reset_contexts[f"{x},{y}"] = {
                "class_id": int(class_id),
                "xhat": int(x),
                "yhat": int(y),
                "hmm_state": 0,
                "substate": 0,
            }

    return {
        "contexts": quotient_contexts,
        "transitions": quotient_transitions,
        "reset_contexts": reset_contexts,
        "structured_to_class": structured_to_class,
        "class_members": {
            str(class_id): [int(v) for v in members]
            for class_id, members in class_members.items()
        },
        "exact_to_class": [int(v) for v in exact_to_class],
        "max_substate": int(max_substate),
        "behavioral_class_count": len(quotient_contexts),
        "minimization": {
            "initial_class_count": int(initial_class_count),
            "final_class_count": int(len(quotient_contexts)),
            "refinement_iterations": int(iterations),
            "reduction_absolute": int(
                state_count - len(quotient_contexts)
            ),
            "reduction_fraction": (
                float(
                    (state_count - len(quotient_contexts))
                    / state_count
                )
                if state_count
                else 0.0
            ),
        },
    }


def build_behavioral_hmm_model(
    model: dict,
    target: Tuple[int, int],
    max_steps: int = DEFAULT_MAX_STEPS,
):
    thresholds = calibrate_thresholds(
        model=model,
        max_steps=max_steps,
    )

    exact = build_exact_reachable_contexts(
        model=model,
        thresholds=thresholds,
        target=target,
    )

    quotient = minimize_behaviorally(
        exact=exact,
        thresholds=thresholds,
        map_size=int(model["n"]) + 1,
    )

    payload = {
        "mode": "behavioral_structured_hmm_beliefs",
        "representation":
            "xhat_yhat_hmm_state_substate_behavioral_quotient",
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
            len(exact["reachable_hidden_states"])
        ),
        "reachable_hidden_states": [
            int(v)
            for v in exact["reachable_hidden_states"]
        ],
        "exact_belief_count": int(exact["exact_belief_count"]),
        "exact_context_count": int(exact["exact_context_count"]),
        "behavioral_class_count": int(
            quotient["behavioral_class_count"]
        ),
        "max_substate": int(quotient["max_substate"]),
        "contexts": quotient["contexts"],
        "belief_transitions": quotient["transitions"],
        "reset_contexts": quotient["reset_contexts"],
        "structured_to_class": quotient["structured_to_class"],
        "class_members": quotient["class_members"],
        "exact_to_class": quotient["exact_to_class"],
        "minimization": quotient["minimization"],
        # Exact beliefs remain available only as offline diagnostics.
        "exact_beliefs": exact["beliefs"],
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

    return payload


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

    result = build_behavioral_hmm_model(
        model=model,
        target=target,
        max_steps=max_steps,
    )

    map_dir = output_dir / f"map_{map_id}"
    map_dir.mkdir(parents=True, exist_ok=True)

    output_path = map_dir / "behavioral_structured.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    reduction = 100.0 * result["minimization"]["reduction_fraction"]

    print(
        f"map {map_id}: "
        f"hidden {result['reachable_hidden_state_count']}/"
        f"{result['hidden_state_count_full']}, "
        f"exact beliefs={result['exact_belief_count']}, "
        f"exact contexts={result['exact_context_count']}, "
        f"behavioral classes={result['behavioral_class_count']}, "
        f"reduction={reduction:.2f}%, "
        f"max_substate={result['max_substate']}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build exact reachable HMM beliefs, minimize them behaviorally, "
            "and write structured xhat/yhat/hmm_state/substate models."
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

    for map_id in range(args.first_map, args.last_map + 1):
        process_map(
            map_id=map_id,
            hmm_dir=args.hmm_dir,
            output_dir=args.output_dir,
            target=target,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
