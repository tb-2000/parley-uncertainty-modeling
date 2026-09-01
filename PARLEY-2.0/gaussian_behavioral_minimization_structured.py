"""
Behavioral minimization of the exact reachable Gaussian Knowledge automaton.

The exact Gaussian model provides reachable contexts

    (xhat, yhat, Gaussian moment state)

where each moment state is

    (bias_x, bias_y, var_x, var_y, cov_xy).

This module does NOT cluster Gaussian moments geometrically and does NOT use K,
medoids, W2, nearest-neighbour projection, etc.

Instead it computes behavioral equivalence by deterministic partition
refinement.

Initial observation signature:
    (xhat, yhat, gaussian_state, MAPE action)

where gaussian_state is the MSE/URC threshold stage 0..10.

Refinement:
    class_{i+1}(s) =
        (initial_signature(s), class_i(successor(s)))

until a fixed point is reached.

For PRISM, every quotient class is represented structurally as

    (xhat, yhat, gaussian_state, substate)

where substate only distinguishes behaviorally different classes that share
the same position and MSE/URC stage.

Perfect-localization contexts always receive
    gaussian_state = 0
    substate = 0.
"""


def uncertainty_stage(raw_uncertainty, thresholds):
    """Return the highest threshold level reached, in 0..10."""
    stage = 0
    for level, threshold in enumerate(thresholds, start=1):
        if raw_uncertainty >= threshold:
            stage = level
        else:
            break
    return stage


def _normalise_transitions(model):
    return {
        int(source): {
            "action": transition["action"],
            "next_context": int(transition["next_context"]),
        }
        for source, transition in model["transitions"].items()
    }


def _canonical_partition(labels):
    blocks = {}
    for state, label in enumerate(labels):
        blocks.setdefault(label, []).append(state)
    return sorted(tuple(members) for members in blocks.values())


def minimize_gaussian_model(exact_model, map_size):
    thresholds = [int(v) for v in exact_model["thresholds"]]
    contexts = exact_model["contexts"]
    transitions = _normalise_transitions(exact_model)
    number_of_states = len(contexts)

    by_id = {
        int(context["gstate"]): context
        for context in contexts
    }

    if set(by_id) != set(range(number_of_states)):
        raise ValueError(
            "Exact Gaussian gstate IDs must be contiguous 0..state_count-1."
        )

    stages = [
        uncertainty_stage(
            int(by_id[state]["uncertainty"]),
            thresholds,
        )
        for state in range(number_of_states)
    ]

    # ------------------------------------------------------------------
    # Initial partition:
    #   same estimated position
    #   same MSE/URC threshold stage
    #   same MAPE action / terminal status
    # ------------------------------------------------------------------
    initial_signatures = []

    for state in range(number_of_states):
        context = by_id[state]
        transition = transitions.get(state)
        action = (
            transition["action"]
            if transition is not None
            else None
        )

        initial_signatures.append((
            int(context["xhat"]),
            int(context["yhat"]),
            int(stages[state]),
            action,
        ))

    def labels_for(signatures):
        mapping = {}
        labels = []

        for signature in signatures:
            if signature not in mapping:
                mapping[signature] = len(mapping)
            labels.append(mapping[signature])

        return labels

    labels = labels_for(initial_signatures)
    initial_class_count = len(set(labels))
    refinement_iterations = 0

    # ------------------------------------------------------------------
    # Deterministic partition refinement.
    # The Gaussian Knowledge successor is deterministic once the commanded
    # MAPE action is fixed, so equality of successor blocks is sufficient.
    # ------------------------------------------------------------------
    while True:
        refined_signatures = []

        for state in range(number_of_states):
            transition = transitions.get(state)

            successor_block = (
                labels[transition["next_context"]]
                if transition is not None
                else None
            )

            refined_signatures.append((
                initial_signatures[state],
                successor_block,
            ))

        new_labels = labels_for(refined_signatures)
        refinement_iterations += 1

        if _canonical_partition(new_labels) == _canonical_partition(labels):
            labels = new_labels
            break

        labels = new_labels

    # ------------------------------------------------------------------
    # Quotient blocks.
    # ------------------------------------------------------------------
    members_by_block = {}
    for state, block in enumerate(labels):
        members_by_block.setdefault(block, []).append(state)

    # Perfect-localization Gaussian zero contexts must remain position-specific.
    zero_block = {}

    for x in range(map_size):
        for y in range(map_size):
            exact_zero_context = int(
                exact_model["zero_contexts"][f"{x},{y}"]
            )
            zero_block[(x, y)] = labels[exact_zero_context]

    if len(set(zero_block.values())) != map_size * map_size:
        raise AssertionError(
            "Zero-uncertainty contexts of different positions were merged."
        )

    # Stable global class IDs are retained only for diagnostics and generation.
    # They are NOT stored as one PRISM variable.
    ordered_blocks = []
    seen = set()

    for x in range(map_size):
        for y in range(map_size):
            block = zero_block[(x, y)]
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
        for state in range(number_of_states)
    ]

    class_members = {}
    for exact_state, class_id in enumerate(exact_to_class):
        class_members.setdefault(class_id, []).append(exact_state)

    quotient_contexts = []
    quotient_transitions = {}

    for class_id in range(len(class_members)):
        members = class_members[class_id]
        representative = members[0]

        positions = {
            (
                int(by_id[state]["xhat"]),
                int(by_id[state]["yhat"]),
            )
            for state in members
        }

        member_stages = {
            int(stages[state])
            for state in members
        }

        actions = {
            transitions[state]["action"]
            if state in transitions
            else None
            for state in members
        }

        successor_classes = {
            exact_to_class[transitions[state]["next_context"]]
            if state in transitions
            else None
            for state in members
        }

        if len(positions) != 1:
            raise AssertionError(
                f"Behavioral class {class_id} mixes positions: {positions}"
            )

        if len(member_stages) != 1:
            raise AssertionError(
                f"Behavioral class {class_id} mixes Gaussian stages."
            )

        if len(actions) != 1:
            raise AssertionError(
                f"Behavioral class {class_id} mixes MAPE actions."
            )

        if len(successor_classes) != 1:
            raise AssertionError(
                f"Behavioral class {class_id} mixes successor classes."
            )

        xhat, yhat = next(iter(positions))
        gaussian_stage = next(iter(member_stages))
        action = next(iter(actions))
        successor_class = next(iter(successor_classes))

        representative_context = by_id[representative]

        quotient_contexts.append({
            "class_id": class_id,
            "xhat": xhat,
            "yhat": yhat,
            "gaussian_state": gaussian_stage,
            "stage": gaussian_stage,
            # Raw MSE is diagnostic only. Different exact contexts in the same
            # class may have different raw MSE values while remaining in the
            # same URC threshold stage.
            "representative_uncertainty": int(
                representative_context["uncertainty"]
            ),
            "member_count": len(members),
            "representative_exact_gstate": representative,
        })

        if action is not None:
            quotient_transitions[str(class_id)] = {
                "action": action,
                "next_context": int(successor_class),
            }

    context_by_class = {
        int(context["class_id"]): context
        for context in quotient_contexts
    }

    # ------------------------------------------------------------------
    # Local substate numbering.
    #
    # Within one (xhat, yhat, gaussian_state) tuple, substate only separates
    # behaviorally different quotient classes.
    #
    # Every perfect-localization zero context is (gaussian_state=0, substate=0).
    # ------------------------------------------------------------------
    zero_class = {}

    for x in range(map_size):
        for y in range(map_size):
            exact_zero_context = int(
                exact_model["zero_contexts"][f"{x},{y}"]
            )
            class_id = exact_to_class[exact_zero_context]
            zero_class[(x, y)] = class_id

            if int(context_by_class[class_id]["gaussian_state"]) != 0:
                raise AssertionError(
                    f"Zero Gaussian context at ({x},{y}) is not stage 0."
                )

    grouped = {}
    for context in quotient_contexts:
        key = (
            int(context["xhat"]),
            int(context["yhat"]),
            int(context["gaussian_state"]),
        )
        grouped.setdefault(key, []).append(
            int(context["class_id"])
        )

    max_substate = 0
    structured_to_class = {}

    for key, class_ids in grouped.items():
        xhat, yhat, gaussian_stage = key
        class_ids = sorted(class_ids)

        ordered = []
        certainty = zero_class.get((xhat, yhat))

        if gaussian_stage == 0 and certainty in class_ids:
            ordered.append(certainty)

        ordered.extend(
            class_id
            for class_id in class_ids
            if class_id not in ordered
        )

        for substate, class_id in enumerate(ordered):
            context_by_class[class_id]["substate"] = substate
            max_substate = max(max_substate, substate)

            structured_key = (
                f"{xhat},{yhat},{gaussian_stage},{substate}"
            )

            if structured_key in structured_to_class:
                raise AssertionError(
                    f"Duplicate structured Gaussian context: {structured_key}"
                )

            structured_to_class[structured_key] = class_id

    # Enrich quotient transitions with the actual structured PRISM values.
    for source, transition in quotient_transitions.items():
        source_id = int(source)
        target_id = int(transition["next_context"])

        src = context_by_class[source_id]
        dst = context_by_class[target_id]

        transition["source"] = {
            "xhat": int(src["xhat"]),
            "yhat": int(src["yhat"]),
            "gaussian_state": int(src["gaussian_state"]),
            "substate": int(src["substate"]),
        }

        transition["target"] = {
            "xhat": int(dst["xhat"]),
            "yhat": int(dst["yhat"]),
            "gaussian_state": int(dst["gaussian_state"]),
            "substate": int(dst["substate"]),
        }

    # Structured tuple must identify quotient classes uniquely.
    structured_keys = {
        (
            int(context["xhat"]),
            int(context["yhat"]),
            int(context["gaussian_state"]),
            int(context["substate"]),
        )
        for context in quotient_contexts
    }

    if len(structured_keys) != len(quotient_contexts):
        raise AssertionError(
            "Structured Gaussian representation is not injective."
        )

    zero_contexts = {}

    for x in range(map_size):
        for y in range(map_size):
            class_id = zero_class[(x, y)]
            context = context_by_class[class_id]

            if (
                int(context["gaussian_state"]) != 0
                or int(context["substate"]) != 0
            ):
                raise AssertionError(
                    f"Zero Gaussian state at ({x},{y}) must be "
                    "(gaussian_state=0, substate=0)."
                )

            zero_contexts[f"{x},{y}"] = {
                "class_id": class_id,
                "xhat": x,
                "yhat": y,
                "gaussian_state": 0,
                "substate": 0,
            }

    position_contexts = {
        f"{x},{y}": []
        for x in range(map_size)
        for y in range(map_size)
    }

    for context in quotient_contexts:
        position_contexts[
            f"{context['xhat']},{context['yhat']}"
        ].append(int(context["class_id"]))

    return {
        "map_id": exact_model.get("map_id"),
        "state_count": len(quotient_contexts),
        "context_count": len(quotient_contexts),
        "exact_context_count": number_of_states,
        "gaussian_count": exact_model.get("gaussian_count"),
        "max_steps": exact_model.get("max_steps"),
        "p": exact_model.get("p"),
        "representation":
            "behavioral_gaussian_structured_xhat_yhat_gaussian_state_substate",
        "uncertainty_metric": exact_model.get("uncertainty_metric", "mse"),
        "mse_scale": exact_model.get("mse_scale"),
        "thresholds": thresholds,
        "stages": [
            int(context["gaussian_state"])
            for context in quotient_contexts
        ],
        "contexts": quotient_contexts,
        "transitions": quotient_transitions,
        "position_contexts": position_contexts,
        "zero_contexts": zero_contexts,
        "max_substate": max_substate,
        "structured_to_class": structured_to_class,
        "exact_to_class": exact_to_class,
        "class_members": class_members,
        # Keep the exact moment diagnostics available offline.
        "gaussian_states": exact_model.get("gaussian_states", []),
        "minimization": {
            "initial_class_count": initial_class_count,
            "final_class_count": len(quotient_contexts),
            "refinement_iterations": refinement_iterations,
            "reduction_absolute":
                number_of_states - len(quotient_contexts),
            "reduction_fraction": (
                (number_of_states - len(quotient_contexts))
                / number_of_states
                if number_of_states
                else 0.0
            ),
        },
    }
