"""
Behavioral partition refinement for the exact reachable PARLEY belief model.

Two exact contexts may be merged only if they have:
- the same estimated position (xhat, yhat),
- the same URC uncertainty stage 0..10,
- the same enabled MAPE action / terminal status,
- and, at the fixpoint, successors in the same behavioral class.

This is a deterministic quotient construction, not distance-based clustering.
"""

def uncertainty_stage(uncertainty, thresholds):
    stage = 0
    for index, threshold in enumerate(thresholds, start=1):
        if uncertainty >= threshold:
            stage = index
        else:
            break
    return stage


def _normalize_transitions(model):
    return {
        int(source): {
            "action": tr["action"],
            "next_context": int(tr["next_context"]),
        }
        for source, tr in model["transitions"].items()
    }


def _canonical_partition(labels):
    blocks = {}
    for state, label in enumerate(labels):
        blocks.setdefault(label, []).append(state)
    return sorted(tuple(members) for members in blocks.values())


def minimize_belief_model(exact_model, map_size):
    thresholds = [int(v) for v in exact_model["thresholds"]]
    contexts = exact_model["contexts"]
    transitions = _normalize_transitions(exact_model)
    n_states = len(contexts)

    by_id = {int(ctx["kstate"]): ctx for ctx in contexts}
    if set(by_id) != set(range(n_states)):
        raise ValueError("Exact kstate IDs must be contiguous.")

    stages = [
        uncertainty_stage(
            int(by_id[state]["uncertainty"]),
            thresholds,
        )
        for state in range(n_states)
    ]

    base_signatures = []
    for state in range(n_states):
        ctx = by_id[state]
        tr = transitions.get(state)
        action = tr["action"] if tr is not None else None
        base_signatures.append((
            int(ctx["xhat"]),
            int(ctx["yhat"]),
            int(stages[state]),
            action,
        ))

    def labels_for(signatures):
        mapping = {}
        result = []
        for signature in signatures:
            if signature not in mapping:
                mapping[signature] = len(mapping)
            result.append(mapping[signature])
        return result

    labels = labels_for(base_signatures)
    initial_class_count = len(set(labels))
    refinement_iterations = 0

    while True:
        refined_signatures = []

        for state in range(n_states):
            tr = transitions.get(state)
            successor_block = (
                labels[tr["next_context"]]
                if tr is not None
                else None
            )
            refined_signatures.append((
                base_signatures[state],
                successor_block,
            ))

        new_labels = labels_for(refined_signatures)
        refinement_iterations += 1

        if _canonical_partition(new_labels) == _canonical_partition(labels):
            labels = new_labels
            break

        labels = new_labels

    members_by_block = {}
    for state, block in enumerate(labels):
        members_by_block.setdefault(block, []).append(state)

    # Certainty classes are kept distinct by position because xhat/yhat are
    # part of every signature.
    certainty_block = {}
    for x in range(map_size):
        for y in range(map_size):
            exact_certainty = int(
                exact_model["certainty_contexts"][f"{x},{y}"]
            )
            certainty_block[(x, y)] = labels[exact_certainty]

    ordered_certainty_blocks = [
        certainty_block[(x, y)]
        for x in range(map_size)
        for y in range(map_size)
    ]

    if len(set(ordered_certainty_blocks)) != map_size * map_size:
        raise AssertionError(
            "Certainty contexts of different positions were merged."
        )

    # Renumber certainty classes first, preserving the existing single-update
    # assignment kstate' = x*(N+1)+y.
    ordered_blocks = []
    seen = set()

    for block in ordered_certainty_blocks:
        if block not in seen:
            ordered_blocks.append(block)
            seen.add(block)

    for block in sorted(members_by_block):
        if block not in seen:
            ordered_blocks.append(block)
            seen.add(block)

    new_id_of_block = {
        block: new_id
        for new_id, block in enumerate(ordered_blocks)
    }

    exact_to_class = [
        new_id_of_block[labels[state]]
        for state in range(n_states)
    ]

    class_members = {}
    for exact_state, class_id in enumerate(exact_to_class):
        class_members.setdefault(class_id, []).append(exact_state)

    quotient_contexts = []
    quotient_stages = []
    quotient_uncertainties = []
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
        member_stages = {stages[s] for s in members}
        actions = {
            transitions[s]["action"] if s in transitions else None
            for s in members
        }
        successor_classes = {
            exact_to_class[transitions[s]["next_context"]]
            if s in transitions else None
            for s in members
        }

        if len(positions) != 1:
            raise AssertionError(f"Class {class_id} mixes positions.")
        if len(member_stages) != 1:
            raise AssertionError(f"Class {class_id} mixes URC stages.")
        if len(actions) != 1:
            raise AssertionError(f"Class {class_id} mixes MAPE actions.")
        if len(successor_classes) != 1:
            raise AssertionError(f"Class {class_id} mixes successor classes.")

        xhat, yhat = next(iter(positions))
        stage = next(iter(member_stages))
        action = next(iter(actions))
        successor_class = next(iter(successor_classes))

        quotient_contexts.append({
            "kstate": class_id,
            "xhat": xhat,
            "yhat": yhat,
            "stage": stage,
            "uncertainty": int(by_id[representative]["uncertainty"]),
            "member_count": len(members),
            "representative_exact_kstate": representative,
        })
        quotient_stages.append(stage)
        quotient_uncertainties.append(
            int(by_id[representative]["uncertainty"])
        )

        if action is not None:
            quotient_transitions[str(class_id)] = {
                "action": action,
                "next_context": int(successor_class),
            }

    position_contexts = {
        f"{x},{y}": []
        for x in range(map_size)
        for y in range(map_size)
    }
    for ctx in quotient_contexts:
        position_contexts[
            f"{ctx['xhat']},{ctx['yhat']}"
        ].append(ctx["kstate"])

    certainty_contexts = {}
    for x in range(map_size):
        for y in range(map_size):
            class_id = new_id_of_block[
                certainty_block[(x, y)]
            ]
            expected = x * map_size + y
            if class_id != expected:
                raise AssertionError(
                    f"Certainty class ({x},{y})={class_id}, expected {expected}."
                )
            certainty_contexts[f"{x},{y}"] = class_id

    return {
        "map_id": exact_model.get("map_id"),
        "state_count": len(quotient_contexts),
        "context_count": len(quotient_contexts),
        "exact_state_count": n_states,
        "belief_count": exact_model.get("belief_count"),
        "thresholds": thresholds,
        "stages": quotient_stages,
        "uncertainties": quotient_uncertainties,
        "transitions": quotient_transitions,
        "contexts": quotient_contexts,
        "position_contexts": position_contexts,
        "certainty_contexts": certainty_contexts,
        "exact_to_class": exact_to_class,
        "class_members": class_members,
        "vectors": exact_model.get("vectors"),
        "representation": "behavioral_quotient_exact_belief",
        "minimization": {
            "initial_class_count": initial_class_count,
            "final_class_count": len(quotient_contexts),
            "refinement_iterations": refinement_iterations,
            "reduction_absolute": n_states - len(quotient_contexts),
            "reduction_fraction": (
                (n_states - len(quotient_contexts)) / n_states
                if n_states else 0.0
            ),
        },
    }
