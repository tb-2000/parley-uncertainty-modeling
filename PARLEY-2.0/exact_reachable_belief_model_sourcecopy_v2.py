"""
Compact exact reachable positional-belief model for PARLEY.

This replaces the former representation
    (xhat, yhat, belief_state)
by one compact PRISM state variable
    kstate

Each kstate is exactly one reachable knowledge context
    (xhat, yhat, exact relative belief).

There is no clustering, medoid selection, nearest-neighbour projection,
or K parameter.

The ten map-specific Gini thresholds are derived exactly as before.
"""

from collections import deque

import full_belief_representatives as belief_impl


ROUND_DIGITS = 14


def _vector_key(vector):
    return tuple(round(float(v), ROUND_DIGITS) for v in vector)


def _scaled_gini(vector):
    return int(round(
        belief_impl._representative_gini(vector) * 10000
    ))


def _absolute_from_relative(vector, xhat, yhat, n):
    relative = belief_impl._vector_to_relative(vector, n)
    absolute = {}

    for (dx, dy), probability in relative.items():
        if probability <= 0.0:
            continue

        x = xhat + dx
        y = yhat + dy

        if not (0 <= x <= n and 0 <= y <= n):
            raise ValueError(
                "Exact reachable belief contains out-of-grid support: "
                f"xhat={xhat}, yhat={yhat}, dx={dx}, dy={dy}"
            )

        absolute[(x, y)] = (
            absolute.get((x, y), 0.0) + probability
        )

    return absolute


def _exact_successor(vector, xhat, yhat, action, n, p):
    absolute = _absolute_from_relative(
        vector, xhat, yhat, n
    )

    propagated = belief_impl._propagate_absolute(
        absolute, action, n, p
    )

    nxhat, nyhat = belief_impl._move(
        xhat, yhat, action, n
    )

    successor = belief_impl._relative_vector(
        propagated, nxhat, nyhat, n
    )

    return successor, nxhat, nyhat


def build_exact_belief_model(
    map_id,
    map_data,
    target,
    p=0.01,
    max_steps=10,
):
    """
    Build one compact exact finite knowledge automaton.

    One PRISM kstate corresponds to one reachable tuple
        (xhat, yhat, relative_belief).

    Perfect localization maps physical position (x,y) to the certainty
    kstate belonging to exactly that position.
    """
    size = len(map_data)
    n = size - 1

    # SAME threshold derivation as the previous exact/K=100 model.
    _, gini_by_age, controller = belief_impl._generate_records(
        map_data,
        target,
        p,
        max_steps,
    )
    thresholds = belief_impl._thresholds(
        gini_by_age,
        max_steps,
    )
    tau10 = int(thresholds[-1])

    certain = belief_impl._relative_vector(
        {(0, 0): 1.0},
        0,
        0,
        n,
    )

    # Keep distinct relative beliefs offline for diagnostics only.
    vectors = []
    belief_id_by_key = {}

    def get_belief_id(vector):
        key = _vector_key(vector)
        if key not in belief_id_by_key:
            belief_id_by_key[key] = len(vectors)
            vectors.append(vector)
        return belief_id_by_key[key]

    certainty_belief_id = get_belief_id(certain)
    if certainty_belief_id != 0:
        raise AssertionError("Certainty must have offline belief ID 0.")

    contexts = []
    context_id_by_key = {}
    queue = deque()

    def get_context_id(xhat, yhat, vector):
        vkey = _vector_key(vector)
        key = (xhat, yhat, vkey)

        if key not in context_id_by_key:
            context_id = len(contexts)
            context_id_by_key[key] = context_id

            belief_id = get_belief_id(vector)
            contexts.append({
                "kstate": context_id,
                "xhat": xhat,
                "yhat": yhat,
                "belief_id": belief_id,
                "vector": vector,
                "uncertainty": _scaled_gini(vector),
            })
            queue.append((xhat, yhat, vector))

        return context_id_by_key[key]

    # Every physical position can be the result of perfect localization.
    # IMPORTANT invariant:
    #   certainty_kstate(x,y) = x * size + y
    # because these certainty contexts are inserted first in row-major order.
    # This lets the PRISM model use one compact update assignment:
    #   kstate' = x*(N+1)+y
    certainty_contexts = {}
    for xhat in range(size):
        for yhat in range(size):
            context_id = get_context_id(
                xhat, yhat, certain
            )
            expected_id = xhat * size + yhat
            if context_id != expected_id:
                raise AssertionError(
                    "Certainty-context numbering invariant violated: "
                    f"({xhat},{yhat}) -> {context_id}, expected {expected_id}"
                )
            certainty_contexts[f"{xhat},{yhat}"] = context_id

    transitions = {}

    while queue:
        xhat, yhat, vector = queue.popleft()
        context_id = context_id_by_key[
            (xhat, yhat, _vector_key(vector))
        ]
        uncertainty = _scaled_gini(vector)

        # At tau_10 every c=1..10 must update, so no skip successor exists.
        if uncertainty >= tau10:
            continue

        action = belief_impl._direction(
            controller, xhat, yhat
        )
        if action is None:
            continue

        successor, nxhat, nyhat = _exact_successor(
            vector,
            xhat,
            yhat,
            action,
            n,
            p,
        )

        next_context = get_context_id(
            nxhat,
            nyhat,
            successor,
        )

        transitions[str(context_id)] = {
            "action": action,
            "next_context": next_context,
            "next_xhat": nxhat,
            "next_yhat": nyhat,
        }

    # Position -> all compact context IDs whose estimate is that position.
    position_contexts = {
        f"{x},{y}": []
        for x in range(size)
        for y in range(size)
    }

    for context in contexts:
        position_contexts[
            f"{context['xhat']},{context['yhat']}"
        ].append(context["kstate"])

    uncertainties = [
        int(context["uncertainty"])
        for context in contexts
    ]

    # Remove large vectors from the compact contexts table; vectors remain once
    # in the offline diagnostics table.
    compact_contexts = [
        {
            "kstate": context["kstate"],
            "xhat": context["xhat"],
            "yhat": context["yhat"],
            "belief_id": context["belief_id"],
            "uncertainty": context["uncertainty"],
        }
        for context in contexts
    ]

    return {
        "map_id": map_id,
        # PRISM state count: one ID per reachable knowledge context.
        "state_count": len(compact_contexts),
        "context_count": len(compact_contexts),
        # Number of distinct relative beliefs is diagnostic only.
        "belief_count": len(vectors),
        "thresholds": thresholds,
        "uncertainties": uncertainties,
        "transitions": transitions,
        "contexts": compact_contexts,
        "position_contexts": position_contexts,
        "certainty_contexts": certainty_contexts,
        "vectors": vectors,
        "representation": "compact_exact_reachable_contexts",
    }
