"""
Exact reachable positional-belief model for PARLEY with position-local state IDs.

PRISM representation:
    (xhat, yhat, belief_state)

Every exact reachable knowledge context is retained. There is:
- no clustering,
- no medoid selection,
- no nearest-neighbour projection,
- no interpolation,
- no K parameter.

belief_state is LOCAL to (xhat, yhat):
    belief_state=0 always denotes the certainty belief at the current estimate.
Different positions may reuse the same local numeric belief_state ID.

The model therefore represents all exact reachable contexts while keeping the
PRISM variable range equal to the maximum number of local contexts at any one
estimated position, rather than the total number of contexts across the map.
"""

from collections import defaultdict, deque

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
    Build the exact finite reachable-belief automaton.

    Threshold calibration still uses ages 1..max_steps exactly as before.
    Reachability enumeration itself is NOT truncated at max_steps: starting
    from every possible perfect-localisation state, exact beliefs are expanded
    until the highest threshold tau_10 is reached or the MAPE path terminates.

    PRISM state:
        (xhat, yhat, belief_state)

    belief_state is a local index inside each estimated position.
    """
    size = len(map_data)
    n = size - 1

    # Same map-specific threshold calibration as the previous belief model.
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
    certain_key = _vector_key(certain)

    # Offline diagnostic table of distinct relative beliefs.
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

    # Per-position exact contexts.
    #
    # position_vectors[(xhat,yhat)] is ordered by local belief_state ID.
    # ID 0 is reserved for certainty at EVERY position.
    position_vectors = defaultdict(list)
    local_id_by_position_and_key = {}

    queue = deque()

    def add_context(xhat, yhat, vector):
        pos = (xhat, yhat)
        vkey = _vector_key(vector)
        key = (xhat, yhat, vkey)

        if key in local_id_by_position_and_key:
            return local_id_by_position_and_key[key], False

        local_id = len(position_vectors[pos])
        local_id_by_position_and_key[key] = local_id
        position_vectors[pos].append(vector)
        get_belief_id(vector)
        queue.append((xhat, yhat, vector))

        return local_id, True

    # Insert certainty first at every position, guaranteeing:
    #   certainty -> belief_state = 0
    for xhat in range(size):
        for yhat in range(size):
            local_id, _ = add_context(
                xhat,
                yhat,
                certain,
            )
            if local_id != 0:
                raise AssertionError(
                    f"Certainty at ({xhat},{yhat}) must have local state 0."
                )

    transitions = {}

    while queue:
        xhat, yhat, vector = queue.popleft()
        state_id = local_id_by_position_and_key[
            (xhat, yhat, _vector_key(vector))
        ]

        uncertainty = _scaled_gini(vector)

        # Once tau_10 is reached, all c in 1..10 require an update.
        # Therefore no skip-successor needs to be represented.
        if uncertainty >= tau10:
            continue

        action = belief_impl._direction(
            controller,
            xhat,
            yhat,
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

        next_state, _ = add_context(
            nxhat,
            nyhat,
            successor,
        )

        transitions[f"{xhat},{yhat},{state_id}"] = {
            "action": action,
            "next_xhat": nxhat,
            "next_yhat": nyhat,
            "next_state": next_state,
        }

    contexts = []
    uncertainties = {}
    local_state_counts = {}

    for xhat in range(size):
        for yhat in range(size):
            pos = (xhat, yhat)
            vectors_here = position_vectors[pos]
            local_state_counts[f"{xhat},{yhat}"] = len(vectors_here)

            uncertainties[f"{xhat},{yhat}"] = [
                _scaled_gini(vector)
                for vector in vectors_here
            ]

            for state_id, vector in enumerate(vectors_here):
                contexts.append({
                    "xhat": xhat,
                    "yhat": yhat,
                    "belief_state": state_id,
                    "belief_id": get_belief_id(vector),
                    "uncertainty": _scaled_gini(vector),
                })

    max_local_states = max(
        local_state_counts.values(),
        default=1,
    )

    return {
        "map_id": map_id,
        "thresholds": thresholds,

        # Number of reachable (xhat,yhat,belief_state) contexts.
        "context_count": len(contexts),

        # Number of distinct relative belief vectors, diagnostic only.
        "belief_count": len(vectors),

        # PRISM uses [0..max_local_states-1].
        "max_local_states": max_local_states,
        "max_belief_state": max_local_states - 1,

        "local_state_counts": local_state_counts,
        "uncertainties": uncertainties,
        "transitions": transitions,
        "contexts": contexts,
        "vectors": vectors,

        "certainty_local_state": 0,
        "representation": "position_local_exact_reachable_beliefs",
    }
