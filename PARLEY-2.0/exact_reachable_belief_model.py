"""
Exact reachable positional-belief model for PARLEY.

This replaces the former K=100 clustering/medoid abstraction.

For a fixed map:
- derive the SAME ten map-specific Gini thresholds as before;
- enumerate exact relative beliefs reachable after a perfect localization;
- assign one integer ID to every distinct exact reachable belief;
- build deterministic context-dependent transitions
      (xhat, yhat, belief_id) -> (next_xhat, next_yhat, next_belief_id)
  for the MAPE-selected action;
- stop skip-expansion once tau_10 is reached, because then every c=1..10
  requires localization.

The resulting PRISM model stores only belief_state, not the full probability
vector. There is no clustering, medoid selection, nearest-neighbour projection,
or K parameter.
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
    Build the exact finite belief automaton for one map.

    Important:
    The ten thresholds are intentionally derived exactly as in the previous
    implementation. Only the belief-state representation changes.

    We seed certainty at every grid position because a perfect localization
    sets (xhat,yhat)=(x,y). In the PRISM robot model, physical motion can enter
    cells marked as crashed before/while the crash flag is set, so a reset
    context must not be restricted to Dijkstra 'free cells' only.
    """
    size = len(map_data)
    n = size - 1

    # SAME threshold derivation as the previous K=100 model.
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

    # Relative certainty is position-independent and is always belief_state=0.
    certain = belief_impl._relative_vector(
        {(0, 0): 1.0},
        0,
        0,
        n,
    )

    vectors = []
    id_by_key = {}

    def get_id(vector):
        key = _vector_key(vector)
        if key not in id_by_key:
            id_by_key[key] = len(vectors)
            vectors.append(vector)
        return id_by_key[key]

    certainty_id = get_id(certain)
    if certainty_id != 0:
        raise AssertionError("Certainty must have belief_state ID 0.")

    queue = deque()
    seen_contexts = set()
    transitions = {}

    # Any actual grid position can become the new estimate after [update].
    for xhat in range(size):
        for yhat in range(size):
            key = (xhat, yhat, _vector_key(certain))
            if key not in seen_contexts:
                seen_contexts.add(key)
                queue.append((xhat, yhat, certain))

    while queue:
        xhat, yhat, vector = queue.popleft()

        state_id = get_id(vector)
        uncertainty = _scaled_gini(vector)

        # If tau_10 is reached, every possible URC stage c=1..10 updates.
        # Therefore no skip/movement successor is reachable from this context.
        if uncertainty >= tau10:
            continue

        action = belief_impl._direction(
            controller, xhat, yhat
        )

        # Target or another position without a MAPE move.
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
        next_state = get_id(successor)

        transitions[f"{xhat},{yhat},{state_id}"] = {
            "action": action,
            "next_xhat": nxhat,
            "next_yhat": nyhat,
            "next_state": next_state,
        }

        next_context = (
            nxhat,
            nyhat,
            _vector_key(successor),
        )

        if next_context not in seen_contexts:
            seen_contexts.add(next_context)
            queue.append(
                (nxhat, nyhat, successor)
            )

    # Must be calculated after exploration because get_id() grows vectors.
    uncertainties = [
        _scaled_gini(vector)
        for vector in vectors
    ]

    return {
        "map_id": map_id,
        "state_count": len(vectors),
        "context_count": len(seen_contexts),
        "thresholds": thresholds,
        "uncertainties": uncertainties,
        "transitions": transitions,
        # Kept for diagnostics / optional offline analysis only.
        "vectors": vectors,
        "certainty_state": 0,
        "representation": "exact_reachable",
    }
