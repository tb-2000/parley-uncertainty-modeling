"""
Exact reachable Gaussian moment model for PARLEY with position-local IDs.

PRISM representation:
    (xhat, yhat, gaussian_state)

Each exact reachable Gaussian moment context is retained. There is:
- no K,
- no medoid selection,
- no projection,
- no behavioral merging,
- no substate.

A Gaussian moment state is:
    g = (bias_x, bias_y, var_x, var_y, cov_xy)

gaussian_state is LOCAL to (xhat, yhat):
    gaussian_state=0 always denotes ZERO_STATE (perfect localisation)
    at the current estimated position.

The complete context identity is therefore:
    (xhat, yhat, gaussian_state)
"""

from collections import defaultdict, deque

from full_gaussian_representatives_bias import (
    ZERO_STATE,
    MSE_SCALE,
    _controller,
    _direction,
    _move,
    _predict_state,
    _state_key,
    _mse,
    _thresholds,
)


def _derive_thresholds(map_data, target, p, max_steps):
    """Derive the same ten median-MSE thresholds as before."""
    map_size = len(map_data)
    n = map_size - 1
    controller = _controller(map_data, target)
    mse_by_age = defaultdict(list)

    for start_x in range(map_size):
        for start_y in range(map_size):
            if int(map_data[start_x][start_y]) > 9:
                continue

            xhat, yhat = start_x, start_y
            state = ZERO_STATE

            for age in range(max_steps + 1):
                state = _state_key(state)
                mse_by_age[age].append(_mse(state))

                if age >= max_steps or (xhat, yhat) == target:
                    break

                action = _direction(controller, xhat, yhat)
                if action is None:
                    break

                state = _predict_state(
                    state,
                    xhat,
                    yhat,
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

    thresholds = _thresholds(
        mse_by_age,
        max_steps,
        scale=MSE_SCALE,
    )

    if len(thresholds) != 10:
        raise ValueError(
            f"Expected exactly 10 Gaussian thresholds, got {len(thresholds)}."
        )

    return thresholds, controller


def build_exact_gaussian_model(
    map_id,
    map_data,
    target,
    p=0.01,
    max_steps=10,
):
    """
    Build the exact finite reachable Gaussian-moment automaton.

    Threshold calibration uses ages 1..max_steps as before.
    Reachability expansion itself continues until tau_10 is reached
    or the fixed MAPE policy terminates.

    PRISM state:
        (xhat, yhat, gaussian_state)

    gaussian_state is a local ID inside the current estimated position.
    """
    size = len(map_data)
    n = size - 1

    thresholds, controller = _derive_thresholds(
        map_data,
        target,
        p,
        max_steps,
    )
    tau10 = int(thresholds[-1])

    # Distinct moment tuples for offline diagnostics only.
    gaussian_states = []
    gaussian_id_by_key = {}

    def get_gaussian_id(state):
        key = _state_key(state)
        if key not in gaussian_id_by_key:
            gaussian_id_by_key[key] = len(gaussian_states)
            gaussian_states.append(key)
        return gaussian_id_by_key[key]

    zero_gaussian_id = get_gaussian_id(ZERO_STATE)
    if zero_gaussian_id != 0:
        raise AssertionError("ZERO_STATE must have offline Gaussian ID 0.")

    # Local state tables. ID 0 is reserved for ZERO_STATE at every position.
    position_states = defaultdict(list)
    local_id_by_context = {}
    queue = deque()

    def add_context(xhat, yhat, state):
        state = _state_key(state)
        key = (xhat, yhat, state)

        if key in local_id_by_context:
            return local_id_by_context[key], False

        pos = (xhat, yhat)
        local_id = len(position_states[pos])

        local_id_by_context[key] = local_id
        position_states[pos].append(state)
        get_gaussian_id(state)

        queue.append((xhat, yhat, state))
        return local_id, True

    # Perfect localisation at every position:
    # gaussian_state = 0.
    for xhat in range(size):
        for yhat in range(size):
            local_id, _ = add_context(
                xhat,
                yhat,
                ZERO_STATE,
            )
            if local_id != 0:
                raise AssertionError(
                    f"ZERO_STATE at ({xhat},{yhat}) must have local ID 0."
                )

    transitions = {}

    while queue:
        xhat, yhat, state = queue.popleft()
        state = _state_key(state)

        state_id = local_id_by_context[
            (xhat, yhat, state)
        ]

        scaled_mse = int(
            round(_mse(state) * MSE_SCALE)
        )

        # At tau_10 every controller setting 1..10 localises.
        if scaled_mse >= tau10:
            continue

        action = _direction(
            controller,
            xhat,
            yhat,
        )
        if action is None:
            continue

        next_state = _predict_state(
            state,
            xhat,
            yhat,
            action,
            n,
            p,
        )
        next_state = _state_key(next_state)

        next_xhat, next_yhat = _move(
            xhat,
            yhat,
            action,
            n,
        )

        next_state_id, _ = add_context(
            next_xhat,
            next_yhat,
            next_state,
        )

        transitions[f"{xhat},{yhat},{state_id}"] = {
            "action": action,
            "next_xhat": next_xhat,
            "next_yhat": next_yhat,
            "next_state": next_state_id,
        }

    contexts = []
    uncertainties = {}
    local_state_counts = {}

    for xhat in range(size):
        for yhat in range(size):
            states_here = position_states[(xhat, yhat)]
            local_state_counts[f"{xhat},{yhat}"] = len(states_here)

            uncertainties[f"{xhat},{yhat}"] = [
                int(round(_mse(state) * MSE_SCALE))
                for state in states_here
            ]

            for state_id, state in enumerate(states_here):
                contexts.append({
                    "xhat": xhat,
                    "yhat": yhat,
                    "gaussian_state": state_id,
                    "gaussian_id": get_gaussian_id(state),
                    "uncertainty": int(
                        round(_mse(state) * MSE_SCALE)
                    ),
                })

    max_local_states = max(
        local_state_counts.values(),
        default=1,
    )

    gaussian_diagnostics = []
    for gaussian_id, state in enumerate(gaussian_states):
        bx, by, var_x, var_y, cov_xy = state
        gaussian_diagnostics.append({
            "gaussian_id": gaussian_id,
            "bias_x": bx,
            "bias_y": by,
            "var_x": var_x,
            "var_y": var_y,
            "cov_xy": cov_xy,
            "trace": var_x + var_y,
            "bias_squared": bx * bx + by * by,
            "mse": _mse(state),
        })

    return {
        "map_id": map_id,
        "context_count": len(contexts),
        "state_count": len(contexts),
        "gaussian_count": len(gaussian_states),

        # PRISM range: [0..max_local_states-1].
        "max_local_states": max_local_states,
        "max_gaussian_state": max_local_states - 1,

        "max_steps": max_steps,
        "p": p,
        "representation": "position_local_exact_reachable_gaussian",
        "uncertainty_metric": "mse",
        "mse_scale": MSE_SCALE,

        "thresholds": thresholds,
        "uncertainties": uncertainties,
        "contexts": contexts,
        "local_state_counts": local_state_counts,
        "zero_local_state": 0,
        "gaussian_states": gaussian_diagnostics,
        "transitions": transitions,
    }
