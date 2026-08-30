"""
Exact reachable Gaussian moment-state model for PARLEY.

This module replaces the former K=100 Gaussian medoid abstraction.

Gaussian knowledge state:
    g = (bias_x, bias_y, var_x, var_y, cov_xy)

For a fixed map:
- derive the SAME ten map-specific Gaussian MSE thresholds as before;
- enumerate all Gaussian moment states reachable under the union of URC choices;
- assign one integer ID to every exact reachable Gaussian state;
- construct exact deterministic knowledge transitions
      (xhat, yhat, gstate_id) -> (next_xhat, next_yhat, next_gstate_id)
  for the MAPE-selected action;
- stop skip expansion when tau_10 is reached, since then every c=1..10
  requires localization.

"Exact" here means exact within the Gaussian moment model. The Gaussian
representation itself remains an approximation of the full positional belief.
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
    """Derive the same 10 median-MSE thresholds as the previous model."""
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

    return thresholds, controller


def build_exact_gaussian_model(
    map_id,
    map_data,
    target,
    p=0.01,
    max_steps=10,
):
    """
    Build the exact reachable Gaussian finite-state knowledge automaton.

    Certainty/zero uncertainty is reusable as gstate 0 at every position after
    perfect localization.
    """
    size = len(map_data)
    n = size - 1

    thresholds, controller = _derive_thresholds(
        map_data,
        target,
        p,
        max_steps,
    )

    if len(thresholds) != 10:
        raise ValueError(
            f"Expected exactly 10 Gaussian thresholds, got {len(thresholds)}."
        )

    tau10 = int(thresholds[-1])

    gaussian_states = []
    id_by_key = {}

    def get_id(state):
        key = _state_key(state)
        if key not in id_by_key:
            id_by_key[key] = len(gaussian_states)
            gaussian_states.append(key)
        return id_by_key[key]

    zero_id = get_id(ZERO_STATE)
    if zero_id != 0:
        raise AssertionError("ZERO_STATE must have gstate ID 0.")

    # Seed zero-state at every grid position because perfect localization can
    # reset xhat,yhat to the concrete robot position.
    queue = deque()
    seen_contexts = set()

    for xhat in range(size):
        for yhat in range(size):
            context = (
                xhat,
                yhat,
                _state_key(ZERO_STATE),
            )
            if context not in seen_contexts:
                seen_contexts.add(context)
                queue.append(
                    (xhat, yhat, ZERO_STATE)
                )

    transitions = {}

    while queue:
        xhat, yhat, state = queue.popleft()

        state = _state_key(state)
        state_id = get_id(state)

        scaled_mse = int(
            round(_mse(state) * MSE_SCALE)
        )

        # Once tau_10 is reached, every URC choice c=1..10 updates.
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

        next_id = get_id(next_state)

        transitions[
            f"{xhat},{yhat},{state_id}"
        ] = {
            "action": action,
            "next_xhat": next_xhat,
            "next_yhat": next_yhat,
            "next_state": next_id,
        }

        next_context = (
            next_xhat,
            next_yhat,
            next_state,
        )

        if next_context not in seen_contexts:
            seen_contexts.add(next_context)
            queue.append(
                (
                    next_xhat,
                    next_yhat,
                    next_state,
                )
            )

    uncertainties = [
        int(round(_mse(state) * MSE_SCALE))
        for state in gaussian_states
    ]

    representatives = []
    for state_id, state in enumerate(gaussian_states):
        bx, by, var_x, var_y, cov_xy = state
        representatives.append(
            {
                "state_id": state_id,
                "bias_x": bx,
                "bias_y": by,
                "var_x": var_x,
                "var_y": var_y,
                "cov_xy": cov_xy,
                "trace": var_x + var_y,
                "bias_squared": bx * bx + by * by,
                "mse": _mse(state),
            }
        )

    return {
        "map_id": map_id,
        "state_count": len(gaussian_states),
        "context_count": len(seen_contexts),
        "max_steps": max_steps,
        "p": p,
        "representation": "exact_reachable_gaussian_moments",
        "uncertainty_metric": "mse",
        "mse_scale": MSE_SCALE,
        "thresholds": thresholds,
        "uncertainties": uncertainties,
        # Name kept for compatibility with existing diagnostics/loaders.
        "representatives": representatives,
        "transitions": transitions,
        "zero_state": 0,
    }
