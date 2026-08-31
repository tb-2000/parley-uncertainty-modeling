"""
Compact exact reachable Gaussian moment model for PARLEY.

PRISM stores only:
    gstate : one ID per reachable (xhat, yhat, Gaussian-moment-state) context
    ready  : Knowledge handshake variable

There are no separate xhat/yhat variables in the Knowledge module.

Each Gaussian moment state is
    g = (bias_x, bias_y, var_x, var_y, cov_xy)

The first (N+1)^2 gstates are the zero-uncertainty contexts in row-major
position order, so perfect localization can be encoded by one PRISM update:
    gstate' = x*(N+1)+y

"Exact" means exact within the Gaussian moment model. No K-medoids or
nearest-representative projection are used.
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
    """Derive exactly the same ten median-MSE thresholds as before."""
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
    Build a compact exact Gaussian Knowledge automaton.

    One PRISM gstate corresponds to exactly one reachable context:
        (xhat, yhat, Gaussian moment state).

    Perfect localization maps physical position (x,y) to the zero-uncertainty
    context at that position.
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

    # Distinct Gaussian moment tuples are kept only for offline diagnostics.
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

    contexts = []
    context_id_by_key = {}
    queue = deque()

    def get_context_id(xhat, yhat, state):
        state = _state_key(state)
        key = (xhat, yhat, state)

        if key not in context_id_by_key:
            context_id = len(contexts)
            context_id_by_key[key] = context_id

            gaussian_id = get_gaussian_id(state)
            contexts.append({
                "gstate": context_id,
                "xhat": xhat,
                "yhat": yhat,
                "gaussian_id": gaussian_id,
                "uncertainty": int(round(_mse(state) * MSE_SCALE)),
            })
            queue.append((xhat, yhat, state))

        return context_id_by_key[key]

    # IMPORTANT numbering invariant for the single Perfect-Localization update:
    # zero_context(x,y) = x*size+y
    zero_contexts = {}
    for xhat in range(size):
        for yhat in range(size):
            context_id = get_context_id(
                xhat,
                yhat,
                ZERO_STATE,
            )
            expected_id = xhat * size + yhat

            if context_id != expected_id:
                raise AssertionError(
                    "Zero-context numbering invariant violated: "
                    f"({xhat},{yhat}) -> {context_id}, expected {expected_id}"
                )

            zero_contexts[f"{xhat},{yhat}"] = context_id

    transitions = {}

    while queue:
        xhat, yhat, state = queue.popleft()
        state = _state_key(state)

        context_id = context_id_by_key[
            (xhat, yhat, state)
        ]
        scaled_mse = int(
            round(_mse(state) * MSE_SCALE)
        )

        # Once tau_10 is reached, every URC choice 1..10 localizes.
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

        next_context = get_context_id(
            next_xhat,
            next_yhat,
            next_state,
        )

        transitions[str(context_id)] = {
            "action": action,
            "next_context": next_context,
        }

    position_contexts = {
        f"{x},{y}": []
        for x in range(size)
        for y in range(size)
    }

    for context in contexts:
        position_contexts[
            f"{context['xhat']},{context['yhat']}"
        ].append(context["gstate"])

    uncertainties = [
        int(context["uncertainty"])
        for context in contexts
    ]

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
        # PRISM gstate count = reachable (xhat,yhat,Gaussian) contexts.
        "state_count": len(contexts),
        "context_count": len(contexts),
        # Distinct moment tuples are diagnostic only.
        "gaussian_count": len(gaussian_states),
        "max_steps": max_steps,
        "p": p,
        "representation": "compact_exact_reachable_gaussian_contexts",
        "uncertainty_metric": "mse",
        "mse_scale": MSE_SCALE,
        "thresholds": thresholds,
        "uncertainties": uncertainties,
        "contexts": contexts,
        "position_contexts": position_contexts,
        "zero_contexts": zero_contexts,
        "gaussian_states": gaussian_diagnostics,
        "transitions": transitions,
    }
