"""
Build the exact reachable Gaussian moment model, then quotient it by
PARLEY-observable behavioral equivalence.

This preserves the exact Gaussian prediction model offline while exposing the
quotient to PRISM as

    (xhat, yhat, gaussian_state, substate)

instead of one global gstate ID.
"""

from exact_reachable_gaussian_model import (
    build_exact_gaussian_model as build_unminimized_exact_gaussian_model,
)
from gaussian_behavioral_minimization_structured import (
    minimize_gaussian_model,
)


def build_exact_gaussian_model(
    map_id,
    map_data,
    target,
    p=0.01,
    max_steps=10,
):
    exact_model = build_unminimized_exact_gaussian_model(
        map_id=map_id,
        map_data=map_data,
        target=target,
        p=p,
        max_steps=max_steps,
    )

    reduced = minimize_gaussian_model(
        exact_model,
        map_size=len(map_data),
    )

    reduced["exact_context_count"] = exact_model["context_count"]
    reduced["exact_gaussian_count"] = exact_model["gaussian_count"]

    return reduced
