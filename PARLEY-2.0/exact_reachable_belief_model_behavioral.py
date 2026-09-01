"""
Exact reachable belief model + behavioral quotient for PARLEY.
"""

from exact_reachable_belief_model_sourcecopy_v2 import (
    build_exact_belief_model as build_unminimized_exact_belief_model,
)
from belief_behavioral_minimization import minimize_belief_model


def build_exact_belief_model(
    map_id,
    map_data,
    target,
    p=0.01,
    max_steps=10,
):
    exact_model = build_unminimized_exact_belief_model(
        map_id=map_id,
        map_data=map_data,
        target=target,
        p=p,
        max_steps=max_steps,
    )

    reduced = minimize_belief_model(
        exact_model,
        map_size=len(map_data),
    )

    reduced["exact_context_count"] = exact_model["context_count"]
    reduced["exact_belief_count"] = exact_model["belief_count"]
    return reduced
