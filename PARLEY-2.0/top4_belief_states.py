from collections import defaultdict
from functools import lru_cache
from itertools import combinations_with_replacement
import math

# Direction order is also used for the history counters written to PRISM.
DIRECTIONS = ("east", "west", "north", "south")
OFFSETS = {
    "east": (1, 0),
    "west": (-1, 0),
    "north": (0, 1),
    "south": (0, -1),
}


def _propagate(distribution, commanded_direction, p):
    """Propagate a full position distribution by one commanded movement."""
    result = defaultdict(float)

    for (x, y), prior in distribution.items():
        for actual_direction in DIRECTIONS:
            probability = (1.0 - 3.0 * p) if actual_direction == commanded_direction else p
            dx, dy = OFFSETS[actual_direction]
            result[(x + dx, y + dy)] += prior * probability

    return dict(result)


@lru_cache(maxsize=None)
def _distribution_for_counts_cached(east, west, north, south, p):
    """
    Compute the complete relative position distribution for a command-count vector.

    On the homogeneous relative grid used by the abstract model, the movement kernels
    commute. Therefore only the number of E/W/N/S commands matters, not their order.
    """
    distribution = {(0, 0): 1.0}

    for direction, count in (
        ("east", east),
        ("west", west),
        ("north", north),
        ("south", south),
    ):
        for _ in range(count):
            distribution = _propagate(distribution, direction, p)

    return distribution


def distribution_for_counts(counts, p=0.01):
    return _distribution_for_counts_cached(*counts, float(p))


def _largest_remainder_percent(values):
    """
    Convert probabilities to integer percentages that sum exactly to 100.
    Uses the largest-remainder method.
    """
    raw = [100.0 * v for v in values]
    floors = [int(math.floor(v + 1e-12)) for v in raw]
    missing = 100 - sum(floors)

    order = sorted(
        range(len(raw)),
        key=lambda i: (raw[i] - floors[i], raw[i]),
        reverse=True,
    )

    rounded = floors[:]
    for i in order[:missing]:
        rounded[i] += 1

    return tuple(rounded)


def top4_signature(distribution):
    """
    Return a position-independent abstract Top-4 belief:
        (b1, b2, b3, b4, other)
    with integer percentages summing to 100.
    """
    probabilities = sorted(distribution.values(), reverse=True)
    top4 = probabilities[:4] + [0.0] * max(0, 4 - len(probabilities))
    top4 = top4[:4]
    other = max(0.0, 1.0 - sum(top4))

    return _largest_remainder_percent(top4 + [other])


def _count_vectors(total):
    """Yield all (E,W,N,S) count vectors whose entries sum to total."""
    for east in range(total + 1):
        for west in range(total - east + 1):
            for north in range(total - east - west + 1):
                south = total - east - west - north
                yield (east, west, north, south)


def build_belief_automaton(max_steps=10, p=0.01):
    """
    Build all abstract Top-4 belief classes reachable up to max_steps.

    Returns:
        states:
            {state_id: {"signature": (b1,b2,b3,b4,other), "first_age": age}}
        count_to_state:
            {(e,w,n,s): state_id}
        transitions:
            {((e,w,n,s), direction): {
                "next_counts": (...),
                "next_state": id,
                "signature": (...)
            }}
    """
    signature_to_state = {}
    states = {}
    count_to_state = {}

    for age in range(max_steps + 1):
        for counts in _count_vectors(age):
            signature = top4_signature(distribution_for_counts(counts, p))

            if signature not in signature_to_state:
                state_id = len(signature_to_state)
                signature_to_state[signature] = state_id
                states[state_id] = {
                    "signature": signature,
                    "first_age": age,
                }

            count_to_state[counts] = signature_to_state[signature]

    transitions = {}

    for counts, state_id in count_to_state.items():
        age = sum(counts)
        if age >= max_steps:
            continue

        for index, direction in enumerate(DIRECTIONS):
            next_counts = list(counts)
            next_counts[index] += 1
            next_counts = tuple(next_counts)

            next_state = count_to_state[next_counts]
            transitions[(counts, direction)] = {
                "next_counts": next_counts,
                "next_state": next_state,
                "signature": states[next_state]["signature"],
            }

    return states, count_to_state, transitions


def write_summary(path, max_steps=10, p=0.01):
    states, count_to_state, _ = build_belief_automaton(max_steps=max_steps, p=p)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Abstract Top-4 belief states, p={p}, max_steps={max_steps}\n")
        f.write("# Format: id: b1 / b2 / b3 / b4 / other ; first_age\n\n")

        for state_id in sorted(states):
            signature = states[state_id]["signature"]
            age = states[state_id]["first_age"]
            f.write(
                f"{state_id}: "
                f"{signature[0]} / {signature[1]} / {signature[2]} / "
                f"{signature[3]} / {signature[4]} ; first_age={age}\n"
            )

        f.write(f"\n# Number of belief states: {len(states)}\n")
        f.write(f"# Number of count vectors: {len(count_to_state)}\n")


if __name__ == "__main__":
    write_summary("top4_belief_states.txt", max_steps=10, p=0.01)
