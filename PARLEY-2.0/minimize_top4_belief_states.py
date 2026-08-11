from collections import defaultdict
from functools import lru_cache
import math
import csv

DIRECTIONS = ("east", "west", "north", "south")
OFFSETS = {
    "east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)
}


def propagate(distribution, commanded_direction, p):
    result = defaultdict(float)
    for (x, y), prior in distribution.items():
        for actual_direction in DIRECTIONS:
            prob = (1.0 - 3.0 * p) if actual_direction == commanded_direction else p
            dx, dy = OFFSETS[actual_direction]
            result[(x + dx, y + dy)] += prior * prob
    return dict(result)


@lru_cache(maxsize=None)
def distribution_for_counts_cached(east, west, north, south, p):
    distribution = {(0, 0): 1.0}
    for direction, count in (("east", east), ("west", west), ("north", north), ("south", south)):
        for _ in range(count):
            distribution = propagate(distribution, direction, p)
    return distribution


def distribution_for_counts(counts, p=0.01):
    return distribution_for_counts_cached(*counts, float(p))


def largest_remainder_percent(values):
    raw = [100.0 * v for v in values]
    floors = [int(math.floor(v + 1e-12)) for v in raw]
    missing = 100 - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - floors[i], raw[i]), reverse=True)
    rounded = floors[:]
    for i in order[:missing]:
        rounded[i] += 1
    return tuple(rounded)


def top4_signature(distribution):
    probs = sorted(distribution.values(), reverse=True)
    top4 = probs[:4] + [0.0] * max(0, 4 - len(probs))
    top4 = top4[:4]
    other = max(0.0, 1.0 - sum(top4))
    return largest_remainder_percent(top4 + [other])


def gini_uncertainty(signature):
    return 10000 - sum(v * v for v in signature)


def count_vectors_for_age(age):
    for e in range(age + 1):
        for w in range(age - e + 1):
            for n in range(age - e - w + 1):
                s = age - e - w - n
                yield (e, w, n, s)


def all_count_vectors(max_steps):
    result = []
    for age in range(max_steps + 1):
        result.extend(count_vectors_for_age(age))
    return result


def increment_counts(counts, direction):
    result = list(counts)
    result[DIRECTIONS.index(direction)] += 1
    return tuple(result)


def build_raw_automaton(max_steps=10, p=0.01):
    raw_states = {}
    counts_list = all_count_vectors(max_steps)
    counts_set = set(counts_list)

    for counts in counts_list:
        age = sum(counts)
        signature = top4_signature(distribution_for_counts(counts, p))
        raw_states[counts] = {
            "counts": counts,
            "age": age,
            "signature": signature,
            "gini": gini_uncertainty(signature),
            "successors": {},
        }

    for counts in counts_list:
        if sum(counts) >= max_steps:
            continue
        for direction in DIRECTIONS:
            successor = increment_counts(counts, direction)
            if successor not in counts_set:
                raise RuntimeError(f"Missing successor {successor}")
            raw_states[counts]["successors"][direction] = successor

    return raw_states


def initial_partition(raw_states):
    groups = defaultdict(list)
    for counts, state in raw_states.items():
        terminal = not bool(state["successors"])
        groups[(state["signature"], terminal)].append(counts)
    return list(groups.values())


def class_map_from_partition(partition):
    result = {}
    for class_id, group in enumerate(partition):
        for counts in group:
            result[counts] = class_id
    return result


def partition_signature(partition):
    return frozenset(frozenset(group) for group in partition)


def refine_partition(raw_states, partition):
    class_map = class_map_from_partition(partition)
    groups = defaultdict(list)

    for counts, state in raw_states.items():
        if state["successors"]:
            succ_classes = tuple(class_map[state["successors"][d]] for d in DIRECTIONS)
        else:
            succ_classes = (None, None, None, None)
        key = (state["signature"], bool(state["successors"]), succ_classes)
        groups[key].append(counts)

    return list(groups.values())


def minimize_automaton(raw_states):
    partition = initial_partition(raw_states)
    iterations = 0
    while True:
        iterations += 1
        refined = refine_partition(raw_states, partition)
        if partition_signature(refined) == partition_signature(partition):
            return refined, iterations
        partition = refined


def build_minimized_belief_automaton(max_steps=10, p=0.01):
    raw_states = build_raw_automaton(max_steps=max_steps, p=p)
    partition, iterations = minimize_automaton(raw_states)
    raw_to_min = class_map_from_partition(partition)

    states = {}
    for class_id, group in enumerate(partition):
        rep = group[0]
        raw = raw_states[rep]
        successors = {}
        if raw["successors"]:
            for d in DIRECTIONS:
                successors[d] = raw_to_min[raw["successors"][d]]
        states[class_id] = {
            "signature": raw["signature"],
            "gini": raw["gini"],
            "terminal": not bool(raw["successors"]),
            "successors": successors,
            "raw_states": sorted(group),
            "ages": sorted({raw_states[c]["age"] for c in group}),
        }

    return {
        "raw_states": raw_states,
        "partition": partition,
        "raw_to_minimized": raw_to_min,
        "states": states,
        "initial_state": raw_to_min[(0, 0, 0, 0)],
        "iterations": iterations,
    }


def write_states_csv(path, automaton):
    states = automaton["states"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "belief_state", "b1", "b2", "b3", "b4", "other", "gini_uncertainty",
            "ages", "terminal", "east", "west", "north", "south", "raw_state_count", "raw_states"
        ])
        for sid in sorted(states):
            st = states[sid]
            b1, b2, b3, b4, other = st["signature"]
            writer.writerow([
                sid, b1, b2, b3, b4, other, st["gini"],
                "|".join(map(str, st["ages"])), int(st["terminal"]),
                st["successors"].get("east", ""), st["successors"].get("west", ""),
                st["successors"].get("north", ""), st["successors"].get("south", ""),
                len(st["raw_states"]), ";".join(map(str, st["raw_states"]))
            ])


def write_prism_snippet(path, automaton):
    states = automaton["states"]
    max_state = max(states)
    initial_state = automaton["initial_state"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"const int MAX_BELIEF_STATE = {max_state};\n")
        f.write(f"// belief_state : [0..{max_state}] init {initial_state};\n\n")
        for idx, name in enumerate(("b1", "b2", "b3", "b4", "other")):
            f.write(f"formula {name} =\n")
            for sid in sorted(states):
                f.write(f"    belief_state={sid} ? {states[sid]['signature'][idx]} :\n")
            f.write("    0;\n\n")
        f.write("formula belief_uncertainty = 10000 - (b1*b1 + b2*b2 + b3*b3 + b4*b4 + other*other);\n\n")
        for sid in sorted(states):
            st = states[sid]
            if st["terminal"]:
                continue
            for d in DIRECTIONS:
                f.write(f"// [{d}] belief_state={sid} -> (belief_state'={st['successors'][d]});\n")


def print_summary(automaton, max_steps, p):
    raw_count = len(automaton["raw_states"])
    dist_count = len({s["signature"] for s in automaton["raw_states"].values()})
    min_count = len(automaton["states"])
    print("Top-4 belief automaton")
    print("======================")
    print(f"p: {p}")
    print(f"max_steps: {max_steps}")
    print(f"raw history/count states: {raw_count}")
    print(f"distinct Top-4 probability signatures: {dist_count}")
    print(f"minimized Markov-compatible belief states: {min_count}")
    print(f"partition-refinement iterations: {automaton['iterations']}")
    print(f"initial minimized state: {automaton['initial_state']}")


if __name__ == "__main__":
    MAX_STEPS = 10
    P = 0.01
    automaton = build_minimized_belief_automaton(MAX_STEPS, P)
    print_summary(automaton, MAX_STEPS, P)
    write_states_csv("minimized_top4_belief_states.csv", automaton)
    write_prism_snippet("minimized_top4_belief_prism_snippet.txt", automaton)
    print("Written minimized_top4_belief_states.csv and minimized_top4_belief_prism_snippet.txt")
