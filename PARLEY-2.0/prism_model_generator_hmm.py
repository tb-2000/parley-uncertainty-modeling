import csv
import json
import os
import dijkstra

startX = 0
startY = 0
targetX = 4
targetY = 4
p = 0.01

directions = ['west', 'east', 'south', 'north']
directions_effects = [
    "(xhat'=max(xhat-1, 0))",
    "(xhat'=min(xhat+1, N))",
    "(yhat'=max(yhat-1, 0))",
    "(yhat'=min(yhat+1, N))",
]

obstacles = []
updates = [5]

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)
corridor = 1

prism_file = ""
period = 1

# Exact reachable HMM-belief models produced by
# full_hmm_belief_representatives.py.
#
# Expected:
#   hmm_belief_models/map_10/exact_reachable_beliefs.json
#   ...
HMM_BELIEF_ROOT = "hmm_belief_models"


def build_map(filename):
    n = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            n.append(row)

    global mapSize
    global map_data
    global obstacles

    mapSize = len(n)
    transposed = list(zip(*n))
    map_data = [row[::-1] for row in transposed]

    obstacles = []
    for x in range(0, mapSize):
        for y in range(0, mapSize):
            if int(map_data[x][y]) > 9:
                obstacles.append([x, y])


def hmm_belief_path(map_id):
    return os.path.join(
        HMM_BELIEF_ROOT,
        f"map_{map_id}",
        "exact_reachable_beliefs.json",
    )


def load_exact_hmm_beliefs(map_id):
    """
    Load the LOSSLESS reachable-belief abstraction.

    There is no K clustering here. Each hstate corresponds one-to-one to one
    actually reachable HMM belief beta.
    """
    path = hmm_belief_path(map_id)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing exact HMM belief model: {path}\n"
            "Run full_hmm_belief_representatives.py first and make sure "
            "exact_reachable_beliefs.json is generated for this map."
        )

    with open(path, 'r') as f:
        data = json.load(f)

    if data.get("mode") != "exact_reachable_beliefs":
        raise ValueError(
            f"{path} is not an exact reachable-belief model. "
            "Do not pass a k_XXX.json clustering file."
        )

    if int(data["map_id"]) != int(map_id):
        raise ValueError(
            f"HMM belief file map_id={data['map_id']} "
            f"does not match requested map {map_id}."
        )

    representatives = data.get("representatives", [])
    state_count = int(data.get("belief_state_count", len(representatives)))

    if state_count != len(representatives):
        raise ValueError(
            f"{path}: belief_state_count={state_count}, but "
            f"{len(representatives)} representatives are stored."
        )

    ids = sorted(int(r["belief_state"]) for r in representatives)
    if ids != list(range(state_count)):
        raise ValueError(
            f"{path}: exact hstate IDs must be consecutive 0..{state_count-1}."
        )

    reset_state = int(data.get("reset_belief_state", 0))
    if reset_state != 0:
        raise ValueError(
            f"{path}: expected reset_belief_state=0, got {reset_state}."
        )

    return data


def preambel(hmm):
    with open(prism_file, 'a') as f:
        f.write('dtmc\n')
        f.write(f'const int c = {period};\n')
        f.write('const int N=' + str(mapSize - 1) + ';\n')
        f.write('const int xstart = ' + str(startX) + ';\n')
        f.write('const int ystart = ' + str(startY) + ';\n')
        f.write('const int xtarget = ' + str(targetX) + ';\n')
        f.write('const int ytarget = ' + str(targetY) + ';\n')
        f.write('const double p = ' + str(p) + ';\n')

        state_count = int(hmm["belief_state_count"])
        f.write(f'const int HMM_STATES = {state_count};\n\n')

        f.write('formula hasCrashed = (1=0) ')
        for x, y in obstacles:
            f.write('| (x={0} & y={1}) '.format(str(x), str(y)))
        f.write(';\n\n')

        write_hmm_group_formulas(f, hmm)



def write_hmm_group_formulas(f, hmm):
    """
    PRISM-4.7-friendly exact HMM uncertainty formulas.

    Every exact reachable hstate is assigned to exactly one uncertainty
    level 0..10.  We use OR-groups rather than one huge ternary formula.

    The offline reachable-belief horizon is exactly 10 prediction steps.
    A separate PRISM variable `step` enforces a mandatory localization
    after the 10th movement, so no horizon/frontier formula is required.
    """
    state_entries = hmm.get("states", hmm.get("representatives", []))
    thresholds = [float(v) for v in hmm["thresholds"]]
    state_count = int(hmm["belief_state_count"])

    groups = {level: [] for level in range(11)}
    seen = set()

    for entry in state_entries:
        hstate = int(entry.get("hstate", entry.get("belief_state")))

        if "urc_level" in entry:
            level = int(entry["urc_level"])
        else:
            mse = float(entry["mse"])
            level = 0
            for idx, threshold in enumerate(thresholds[:10], start=1):
                if mse + 1e-15 >= threshold:
                    level = idx
                else:
                    break

        level = min(max(level, 0), 10)
        groups[level].append(hstate)
        seen.add(hstate)

    expected = set(range(state_count))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            "Exact HMM state table does not match hstate domain. "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    # Base groups: every hstate occurs exactly once.
    for level in range(11):
        states = sorted(groups[level])
        expr = (
            "false"
            if not states
            else " | ".join(f"(hstate={hstate})" for hstate in states)
        )
        f.write(f'formula hmm_l{level} = {expr};\n')

    f.write('\n')

    # Cumulative threshold predicates.
    for level in range(1, 11):
        expr = " | ".join(f"hmm_l{k}" for k in range(level, 11))
        f.write(f'formula hmm_ge_{level} = {expr};\n')

    f.write('\n')


def hmm_update_guard():
    """
    Return a short guard that compares synthesized c with the grouped HMM
    uncertainty formulas.

    c is a PRISM variable, therefore it cannot be used to index formula names.
    We use ten small cases, each referring to one cumulative formula.
    """
    cases = [
        f"(c={level} & hmm_ge_{level})"
        for level in range(1, 11)
    ]
    return "(" + " | ".join(cases) + ")"


def hmm_skip_guard():
    """
    Exact logical complement of hmm_update_guard() for c in 1..10.
    """
    cases = [
        f"(c={level} & !hmm_ge_{level})"
        for level in range(1, 11)
    ]
    return "(" + " | ".join(cases) + ")"


def robot():
    with open(prism_file, 'a') as f:
        f.write('module Robot \n')
        f.write('  x : [0..N] init xstart;\n')
        f.write('  y : [0..N] init ystart;\n')
        f.write('  move_ready : [0..1] init 1;\n')
        f.write('  crashed : [0..1] init 0;\n\n')

        f.write(
            '  [east] (move_ready=1) -> \n'
            '    (1-3*p): (x\'=min(x+1, N)) & (move_ready\'=0) + \n'
            '    p: (y\'=min(y+1, N)) & (move_ready\'=0) + \n'
            '    p: (y\'=max(y-1, 0)) & (move_ready\'=0) + \n'
            '    p: (x\'=max(x-1, 0)) & (move_ready\'=0); \n'
        )

        f.write(
            '  [west] (move_ready=1) -> \n'
            '    p: (x\'=min(x+1, N)) & (move_ready\'=0) + \n'
            '    p: (y\'=min(y+1, N)) & (move_ready\'=0) + \n'
            '    p: (y\'=max(y-1, 0)) & (move_ready\'=0) + \n'
            '    (1-3*p): (x\'=max(x-1, 0)) & (move_ready\'=0); \n'
        )

        f.write(
            '  [north] (move_ready=1) -> \n'
            '    p: (x\'=min(x+1, N)) & (move_ready\'=0) + \n'
            '    (1-3*p): (y\'=min(y+1, N)) & (move_ready\'=0) + \n'
            '    p: (y\'=max(y-1, 0)) & (move_ready\'=0) + \n'
            '    p: (x\'=max(x-1, 0)) & (move_ready\'=0); \n'
        )

        f.write(
            '  [south] (move_ready=1) -> \n'
            '    p: (x\'=min(x+1, N)) & (move_ready\'=0) + \n'
            '    p: (y\'=min(y+1, N)) & (move_ready\'=0) + \n'
            '    (1-3*p): (y\'=max(y-1, 0)) & (move_ready\'=0) + \n'
            '    p: (x\'=max(x-1, 0)) & (move_ready\'=0); \n'
        )

        f.write('\n')
        f.write(
            '  [check] (move_ready=0) & hasCrashed -> '
            '(crashed\'=1) & (move_ready\'=1); \n'
        )
        f.write(
            '  [check] (move_ready=0) & !hasCrashed -> '
            '(move_ready\'=1); \n'
        )
        f.write('endmodule\n\n')


def adaptation_mape_controller(d):
    with open(prism_file, 'a') as f:
        f.write('module Adaptation_MAPE_controller\n')

        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])
                if direction < 4:
                    f.write('  [{0}] '.format(directions[direction]))
                    f.write(
                        '(xhat={0}) & (yhat={1}) -> true;\n'.format(
                            str(x),
                            str(y),
                        )
                    )

        f.write('endmodule\n\n')


def knowledge_hmm(hmm):
    """
    Exact finite HMM Knowledge module with a hard 10-step update horizon.

    Semantics
    ---------
    step=0 immediately after a perfect localization.
    Every movement increments step by one.
    An update is performed if either
        1) the HMM uncertainty reaches the URC-selected threshold c, or
        2) step=10.
    At step=10, skip_update is impossible.

    Hence at most 10 movements can occur between two localizations.
    This also means that exact reachable HMM beliefs only need to be
    precomputed through prediction step 10.
    """
    transitions = hmm.get("belief_transitions", {})
    state_count = int(hmm["belief_state_count"])

    max_steps = int(hmm.get("max_steps", 10))
    if max_steps != 10:
        raise ValueError(
            "This PRISM generator requires exact HMM beliefs generated "
            f"with max_steps=10, but the input contains max_steps={max_steps}."
        )

    update_guard = hmm_update_guard()
    skip_guard = hmm_skip_guard()

    with open(prism_file, 'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n')
        f.write('  yhat : [0..N] init ystart;\n')
        f.write(f'  hstate : [0..{state_count - 1}] init 0;\n')
        f.write('  step : [0..10] init 0;\n')
        f.write('  ready : [0..1] init 1;\n\n')

        def sort_key(item):
            key, value = item
            xhat_s, yhat_s, hstate_s = key.split(",")
            return (
                int(xhat_s),
                int(yhat_s),
                int(hstate_s),
                directions.index(value["action"]),
            )

        for key, value in sorted(transitions.items(), key=sort_key):
            xhat_s, yhat_s, hstate_s = key.split(",")
            xhat = int(xhat_s)
            yhat = int(yhat_s)
            hstate = int(hstate_s)
            action = value["action"]
            next_hstate = int(
                value.get("next_hstate", value["next_belief_state"])
            )

            if action not in directions:
                raise ValueError(
                    f"Unknown HMM transition action '{action}' in {key}."
                )

            effect = directions_effects[directions.index(action)]

            f.write(
                f'  [{action}] ready=1 & step<10'
                f' & xhat={xhat}'
                f' & yhat={yhat}'
                f' & hstate={hstate}'
                f' -> {effect}'
                f' & (hstate\'={next_hstate})'
                f' & (step\'=step+1)'
                f' & (ready\'=0);\n'
            )

        f.write('\n')

        # Mandatory update at the 10th movement, otherwise uncertainty-driven.
        f.write(
            '  [update] ready=0 & '
            f'(step=10 | {update_guard}) -> '
            '(xhat\'=x) & (yhat\'=y) & '
            '(hstate\'=0) & (step\'=0) & (ready\'=1);\n'
        )

        # At step=10 this command is disabled, so update is compulsory.
        f.write(
            '  [skip_update] ready=0 & step<10 & '
            f'{skip_guard} -> '
            '(ready\'=1);\n'
        )

        f.write('endmodule\n\n')


def rewards():
    with open(prism_file, 'a') as f:
        f.write('rewards "cost" \n')
        f.write('  [east] true : 1; \n')
        f.write('  [west] true : 1; \n')
        f.write('  [north] true : 1; \n')
        f.write('  [south] true : 1; \n')
        f.write('  [update] true : 5;\n')
        f.write('endrewards \n\n')


def read_params_from_file():
    with open('input.json', 'r') as file:
        params = json.load(file)

    global startX, startY, targetX, targetY, map_file, p, updates

    startX = params["startX"]
    startY = params["startY"]
    targetX = params["targetX"]
    targetY = params["targetY"]
    p = params["p"]
    map_file = params["map_file"]
    updates = params["updates"]


# i depicts which map should be used
def generate_model(i):
    global prism_file

    prism_file = (
        "Applications/EvoChecker-master/models/model_"
        + str(i)
        + ".prism"
    )

    read_params_from_file()
    build_map("maps/map_" + str(i) + ".csv")

    hmm = load_exact_hmm_beliefs(i)

    target_pos = (targetX, targetY)
    _d = dijkstra.compute_directions(map_data, target_pos)
    d = list(zip(*_d))

    open(prism_file, "w").close()

    preambel(hmm)
    robot()
    adaptation_mape_controller(d)
    knowledge_hmm(hmm)
    rewards()

    print(
        "finished map "
        + str(i)
        + " with "
        + str(hmm["belief_state_count"])
        + " exact reachable HMM belief states"
    )
