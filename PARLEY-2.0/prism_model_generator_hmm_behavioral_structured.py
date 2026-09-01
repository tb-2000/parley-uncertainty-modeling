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
        "behavioral_structured.json",
    )


def load_behavioral_hmm_beliefs(map_id):
    """
    Load the exact-reachable behavioral HMM quotient.

    PRISM stores no global hstate ID. Each quotient class is represented by:
        xhat, yhat, hmm_state, substate.
    """
    path = hmm_belief_path(map_id)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing behavioral HMM model: {path}\n"
            "Run full_hmm_belief_behavioral_structured.py first."
        )

    with open(path, 'r') as f:
        data = json.load(f)

    if data.get("mode") != "behavioral_structured_hmm_beliefs":
        raise ValueError(
            f"{path} is not a structured behavioral HMM model."
        )

    if int(data["map_id"]) != int(map_id):
        raise ValueError(
            f"{path}: map_id={data['map_id']} does not match map {map_id}."
        )

    if len(data.get("thresholds", [])) != 10:
        raise ValueError(
            f"{path}: expected exactly ten HMM-MSE thresholds."
        )

    if int(data.get("max_steps", 10)) != 10:
        raise ValueError(
            f"{path}: expected threshold calibration with max_steps=10."
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
        f.write('const double p = ' + str(p) + ';\n\n')

        f.write('// HMM uncertainty level -> raw expected-squared-error threshold\n')
        for level, threshold in enumerate(hmm["thresholds"], start=1):
            f.write(
                f'// level {level} -> HMM-MSE >= {threshold:.17g}\n'
            )
        f.write('\n')

        f.write('formula hasCrashed = (1=0) ')
        for x, y in obstacles:
            f.write('| (x={0} & y={1}) '.format(str(x), str(y)))
        f.write(';\n\n')

        # hmm_state is directly the highest HMM-MSE threshold reached.
        # Therefore no hstate groups, hmm_u_X, hmm_ge_X, or frontier formula
        # are necessary.
        terms = [
            f'(hmm_state={level} & c<={level})'
            for level in range(1, 10)
        ]
        terms.append('(hmm_state=10)')

        f.write(
            'formula update_required = '
            + ' | '.join(terms)
            + ';\n\n'
        )

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
    Behavioral HMM Knowledge module.

    State:
        (xhat, yhat, hmm_state, substate)

    hmm_state:
        highest HMM-MSE threshold level reached, 0..10.

    substate:
        only distinguishes behaviorally different quotient classes sharing the
        same position and uncertainty level.
    """
    transitions = hmm.get("belief_transitions", {})
    max_substate = int(hmm["max_substate"])

    with open(prism_file, 'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n')
        f.write('  yhat : [0..N] init ystart;\n')
        f.write('  hmm_state : [0..10] init 0;\n')
        f.write(
            f'  substate : [0..{max_substate}] init 0;\n'
        )
        f.write('  ready : [0..1] init 1;\n\n')

        def sort_key(item):
            key, value = item
            src = value["source"]
            return (
                int(src["xhat"]),
                int(src["yhat"]),
                int(src["hmm_state"]),
                int(src["substate"]),
                directions.index(value["action"]),
            )

        for key, value in sorted(
            transitions.items(),
            key=sort_key,
        ):
            action = value["action"]
            src = value["source"]
            dst = value["target"]

            f.write(
                f'  [{action}] ready=1'
                f' & xhat={src["xhat"]}'
                f' & yhat={src["yhat"]}'
                f' & hmm_state={src["hmm_state"]}'
                f' & substate={src["substate"]}'
                f' -> (xhat\'={dst["xhat"]})'
                f' & (yhat\'={dst["yhat"]})'
                f' & (hmm_state\'={dst["hmm_state"]})'
                f' & (substate\'={dst["substate"]})'
                f' & (ready\'=0);\n'
            )

        f.write('\n')

        # Perfect localization: beta <- pi, therefore uncertainty level 0 and
        # the unique reset substate 0 at the actual position.
        f.write(
            '  [update] ready=0 & update_required -> '
            '(xhat\'=x) & (yhat\'=y) & '
            '(hmm_state\'=0) & (substate\'=0) & '
            '(ready\'=1);\n'
        )

        f.write(
            '  [skip_update] ready=0 & !update_required -> '
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

    hmm = load_behavioral_hmm_beliefs(i)

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
        f"finished map {i}: "
        f"exact contexts={hmm['exact_context_count']}, "
        f"behavioral classes={hmm['behavioral_class_count']}, "
        f"max_substate={hmm['max_substate']}, "
        f"reachable hidden states="
        f"{hmm['reachable_hidden_state_count']}/"
        f"{hmm['hidden_state_count_full']}"
    )
