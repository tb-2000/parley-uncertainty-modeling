import csv
import json
import dijkstra
from exact_reachable_belief_model_behavioral_structured import build_exact_belief_model

startX = 0
startY = 0
targetX = 4
targetY = 4
p = 0.01
directions = ['west', 'east', 'south', 'north']
obstacles = []

updates = [5]  # cost of updates

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)
corridor = 1

prism_file = ""
belief_model = None
# period of updates
period = 1


def build_map(filename):
    n = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            n.append(row)
    global mapSize
    global map_data
    mapSize = len(n)
    transposed = list(zip(*n))  # Transpose the matrix
    map_data = [row[::-1] for row in transposed]  # Reverse the order of each row
    global obstacles
    obstacles = []
    for x in range(0, mapSize):
        for y in range(0, mapSize):
            if int(map_data[x][y]) > 9:
                obstacles.append([x, y])
    # for j in range(0, mapSize):
    #    map_data.append([i[j] for i in n])


def preambel():
    with open(prism_file, 'a') as f:
        f.write('dtmc\n')

        thresholds = belief_model["thresholds"]

        # urc_synthesis_belief_behavioral_structured.py removes this constant
        # and replaces it with the synthesized c:[1..10].
        f.write(f'const int c = {period};\n')

        f.write('// Map-specific belief-uncertainty thresholds:\n')
        for index, threshold in enumerate(thresholds, start=1):
            f.write(
                f'// c={index} -> max_belief_uncertainty={threshold}\n'
            )
            f.write(
                f'const int belief_threshold_{index} = {threshold};\n'
            )

        f.write('const int N=' + str(mapSize - 1) + ';\n')
        f.write('const int xstart = ' + str(startX) + ';\n')
        f.write('const int ystart = ' + str(startY) + ';\n')
        f.write('const int xtarget = ' + str(targetX) + ';\n')
        f.write('const int ytarget = ' + str(targetY) + ';\n')
        f.write('const double p = ' + str(p) + ';\n \n')

        f.write('formula hasCrashed = (1=0) ')
        for x, y in obstacles:
            f.write(
                '| (x={0} & y={1}) '.format(str(x), str(y))
            )
        f.write(';\n\n')

        # belief_state IS the already preserved threshold/URC stage.
        # No kstate disjunction is necessary anymore.
        f.write('// belief_state is the behavioral URC stage 0..10.\n')
        f.write('formula update_required = ')
        terms = []
        for stage in range(1, 10):
            terms.append(
                f'(belief_state={stage} & c<={stage})'
            )
        terms.append('(belief_state=10)')
        f.write(' | '.join(terms))
        f.write(';\n\n')

def robot():
    with open(prism_file, 'a') as f:
        # Keep the original Robot structure, including crashed and
        # the explicit [check] phase after every physical movement.
        f.write('module Robot \n')
        f.write('  x : [0..N] init xstart;\n')
        f.write('  y : [0..N] init ystart;\n')
        f.write('  move_ready : [0..1] init 1;\n')
        f.write('  crashed : [0..1] init 0;\n\n')
        f.write('  [east] (move_ready=1) -> \n'
                '    (1-3*p): (x\'=min(x+1, N)) & (move_ready\'=0) + \n'
                '    p: (y\'=min(y+1, N)) & (move_ready\'=0) + \n'
                '    p: (y\'=max(y-1, 0)) & (move_ready\'=0) + \n'
                '    p: (x\'=max(x-1, 0)) & (move_ready\'=0); \n')

        f.write('  [west] (move_ready=1) -> \n'
                '    p: (x\'=min(x+1, N)) & (move_ready\'=0) + \n'
                '    p: (y\'=min(y+1, N)) & (move_ready\'=0) + \n'
                '    p: (y\'=max(y-1, 0)) & (move_ready\'=0) + \n'
                '    (1-3*p): (x\'=max(x-1, 0)) & (move_ready\'=0); \n')

        f.write('  [north] (move_ready=1) -> \n'
                '    p: (x\'=min(x+1, N)) & (move_ready\'=0) + \n'
                '    (1-3*p): (y\'=min(y+1, N)) & (move_ready\'=0) + \n'
                '    p: (y\'=max(y-1, 0)) & (move_ready\'=0) + \n'
                '    p: (x\'=max(x-1, 0)) & (move_ready\'=0); \n')

        f.write('  [south] (move_ready=1) -> \n'
                '    p: (x\'=min(x+1, N)) & (move_ready\'=0) + \n'
                '    p: (y\'=min(y+1, N)) & (move_ready\'=0) + \n'
                '    (1-3*p): (y\'=max(y-1, 0)) & (move_ready\'=0) + \n'
                '    p: (x\'=max(x-1, 0)) & (move_ready\'=0); \n')
        f.write('\n')
        f.write('  [check] (move_ready=0) & hasCrashed -> (crashed\'=1) & (move_ready\'=1); \n')
        f.write('  [check] (move_ready=0) & !hasCrashed -> (move_ready\'=1); \n')
        f.write('endmodule\n\n')


def adaptation_mape_controller(d):
    with open(prism_file, 'a') as f:
        # xhat/yhat are explicit again.  Therefore the large estimate_X_Y
        # formulas of the compact kstate encoding disappear completely.
        f.write('module Adaptation_MAPE_controller\n')
        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])
                if direction < 4:
                    f.write(
                        f'  [{directions[direction]}] '
                        f'xhat={x} & yhat={y} -> true;\n'
                    )
        f.write('endmodule\n\n')

def knowledge():
    with open(prism_file, 'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n')
        f.write('  yhat : [0..N] init ystart;\n')
        f.write('  belief_state : [0..10] init 0;\n')
        f.write(
            f'  substate : [0..{belief_model["max_substate"]}] init 0;\n'
        )
        f.write('  ready : [0..1] init 1;\n\n')

        # One command per reachable behavioral quotient class with a
        # successor.  The global class ID is only a Python-side identifier;
        # it does not appear in PRISM.
        for context_id, transition in belief_model["transitions"].items():
            action = transition["action"]
            src = transition["source"]
            dst = transition["target"]

            f.write(
                f'  [{action}] ready=1'
                f' & xhat={src["xhat"]}'
                f' & yhat={src["yhat"]}'
                f' & belief_state={src["belief_state"]}'
                f' & substate={src["substate"]} -> '
                f"(xhat'={dst['xhat']})"
                f" & (yhat'={dst['yhat']})"
                f" & (belief_state'={dst['belief_state']})"
                f" & (substate'={dst['substate']})"
                f" & (ready'=0);\n"
            )

        # Perfect localization resets the structured knowledge state to the
        # unique certainty state of the actual robot position.
        #
        # By construction certainty is always:
        #   xhat=x, yhat=y, belief_state=0, substate=0.
        f.write(
            '  [update] update_required & ready=0 -> '
            "(xhat'=x) & (yhat'=y) "
            "& (belief_state'=0) & (substate'=0) "
            "& (ready'=1);\n"
        )

        f.write(
            '  [skip_update] !update_required & ready=0 -> '
            "(ready'=1);\n"
        )
        f.write('endmodule\n\n')

def rewards():
    with open(prism_file, 'a') as f:
        f.write('rewards \"cost\" \n')
        f.write('  [east] true : 1; \n')
        f.write('  [west] true : 1; \n')
        f.write('  [north] true : 1; \n')
        f.write('  [south] true : 1; \n')
        f.write('  [update] true : 5;\n')
        f.write('endrewards \n\n')

        # f.write('label \"mission_success\" = (x=xtarget) & (y=ytarget) & (!hasCrashed);\n')


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
    global prism_file, belief_model
    prism_file = "Applications/EvoChecker-master/models/model_" + str(i) + ".prism"
    read_params_from_file()
    build_map("maps/map_" + str(i) + ".csv")
    target_pos = (targetX, targetY)
    _d = dijkstra.compute_directions(map_data, target_pos)
    # we have to transpose the matrix in the end, similar to what we did when reading the map_data in the first place
    d = list(zip(*_d))  # Transpose the matrix

    belief_model = build_exact_belief_model(
        map_id=i,
        map_data=map_data,
        target=target_pos,
        p=p,
        max_steps=10,
    )

    # Structured update invariant: every perfect-localization state is
    # represented by (xhat=x, yhat=y, belief_state=0, substate=0).
    for x in range(mapSize):
        for y in range(mapSize):
            certainty = belief_model["certainty_contexts"][f"{x},{y}"]
            if (
                certainty["xhat"] != x
                or certainty["yhat"] != y
                or certainty["belief_state"] != 0
                or certainty["substate"] != 0
            ):
                raise AssertionError(
                    f"Invalid structured certainty state for ({x},{y}): "
                    f"{certainty}"
                )

    exact_count = belief_model["exact_context_count"]
    reduced_count = belief_model["state_count"]

    print(
        f"map {i}: exact contexts={exact_count}, "
        f"behavioral classes={reduced_count}, "
        f"reduction={100.0 * (exact_count - reduced_count) / exact_count:.2f}%, "
        f"distinct relative beliefs={belief_model['exact_belief_count']}, "
        f"max substate={belief_model['max_substate']}"
    )

    open(prism_file, "w").close()
    preambel()
    robot()
    adaptation_mape_controller(d)
    knowledge()
    rewards()

    print("finished map " + str(i))
