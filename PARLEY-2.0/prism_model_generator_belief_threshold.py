import csv
import json
import dijkstra

from top4_belief_states import DIRECTIONS as BELIEF_DIRECTIONS
from top4_belief_states import build_belief_automaton

startX = 0
startY = 0
targetX = 4
targetY = 4
p = 0.01

# Matches the old c range: a belief is never propagated beyond 10 moves
# without a perfect Ground-Truth update.
max_belief_steps = 10

# Default Gini threshold used before urc_synthesis_belief_threshold.py injects
# the real URC module. The calibrated thresholds from maps 10-99 are:
# 588, 1154, 1698, 2218, 2552, 3034, 3342, 3642, 4076, 4354.
#
# Belief uncertainty uses all five masses:
#   G = 10000 - (b1^2 + b2^2 + b3^2 + b4^2 + other^2)
# Larger values mean a more distributed / uncertain belief.
default_belief_threshold = 2552

directions = ['west', 'east', 'south', 'north']
obstacles = []

updates = [5]

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)

prism_file = ""

belief_states = {}
count_to_state = {}
belief_transitions = {}


def build_map(filename):
    n = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            n.append(row)

    global mapSize, map_data, obstacles

    mapSize = len(n)
    transposed = list(zip(*n))
    map_data = [row[::-1] for row in transposed]

    obstacles = []
    for x in range(mapSize):
        for y in range(mapSize):
            if int(map_data[x][y]) > 9:
                obstacles.append([x, y])


def build_beliefs():
    global belief_states, count_to_state, belief_transitions

    belief_states, count_to_state, belief_transitions = build_belief_automaton(
        max_steps=max_belief_steps,
        p=p,
    )


def preambel():
    max_state = max(belief_states)

    with open(prism_file, 'a') as f:
        f.write('dtmc\n')
        f.write(f'const int MAX_BELIEF_STEPS = {max_belief_steps};\n')
        f.write(f'const int MAX_BELIEF_STATE = {max_state};\n')

        # This constant is removed by urc_synthesis_belief_threshold.py
        # and replaced by a URC-controlled state variable.
        f.write(f'const int max_belief_uncertainty = {default_belief_threshold};\n')

        f.write('const int N=' + str(mapSize - 1) + ';\n')
        f.write('const int xstart = ' + str(startX) + ';\n')
        f.write('const int ystart = ' + str(startY) + ';\n')
        f.write('const int xtarget = ' + str(targetX) + ';\n')
        f.write('const int ytarget = ' + str(targetY) + ';\n')
        f.write('const double p = ' + str(p) + ';\n\n')

        f.write('formula hasCrashed = (1=0) ')
        for x, y in obstacles:
            f.write('| (x={0} & y={1}) '.format(x, y))
        f.write(';\n\n')

        # Full Top-4 state is represented by belief_state and b1..b4/other.
        # Gini uncertainty uses all five masses. Values are integer-scaled
        # to avoid floating-point/logarithmic calculations in PRISM.
        f.write(
            'formula belief_uncertainty = 10000 - ' 
            '(b1*b1 + b2*b2 + b3*b3 + b4*b4 + other*other);\n'
        )
        f.write(
            'formula update_required = '
            '(belief_uncertainty>=max_belief_uncertainty) | '
            '(belief_age>=MAX_BELIEF_STEPS);\n\n'
        )

        f.write('// Abstract Top-4 belief-state catalogue:\n')
        for state_id in sorted(belief_states):
            b1, b2, b3, b4, other = belief_states[state_id]["signature"]
            f.write(
                f'// belief_state {state_id}: '
                f'{b1}/{b2}/{b3}/{b4}/{other}\n'
            )
        f.write('\n')


def robot():
    with open(prism_file, 'a') as f:
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
        f.write('  [check] (move_ready=0) & hasCrashed -> '
                '(crashed\'=1) & (move_ready\'=1); \n')
        f.write('  [check] (move_ready=0) & !hasCrashed -> '
                '(move_ready\'=1); \n')
        f.write('endmodule\n\n')


def adaptation_mape_controller(d):
    # Unchanged: Dijkstra/MAPE still uses xhat,yhat.
    with open(prism_file, 'a') as f:
        f.write('module Adaptation_MAPE_controller\n')
        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])
                if direction < 4:
                    f.write('  [{0}] '.format(directions[direction]))
                    f.write(
                        '(xhat={0}) & (yhat={1}) -> true;\n'.format(
                            x, y
                        )
                    )
        f.write('endmodule\n\n')


def _knowledge_movement_effect(direction):
    if direction == "east":
        return "(xhat'=min(xhat+1, N))"
    if direction == "west":
        return "(xhat'=max(xhat-1, 0))"
    if direction == "north":
        return "(yhat'=min(yhat+1, N))"
    if direction == "south":
        return "(yhat'=max(yhat-1, 0))"
    raise ValueError(direction)


def knowledge():
    max_state = max(belief_states)

    with open(prism_file, 'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n')
        f.write('  yhat : [0..N] init ystart;\n')

        f.write(f'  belief_state : [0..{max_state}] init 0;\n')

        f.write('  b1 : [0..100] init 100;\n')
        f.write('  b2 : [0..100] init 0;\n')
        f.write('  b3 : [0..100] init 0;\n')
        f.write('  b4 : [0..100] init 0;\n')
        f.write('  other : [0..100] init 0;\n')

        # Internal history only. It is not a URC policy dimension.
        f.write(f'  cnt_e : [0..{max_belief_steps}] init 0;\n')
        f.write(f'  cnt_w : [0..{max_belief_steps}] init 0;\n')
        f.write(f'  cnt_n : [0..{max_belief_steps}] init 0;\n')
        f.write(f'  cnt_s : [0..{max_belief_steps}] init 0;\n')
        f.write(f'  belief_age : [0..{max_belief_steps}] init 0;\n')

        f.write('  ready : [0..1] init 1;\n\n')

        counter_names = ("cnt_e", "cnt_w", "cnt_n", "cnt_s")
        dir_to_index = {
            "east": 0,
            "west": 1,
            "north": 2,
            "south": 3,
        }

        for counts in sorted(count_to_state, key=lambda c: (sum(c), c)):
            age = sum(counts)
            if age >= max_belief_steps:
                continue

            guard_counts = ' & '.join(
                f'{name}={value}'
                for name, value in zip(counter_names, counts)
            )

            for direction in BELIEF_DIRECTIONS:
                transition = belief_transitions[(counts, direction)]
                next_counts = transition["next_counts"]
                next_state = transition["next_state"]
                b1, b2, b3, b4, other = transition["signature"]

                updates_parts = [
                    _knowledge_movement_effect(direction),
                    f"(belief_state'={next_state})",
                    f"(b1'={b1})",
                    f"(b2'={b2})",
                    f"(b3'={b3})",
                    f"(b4'={b4})",
                    f"(other'={other})",
                    f"(belief_age'={age + 1})",
                ]

                changed_index = dir_to_index[direction]
                updates_parts.append(
                    f"({counter_names[changed_index]}'="
                    f"{next_counts[changed_index]})"
                )
                updates_parts.append("(ready'=0)")

                f.write(
                    f'  [{direction}] ready=1 & belief_age={age} & '
                    f'{guard_counts} ->\n'
                    f'    ' + ' &\n    '.join(updates_parts) + ';\n'
                )

        # Perfect Ground-Truth update.
        f.write('\n')
        f.write('  [update] update_required & ready=0 ->\n')
        f.write("    (xhat'=x) & (yhat'=y) &\n")
        f.write("    (belief_state'=0) &\n")
        f.write("    (b1'=100) & (b2'=0) & (b3'=0) & (b4'=0) & (other'=0) &\n")
        f.write("    (cnt_e'=0) & (cnt_w'=0) & (cnt_n'=0) & (cnt_s'=0) &\n")
        f.write("    (belief_age'=0) & (ready'=1);\n")

        f.write(
            '  [skip_update] !update_required & ready=0 -> '
            '(ready\'=1);\n'
        )
        f.write('endmodule\n\n')


def rewards():
    # Same objectives as the point-estimate and interval models.
    with open(prism_file, 'a') as f:
        f.write('rewards "cost" \n')
        f.write('  [east] true : 1; \n')
        f.write('  [west] true : 1; \n')
        f.write('  [north] true : 1; \n')
        f.write('  [south] true : 1; \n')
        f.write(f'  [update] true : {updates[0]};\n')
        f.write('endrewards \n\n')


def read_params_from_file():
    with open('input.json', 'r') as file:
        params = json.load(file)

    global startX, startY, targetX, targetY
    global map_file, p, updates, max_belief_steps
    global default_belief_threshold

    startX = params["startX"]
    startY = params["startY"]
    targetX = params["targetX"]
    targetY = params["targetY"]
    p = params["p"]
    map_file = params["map_file"]
    updates = params["updates"]

    max_belief_steps = params.get(
        "max_belief_steps",
        max_belief_steps,
    )

    default_belief_threshold = params.get(
        "default_belief_threshold",
        default_belief_threshold,
    )


def generate_model(i):
    global prism_file

    prism_file = (
        "Applications/EvoChecker-master/models/model_"
        + str(i)
        + ".prism"
    )

    read_params_from_file()
    build_map("maps/map_" + str(i) + ".csv")
    build_beliefs()

    target_pos = (targetX, targetY)
    _d = dijkstra.compute_directions(map_data, target_pos)
    d = list(zip(*_d))

    open(prism_file, "w").close()
    preambel()
    robot()
    adaptation_mape_controller(d)
    knowledge()
    rewards()

    print(
        f"finished map {i}: "
        f"{len(belief_states)} abstract Top-4 belief states "
        f"up to {max_belief_steps} movements"
    )


if __name__ == "__main__":
    # Local generator test.
    generate_model(10)
