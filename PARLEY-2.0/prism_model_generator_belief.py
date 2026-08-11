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

# The B2 model keeps an abstract Top-4 belief for at most this many
# movements since the last perfect Ground-Truth update.
max_belief_steps = 10

# Placeholder used before urc_synthesis_belief.py adds the real URC module.
# 0 = skip update, 1 = update.
default_update_decision = 0

directions = ['west', 'east', 'south', 'north']
obstacles = []

updates = [5]  # cost of updates

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)

prism_file = ""

# Filled by build_beliefs().
belief_states = {}
count_to_state = {}
belief_transitions = {}


def build_map(filename):
    n = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            n.append(row)

    global mapSize
    global map_data
    mapSize = len(n)
    transposed = list(zip(*n))
    map_data = [row[::-1] for row in transposed]

    global obstacles
    obstacles = []
    for x in range(0, mapSize):
        for y in range(0, mapSize):
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

        # urc_synthesis_belief.py removes this constant and replaces it by
        # the variable urc_update in the generated URC module.
        f.write(f'const int urc_update = {default_update_decision};\n')

        f.write('const int N=' + str(mapSize - 1) + ';\n')
        f.write('const int xstart = ' + str(startX) + ';\n')
        f.write('const int ystart = ' + str(startY) + ';\n')
        f.write('const int xtarget = ' + str(targetX) + ';\n')
        f.write('const int ytarget = ' + str(targetY) + ';\n')
        f.write('const double p = ' + str(p) + ';\n\n')

        f.write('formula hasCrashed = (1=0) ')
        for x, y in obstacles:
            f.write('| (x={0} & y={1}) '.format(str(x), str(y)))
        f.write(';\n\n')

        f.write('formula belief_uncertainty = 100-b1;\n')
        f.write('formula update_required = (urc_update=1) | (belief_age>=MAX_BELIEF_STEPS);\n\n')

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
    # Unchanged: the shortest-path MAPE controller still uses xhat/yhat.
    with open(prism_file, 'a') as f:
        f.write('module Adaptation_MAPE_controller\n')
        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])
                if direction < 4:
                    f.write('  [{0}] '.format(directions[direction]))
                    f.write(
                        '(xhat={0}) & (yhat={1}) -> true;\n'.format(
                            str(x), str(y)
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

        # This is the Knowledge feature that the B2 URC will use.
        f.write(f'  belief_state : [0..{max_state}] init 0;\n')

        # The actual discrete Top-4 masses are kept explicitly so that
        # b2, b3, b4 and other remain part of the model state.
        f.write('  b1 : [0..100] init 100;\n')
        f.write('  b2 : [0..100] init 0;\n')
        f.write('  b3 : [0..100] init 0;\n')
        f.write('  b4 : [0..100] init 0;\n')
        f.write('  other : [0..100] init 0;\n')

        # Internal history only. These variables are NOT URC policy inputs.
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

        # Generate the finite abstract belief automaton.  The guards use
        # only the internal count vector; the resulting abstract state is
        # the position- and direction-independent Top-4 distribution class.
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
                    f"({counter_names[changed_index]}'={next_counts[changed_index]})"
                )
                updates_parts.append("(ready'=0)")

                f.write(
                    f'  [{direction}] ready=1 & belief_age={age} & '
                    f'{guard_counts} ->\n'
                    f'    ' + ' &\n    '.join(updates_parts) + ';\n'
                )

        # Perfect Ground-Truth observation.  At max_belief_steps the update
        # is forced so the finite belief abstraction cannot overflow.
        f.write('\n')
        f.write('  [update] update_required & ready=0 ->\n')
        f.write("    (xhat'=x) & (yhat'=y) &\n")
        f.write("    (belief_state'=0) &\n")
        f.write("    (b1'=100) & (b2'=0) & (b3'=0) & (b4'=0) & (other'=0) &\n")
        f.write("    (cnt_e'=0) & (cnt_w'=0) & (cnt_n'=0) & (cnt_s'=0) &\n")
        f.write("    (belief_age'=0) & (ready'=1);\n")

        f.write('  [skip_update] !update_required & ready=0 -> (ready\'=1);\n')
        f.write('endmodule\n\n')


def rewards():
    # Same objectives as point-estimate and interval models:
    # maximize mission success and minimize this cost reward.
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

    global startX, startY, targetX, targetY, map_file, p, updates, max_belief_steps

    startX = params["startX"]
    startY = params["startY"]
    targetX = params["targetX"]
    targetY = params["targetY"]
    p = params["p"]
    map_file = params["map_file"]
    updates = params["updates"]

    # Optional override, default remains 10 to match the former c range.
    max_belief_steps = params.get("max_belief_steps", max_belief_steps)


def generate_model(i):
    global prism_file

    prism_file = "Applications/EvoChecker-master/models/model_" + str(i) + ".prism"

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
    generate_model(10)
