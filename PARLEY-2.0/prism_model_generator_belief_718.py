import csv
import json
import dijkstra
from minimize_top4_belief_states import build_minimized_belief_automaton

startX = 0
startY = 0
targetX = 4
targetY = 4
p = 0.01
directions = ['west', 'east', 'south', 'north']
directions_effects = ['(xhat\'=max(xhat-1, 0))', '(xhat\'=min(xhat+1, N))', '(yhat\'=max(yhat-1, 0))', '(yhat\'=min(yhat+1, N))']
obstacles = []

updates = [5]  # cost of updates

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)
corridor = 1

prism_file = ""
# period of updates
period = 1
max_belief_steps = 10
default_belief_threshold = 2552
belief_automaton = None
BELIEF_THRESHOLDS = [588, 1154, 1698, 2218, 2552, 3034, 3342, 3642, 4076, 4354]


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


def build_beliefs():
    global belief_automaton
    belief_automaton = build_minimized_belief_automaton(
        max_steps=max_belief_steps,
        p=p
    )


def preambel():
    with open(prism_file, 'a') as f:
        f.write('dtmc\n')
        f.write(f'const int max_belief_uncertainty = {default_belief_threshold};\n')
        f.write(f'const int MAX_BELIEF_STATE = {max(belief_automaton["states"])};\n')
        f.write('const int N=' + str(mapSize - 1) + ';\n')
        f.write('const int xstart = ' + str(startX) + ';\n')
        f.write('const int ystart = ' + str(startY) + ';\n')
        f.write('const int xtarget = ' + str(targetX) + ';\n')
        f.write('const int ytarget = ' + str(targetY) + ';\n')
        f.write('const double p = ' + str(p) + ';\n \n')
        # formula for obstacles
        f.write('formula hasCrashed = (1=0) ')
        for x, y in obstacles:
            f.write('| (x={0} & y={1}) '.format(str(x), str(y)))
        f.write(';\n\n')

        # Map every belief_state offline to one of 10 uncertainty classes.
        # Class 1 is the lowest uncertainty, class 10 the highest.
        class_to_states = {i: [] for i in range(1, 11)}

        for state_id, state in belief_automaton["states"].items():
            gini = state["gini"]

            belief_class = 10
            for i, threshold in enumerate(BELIEF_THRESHOLDS, start=1):
                if gini <= threshold:
                    belief_class = i
                    break

            class_to_states[belief_class].append(state_id)

        for belief_class in range(1, 11):
            f.write(f'formula belief_class_{belief_class} = ')
            states_in_class = sorted(class_to_states[belief_class])

            if states_in_class:
                f.write(' | '.join(
                    f'belief_state={state_id}'
                    for state_id in states_in_class
                ))
            else:
                f.write('false')

            f.write(';\n')

        terminal_states = [
            state_id
            for state_id, state in belief_automaton["states"].items()
            if state["terminal"]
        ]

        f.write('formula terminal_belief = ')
        if terminal_states:
            f.write(' | '.join(
                f'belief_state={state_id}'
                for state_id in sorted(terminal_states)
            ))
        else:
            f.write('false')
        f.write(';\n')

        # max_belief_uncertainty is still selected by the URC using decision_x_y.
        # We translate the selected empirical Gini threshold to a class index.
        f.write('formula max_belief_class = ')
        for i, threshold in enumerate(BELIEF_THRESHOLDS[:-1], start=1):
            f.write(
                f'(max_belief_uncertainty={threshold} ? {i} : '
            )
        f.write('10' + ')' * (len(BELIEF_THRESHOLDS) - 1) + ';\n')

        # Current belief class is encoded without introducing a new state variable.
        f.write('formula current_belief_class = ')
        for belief_class in range(1, 10):
            f.write(
                f'(belief_class_{belief_class} ? {belief_class} : '
            )
        f.write('10' + ')' * 9 + ';\n')

        f.write(
            'formula update_required = '
            '(current_belief_class>=max_belief_class) | terminal_belief;\n\n'
        )


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
        f.write('  [check] (move_ready=0) & hasCrashed -> (crashed\'=1) & (move_ready\'=1); \n')
        f.write('  [check] (move_ready=0) & !hasCrashed -> (move_ready\'=1); \n')
        f.write('endmodule\n\n')


def adaptation_mape_controller(d):
    with open(prism_file, 'a') as f:
        f.write('module Adaptation_MAPE_controller\n')
        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])
                if direction < 4:
                    f.write('  [{0}] '.format(directions[direction]))
                    f.write('(xhat={0}) & (yhat={1}) -> true;\n'.format(str(x), str(y)))
        f.write('endmodule\n\n')


def knowledge():
    with open(prism_file, 'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n')
        f.write('  yhat : [0..N] init ystart;\n')
        f.write(
            f'  belief_state : [0..{max(belief_automaton["states"])}] '
            f'init {belief_automaton["initial_state"]};\n\n'
        )
        f.write('  ready : [0..1] init 1;\n')

        for state_id in sorted(belief_automaton["states"]):
            state = belief_automaton["states"][state_id]

            if state["terminal"]:
                continue

            for direction in directions:
                successor = state["successors"][direction]

                if direction == 'west':
                    effect = "(xhat'=max(xhat-1, 0))"
                elif direction == 'east':
                    effect = "(xhat'=min(xhat+1, N))"
                elif direction == 'south':
                    effect = "(yhat'=max(yhat-1, 0))"
                else:
                    effect = "(yhat'=min(yhat+1, N))"

                f.write(
                    f'  [{direction}] ready=1 & belief_state={state_id} -> '
                    f'{effect} & (belief_state\'={successor}) & (ready\'=0);\n'
                )

        f.write(
            '  [update] update_required & ready=0 -> '
            "(xhat'=x) & (yhat'=y) & "
            f"(belief_state'={belief_automaton['initial_state']}) & "
            "(ready'=1);\n"
        )
        f.write(
            '  [skip_update] !update_required & ready=0 -> (ready\'=1);\n'
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
    global startX, startY, targetX, targetY, map_file, p, updates, max_belief_steps, default_belief_threshold
    startX = params["startX"]
    startY = params["startY"]
    targetX = params["targetX"]
    targetY = params["targetY"]
    p = params["p"]
    map_file = params["map_file"]
    updates = params["updates"]
    max_belief_steps = params.get("max_belief_steps", max_belief_steps)
    default_belief_threshold = params.get("default_belief_threshold", default_belief_threshold)


# i depicts which map should be used
def generate_model(i):
    global prism_file
    prism_file = "Applications/EvoChecker-master/models/model_" + str(i) + ".prism"
    read_params_from_file()
    build_map("maps/map_" + str(i) + ".csv")
    build_beliefs()
    target_pos = (targetX, targetY)
    _d = dijkstra.compute_directions(map_data, target_pos)
    # we have to transpose the matrix in the end, similar to what we did when reading the map_data in the first place
    d = list(zip(*_d))  # Transpose the matrix

    open(prism_file, "w").close()
    preambel()
    robot()
    adaptation_mape_controller(d)
    knowledge()
    rewards()

    print("finished map " + str(i))


if __name__ == "__main__":
    generate_model(10)
