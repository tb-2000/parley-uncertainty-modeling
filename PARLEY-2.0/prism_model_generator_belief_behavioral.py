import csv
import json
import dijkstra
from exact_reachable_belief_model_behavioral import build_exact_belief_model

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

        # Same discrete URC control variable as in the original
        # point-estimate model. urc_synthesis_belief_full_stages_grouped.py
        # removes this constant and replaces it by c:[1..10].
        f.write(f'const int c = {period};\n')

        # Numerical Gini thresholds are map-specific.
        f.write('// Map-specific belief-uncertainty thresholds:\n')
        for index, threshold in enumerate(
            thresholds,
            start=1,
        ):
            f.write(
                f'// c={index} -> '
                f'max_belief_uncertainty={threshold}\n'
            )
            f.write(
                f'const int belief_threshold_{index} = '
                f'{threshold};\n'
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
                '| (x={0} & y={1}) '.format(
                    str(x),
                    str(y),
                )
            )
        f.write(';\n\n')

        # Group every exact reachable knowledge context by the highest
        # map-specific threshold it has reached.
        #
        # class 0: uncertainty < threshold_1
        # class 1: threshold_1 <= uncertainty < threshold_2
        # ...
        # class 10: uncertainty >= threshold_10
        #
        # Only classes 1..10 are relevant for update_required.
        belief_classes = {
            stage: []
            for stage in range(0, 11)
        }

        # Behavioral quotient classes preserve the URC stage exactly.
        stages = belief_model.get("stages")

        if stages is None:
            # Backward-compatible fallback.
            stages = []
            for uncertainty in belief_model["uncertainties"]:
                reached_stage = 0
                for index, threshold in enumerate(
                    thresholds,
                    start=1,
                ):
                    if uncertainty >= threshold:
                        reached_stage = index
                stages.append(reached_stage)

        for state_id, reached_stage in enumerate(stages):
            belief_classes[reached_stage].append(state_id)

        # Write at most ten formulas for the update-relevant classes.
        # Empty classes are omitted.
        written_stages = []

        for stage in range(1, 11):
            state_ids = belief_classes[stage]

            if not state_ids:
                continue

            written_stages.append(stage)

            f.write(
                f'formula belief_u_{stage} = '
            )
            f.write(
                ' | '.join(
                    f'kstate={state_id}'
                    for state_id in state_ids
                )
            )
            f.write(';\n')

        # Optional documentation for completely certain / low-uncertainty
        # states that do not reach even threshold 1.
        if belief_classes[0]:
            f.write(
                '// Knowledge states below threshold 1 '
                '(never trigger update by uncertainty): '
            )
            f.write(
                ','.join(
                    str(state_id)
                    for state_id in belief_classes[0]
                )
            )
            f.write('\n')

        # Compact stage-based update formula.
        #
        # A belief in class j has reached thresholds 1..j, hence an update
        # is required iff the URC-selected stage c is <= j.
        f.write('formula update_required = ')

        update_terms = []

        for stage in written_stages:
            if stage < 10:
                update_terms.append(
                    f'(belief_u_{stage} & c<={stage})'
                )
            else:
                # Since c is restricted to 1..10, c<=10 is always true.
                update_terms.append(
                    f'belief_u_{stage}'
                )

        if update_terms:
            f.write(' | '.join(update_terms))
        else:
            f.write('false')

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
        # xhat/yhat are encoded inside kstate. Expose one compact formula for
        # every estimate position so the MAPE controller and URC can keep their
        # original position-dependent semantics.
        for x in range(mapSize):
            for y in range(mapSize):
                ids = belief_model["position_contexts"][f"{x},{y}"]
                if ids:
                    f.write(
                        f'formula estimate_{x}_{y} = '
                        + ' | '.join(f'kstate={sid}' for sid in ids)
                        + ';\n'
                    )
        f.write('\nmodule Adaptation_MAPE_controller\n')
        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])
                if direction < 4:
                    f.write(
                        f'  [{directions[direction]}] '
                        f'estimate_{x}_{y} -> true;\n'
                    )
        f.write('endmodule\n\n')


def knowledge():
    with open(prism_file, 'a') as f:
        initial_context = belief_model["certainty_contexts"][
            f"{startX},{startY}"
        ]

        f.write('module Knowledge\n')
        f.write(
            f'  kstate : [0..{belief_model["state_count"] - 1}] '
            f'init {initial_context};\n'
        )
        f.write('  ready : [0..1] init 1;\n\n')

        # One exact transition per reachable compact knowledge context.
        for context_id, transition in belief_model["transitions"].items():
            action = transition["action"]
            next_context = transition["next_context"]

            f.write(
                f'  [{action}] ready=1 & kstate={context_id} -> '
                f"(kstate'={next_context}) & (ready'=0);\n"
            )

        # Perfect localization.
        # The first (N+1)^2 kstates are certainty contexts in row-major order:
        #   certainty_kstate(x,y) = x*(N+1)+y
        # Therefore one update command is sufficient.
        f.write(
            '  [update] update_required & ready=0 -> '
            "(kstate'=x*(N+1)+y) & (ready'=1);\n"
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

    # The compact single-update encoding relies on certainty contexts
    # occupying IDs 0..(N+1)^2-1 in row-major order.
    for x in range(mapSize):
        for y in range(mapSize):
            expected = x * mapSize + y
            actual = belief_model["certainty_contexts"][f"{x},{y}"]
            if actual != expected:
                raise AssertionError(
                    f"Invalid certainty kstate for ({x},{y}): "
                    f"{actual}, expected {expected}"
                )

    exact_count = belief_model["exact_context_count"]
    reduced_count = belief_model["state_count"]

    print(
        f"map {i}: exact contexts={exact_count}, "
        f"behavioral classes={reduced_count}, "
        f"reduction={100.0 * (exact_count - reduced_count) / exact_count:.2f}%, "
        f"distinct relative beliefs={belief_model['exact_belief_count']}"
    )

    open(prism_file, "w").close()
    preambel()
    robot()
    adaptation_mape_controller(d)
    knowledge()
    rewards()

    print("finished map " + str(i))
