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

obstacles = []
updates = [5]

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)

prism_file = ""
period = 1

HMM_BELIEF_ROOT = "hmm_belief_models"


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


def hmm_belief_path(map_id):
    return os.path.join(
        HMM_BELIEF_ROOT,
        f"map_{map_id}",
        "exact_local.json",
    )


def load_exact_local_hmm_beliefs(map_id):
    path = hmm_belief_path(map_id)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing exact local HMM model: {path}\n"
            "Run full_hmm_belief_exact_local.py first."
        )

    with open(path, 'r') as f:
        data = json.load(f)

    if data.get("mode") != "exact_local_hmm_beliefs":
        raise ValueError(
            f"{path} is not an exact local HMM-belief model."
        )

    if int(data["map_id"]) != int(map_id):
        raise ValueError(
            f"{path}: map_id={data['map_id']} does not match map {map_id}."
        )

    if len(data.get("thresholds", [])) != 10:
        raise ValueError(
            f"{path}: expected exactly ten HMM-MSE thresholds."
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

        thresholds = [
            float(v)
            for v in hmm["thresholds"]
        ]

        f.write(
            '// HMM uncertainty level -> raw expected-squared-error threshold\n'
        )

        for level, threshold in enumerate(
            thresholds,
            start=1,
        ):
            f.write(
                f'// level {level} -> HMM-MSE >= {threshold:.17g}\n'
            )

        f.write('\n')

        f.write('formula hasCrashed = (1=0) ')
        for x, y in obstacles:
            f.write(
                '| (x={0} & y={1}) '.format(
                    str(x),
                    str(y),
                )
            )
        f.write(';\n\n')

        # hmm_state is a POSITION-LOCAL exact-belief ID, not a threshold level.
        # Therefore update formulas must include xhat/yhat/hmm_state.
        classes = {
            stage: []
            for stage in range(11)
        }

        for position, levels in hmm["urc_levels"].items():
            xhat, yhat = [
                int(v)
                for v in position.split(',')
            ]

            for state_id, level in enumerate(levels):
                classes[int(level)].append(
                    f'(xhat={xhat} & yhat={yhat} '
                    f'& hmm_state={state_id})'
                )

        written_stages = []

        for stage in range(1, 11):
            terms = classes[stage]

            if not terms:
                continue

            written_stages.append(stage)
            f.write(
                f'formula hmm_u_{stage} = '
                + ' | '.join(terms)
                + ';\n'
            )

        update_terms = []

        for stage in written_stages:
            if stage < 10:
                update_terms.append(
                    f'(hmm_u_{stage} & c<={stage})'
                )
            else:
                update_terms.append(
                    'hmm_u_10'
                )

        f.write(
            'formula update_required = '
            + (
                ' | '.join(update_terms)
                if update_terms
                else 'false'
            )
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
                    f.write(
                        f'  [{directions[direction]}] '
                        f'(xhat={x}) & (yhat={y}) -> true;\n'
                    )

        f.write('endmodule\n\n')


def knowledge_hmm(hmm):
    """
    Exact local HMM Knowledge state:
        (xhat, yhat, hmm_state)

    hmm_state is a local ID of one exact 361-dimensional HMM belief beta.
    """
    transitions = hmm.get(
        "belief_transitions",
        {},
    )

    max_hmm_state = int(
        hmm["max_hmm_state"]
    )

    with open(prism_file, 'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n')
        f.write('  yhat : [0..N] init ystart;\n')
        f.write(
            f'  hmm_state : [0..{max_hmm_state}] init 0;\n'
        )
        f.write('  ready : [0..1] init 1;\n\n')

        def sort_key(item):
            key, value = item
            xhat, yhat, state_id = [
                int(v)
                for v in key.split(',')
            ]

            return (
                xhat,
                yhat,
                state_id,
                directions.index(value["action"]),
            )

        for key, value in sorted(
            transitions.items(),
            key=sort_key,
        ):
            xhat, yhat, state_id = [
                int(v)
                for v in key.split(',')
            ]

            action = value["action"]
            nxhat = int(value["next_xhat"])
            nyhat = int(value["next_yhat"])
            next_state = int(value["next_state"])

            f.write(
                f'  [{action}] ready=1'
                f' & xhat={xhat}'
                f' & yhat={yhat}'
                f' & hmm_state={state_id}'
                f' -> (xhat\'={nxhat})'
                f' & (yhat\'={nyhat})'
                f' & (hmm_state\'={next_state})'
                f' & (ready\'=0);\n'
            )

        f.write('\n')

        # beta <- pi. Local state 0 denotes pi at EVERY position.
        f.write(
            '  [update] ready=0 & update_required -> '
            '(xhat\'=x) & (yhat\'=y) & '
            '(hmm_state\'=0) & '
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

    global startX, startY, targetX, targetY
    global map_file, p, updates

    startX = params["startX"]
    startY = params["startY"]
    targetX = params["targetX"]
    targetY = params["targetY"]
    p = params["p"]
    map_file = params["map_file"]
    updates = params["updates"]


def generate_model(i):
    global prism_file

    prism_file = (
        "Applications/EvoChecker-master/models/model_"
        + str(i)
        + ".prism"
    )

    read_params_from_file()
    build_map(
        "maps/map_" + str(i) + ".csv"
    )

    hmm = load_exact_local_hmm_beliefs(i)

    target_pos = (
        targetX,
        targetY,
    )

    _d = dijkstra.compute_directions(
        map_data,
        target_pos,
    )
    d = list(zip(*_d))

    open(
        prism_file,
        "w",
    ).close()

    preambel(hmm)
    robot()
    adaptation_mape_controller(d)
    knowledge_hmm(hmm)
    rewards()

    # Structural checks: exact local model must have xhat/yhat/hmm_state,
    # but neither global hstate nor substate.
    with open(
        prism_file,
        "r",
        encoding="utf-8",
    ) as generated_file:
        generated = generated_file.read()

    required = [
        "  xhat : [0..N] init xstart;",
        "  yhat : [0..N] init ystart;",
        "  hmm_state :",
        "  ready : [0..1] init 1;",
    ]

    for fragment in required:
        if fragment not in generated:
            raise ValueError(
                f"{prism_file}: missing required fragment {fragment}"
            )

    forbidden = [
        "  substate :",
        "  hstate :",
    ]

    for fragment in forbidden:
        if fragment in generated:
            raise ValueError(
                f"{prism_file}: obsolete HMM structure found: {fragment}"
            )

    print(
        f"finished map {i}: "
        f"exact contexts={hmm['exact_context_count']}, "
        f"exact HMM beliefs={hmm['exact_belief_count']}, "
        f"max local HMM states={hmm['max_local_states']}, "
        f"reachable hidden states="
        f"{hmm['reachable_hidden_state_count']}/"
        f"{hmm['hidden_state_count_full']}"
    )
