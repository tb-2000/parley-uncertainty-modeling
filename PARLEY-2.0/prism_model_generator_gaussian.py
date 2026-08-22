import csv
import json
import os
import dijkstra

startX = 0
startY = 0
targetX = 4
targetY = 4
p = 0.01

directions = [
    'west',
    'east',
    'south',
    'north'
]

obstacles = []
updates = [5]

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)
corridor = 1

prism_file = ""
period = 1

# Output directory of refine_gaussian_markov_states.py
gaussian_refined_dir = "gaussian_refined"
gaussian_trace_dir = "gaussian_trace"

gaussian_gvars = []
gaussian_gstates = []
gaussian_lookup = []

gvar_max = 0
gstate_max = 0

# Position-dependent gstate used after a perfect update.
reset_gstate_by_position = {}
gaussian_trace_thresholds = []
gaussian_trace_groups = {}


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
    map_data = [
        row[::-1]
        for row in transposed
    ]

    global obstacles
    obstacles = []

    for x in range(0, mapSize):
        for y in range(0, mapSize):
            if int(map_data[x][y]) > 9:
                obstacles.append([x, y])


def load_gaussian_refined(map_id):
    """
    Loads:
        gaussian_refined/gaussian_refined_<map>.json

    Required content:
      gvars:
        gvar -> quantized Sigma
      gstates:
        gstate -> gvar
      lookup:
        (xhat,yhat,gstate,action)
          -> (xhat_next,yhat_next,gstate_next,gvar_next)

    gvar is the URC decision variable.
    gstate is only used for Markov-compatible Knowledge transitions.
    """
    path = os.path.join(
        gaussian_refined_dir,
        f"gaussian_refined_{map_id}.json"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Refined Gaussian file not found: {path}. "
            f"Run refine_gaussian_markov_states.py first."
        )

    with open(path, 'r') as file:
        data = json.load(file)

    gvars = data.get("gvars", [])
    gstates = data.get("gstates", [])
    lookup = data.get("lookup", [])

    if not gvars:
        raise ValueError(
            f"{path} contains no gvars."
        )

    if not gstates:
        raise ValueError(
            f"{path} contains no gstates."
        )

    # Validate state IDs and gstate->gvar mapping.
    gvar_ids = {
        int(row["gvar"])
        for row in gvars
    }

    gstate_to_gvar = {}

    for row in gstates:
        gstate = int(row["gstate"])
        gvar = int(row["gvar"])

        if gvar not in gvar_ids:
            raise ValueError(
                f"{path}: gstate {gstate} references "
                f"unknown gvar {gvar}."
            )

        gstate_to_gvar[gstate] = gvar

    # After update Sigma=0 => gvar=0, but the refined gstate may
    # depend on the actual robot position. Load the explicit reset mapping
    # produced by refine_gaussian_markov_states_fixed.py.
    reset_mapping_raw = data.get("reset_gstate_by_position", {})

    if not reset_mapping_raw:
        raise ValueError(
            f"{path}: missing 'reset_gstate_by_position'. "
            "Regenerate gaussian_refined files with "
            "refine_gaussian_markov_states_fixed.py."
        )

    reset_mapping = {}

    for key, gstate in reset_mapping_raw.items():
        x_str, y_str = key.split(",")
        pos = (int(x_str), int(y_str))
        gstate = int(gstate)

        if gstate not in gstate_to_gvar:
            raise ValueError(
                f"{path}: reset gstate {gstate} for position {pos} "
                "is not a known gstate."
            )

        if gstate_to_gvar[gstate] != 0:
            raise ValueError(
                f"{path}: reset gstate {gstate} for position {pos} "
                f"maps to gvar={gstate_to_gvar[gstate]}, expected 0."
            )

        reset_mapping[pos] = gstate

    # Validate deterministic Markov lookup.
    transitions = {}

    for row in lookup:
        source = (
            int(row["xhat"]),
            int(row["yhat"]),
            int(row["gstate"]),
            str(row["action"])
        )

        successor = (
            int(row["xhat_next"]),
            int(row["yhat_next"]),
            int(row["gstate_next"])
        )

        if source in transitions and transitions[source] != successor:
            raise ValueError(
                f"{path}: refined lookup is still ambiguous: "
                f"{source} -> {transitions[source]} / {successor}"
            )

        transitions[source] = successor

    # Remove exact duplicate rows.
    dedup_lookup = []
    seen = set()

    for row in lookup:
        key = (
            int(row["xhat"]),
            int(row["yhat"]),
            int(row["gstate"]),
            str(row["action"]),
            int(row["xhat_next"]),
            int(row["yhat_next"]),
            int(row["gstate_next"]),
            int(row["gvar_next"])
        )

        if key in seen:
            continue

        seen.add(key)
        dedup_lookup.append(row)

    global gaussian_gvars
    global gaussian_gstates
    global gaussian_lookup
    global gvar_max
    global gstate_max
    global reset_gstate_by_position

    gaussian_gvars = gvars
    gaussian_gstates = gstates
    gaussian_lookup = dedup_lookup

    gvar_max = max(gvar_ids)
    gstate_max = max(gstate_to_gvar)
    reset_gstate_by_position = reset_mapping



def load_gaussian_trace(map_id):
    path = os.path.join(
        gaussian_trace_dir,
        f"gaussian_trace_{map_id}.json"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gaussian trace file not found: {path}. "
            "Run compute_gaussian_trace_thresholds.py first."
        )

    with open(path, 'r') as file:
        data = json.load(file)

    thresholds_raw = data.get("thresholds", {})
    thresholds = [
        int(thresholds_raw[str(i)])
        for i in range(1, 11)
    ]

    if any(
        thresholds[i] > thresholds[i + 1]
        for i in range(len(thresholds) - 1)
    ):
        raise ValueError(
            f"{path}: Gaussian trace thresholds are not monotonic."
        )

    groups_raw = data.get("trace_groups", {})
    groups = {
        int(trace_value): [int(g) for g in gvars]
        for trace_value, gvars in groups_raw.items()
    }

    if not groups:
        raise ValueError(
            f"{path}: no trace_groups found."
        )

    global gaussian_trace_thresholds
    global gaussian_trace_groups

    gaussian_trace_thresholds = thresholds
    gaussian_trace_groups = groups

def preambel():
    with open(prism_file, 'a') as f:
        f.write('dtmc\n')

        for i, threshold in enumerate(
                gaussian_trace_thresholds, start=1):
            f.write(
                f'const int gaussian_threshold_{i} = '
                f'{threshold};\n'
            )

        f.write('const int N=' + str(mapSize - 1) + ';\n')
        f.write('const int xstart = ' + str(startX) + ';\n')
        f.write('const int ystart = ' + str(startY) + ';\n')
        f.write('const int xtarget = ' + str(targetX) + ';\n')
        f.write('const int ytarget = ' + str(targetY) + ';\n')
        f.write('const double p = ' + str(p) + ';\n')
        f.write('const int GVAR_MAX = ' + str(gvar_max) + ';\n')
        f.write('const int GSTATE_MAX = ' + str(gstate_max) + ';\n\n')

        f.write('formula hasCrashed = (1=0) ')
        for x, y in obstacles:
            f.write('| (x={0} & y={1}) '.format(str(x), str(y)))
        f.write(';\n\n')

        sorted_trace_values = sorted(gaussian_trace_groups.keys())

        for index, trace_value in enumerate(sorted_trace_values):
            gvars = gaussian_trace_groups[trace_value]
            expression = ' | '.join(
                f'gvar={gvar}' for gvar in gvars
            )
            f.write(
                f'formula gaussian_u_{index} = '
                f'{expression};\n'
            )

        terms = []
        for index, trace_value in enumerate(sorted_trace_values):
            terms.append(
                f'(gaussian_u_{index} & '
                f'max_gaussian_uncertainty<={trace_value})'
            )

        f.write(
            'formula update_required = '
            + ' | '.join(terms)
            + ';\n\n'
        )

        f.write('// Gaussian covariance quantization: h=0.1\n')
        f.write('// trace(Sigma)=var_x+var_y, integer-scaled offline\n')
        f.write('// gstate = refined Markov state; gvar = quantized covariance class\n\n')


def robot():
    with open(prism_file, 'a') as f:
        f.write('module Robot \n')
        f.write(
            '  x : [0..N] init xstart;\n'
        )
        f.write(
            '  y : [0..N] init ystart;\n'
        )
        f.write(
            '  move_ready : [0..1] init 1;\n'
        )
        f.write(
            '  crashed : [0..1] init 0;\n\n'
        )

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
        f.write(
            'module Adaptation_MAPE_controller\n'
        )

        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])

                if direction < 4:
                    f.write(
                        '  [{0}] '.format(
                            directions[direction]
                        )
                    )

                    f.write(
                        '(xhat={0}) & (yhat={1}) -> true;\n'.format(
                            str(x),
                            str(y)
                        )
                    )

        f.write('endmodule\n\n')


def knowledge():
    """
    Refined Gaussian Knowledge module with trace-based uncertainty updates.

    The URC chooses a position-dependent Gaussian trace threshold.
    The current quantized covariance trace is compared with this threshold.
    A hard cap at 10 moves is retained because the refined lookup was
    constructed for at most 10 steps since the last update.
    """
    with open(prism_file, 'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n')
        f.write('  yhat : [0..N] init ystart;\n')

        start_reset_gstate = reset_gstate_by_position.get((startX, startY))
        if start_reset_gstate is None:
            raise ValueError(
                f'No reset gstate for start position ({startX},{startY}).'
            )

        f.write(
            '  gstate : [0..GSTATE_MAX] init '
            + str(start_reset_gstate)
            + ';\n'
        )
        f.write('  gvar : [0..GVAR_MAX] init 0;\n')
        f.write('  step : [1..10] init 1;\n\n')
        f.write('  ready : [0..1] init 1;\n\n')

        f.write('  // Refined Gaussian Markov transitions (h=0.1)\n')

        for row in gaussian_lookup:
            action = str(row["action"])
            xhat = int(row["xhat"])
            yhat = int(row["yhat"])
            gstate = int(row["gstate"])
            xhat_next = int(row["xhat_next"])
            yhat_next = int(row["yhat_next"])
            gstate_next = int(row["gstate_next"])
            gvar_next = int(row["gvar_next"])

            f.write(
                f'  [{action}] ready=1 '
                f'& xhat={xhat} & yhat={yhat} & gstate={gstate} -> '
                f'(xhat\'={xhat_next}) & '
                f'(yhat\'={yhat_next}) & '
                f'(gstate\'={gstate_next}) & '
                f'(gvar\'={gvar_next}) & '
                f'(ready\'=0);\n'
            )

        f.write('\n')

        for (reset_x, reset_y), reset_state in sorted(
                reset_gstate_by_position.items()):
            f.write(
                f'  [update] ready=0 & (update_required | step>=10) '
                f'& x={reset_x} & y={reset_y} -> '
                f'(xhat\'={reset_x}) & '
                f'(yhat\'={reset_y}) & '
                f'(gstate\'={reset_state}) & '
                f'(gvar\'=0) & '
                f'(step\'=1) & '
                f'(ready\'=1);\n'
            )

        f.write(
            '  [skip_update] ready=0 & !update_required & step<10 -> '
            '(ready\'=1) & (step\'=step+1);\n'
        )

        f.write('endmodule\n\n')


def rewards():
    with open(prism_file, 'a') as f:
        f.write('rewards "cost" \n')
        f.write(
            '  [east] true : 1; \n'
        )
        f.write(
            '  [west] true : 1; \n'
        )
        f.write(
            '  [north] true : 1; \n'
        )
        f.write(
            '  [south] true : 1; \n'
        )
        f.write(
            '  [update] true : 5;\n'
        )
        f.write('endrewards \n\n')


def read_params_from_file():
    with open('input.json', 'r') as file:
        params = json.load(file)

    global startX
    global startY
    global targetX
    global targetY
    global map_file
    global p
    global updates

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
        "maps/map_"
        + str(i)
        + ".csv"
    )

    load_gaussian_refined(i)
    load_gaussian_trace(i)

    target_pos = (
        targetX,
        targetY
    )

    _d = dijkstra.compute_directions(
        map_data,
        target_pos
    )

    d = list(zip(*_d))

    open(
        prism_file,
        "w"
    ).close()

    preambel()
    robot()
    adaptation_mape_controller(d)
    knowledge()
    rewards()

    print(
        "finished map "
        + str(i)
        + " with "
        + str(gvar_max + 1)
        + " gvars, "
        + str(gstate_max + 1)
        + " refined gstates and "
        + str(len(gaussian_lookup))
        + " Gaussian transitions"
    )
