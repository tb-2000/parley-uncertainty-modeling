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
corridor = 1

prism_file = ""
period = 1

# Output directory of build_gaussian_lookup_fixed.py
gaussian_lookup_dir = "gaussian_lookup"

gaussian_states = []
gaussian_lookup = []
gvar_max = 0


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


def load_gaussian_lookup(map_id):
    """
    Loads gaussian_lookup/gaussian_lookup_<map_id>.json.

    The lookup is expected to contain:
        gaussian_states: gvar -> quantized covariance
        lookup:
          (xhat,yhat,gvar,action)
             -> (xhat_next,yhat_next,gvar_next)

    A quantized DTMC state must be Markov. Therefore each source tuple
    (xhat,yhat,gvar,action) must have exactly one successor.
    """
    path = os.path.join(
        gaussian_lookup_dir,
        f"gaussian_lookup_{map_id}.json"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gaussian lookup file not found: {path}. "
            f"Run build_gaussian_lookup_fixed.py first."
        )

    with open(path, 'r') as f:
        data = json.load(f)

    states = data.get("gaussian_states", [])
    lookup = data.get("lookup", [])

    if not states:
        raise ValueError(
            f"{path} contains no gaussian_states."
        )

    state_by_id = {int(s["gvar"]): s for s in states}

    if 0 not in state_by_id:
        raise ValueError(f"{path}: gvar 0 is missing.")

    zero = state_by_id[0]

    if (
        abs(float(zero["var_x"])) > 1e-12
        or abs(float(zero["var_y"])) > 1e-12
        or abs(float(zero.get("cov_xy", 0.0))) > 1e-12
    ):
        raise ValueError(
            f"{path}: gvar 0 must represent Sigma=0."
        )

    # Validate unique successor per quantized source state.
    successors = {}

    for row in lookup:
        source = (
            int(row["xhat"]),
            int(row["yhat"]),
            int(row["gvar"]),
            str(row["action"])
        )
        successor = (
            int(row["xhat_next"]),
            int(row["yhat_next"]),
            int(row["gvar_next"])
        )

        if source in successors and successors[source] != successor:
            raise ValueError(
                "Ambiguous Gaussian lookup: the quantized state is "
                "not Markov. "
                f"Map {map_id}, source={source}, successors="
                f"{successors[source]} and {successor}. "
                "Use a finer quantization or add another state variable."
            )

        successors[source] = successor

    # Remove exact duplicate lookup entries.
    dedup_lookup = []
    seen = set()

    for row in lookup:
        key = (
            int(row["xhat"]),
            int(row["yhat"]),
            int(row["gvar"]),
            str(row["action"]),
            int(row["xhat_next"]),
            int(row["yhat_next"]),
            int(row["gvar_next"])
        )

        if key in seen:
            continue

        seen.add(key)
        dedup_lookup.append(row)

    global gaussian_states
    global gaussian_lookup
    global gvar_max

    gaussian_states = states
    gaussian_lookup = dedup_lookup
    gvar_max = max(int(s["gvar"]) for s in states)


def preambel():
    with open(prism_file, 'a') as f:
        f.write('dtmc\n')
        f.write(f'const int c = {period};\n')
        f.write('const int N=' + str(mapSize - 1) + ';\n')
        f.write('const int xstart = ' + str(startX) + ';\n')
        f.write('const int ystart = ' + str(startY) + ';\n')
        f.write('const int xtarget = ' + str(targetX) + ';\n')
        f.write('const int ytarget = ' + str(targetY) + ';\n')
        f.write('const double p = ' + str(p) + ';\n')
        f.write('const int GMAX = ' + str(gvar_max) + ';\n\n')

        f.write('formula hasCrashed = (1=0) ')

        for x, y in obstacles:
            f.write(
                '| (x={0} & y={1}) '.format(
                    str(x), str(y)
                )
            )

        f.write(';\n\n')
        f.write('// Gaussian grid quantization: h=0.1\n')
        f.write('// gvar=0 represents Sigma=0 after update.\n\n')


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
                        '  [{0}] '.format(
                            directions[direction]
                        )
                    )
                    f.write(
                        '(xhat={0}) & (yhat={1}) -> true;\n'.format(
                            str(x), str(y)
                        )
                    )

        f.write('endmodule\n\n')


def knowledge():
    """
    Gaussian Knowledge module.

    gvar stores the map-specific quantized covariance state.
    It is also the URC decision variable.
    """
    with open(prism_file, 'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n')
        f.write('  yhat : [0..N] init ystart;\n')
        f.write('  gvar : [0..GMAX] init 0;\n')
        f.write('  step : [1..20] init 1;\n\n')
        f.write('  ready : [0..1] init 1;\n\n')

        f.write(
            '  // Map-specific quantized Gaussian transitions\n'
        )

        for row in gaussian_lookup:
            action = str(row["action"])
            xhat = int(row["xhat"])
            yhat = int(row["yhat"])
            gvar = int(row["gvar"])
            xhat_next = int(row["xhat_next"])
            yhat_next = int(row["yhat_next"])
            gvar_next = int(row["gvar_next"])

            f.write(
                f'  [{action}] '
                f'ready=1 & xhat={xhat} & yhat={yhat} '
                f'& gvar={gvar} -> '
                f'(xhat\'={xhat_next}) & '
                f'(yhat\'={yhat_next}) & '
                f'(gvar\'={gvar_next}) & '
                f'(ready\'=0);\n'
            )

        f.write('\n')

        # Perfect observation resets the Gaussian covariance.
        f.write(
            '  [update] step>=c & ready=0 -> '
            '(xhat\'=x) & (yhat\'=y) & (gvar\'=0) & '
            '(step\'=1) & (ready\'=1);\n'
        )

        # gvar already changed in the preceding movement command.
        f.write(
            '  [skip_update] step<c & ready=0 -> '
            '(ready\'=1) & (step\'=step+1);\n'
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
    build_map("maps/map_" + str(i) + ".csv")
    load_gaussian_lookup(i)

    target_pos = (targetX, targetY)
    _d = dijkstra.compute_directions(
        map_data,
        target_pos
    )

    d = list(zip(*_d))

    open(prism_file, "w").close()

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
        + " Gaussian states and "
        + str(len(gaussian_lookup))
        + " lookup transitions"
    )
