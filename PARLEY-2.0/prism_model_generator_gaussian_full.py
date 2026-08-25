import csv
import json
from pathlib import Path
import dijkstra



startX = 0
startY = 0
targetX = 4
targetY = 4
p = 0.01

directions = [
    "west",
    "east",
    "south",
    "north",
]

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

prism_file = ""
gaussian_model = None

# Fixed finite Gaussian abstraction budget, analogous to belief model.
GAUSSIAN_K = 100
GAUSSIAN_MAX_STEPS = 10
GAUSSIAN_METRIC = "bures_wasserstein"
GAUSSIAN_MODEL_DIR = Path("gaussian_models")


def build_map(filename):
    rows = []

    with open(
        filename,
        "r",
    ) as file:
        reader = csv.reader(file)
        rows.extend(reader)

    global mapSize
    global map_data
    global obstacles

    mapSize = len(rows)

    transposed = list(
        zip(*rows)
    )
    map_data = [
        row[::-1]
        for row in transposed
    ]

    obstacles = []

    for x in range(mapSize):
        for y in range(mapSize):
            if int(map_data[x][y]) > 9:
                obstacles.append(
                    [x, y]
                )


def preambel():
    with open(
        prism_file,
        "a",
    ) as f:
        f.write("dtmc\n")

        thresholds = gaussian_model[
            "thresholds"
        ]

        f.write(
            f"const int max_gaussian_uncertainty = "
            f"{thresholds[0]};\n"
        )

        for (
            index,
            threshold,
        ) in enumerate(
            thresholds,
            start=1,
        ):
            f.write(
                f"const int gaussian_threshold_{index} = "
                f"{threshold};\n"
            )

        f.write(
            f"const int N={mapSize - 1};\n"
        )
        f.write(
            f"const int xstart={startX};\n"
        )
        f.write(
            f"const int ystart={startY};\n"
        )
        f.write(
            f"const int xtarget={targetX};\n"
        )
        f.write(
            f"const int ytarget={targetY};\n"
        )
        f.write(
            f"const double p={p};\n\n"
        )

        f.write(
            "formula hasCrashed = (1=0) "
        )

        for x, y in obstacles:
            f.write(
                f"| (x={x} & y={y}) "
            )

        f.write(";\n\n")

        # PRISM 4.7-friendly uncertainty encoding:
        # group gstates by offline-computed trace(Sigma).
        uncertainty_groups = {}

        for (
            state_id,
            uncertainty,
        ) in enumerate(
            gaussian_model[
                "uncertainties"
            ]
        ):
            uncertainty_groups.setdefault(
                uncertainty,
                [],
            ).append(
                state_id
            )

        for (
            group_index,
            (
                uncertainty,
                state_ids,
            ),
        ) in enumerate(
            sorted(
                uncertainty_groups.items()
            )
        ):
            f.write(
                f"formula gaussian_u_{group_index} = "
            )

            f.write(
                " | ".join(
                    f"gstate={state_id}"
                    for state_id in state_ids
                )
            )

            f.write(";\n")

        f.write(
            "formula update_required = "
        )

        update_terms = []

        for (
            group_index,
            (
                uncertainty,
                _,
            ),
        ) in enumerate(
            sorted(
                uncertainty_groups.items()
            )
        ):
            update_terms.append(
                f"(gaussian_u_{group_index} & "
                f"max_gaussian_uncertainty<={uncertainty})"
            )

        if update_terms:
            f.write(
                " | ".join(
                    update_terms
                )
            )
        else:
            f.write("false")

        f.write(";\n\n")


def robot():
    with open(
        prism_file,
        "a",
    ) as f:
        f.write(
            "module Robot\n"
        )

        f.write(
            "  x : [0..N] init xstart;\n"
        )
        f.write(
            "  y : [0..N] init ystart;\n"
        )
        f.write(
            "  move_ready : [0..1] init 1;\n"
        )
        f.write(
            "  crashed : [0..1] init 0;\n\n"
        )

        f.write(
            "  [east] (move_ready=1) ->\n"
            "    (1-3*p): (x'=min(x+1,N)) & (move_ready'=0) +\n"
            "    p: (y'=min(y+1,N)) & (move_ready'=0) +\n"
            "    p: (y'=max(y-1,0)) & (move_ready'=0) +\n"
            "    p: (x'=max(x-1,0)) & (move_ready'=0);\n"
        )

        f.write(
            "  [west] (move_ready=1) ->\n"
            "    p: (x'=min(x+1,N)) & (move_ready'=0) +\n"
            "    p: (y'=min(y+1,N)) & (move_ready'=0) +\n"
            "    p: (y'=max(y-1,0)) & (move_ready'=0) +\n"
            "    (1-3*p): (x'=max(x-1,0)) & (move_ready'=0);\n"
        )

        f.write(
            "  [north] (move_ready=1) ->\n"
            "    p: (x'=min(x+1,N)) & (move_ready'=0) +\n"
            "    (1-3*p): (y'=min(y+1,N)) & (move_ready'=0) +\n"
            "    p: (y'=max(y-1,0)) & (move_ready'=0) +\n"
            "    p: (x'=max(x-1,0)) & (move_ready'=0);\n"
        )

        f.write(
            "  [south] (move_ready=1) ->\n"
            "    p: (x'=min(x+1,N)) & (move_ready'=0) +\n"
            "    p: (y'=min(y+1,N)) & (move_ready'=0) +\n"
            "    (1-3*p): (y'=max(y-1,0)) & (move_ready'=0) +\n"
            "    p: (x'=max(x-1,0)) & (move_ready'=0);\n"
        )

        f.write(
            "  [check] (move_ready=0) & hasCrashed -> "
            "(crashed'=1) & (move_ready'=1);\n"
        )
        f.write(
            "  [check] (move_ready=0) & !hasCrashed -> "
            "(move_ready'=1);\n"
        )

        f.write(
            "endmodule\n\n"
        )


def adaptation_mape_controller(d):
    with open(
        prism_file,
        "a",
    ) as f:
        f.write(
            "module Adaptation_MAPE_controller\n"
        )

        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(
                    d[y][x]
                )

                if direction < 4:
                    f.write(
                        f"  [{directions[direction]}] "
                        f"(xhat={x}) & (yhat={y}) -> true;\n"
                    )

        f.write(
            "endmodule\n\n"
        )


def knowledge():
    with open(
        prism_file,
        "a",
    ) as f:
        f.write(
            "module Knowledge\n"
        )
        f.write(
            "  xhat : [0..N] init xstart;\n"
        )
        f.write(
            "  yhat : [0..N] init ystart;\n"
        )
        f.write(
            f"  gstate : [0..{gaussian_model['state_count'] - 1}] "
            "init 0;\n\n"
        )
        f.write(
            "  ready : [0..1] init 1;\n"
        )

        # Map-specific deterministic finite Gaussian automaton.
        # Only the MAPE-selected action at each xhat,yhat is written.
        for (
            key,
            transition,
        ) in gaussian_model[
            "transitions"
        ].items():
            x, y, state_id = [
                int(value)
                for value in key.split(",")
            ]

            action = transition[
                "action"
            ]
            next_state = transition[
                "next_state"
            ]

            effect = directions_effects[
                directions.index(
                    action
                )
            ]

            f.write(
                f"  [{action}] ready=1 & "
                f"xhat={x} & yhat={y} & "
                f"gstate={state_id} -> "
                f"{effect} & "
                f"(gstate'={next_state}) & "
                f"(ready'=0);\n"
            )

        # Perfect Ground-Truth update:
        # exact position and exact covariance Sigma=0 = gstate 0.
        f.write(
            "  [update] update_required & ready=0 -> "
            "(xhat'=x) & (yhat'=y) & "
            "(gstate'=0) & (ready'=1);\n"
        )

        f.write(
            "  [skip_update] !update_required & ready=0 -> "
            "(ready'=1);\n"
        )

        f.write(
            "endmodule\n\n"
        )


def rewards():
    with open(
        prism_file,
        "a",
    ) as f:
        f.write(
            'rewards "cost"\n'
        )

        for action in (
            "east",
            "west",
            "north",
            "south",
        ):
            f.write(
                f"  [{action}] true : 1;\n"
            )

        f.write(
            "  [update] true : 5;\n"
        )
        f.write(
            "endrewards\n\n"
        )


def read_params_from_file():
    global startX
    global startY
    global targetX
    global targetY
    global map_file
    global p
    global updates

    with open(
        "input.json",
        "r",
    ) as file:
        params = json.load(file)

    startX = params[
        "startX"
    ]
    startY = params[
        "startY"
    ]
    targetX = params[
        "targetX"
    ]
    targetY = params[
        "targetY"
    ]
    p = params["p"]
    map_file = params[
        "map_file"
    ]
    updates = params[
        "updates"
    ]



def load_precomputed_gaussian_model(map_id):
    """
    Load an already precomputed finite Gaussian model.

    The PRISM generator deliberately does NOT perform clustering.
    This keeps offline abstraction time separate from PARLEY synthesis time.

    Expected file:
        gaussian_models/map_<id>.json
    """
    path = (
        GAUSSIAN_MODEL_DIR
        / f"map_{map_id}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Precomputed Gaussian model not found: {path}. "
            "Run full_gaussian_representatives.py first."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        model = json.load(file)

    required = {
        "map_id",
        "state_count",
        "thresholds",
        "uncertainties",
        "transitions",
        "representatives",
        "metric",
    }

    missing = sorted(
        required - set(model)
    )

    if missing:
        raise ValueError(
            f"{path}: missing required fields: {missing}"
        )

    if int(model["map_id"]) != int(map_id):
        raise ValueError(
            f"{path}: contains map_id={model['map_id']}, "
            f"expected {map_id}."
        )

    if len(model["thresholds"]) != 10:
        raise ValueError(
            f"{path}: expected exactly 10 Gaussian thresholds, "
            f"got {len(model['thresholds'])}."
        )

    if len(model["uncertainties"]) != int(model["state_count"]):
        raise ValueError(
            f"{path}: uncertainties/state_count mismatch."
        )

    if len(model["representatives"]) != int(model["state_count"]):
        raise ValueError(
            f"{path}: representatives/state_count mismatch."
        )

    # Strong reset invariant: representative 0 must be Sigma=0.
    zero = model["representatives"][0]

    for field in ("var_x", "var_y", "cov_xy"):
        if abs(float(zero[field])) > 1e-15:
            raise ValueError(
                f"{path}: representative 0 is not exact Sigma=0."
            )

    return model

def generate_model(i):
    global prism_file
    global gaussian_model

    prism_file = (
        "Applications/EvoChecker-master/models/"
        f"model_{i}.prism"
    )

    read_params_from_file()

    build_map(
        f"maps/map_{i}.csv"
    )

    target_pos = (
        targetX,
        targetY,
    )

    raw_directions = (
        dijkstra.compute_directions(
            map_data,
            target_pos,
        )
    )
    d = list(
        zip(*raw_directions)
    )

    gaussian_model = load_precomputed_gaussian_model(
        i
    )

    open(
        prism_file,
        "w",
    ).close()

    preambel()
    robot()
    adaptation_mape_controller(
        d
    )
    knowledge()
    rewards()

    print(
        f"finished map {i}: "
        f"{gaussian_model['state_count']} Gaussian states, "
        f"metric={gaussian_model['metric']}"
    )
