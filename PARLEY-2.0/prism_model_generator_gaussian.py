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
GAUSSIAN_METRIC = "frobenius"
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

        raw_thresholds = [
            int(value)
            for value in gaussian_model[
                "thresholds"
            ]
        ]

        if len(raw_thresholds) != 10:
            raise ValueError(
                "Expected exactly 10 Gaussian uncertainty thresholds."
            )

        if any(
            raw_thresholds[index]
            > raw_thresholds[index + 1]
            for index in range(9)
        ):
            raise ValueError(
                "Gaussian thresholds must be monotonically nondecreasing."
            )

        # PRISM stores only the discrete uncertainty LEVEL 1..10.
        # The actual Gaussian trace thresholds remain offline information.
        # The mapping is written as comments for traceability.
        f.write(
            "// Gaussian uncertainty level -> raw trace(Sigma) threshold\n"
        )

        for (
            level,
            threshold,
        ) in enumerate(
            raw_thresholds,
            start=1,
        ):
            f.write(
                f"// level {level} -> trace(Sigma) >= {threshold}\n"
            )

        # Baseline/default value. urc_synthesis removes this constant and
        # replaces it by the mutable [1..10] variable in the UMC.
        f.write(
            "const int max_gaussian_uncertainty = 1;\n"
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

        # Assign every Gaussian representative to the highest uncertainty
        # threshold level reached by its raw trace(Sigma).
        # level 0 means: below threshold 1.
        states_by_level = {
            level: []
            for level in range(0, 11)
        }

        for (
            state_id,
            raw_uncertainty,
        ) in enumerate(
            gaussian_model[
                "uncertainties"
            ]
        ):
            reached_level = 0

            for (
                level,
                threshold,
            ) in enumerate(
                raw_thresholds,
                start=1,
            ):
                if raw_uncertainty >= threshold:
                    reached_level = level
                else:
                    break

            states_by_level[
                reached_level
            ].append(
                state_id
            )

        # At most 11 compact formulas are required, independently of the
        # absolute magnitude of trace(Sigma).
        for level in range(0, 11):
            state_ids = states_by_level[
                level
            ]

            if not state_ids:
                continue

            f.write(
                f"formula gaussian_u_level_{level} = "
            )
            f.write(
                " | ".join(
                    f"gstate={state_id}"
                    for state_id in state_ids
                )
            )
            f.write(";\n")

        # Semantics:
        # selected URC level c means the raw threshold threshold[c].
        # A Gaussian state whose uncertainty has reached level L requires
        # an update iff c <= L.
        # This is exactly equivalent to:
        #   trace(Sigma_gstate) >= raw_threshold[c]
        # but PRISM stores only integers 1..10.
        update_terms = []

        for reached_level in range(1, 11):
            if not states_by_level[
                reached_level
            ]:
                continue

            selected_levels = " | ".join(
                f"max_gaussian_uncertainty={level}"
                for level in range(
                    1,
                    reached_level + 1,
                )
            )

            update_terms.append(
                f"(gaussian_u_level_{reached_level} & "
                f"({selected_levels}))"
            )

        f.write(
            "formula update_required = "
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

        # Keep the Robot state structure aligned with the original
        # point-estimate PARLEY model.
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

        # Same handshake variable as in the point-estimate Knowledge module.
        # There is deliberately NO separate variable named `read`.
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

    # Structural sanity check against the original point-estimate model.
    with open(
        prism_file,
        "r",
        encoding="utf-8",
    ) as generated_file:
        generated_model = generated_file.read()

    required_fragments = [
        "crashed : [0..1] init 0;",
        "ready : [0..1] init 1;",
        "[check] (move_ready=0) & hasCrashed",
        "[check] (move_ready=0) & !hasCrashed",
    ]

    for fragment in required_fragments:
        if fragment not in generated_model:
            raise ValueError(
                f"{prism_file}: missing required point-estimate-compatible "
                f"structure: {fragment}"
            )

    if "  read :" in generated_model:
        raise ValueError(
            f"{prism_file}: unexpected Knowledge variable `read` found."
        )

    print(
        f"finished map {i}: "
        f"{gaussian_model['state_count']} Gaussian states, "
        f"metric={gaussian_model['metric']}"
    )
