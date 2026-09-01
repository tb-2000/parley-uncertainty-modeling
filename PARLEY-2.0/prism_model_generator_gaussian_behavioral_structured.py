import csv
import json
from pathlib import Path
import dijkstra
from exact_reachable_gaussian_model_behavioral_structured import build_exact_gaussian_model



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


obstacles = []
updates = [5]

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)

prism_file = ""
gaussian_model = None

# Exact reachable Gaussian moment model; no K, medoids, or projection.
GAUSSIAN_MAX_STEPS = 10


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
            for value in gaussian_model["thresholds"]
        ]

        if len(raw_thresholds) != 10:
            raise ValueError(
                "Expected exactly 10 Gaussian uncertainty thresholds."
            )

        if any(
            raw_thresholds[index] > raw_thresholds[index + 1]
            for index in range(9)
        ):
            raise ValueError(
                "Gaussian thresholds must be monotonically nondecreasing."
            )

        # Keep raw MSE values as traceable offline semantics.
        f.write(
            "// Gaussian uncertainty level -> raw MSE threshold\n"
        )

        for level, threshold in enumerate(
            raw_thresholds,
            start=1,
        ):
            f.write(
                f"// level {level} -> MSE >= {threshold}\n"
            )

        # Baseline/default value. urc_synthesis removes this constant and
        # replaces it with mutable max_gaussian_uncertainty:[1..10].
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

        # gaussian_state is now directly the highest MSE/URC threshold level
        # reached (0..10). Therefore all large gstate disjunctions disappear.
        #
        # Selected level c requires localization iff c <= gaussian_state.
        update_terms = []

        for stage in range(1, 10):
            update_terms.append(
                f"(gaussian_state={stage} & "
                f"max_gaussian_uncertainty<={stage})"
            )

        # At stage 10 every selectable controller level 1..10 must update.
        update_terms.append(
            "(gaussian_state=10)"
        )

        f.write(
            "formula update_required = "
            + " | ".join(update_terms)
            + ";\n\n"
        )

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
        # xhat/yhat are explicit state variables again, so no estimate_X_Y
        # disjunctions over gstate are required.
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
                        f"xhat={x} & yhat={y} -> true;\n"
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
            "  gaussian_state : [0..10] init 0;\n"
        )
        f.write(
            f"  substate : [0..{gaussian_model['max_substate']}] init 0;\n"
        )
        f.write(
            "  ready : [0..1] init 1;\n\n"
        )

        # Exactly one transition per reachable behavioral quotient class with
        # a MAPE successor. No global gstate is stored by PRISM.
        for (
            context_id,
            transition,
        ) in gaussian_model["transitions"].items():
            action = transition["action"]
            src = transition["source"]
            dst = transition["target"]

            f.write(
                f"  [{action}] ready=1"
                f" & xhat={src['xhat']}"
                f" & yhat={src['yhat']}"
                f" & gaussian_state={src['gaussian_state']}"
                f" & substate={src['substate']} -> "
                f"(xhat'={dst['xhat']})"
                f" & (yhat'={dst['yhat']})"
                f" & (gaussian_state'={dst['gaussian_state']})"
                f" & (substate'={dst['substate']})"
                f" & (ready'=0);\n"
            )

        # Perfect localization resets estimate and Gaussian uncertainty to the
        # unique zero context. By construction every position's zero context
        # is encoded as gaussian_state=0, substate=0.
        f.write(
            "  [update] update_required & ready=0 -> "
            "(xhat'=x) & (yhat'=y) "
            "& (gaussian_state'=0) & (substate'=0) "
            "& (ready'=1);\n"
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

    gaussian_model = build_exact_gaussian_model(
        map_id=i,
        map_data=map_data,
        target=target_pos,
        p=p,
        max_steps=GAUSSIAN_MAX_STEPS,
    )

    # Structured Perfect-Localization invariant:
    # every position resets to
    #   (xhat=x, yhat=y, gaussian_state=0, substate=0).
    for x in range(mapSize):
        for y in range(mapSize):
            zero = gaussian_model["zero_contexts"][f"{x},{y}"]

            if (
                zero["xhat"] != x
                or zero["yhat"] != y
                or zero["gaussian_state"] != 0
                or zero["substate"] != 0
            ):
                raise AssertionError(
                    f"Invalid structured zero Gaussian context "
                    f"for ({x},{y}): {zero}"
                )

    print(
        f"map {i}: behavioral Gaussian classes="
        f"{gaussian_model['state_count']}, "
        f"exact reachable contexts={gaussian_model['exact_context_count']}, "
        f"distinct Gaussian moment states={gaussian_model['gaussian_count']}, "
        f"max substate={gaussian_model['max_substate']}"
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

    required_structured_fragments = [
        "  xhat : [0..N] init xstart;",
        "  yhat : [0..N] init ystart;",
        "  gaussian_state : [0..10] init 0;",
        "  substate :",
    ]

    for fragment in required_structured_fragments:
        if fragment not in generated_model:
            raise ValueError(
                f"{prism_file}: missing structured Gaussian variable: "
                f"{fragment}"
            )

    if "  gstate :" in generated_model:
        raise ValueError(
            f"{prism_file}: obsolete global gstate variable found."
        )

    if "formula estimate_" in generated_model:
        raise ValueError(
            f"{prism_file}: obsolete estimate_X_Y gstate formulas found."
        )

    print(
        f"finished map {i}: "
        f"{gaussian_model['state_count']} behavioral Gaussian classes "
        f"from {gaussian_model['exact_context_count']} exact contexts"
    )
