import csv
import json

import dijkstra
from exact_reachable_gaussian_model_local import build_exact_gaussian_model


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

GAUSSIAN_MAX_STEPS = 10


def build_map(filename):
    rows = []

    with open(filename, "r") as file:
        reader = csv.reader(file)
        rows.extend(reader)

    global mapSize, map_data, obstacles

    mapSize = len(rows)
    transposed = list(zip(*rows))
    map_data = [
        row[::-1]
        for row in transposed
    ]

    obstacles = []

    for x in range(mapSize):
        for y in range(mapSize):
            if int(map_data[x][y]) > 9:
                obstacles.append([x, y])


def _stage_for_uncertainty(uncertainty, thresholds):
    stage = 0

    for index, threshold in enumerate(
        thresholds,
        start=1,
    ):
        if uncertainty >= threshold:
            stage = index
        else:
            break

    return stage


def preambel():
    with open(prism_file, "a") as f:
        f.write("dtmc\n")

        thresholds = [
            int(value)
            for value in gaussian_model["thresholds"]
        ]

        if len(thresholds) != 10:
            raise ValueError(
                "Expected exactly 10 Gaussian thresholds."
            )

        if any(
            thresholds[i] > thresholds[i + 1]
            for i in range(9)
        ):
            raise ValueError(
                "Gaussian thresholds must be monotonically nondecreasing."
            )

        f.write(
            "// Gaussian uncertainty level -> raw scaled MSE threshold\n"
        )

        for level, threshold in enumerate(
            thresholds,
            start=1,
        ):
            f.write(
                f"// level {level} -> MSE >= {threshold}\n"
            )
            f.write(
                f"const int gaussian_threshold_{level} = {threshold};\n"
            )

        # Base/default value; URC synthesis replaces this constant.
        f.write(
            "const int max_gaussian_uncertainty = 1;\n"
        )

        f.write(f"const int N={mapSize - 1};\n")
        f.write(f"const int xstart={startX};\n")
        f.write(f"const int ystart={startY};\n")
        f.write(f"const int xtarget={targetX};\n")
        f.write(f"const int ytarget={targetY};\n")
        f.write(f"const double p={p};\n\n")

        f.write("formula hasCrashed = (1=0) ")
        for x, y in obstacles:
            f.write(
                f"| (x={x} & y={y}) "
            )
        f.write(";\n\n")

        # gaussian_state IDs are local. Hence every MSE class atom includes
        # xhat and yhat as well as gaussian_state.
        gaussian_classes = {
            stage: []
            for stage in range(11)
        }

        for position, values in gaussian_model["uncertainties"].items():
            xhat, yhat = [
                int(v)
                for v in position.split(",")
            ]

            for state_id, uncertainty in enumerate(values):
                stage = _stage_for_uncertainty(
                    uncertainty,
                    thresholds,
                )

                gaussian_classes[stage].append(
                    f"(xhat={xhat} & yhat={yhat} "
                    f"& gaussian_state={state_id})"
                )

        written_stages = []

        for stage in range(1, 11):
            terms = gaussian_classes[stage]

            if not terms:
                continue

            written_stages.append(stage)

            f.write(
                f"formula gaussian_u_{stage} = "
                + " | ".join(terms)
                + ";\n"
            )

        update_terms = []

        for stage in written_stages:
            if stage < 10:
                update_terms.append(
                    f"(gaussian_u_{stage} & "
                    f"max_gaussian_uncertainty<={stage})"
                )
            else:
                update_terms.append(
                    "gaussian_u_10"
                )

        f.write(
            "formula update_required = "
            + (
                " | ".join(update_terms)
                if update_terms
                else "false"
            )
            + ";\n\n"
        )


def robot():
    with open(prism_file, "a") as f:
        f.write("module Robot\n")
        f.write("  x : [0..N] init xstart;\n")
        f.write("  y : [0..N] init ystart;\n")
        f.write("  move_ready : [0..1] init 1;\n")
        f.write("  crashed : [0..1] init 0;\n\n")

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

        f.write("endmodule\n\n")


def adaptation_mape_controller(d):
    with open(prism_file, "a") as f:
        f.write(
            "module Adaptation_MAPE_controller\n"
        )

        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])

                if direction < 4:
                    f.write(
                        f"  [{directions[direction]}] "
                        f"xhat={x} & yhat={y} -> true;\n"
                    )

        f.write("endmodule\n\n")


def knowledge():
    with open(prism_file, "a") as f:
        f.write("module Knowledge\n")

        f.write(
            "  xhat : [0..N] init xstart;\n"
        )
        f.write(
            "  yhat : [0..N] init ystart;\n"
        )
        f.write(
            f"  gaussian_state : "
            f"[0..{gaussian_model['max_gaussian_state']}] init 0;\n"
        )
        f.write(
            "  ready : [0..1] init 1;\n\n"
        )

        # Exact deterministic moment successor for every reachable local context.
        for key, transition in gaussian_model["transitions"].items():
            xhat, yhat, state_id = [
                int(value)
                for value in key.split(",")
            ]

            action = transition["action"]
            nxhat = transition["next_xhat"]
            nyhat = transition["next_yhat"]
            next_state = transition["next_state"]

            f.write(
                f"  [{action}] ready=1"
                f" & xhat={xhat}"
                f" & yhat={yhat}"
                f" & gaussian_state={state_id} -> "
                f"(xhat'={nxhat})"
                f" & (yhat'={nyhat})"
                f" & (gaussian_state'={next_state})"
                f" & (ready'=0);\n"
            )

        # Perfect localisation:
        # local gaussian_state=0 means ZERO_STATE at every position.
        f.write(
            "  [update] update_required & ready=0 -> "
            "(xhat'=x) & (yhat'=y) "
            "& (gaussian_state'=0) "
            "& (ready'=1);\n"
        )

        f.write(
            "  [skip_update] !update_required & ready=0 -> "
            "(ready'=1);\n"
        )

        f.write("endmodule\n\n")


def rewards():
    with open(prism_file, "a") as f:
        f.write('rewards "cost"\n')

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
    global startX, startY, targetX, targetY
    global map_file, p, updates

    with open("input.json", "r") as file:
        params = json.load(file)

    startX = params["startX"]
    startY = params["startY"]
    targetX = params["targetX"]
    targetY = params["targetY"]
    p = params["p"]
    map_file = params["map_file"]
    updates = params["updates"]


def generate_model(i):
    global prism_file, gaussian_model

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
    d = list(zip(*raw_directions))

    gaussian_model = build_exact_gaussian_model(
        map_id=i,
        map_data=map_data,
        target=target_pos,
        p=p,
        max_steps=GAUSSIAN_MAX_STEPS,
    )

    print(
        f"map {i}: exact reachable Gaussian contexts="
        f"{gaussian_model['context_count']}, "
        f"distinct Gaussian moment states="
        f"{gaussian_model['gaussian_count']}, "
        f"max local Gaussian states="
        f"{gaussian_model['max_local_states']}"
    )

    open(
        prism_file,
        "w",
    ).close()

    preambel()
    robot()
    adaptation_mape_controller(d)
    knowledge()
    rewards()

    # Structural checks.
    with open(
        prism_file,
        "r",
        encoding="utf-8",
    ) as generated_file:
        generated_model = generated_file.read()

    required = [
        "  xhat : [0..N] init xstart;",
        "  yhat : [0..N] init ystart;",
        "  gaussian_state :",
        "  ready : [0..1] init 1;",
    ]

    for fragment in required:
        if fragment not in generated_model:
            raise ValueError(
                f"{prism_file}: missing required structure: {fragment}"
            )

    forbidden = [
        "  gstate :",
        "  substate :",
        "formula estimate_",
    ]

    for fragment in forbidden:
        if fragment in generated_model:
            raise ValueError(
                f"{prism_file}: obsolete Gaussian structure found: {fragment}"
            )

    print(
        f"finished map {i}: "
        f"{gaussian_model['context_count']} exact Gaussian contexts, "
        f"local state range 0..{gaussian_model['max_gaussian_state']}"
    )
