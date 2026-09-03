"""
Grid/interpolation variant of prism_model_generator_belief_full.py.

Only the belief-model construction and Knowledge transitions differ from the
full-medoid version. URC thresholds and the original PARLEY handshakes remain
unchanged.

IMPORTANT:
The probabilistic branches in Knowledge encode convex interpolation weights.
They are an experimental abstraction device, not additional physical
robot-motion probabilities.
"""

import csv
import json

import dijkstra
from full_belief_grid_interpolation import build_grid_belief_model


startX = 0
startY = 0
targetX = 4
targetY = 4
p = 0.01
directions = ["west", "east", "south", "north"]
obstacles = []
updates = [5]

map_file = "maps/map_1.csv"
map_data = []
mapSize = len(map_data)
corridor = 1

prism_file = ""
belief_model = None
period = 1

# Test parameters. A sweep should be done with the separate analysis script
# before choosing these for PARLEY synthesis.
GRID_PER_POSITION = 5
INTERPOLATION_NEIGHBOURS = 5


def build_map(filename):
    rows = []
    with open(filename, "r") as file:
        rows.extend(csv.reader(file))

    global mapSize, map_data, obstacles
    mapSize = len(rows)
    transposed = list(zip(*rows))
    map_data = [row[::-1] for row in transposed]

    obstacles = []
    for x in range(mapSize):
        for y in range(mapSize):
            if int(map_data[x][y]) > 9:
                obstacles.append([x, y])


def preambel():
    with open(prism_file, "a") as f:
        f.write("dtmc\n")

        thresholds = belief_model["thresholds"]
        f.write(f"const int c = {period};\n")
        f.write("// Map-specific belief-uncertainty thresholds:\n")

        for index, threshold in enumerate(thresholds, start=1):
            f.write(
                f"// c={index} -> "
                f"max_belief_uncertainty={threshold}\n"
            )
            f.write(
                f"const int belief_threshold_{index} = "
                f"{threshold};\n"
            )

        f.write(f"const int N={mapSize - 1};\n")
        f.write(f"const int xstart = {startX};\n")
        f.write(f"const int ystart = {startY};\n")
        f.write(f"const int xtarget = {targetX};\n")
        f.write(f"const int ytarget = {targetY};\n")
        f.write(f"const double p = {p};\n\n")

        f.write("formula hasCrashed = (1=0) ")
        for x, y in obstacles:
            f.write(f"| (x={x} & y={y}) ")
        f.write(";\n\n")

        # Local state IDs are reused at different xhat/yhat positions.
        # Therefore the uncertainty formulas must include xhat/yhat.
        uncertainty_terms = {stage: [] for stage in range(11)}

        for position, values in belief_model["uncertainties"].items():
            xhat, yhat = [int(v) for v in position.split(",")]

            for state_id, uncertainty in enumerate(values):
                level = 0
                for index, threshold in enumerate(
                    thresholds,
                    start=1,
                ):
                    if uncertainty >= threshold:
                        level = index
                    else:
                        break

                uncertainty_terms[level].append(
                    f"(xhat={xhat} & yhat={yhat} "
                    f"& belief_state={state_id})"
                )

        written_stages = []
        for stage in range(1, 11):
            terms = uncertainty_terms[stage]
            if not terms:
                continue
            written_stages.append(stage)
            f.write(
                f"formula belief_u_{stage} = "
                + " | ".join(terms)
                + ";\n"
            )

        f.write("formula update_required = ")
        update_terms = []
        for stage in written_stages:
            if stage < 10:
                update_terms.append(
                    f"(belief_u_{stage} & c<={stage})"
                )
            else:
                update_terms.append(f"belief_u_{stage}")

        f.write(" | ".join(update_terms) if update_terms else "false")
        f.write(";\n\n")


def robot():
    with open(prism_file, "a") as f:
        f.write("module Robot\n")
        f.write("  x : [0..N] init xstart;\n")
        f.write("  y : [0..N] init ystart;\n")
        f.write("  move_ready : [0..1] init 1;\n")
        f.write("  crashed : [0..1] init 0;\n\n")

        f.write(
            "  [east] (move_ready=1) ->\n"
            "    (1-3*p): (x'=min(x+1, N)) & (move_ready'=0) +\n"
            "    p: (y'=min(y+1, N)) & (move_ready'=0) +\n"
            "    p: (y'=max(y-1, 0)) & (move_ready'=0) +\n"
            "    p: (x'=max(x-1, 0)) & (move_ready'=0);\n"
        )
        f.write(
            "  [west] (move_ready=1) ->\n"
            "    p: (x'=min(x+1, N)) & (move_ready'=0) +\n"
            "    p: (y'=min(y+1, N)) & (move_ready'=0) +\n"
            "    p: (y'=max(y-1, 0)) & (move_ready'=0) +\n"
            "    (1-3*p): (x'=max(x-1, 0)) & (move_ready'=0);\n"
        )
        f.write(
            "  [north] (move_ready=1) ->\n"
            "    p: (x'=min(x+1, N)) & (move_ready'=0) +\n"
            "    (1-3*p): (y'=min(y+1, N)) & (move_ready'=0) +\n"
            "    p: (y'=max(y-1, 0)) & (move_ready'=0) +\n"
            "    p: (x'=max(x-1, 0)) & (move_ready'=0);\n"
        )
        f.write(
            "  [south] (move_ready=1) ->\n"
            "    p: (x'=min(x+1, N)) & (move_ready'=0) +\n"
            "    p: (y'=min(y+1, N)) & (move_ready'=0) +\n"
            "    (1-3*p): (y'=max(y-1, 0)) & (move_ready'=0) +\n"
            "    p: (x'=max(x-1, 0)) & (move_ready'=0);\n"
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
        f.write("module Adaptation_MAPE_controller\n")
        for x in range(mapSize):
            for y in range(mapSize):
                direction = int(d[y][x])
                if direction < 4:
                    f.write(
                        f"  [{directions[direction]}] "
                        f"(xhat={x}) & (yhat={y}) -> true;\n"
                    )
        f.write("endmodule\n\n")


def _format_interpolated_update(
    action,
    next_xhat,
    next_yhat,
    successors,
):
    branches = []

    for successor in successors:
        weight = successor["weight"]
        state = successor["state"]

        branches.append(
            f"{weight:.15g}: "
            f"(xhat'={next_xhat}) & "
            f"(yhat'={next_yhat}) & "
            f"(belief_state'={state}) & "
            f"(ready'=0)"
        )

    return " + ".join(branches)


def knowledge():
    with open(prism_file, "a") as f:
        f.write("module Knowledge\n")
        f.write("  xhat : [0..N] init xstart;\n")
        f.write("  yhat : [0..N] init ystart;\n")
        f.write(
            f"  belief_state : "
            f"[0..{belief_model['max_local_states'] - 1}] "
            "init 0;\n"
        )
        f.write("  ready : [0..1] init 1;\n\n")

        for key, transition in belief_model["transitions"].items():
            xhat, yhat, state_id = [
                int(value)
                for value in key.split(",")
            ]

            rhs = _format_interpolated_update(
                transition["action"],
                transition["next_xhat"],
                transition["next_yhat"],
                transition["successors"],
            )

            f.write(
                f"  [{transition['action']}] "
                f"ready=1 & xhat={xhat} & yhat={yhat} "
                f"& belief_state={state_id} -> "
                f"{rhs};\n"
            )

        # Local state 0 is the exact certainty belief at every reset position.
        f.write(
            "  [update] update_required & ready=0 -> "
            "(xhat'=x) & (yhat'=y) & "
            "(belief_state'=0) & (ready'=1);\n"
        )
        f.write(
            "  [skip_update] !update_required & ready=0 -> "
            "(ready'=1);\n"
        )
        f.write("endmodule\n\n")


def rewards():
    with open(prism_file, "a") as f:
        f.write('rewards "cost"\n')
        f.write("  [east] true : 1;\n")
        f.write("  [west] true : 1;\n")
        f.write("  [north] true : 1;\n")
        f.write("  [south] true : 1;\n")
        f.write("  [update] true : 5;\n")
        f.write("endrewards\n\n")


def read_params_from_file():
    with open("input.json", "r") as file:
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
    global prism_file, belief_model

    prism_file = (
        "Applications/EvoChecker-master/models/"
        f"model_{i}.prism"
    )

    read_params_from_file()
    build_map(f"maps/map_{i}.csv")
    target_pos = (targetX, targetY)

    d_raw = dijkstra.compute_directions(
        map_data,
        target_pos,
    )
    d = list(zip(*d_raw))

    belief_model = build_grid_belief_model(
        map_id=i,
        map_data=map_data,
        target=target_pos,
        p=p,
        grid_per_position=GRID_PER_POSITION,
        neighbours=INTERPOLATION_NEIGHBOURS,
        max_steps=10,
    )

    open(prism_file, "w").close()
    preambel()
    robot()
    adaptation_mape_controller(d)
    knowledge()
    rewards()

    print("finished map " + str(i))
    print(
        "grid analysis:",
        json.dumps(
            belief_model["analysis"],
            indent=2,
        ),
    )
