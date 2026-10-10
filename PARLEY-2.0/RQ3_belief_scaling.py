import json
import csv
from pathlib import Path
import os
import time

import prism_model_generator_belief_exact_local_scaling
import urc_synthesis_belief_exact_local_scaling
import run_evochecker_belief_rq3_200 as run_evochecker
import plot_fronts


max_replications = 10

SCALING_MAPS = {
    0: 5,
    1: 10,
    2: 15,
    3: 20,
}


def validate_map(i, size):
    path = Path(f"maps/map_{i}.csv")
    with path.open(newline="") as f:
        grid = [[int(cell) for cell in row] for row in csv.reader(f)]
    if len(grid) != size or any(len(row) != size for row in grid):
        raise ValueError(f"{path}: expected {size}x{size} grid")
    # build_map transposes and reverses the CSV: (0,0) is bottom-left.
    if grid[size - 1][0] > 9 or grid[0][size - 1] > 9:
        raise ValueError(f"{path}: start (0,0) or target ({size-1},{size-1}) blocked")


def update_input(i, size):
    """Set map and target coordinates for the current scaling experiment."""
    with open("input_belief_scaling.json", "r") as f:
        params = json.load(f)

    params["startX"] = 0
    params["startY"] = 0
    params["targetX"] = size - 1
    params["targetY"] = size - 1
    params["map_file"] = f"maps/map_{i}.csv"

    with open("input_belief_scaling.json", "w") as f:
        json.dump(params, f, indent=4)


def write_properties(i, size):
    """Create model-specific PCTL objective file for this map."""
    path = Path("Applications/EvoChecker-master") / f"robot_rq3_belief_map_{i}.pctl"
    path.parent.mkdir(parents=True, exist_ok=True)
    goal = f"x={size - 1} & y={size - 1} & crashed=0"
    path.write_text(
        f"//Objective, max\nP=? [ F ({goal}) ]\n\n"
        '//Objective, min\nR{"cost"}=? [ C<=100 ]\n',
        encoding="utf-8",
    )
    print(f"Using {path} with target ({size - 1},{size - 1})")


def models(i):
    """Generate the exact-local-belief model and its URC synthesis model."""
    prism_model_generator_belief_exact_local_scaling.generate_model(i)

    infile = f"Applications/EvoChecker-master/models/model_belief_{i}.prism"
    outfile = f"Applications/EvoChecker-master/models/model_belief_{i}_umc.prism"

    urc_synthesis_belief_exact_local_scaling.manipulate_prism_model(
        infile,
        outfile,
        baseline=False,
    )


def evo_checker(i):
    """Run all EvoChecker replications and return the wall-clock runtime."""
    return run_evochecker.run(i, max_replications)


def fronts(i):
    for period in range(max_replications):
        plot_fronts.plot_pareto_front(i, period)


def save_runtime(i, size, runtime):
    """
    Store one runtime file per scaling map.
    A rerun replaces the previous measurement for that map.
    """
    times_dir = "times_belief_scaling"
    os.makedirs(times_dir, exist_ok=True)

    times_file = os.path.join(
        times_dir,
        f"map_{i}_{size}x{size}.txt",
    )

    with open(times_file, "w") as f:
        f.write(f"Map: {i}\n")
        f.write(f"Size: {size}x{size}\n")
        f.write(f"WallClock: {runtime:.3f}\n")


def main():
    # Existing scaling maps are reused unchanged.

    for i, size in SCALING_MAPS.items():
        validate_map(i, size)
        print("=" * 70)
        print(f"Starting belief scalability experiment: map {i}, {size}x{size}")
        print("=" * 70)

        # The belief generator reads these values from input_belief_scaling.json.
        update_input(i, size)

        write_properties(i, size)

        # Generate only the belief model on the belief-state branch.
        models(i)

        print(f"Starting EvoChecker for map {i} ({size}x{size})")

        wall_start = time.time()
        evochecker_runtime = evo_checker(i)
        outer_wall_runtime = time.time() - wall_start

        print(
            f"EvoChecker wall-clock runtime for map {i} "
            f"({size}x{size}): {evochecker_runtime:.3f} seconds"
        )
        print(
            f"Measured outer wall-clock runtime for map {i} "
            f"({size}x{size}): {outer_wall_runtime:.3f} seconds"
        )

        save_runtime(i, size, evochecker_runtime)

        #fronts(i)

        print(f"Finished map {i} ({size}x{size})")


if __name__ == "__main__":
    os.makedirs("plots/fronts", exist_ok=True)
    os.makedirs("plots/box-plots", exist_ok=True)
    main()
