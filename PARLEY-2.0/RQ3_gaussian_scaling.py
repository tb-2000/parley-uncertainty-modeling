import json
import os
import time

import csv
from pathlib import Path
import prism_model_generator_gaussian_exact_local_scaling
import urc_synthesis_gaussian_exact_local_scaling
import run_evochecker_gaussian_rq3_200 as run_evochecker
import plot_fronts


max_replications = 10

# Existing maps: 5x5, 10x10, 15x15, 20x20
SCALING_MAPS = {
    3: 20,
}


def validate_map(i, size):
    path = Path(f"maps/map_{i}.csv")
    with path.open(newline="") as f:
        grid = [[int(v) for v in row] for row in csv.reader(f)]
    if len(grid) != size or any(len(row) != size for row in grid):
        raise ValueError(f"{path}: expected {size}x{size} map")
    if grid[size - 1][0] > 9 or grid[0][size - 1] > 9:
        raise ValueError(f"{path}: start or target is blocked")


def write_properties(i, size):
    # Generate properties for each map; avoid stale coordinates from other runs.
    path = Path("Applications/EvoChecker-master") / f"robot_rq3_gaussian_map_{i}.pctl"
    path.parent.mkdir(parents=True, exist_ok=True)
    goal = f"x={size - 1} & y={size - 1} & crashed=0"
    path.write_text(
        f"//Objective, max\nP=? [ F ({goal}) ]\n\n"
        '//Objective, min\nR{"cost"}=? [ C<=100 ]\n',
        encoding="utf-8",
    )
    print(f"Using {path} with target ({size-1},{size-1})")


def update_input(i, size):
    """Set the current scaling map and its opposite-corner target."""
    with open("input_gaussian_scaling.json", "r") as f:
        params = json.load(f)

    params["startX"] = 0
    params["startY"] = 0
    params["targetX"] = size - 1
    params["targetY"] = size - 1
    params["map_file"] = f"maps/map_{i}.csv"

    with open("input_gaussian_scaling.json", "w") as f:
        json.dump(params, f, indent=4)

    print(
        f"Map {i}: {size}x{size}, "
        f"start=(0, 0), target=({size - 1}, {size - 1})"
    )


def models(i):
    """Generate Gaussian base PRISM model and synthesized UMC model."""
    prism_model_generator_gaussian_exact_local_scaling.generate_model(i)

    infile = f"Applications/EvoChecker-master/models/model_gaussian_{i}.prism"
    outfile = f"Applications/EvoChecker-master/models/model_gaussian_{i}_umc.prism"

    urc_synthesis_gaussian_exact_local_scaling.manipulate_prism_model(
        infile,
        outfile,
        baseline=False,
    )


def evo_checker(i):
    return run_evochecker.run(i, max_replications)


def fronts(i):
    for period in range(max_replications):
        plot_fronts.plot_pareto_front(i, period)


def save_runtime(i, size, runtime):
    times_dir = "times_gaussian_scaling"
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
    # Reuse the same maps as the belief-scaling experiment; never overwrite them.

    for i, size in SCALING_MAPS.items():
        validate_map(i, size)
        write_properties(i, size)
        print("=" * 70)
        print(
            f"Starting Gaussian scalability experiment: "
            f"map {i}, {size}x{size}"
        )
        print("=" * 70)

        update_input(i, size)

        # Creates model_i.prism and model_i_umc.prism.
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
        # fronts(i)  # optional; avoid missing/obsolete front files during scaling

        print(f"Finished map {i} ({size}x{size})")


if __name__ == "__main__":
    os.makedirs("plots/fronts", exist_ok=True)
    os.makedirs("plots/box-plots", exist_ok=True)
    main()
