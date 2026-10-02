import json
import os
import time

import create_maps
import prism_model_generator_belief_exact_local_scaling
import urc_synthesis_belief_exact_local_scaling
import run_evochecker
import plot_fronts


max_replications = 10

# map_0 = 5x5, map_1 = 15x15, map_2 = 20x20
SCALING_MAPS = {
    0: 5,
    1: 15,
    2: 20,
}


def update_input(i, size):
    """Set map and target coordinates for the current scaling experiment."""
    with open("input.json", "r") as f:
        params = json.load(f)

    params["startX"] = 0
    params["startY"] = 0
    params["targetX"] = size - 1
    params["targetY"] = size - 1
    params["map_file"] = f"maps/map_{i}.csv"

    with open("input.json", "w") as f:
        json.dump(params, f, indent=4)


def models(i):
    """Generate the exact-local-belief model and its URC synthesis model."""
    prism_model_generator_belief_exact_local_scaling.generate_model(i)

    infile = f"Applications/EvoChecker-master/models/model_{i}.prism"
    outfile = f"Applications/EvoChecker-master/models/model_{i}_umc.prism"

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
    # Generate the 5x5, 15x15 and 20x20 maps.
    create_maps.create_3_maps()

    for i, size in SCALING_MAPS.items():
        print("=" * 70)
        print(f"Starting belief scalability experiment: map {i}, {size}x{size}")
        print("=" * 70)

        # The belief generator reads these values from input.json.
        update_input(i, size)

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

        fronts(i)

        print(f"Finished map {i} ({size}x{size})")


if __name__ == "__main__":
    os.makedirs("plots/fronts", exist_ok=True)
    os.makedirs("plots/box-plots", exist_ok=True)
    main()
