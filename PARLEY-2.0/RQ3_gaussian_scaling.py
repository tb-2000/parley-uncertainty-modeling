import json
import os
import time

import create_maps
import prism_model_generator_gaussian_exact_local_scaling
import urc_synthesis_gaussian_exact_local_scaling
import run_evochecker_rq3_200 as run_evochecker
import plot_fronts


max_replications = 10

# map_0 = 5x5, map_1 = 15x15, map_2 = 20x20
SCALING_MAPS = {
    0: 5,
    1: 10,
    2: 15,
    3: 20,
}


def update_input(i, size):
    """Set the current scaling map and its opposite-corner target."""
    with open("input.json", "r") as f:
        params = json.load(f)

    params["startX"] = 0
    params["startY"] = 0
    params["targetX"] = size - 1
    params["targetY"] = size - 1
    params["map_file"] = f"maps/map_{i}.csv"

    with open("input.json", "w") as f:
        json.dump(params, f, indent=4)

    print(
        f"Map {i}: {size}x{size}, "
        f"start=(0, 0), target=({size - 1}, {size - 1})"
    )


def models(i):
    """Generate Gaussian base PRISM model and synthesized UMC model."""
    prism_model_generator_gaussian_exact_local_scaling.generate_model(i)

    infile = f"Applications/EvoChecker-master/models/model_{i}.prism"
    outfile = f"Applications/EvoChecker-master/models/model_{i}_umc.prism"

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
    # Generate map_0=5x5, map_1=15x15 and map_2=20x20.
    create_maps.create_3_maps()

    for i, size in SCALING_MAPS.items():
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
        fronts(i)

        print(f"Finished map {i} ({size}x{size})")


if __name__ == "__main__":
    os.makedirs("plots/fronts", exist_ok=True)
    os.makedirs("plots/box-plots", exist_ok=True)
    main()
