import json
import os
import time
from pathlib import Path

import csv
import prism_model_generator_hmm_exact_local_scaling
import urc_synthesis_hmm_exact_local_scaling
import run_evochecker_hmm_scaling
import plot_fronts


max_replications = 10

SCALING_MAPS = {
    0: 5,
    1: 10,
    2: 15,
    3: 20,
}

HMM_BELIEF_ROOT = Path("hmm_belief_models")



def validate_map_size(map_id, size):
    path = Path(f"maps/map_{map_id}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; create the {size}x{size} scaling map first.")
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError(f"{path}: expected {size}x{size}, got {len(rows)} rows.")

def update_input(i, size):
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


def validate_hmm_belief_model(i):
    """
    RQ3 requires exact_local.json to have been generated with max_steps=10,
    resulting in exactly 10 HMM-MSE thresholds.
    """
    path = HMM_BELIEF_ROOT / f"map_{i}" / "exact_local.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate the exact-local HMM belief model "
            f"for map {i} with max_steps=10 first."
        )

    with path.open("r") as f:
        data = json.load(f)

    thresholds = data.get("thresholds", [])
    if int(data.get("max_steps", -1)) != 10:
        raise ValueError(f"{path}: expected max_steps=10; regenerate the HMM belief file.")
    if len(thresholds) != 10:
        raise ValueError(
            f"{path} contains {len(thresholds)} thresholds, but RQ3 scaling "
            f"requires 10. Regenerate it with max_steps=10."
        )


def models(i):
    """Generate model_hmm_i.prism and model_hmm_i_umc.prism."""
    validate_hmm_belief_model(i)

    prism_model_generator_hmm_exact_local_scaling.generate_model(i)

    infile = f"Applications/EvoChecker-master/models/model_hmm_{i}.prism"
    outfile = f"Applications/EvoChecker-master/models/model_hmm_{i}_umc.prism"

    urc_synthesis_hmm_exact_local_scaling.manipulate_prism_model(
        infile,
        outfile,
        baseline=False,
    )


def write_pctl(i, size):
    """Write map-specific RQ3 properties, matching the target of this grid."""
    evo_dir = Path("Applications/EvoChecker-master")
    pctl = evo_dir / f"robot_rq3_hmm_map_{i}.pctl"
    target = size - 1
    pctl.write_text(
        f"P=? [ F (x={target}&y={target}&crashed=0) ]\n"
        'R{"cost"}=? [ C<=100 ]\n',
        encoding="utf-8",
    )
    print(f"Generated {pctl} (target=({target},{target}), reward horizon=100)")


def evo_checker(i):
    return run_evochecker_hmm_scaling.run(i, max_replications)


def fronts(i):
    for period in range(max_replications):
        plot_fronts.plot_pareto_front(i, period)


def save_runtime(i, size, runtime):
    times_dir = "times_hmm_scaling"
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
    # Existing maps must match SCALING_MAPS; do not overwrite them here.

    for i, size in SCALING_MAPS.items():
        print("=" * 70)
        print(f"Starting HMM scalability experiment: map {i}, {size}x{size}")
        print("=" * 70)

        validate_map_size(i, size)
        update_input(i, size)

        # Requires hmm_belief_models/map_i/exact_local.json with 10 thresholds.
        # Creates both model_hmm_i.prism and model_hmm_i_umc.prism.
        models(i)
        write_pctl(i, size)

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
