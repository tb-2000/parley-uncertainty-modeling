import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import prism_model_generator_interval_scaling
import run_evochecker_interval_rq3_200 as run_evochecker
import plot_fronts


max_replications = 10

SCALING_MAPS = {
    2: 15,
    3: 20,
}

MODELS_DIR = Path("Applications/EvoChecker-master/models")
THRESHOLD_OUTPUT_DIR = Path("interval_thresholds_scaling_analysis")
THRESHOLD_FILE = Path("interval_thresholds_scaling.py")


def update_input(i, size):
    with open("input_interval_scaling.json", "r") as f:
        params = json.load(f)

    map_path = Path(f"maps/map_{i}.csv")
    import csv
    with map_path.open(newline="") as map_file:
        rows = list(csv.reader(map_file))
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError(f"{map_path}: expected {size}x{size} map")
    if int(rows[-1][0]) > 9 or int(rows[0][-1]) > 9:
        raise ValueError(f"{map_path}: start (0,0) or target ({size-1},{size-1}) blocked")
    params["startX"] = 0
    params["startY"] = 0
    params["targetX"] = size - 1
    params["targetY"] = size - 1
    params["map_file"] = f"maps/map_{i}.csv"

    with open("input_interval_scaling.json", "w") as f:
        json.dump(params, f, indent=4)

    print(
        f"Map {i}: {size}x{size}, "
        f"start=(0, 0), target=({size - 1}, {size - 1})"
    )


def generate_base_models():
    """Generate map_0..3 base PRISM models without precomputed thresholds."""
    for i, size in SCALING_MAPS.items():
        update_input(i, size)
        prism_model_generator_interval_scaling.generate_model(i)


def calculate_thresholds():
    """
    Analyse mean interval widths after steps 1..10 for model_0..3.
    The analysis writes thresholds_per_map.py into its output directory.
    """
    subprocess.run(
        [
            sys.executable,
            "analyze_interval_thresholds_scaling.py",
            str(MODELS_DIR),
            "--steps",
            "10",
            "--output-dir",
            str(THRESHOLD_OUTPUT_DIR),
            "--maps",
            *[str(i) for i in SCALING_MAPS],
        ],
        check=True,
    )

    generated = THRESHOLD_OUTPUT_DIR / "thresholds_per_map.py"
    if not generated.exists():
        raise FileNotFoundError(
            f"Threshold analysis did not create {generated}"
        )

    # The scaling URC module imports interval_thresholds_scaling.py.
    THRESHOLD_FILE.write_text(
        generated.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(f"Scaling thresholds written to {THRESHOLD_FILE}")


def synthesize_models():
    """
    Import the URC module only after interval_thresholds_scaling.py exists,
    then create the UMC models for maps 0..3.
    """
    importlib.invalidate_caches()
    urc_module = importlib.import_module("urc_synthesis_interval_scaling")
    urc_module = importlib.reload(urc_module)

    for i in SCALING_MAPS:
        infile = MODELS_DIR / f"model_interval_{i}.prism"
        outfile = MODELS_DIR / f"model_interval_{i}_umc.prism"

        urc_module.manipulate_prism_model(
            str(infile),
            str(outfile),
            baseline=False,
        )


def verify_properties(i, size):
    pctl = Path("Applications/EvoChecker-master") / f"robot_rq3_map_{i}.pctl"
    if not pctl.is_file():
        raise FileNotFoundError(f"Missing {pctl}")
    import re
    compact = re.sub(r"\s+", "", pctl.read_text(encoding="utf-8"))
    if not re.search(rf"P=\?\[F\(x={size-1}&y={size-1}&crashed=0\)\]", compact):
        raise ValueError(f"{pctl}: expected success goal ({size-1},{size-1}); inspect PCTL")

def evo_checker(i):
    return run_evochecker.run(i, max_replications)


def fronts(i):
    for period in range(max_replications):
        plot_fronts.plot_pareto_front(i, period)


def save_runtime(i, size, runtime):
    times_dir = Path("times_interval_scaling")
    times_dir.mkdir(exist_ok=True)

    times_file = times_dir / f"map_{i}_{size}x{size}.txt"

    with times_file.open("w") as f:
        f.write(f"Map: {i}\n")
        f.write(f"Size: {size}x{size}\n")
        f.write(f"WallClock: {runtime:.3f}\n")


def main():
    # 1. Reuse existing scaling maps; never overwrite them.
    #create_maps.create_4_maps()

    # 2. Generate the base interval models with size-dependent targets.
    generate_base_models()

    # 3. Determine map-specific interval thresholds from steps 1..10.
    calculate_thresholds()

    # 4. Build the URC/UMC models using those thresholds.
    synthesize_models()

    # 5. Run EvoChecker and record scalability runtimes.
    for i, size in SCALING_MAPS.items():
        print("=" * 70)
        print(f"Starting EvoChecker: map {i}, {size}x{size}")
        print("=" * 70)

        verify_properties(i, size)
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
