from os import times
import os

import create_maps
import prism_model_generator
import urc_synthesis
import run_evochecker_rq3_200 as run_evochecker

max_replications = 10

def evo_checker(i):
    # Invoke all EvoChecker replications in parallel and return wall-clock runtime.
    return run_evochecker.run(i, max_replications)

def main():
    # wurden schon erzeugt, daher nicht mehr nötig
    # create_maps.create_3_maps()
    # for i in range(0, 3):
    #     prism_model_generator.generate_model(i)
    maps = [0, 1, 2, 3] # selected_maps
    for i in maps:
        prism_model_generator.generate_model(i)
        outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
        infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
        # TODO umc_synthesis.manipulate_prism_model is currently broken
        urc_synthesis.manipulate_prism_model(infile, outfile, baseline=False) # vorher baseline=True, aber das ist nicht sinnvoll, da wir die Baseline ja erst berechnen wollen.
        print('Starting EvoChecker for map {0}'.format(str(i)))
        wall_start = times.time()
        evochecker_runtime = evo_checker(i)
        wall_time = times.time() - wall_start
        print(f"EvoChecker wall-clock runtime for map {i}: {evochecker_runtime:.3f} seconds")
        print(f"Measured outer wall-clock runtime for map {i}: {wall_time:.3f} seconds")

        # Store one file per map. A rerun of the same map replaces the old
        # measurement with the newest complete run.
        times_dir = "times_point_estimates_scaling"
        os.makedirs(times_dir, exist_ok=True)
        times_file = os.path.join(times_dir, f"map_{i}.txt")

        with open(times_file, "w") as f:
            f.write(f"WallClock: {evochecker_runtime:.3f}\n")


if __name__ == '__main__':
    main()
