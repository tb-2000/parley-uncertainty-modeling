import os

import create_maps
import prism_model_generator_interval
import prism_caller
import run_evochecker
import evaluation
import plot_fronts
import urc_synthesis_interval
import time

max_replications = 10 # 10


def maps():
    create_maps.create_90_maps()


def models(i):
    # prism_model_generator.generate_model(i)
    prism_model_generator_interval.generate_model(i)
    infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
    outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'
    # TODO umc_synthesis.manipulate_prism_model is currently broken
    urc_synthesis_interval.manipulate_prism_model(infile, outfile, baseline=False) # vorher baseline=True, aber das ist nicht sinnvoll, da wir die Baseline ja erst berechnen wollen.


def baseline(i):
    baseline_file = f'Applications/EvoChecker-master/data/ROBOT{i}_BASELINE/Front'
    infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
    os.makedirs(f'Applications/EvoChecker-master/data/ROBOT{i}_BASELINE', exist_ok=True)
    with open(baseline_file, 'w') as b_file:
        for period in range(1, 11):
            b_file.write(prism_caller.compute_baseline(infile, period))
            if period < 10:
                b_file.write('\n')
            print('finished baseline map {0}, value {1}'.format(str(i), str(period)))


def evo_checker(i):
    # invoke EvoChecker
    run_evochecker.run(i, max_replications)


def fronts(i):
    for period in range(max_replications):
        plot_fronts.plot_pareto_front(i, period)

## teste auf zufällig gewählten 50 maps
## [14, 16, 20, 21, 22, 23, 29, 30, 31, 32, 33, 36, 37,
#  40, 43, 44, 46, 47, 48, 49, 50, 52, 53, 54, 55, 
# 56, 57, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 
# 75, 76, 79, 81, 82, 83, 85, 86, 87, 89, 90, 97]
def main():
    selected_maps = [14, 16, 20, 21, 22, 23, 29, 30, 31, 32, 33, 36, 37,
                     40, 43, 44, 46, 47, 48, 49, 50, 52, 53, 54, 55,
                     56, 57, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73,
                     75, 76, 79, 81, 82, 83, 85, 86, 87, 89, 90, 97]
                    
    # maps()
    maps = [23] # selected_maps
    for i in maps: # lasse auf map 22 laufen
        models(i)
        #baseline(i)
        print('Starting EvoChecker for map {0}'.format(str(i)))
        start = time.time()
        evo_checker(i)
        end = time.time()
        runtime = end - start
        print(f"Total runtime of EvoChecker for map {i} is {runtime:.3f} seconds")

        # time speichern
        filename = "times.txt"
        times = {}
        try:
            with open(filename, "r") as f:
                for line in f:
                    map_str, time_str = line.strip().split(": ")
                    map_id = int(map_str.replace("Map ", ""))
                    times[map_id] = float(time_str)
        except FileNotFoundError:
            pass
        times[i] = runtime
        with open(filename, "w") as f:
            for map_id in sorted(times):
                f.write(f"Map {map_id}: {times[map_id]:.3f}\n")
        
        fronts(i)
        print(f'Finished map {i}')
    # evaluation
    #evaluation.main()



if __name__ == '__main__':
    os.makedirs('plots/fronts', exist_ok=True)
    os.makedirs('plots/box-plots', exist_ok=True)
    main()
