import os
import time
from multiprocessing import Pool, cpu_count


def run_task(args):
    os.chdir('Applications/EvoChecker-master')
    os.environ['LD_LIBRARY_PATH'] = "libs/runtime"
    i, rep = args
    path = "./{0}_{1}.properties".format(str(i), str(rep))
    open(path, "w").close()

    init_port = 10000 + i * 100 + rep * 10

    with open(path, 'a') as f:
        f.write("PROBLEM = ROBOT{0}_REP{1}\n".format(str(i), str(rep)))
        f.write("       MODEL_TEMPLATE_FILE = models/model_{0}_umc.prism\n".format(str(i)))
        f.write("       PROPERTIES_FILE = robot.pctl\n")
        f.write("       ALGORITHM = NSGAII\n")
        f.write("       POPULATION_SIZE = 100\n") # 100
        f.write("       MAX_EVALUATIONS = 4000\n") # 4000
        f.write("       PROCESSORS = 6\n") # 6 PrismExecutor workers per replication
        f.write("       PLOT_PARETO_FRONT = false\n")
        f.write("       VERBOSE = true\n")
        #f.write("       INIT_PORT = 55{0}\n".format(str(i)))
        f.write("       INIT_PORT = {init_port}\n".format(init_port=init_port))
    # Note: INIT_PORT doesn't have an effect https://github.com/gerasimou/EvoChecker/issues/11

    os.system('java -jar ./target/EvoChecker-1.1.0.jar ' + path)


def run(map_, replications):
    """Run all replications sequentially and return their runtimes in seconds."""
    runtimes = []

    for rep in range(replications):
        print(f"Starting EvoChecker replication {rep + 1}/{replications} for map {map_}")
        start = time.time()

        # Run exactly one replication. run_task blocks until EvoChecker,
        # including its 6 PrismExecutor workers, has finished.
        run_task((map_, rep))

        runtime = time.time() - start
        runtimes.append(runtime)
        print(
            f"Finished replication {rep + 1}/{replications} for map {map_} "
            f"in {runtime:.3f} seconds"
        )

    return runtimes
