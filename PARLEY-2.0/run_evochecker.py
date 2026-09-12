import os
import time
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count


def run_task(args):
    i, rep = args

    # Do not change the working directory of the Python process.
    # Otherwise the second serial replication would try to chdir into
    # Applications/EvoChecker-master a second time.
    evochecker_dir = Path(__file__).resolve().parent / "Applications" / "EvoChecker-master"
    properties_name = f"{i}_{rep}.properties"
    properties_path = evochecker_dir / properties_name

    init_port = 10000 + i * 100 + rep * 10

    with open(properties_path, "w") as f:
        f.write(f"PROBLEM = ROBOT{i}_REP{rep}\n")
        f.write(f"       MODEL_TEMPLATE_FILE = models/model_{i}_umc.prism\n")
        f.write("       PROPERTIES_FILE = robot.pctl\n")
        f.write("       ALGORITHM = NSGAII\n")
        f.write("       POPULATION_SIZE = 100\n")
        f.write("       MAX_EVALUATIONS = 4000\n")
        f.write("       PROCESSORS = 6\n")
        f.write("       PLOT_PARETO_FRONT = false\n")
        f.write("       VERBOSE = true\n")
        f.write(f"       INIT_PORT = {init_port}\n")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "libs/runtime"

    # This blocks until this EvoChecker replication has completely finished.
    # Multiple run_task calls are executed concurrently by run().
    subprocess.run(
        ["java", "-jar", "./target/EvoChecker-1.1.0.jar", f"./{properties_name}"],
        cwd=evochecker_dir,
        env=env,
        check=True,
    )


def run(map_, replications):
    """Run all replications in parallel and return wall-clock runtime in seconds."""
    tasks = [(map_, rep) for rep in range(replications)]
    num_processes = min(replications, cpu_count())

    print(
        f"Starting {replications} EvoChecker replications in parallel "
        f"for map {map_} using {num_processes} Python processes"
    )

    start = time.time()
    with Pool(processes=num_processes) as pool:
        pool.map(run_task, tasks)
    wall_runtime = time.time() - start

    print(
        f"Finished all {replications} EvoChecker replications for map {map_} "
        f"in {wall_runtime:.3f} seconds"
    )
    return wall_runtime
