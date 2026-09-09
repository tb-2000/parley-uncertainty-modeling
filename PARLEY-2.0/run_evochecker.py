import os
import time
import subprocess
from pathlib import Path


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

    # This blocks until the current EvoChecker replication has completely
    # finished. Then run() starts the next replication.
    subprocess.run(
        ["java", "-jar", "./target/EvoChecker-1.1.0.jar", f"./{properties_name}"],
        cwd=evochecker_dir,
        env=env,
        check=True,
    )


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
