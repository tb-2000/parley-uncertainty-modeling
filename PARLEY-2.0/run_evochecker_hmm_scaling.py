"""Dedicated EvoChecker runner for HMM RQ3 scaling (parallel replications)."""
import os
import subprocess
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

EVO_DIR = Path(__file__).resolve().parent / "Applications" / "EvoChecker-master"
PORT_BASE = 30000
MAX_EVALUATIONS = 20
POPULATION_SIZE = 10
GRID_SIZES = {0: 5, 1: 10, 2: 15, 3: 20}


def write_objectives(map_id):
    """Write the EvoChecker-specific objective annotations once per map."""
    if map_id not in GRID_SIZES:
        raise ValueError(f"Unsupported HMM scaling map: {map_id}")
    target = GRID_SIZES[map_id] - 1
    path = EVO_DIR / f"robot_rq3_hmm_map_{map_id}.pctl"
    path.write_text(
        "//Objective, max\n"
        f"P=? [ F (x={target} & y={target} & crashed=0) ]\n\n"
        "//Objective, min\n"
        'R{"cost"}=? [ C<=100 ]\n',
        encoding="utf-8",
    )
    return path


def run_task(args):
    map_id, rep = args
    properties_name = f"hmm_scaling_{map_id}_{rep}.properties"
    properties_path = EVO_DIR / properties_name
    model_name = f"models/model_hmm_{map_id}_umc.prism"
    pctl_name = f"robot_rq3_hmm_map_{map_id}.pctl"

    for required in (EVO_DIR / model_name, EVO_DIR / pctl_name):
        if not required.is_file():
            raise FileNotFoundError(f"Missing HMM scaling input: {required}")

    init_port = PORT_BASE + map_id * 1000 + rep * 20
    properties_path.write_text(
        f"PROBLEM = ROBOT_HMM_RQ3_MAP{map_id}_REP{rep}\n"
        f"MODEL_TEMPLATE_FILE = {model_name}\n"
        f"PROPERTIES_FILE = {pctl_name}\n"
        "ALGORITHM = NSGAII\n"
        f"POPULATION_SIZE = {POPULATION_SIZE}\n"
        f"MAX_EVALUATIONS = {MAX_EVALUATIONS}\n"
        "PROCESSORS = 1\n"
        "PLOT_PARETO_FRONT = false\n"
        "VERBOSE = true\n"
        f"INIT_PORT = {init_port}\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    runtime_libs = str(EVO_DIR / "libs" / "runtime")
    env["LD_LIBRARY_PATH"] = runtime_libs + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    print(f"HMM scaling: map={map_id}, rep={rep}, port={init_port}", flush=True)
    subprocess.run(
        ["java", "-jar", "./target/EvoChecker-1.1.0.jar", f"./{properties_name}"],
        cwd=EVO_DIR,
        env=env,
        check=True,
    )


def run(map_id, replications):
    """Run replications concurrently; return total elapsed wall-clock seconds."""
    if replications < 1:
        raise ValueError("replications must be at least 1")
    if not (EVO_DIR / f"models/model_hmm_{map_id}_umc.prism").is_file():
        raise FileNotFoundError(f"Missing HMM model for map {map_id}")

    # Avoid concurrent writes to the same PCTL file from multiple workers.
    write_objectives(map_id)
    tasks = [(map_id, rep) for rep in range(replications)]
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    num_processes = min(replications, max(1, allocated_cpus))

    print(
        f"Starting {replications} EvoChecker replications in parallel "
        f"for HMM map {map_id} using {num_processes} processes",
        flush=True,
    )
    start = time.perf_counter()
    with Pool(processes=num_processes) as pool:
        pool.map(run_task, tasks)
    elapsed = time.perf_counter() - start
    print(
        f"HMM scaling map {map_id}: {replications} parallel replications "
        f"completed in {elapsed:.3f}s",
        flush=True,
    )
    return elapsed
