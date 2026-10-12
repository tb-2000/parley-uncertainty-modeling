"""Dedicated EvoChecker runner for HMM scaling (sequential replications)."""
import os
import subprocess
import time
from pathlib import Path

EVO_DIR = Path(__file__).resolve().parent / "Applications" / "EvoChecker-master"
PORT_BASE = 30000  # separate from the standard 10000+map*100+rep*10 range

def run_task(map_id, rep):
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
        "POPULATION_SIZE = 10\n"
        "MAX_EVALUATIONS = 20\n"
        "PROCESSORS = 1\n"
        "PLOT_PARETO_FRONT = false\n"
        "VERBOSE = true\n"
        f"INIT_PORT = {init_port}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(EVO_DIR / "libs" / "runtime") + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    print(f"HMM scaling: map={map_id}, rep={rep}, port={init_port}", flush=True)
    subprocess.run(
        ["java", "-jar", "./target/EvoChecker-1.1.0.jar", f"./{properties_name}"],
        cwd=EVO_DIR, env=env, check=True,
    )

def run(map_id, replications):
    start = time.perf_counter()
    for rep in range(replications):
        run_task(map_id, rep)
    elapsed = time.perf_counter() - start
    print(f"HMM scaling map {map_id}: {replications} serial replications in {elapsed:.3f}s", flush=True)
    return elapsed
