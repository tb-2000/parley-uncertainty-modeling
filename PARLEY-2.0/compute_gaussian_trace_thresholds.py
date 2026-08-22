#!/usr/bin/env python3
"""
compute_gaussian_trace_thresholds.py

Berechnet pro Map die skalare Gaussian-Unsicherheit

    U_G(Sigma) = trace(Sigma) = var_x + var_y

fuer die h=0.1-quantisierten Kovarianzzustaende.

Analog zum Belief-State-Modell werden 10 Unsicherheitsschwellwerte erzeugt,
die den Schritten 1..10 seit dem letzten Knowledge-Update entsprechen.

Fuer jeden exakten Schritt k wird ueber alle moeglichen Update-Startpositionen
der maximale quantisierte Trace-Wert bestimmt. Anschliessend wird ein
kumulatives Maximum verwendet, damit die 10 Thresholds monoton nicht fallend
sind.

PRISM verwendet Integerwerte:
    trace_int = round(TRACE_SCALE * trace)
mit TRACE_SCALE=1000.

Eingaben:
    maps/map_<id>.csv
    gaussian_refined/gaussian_refined_<id>.json
    dijkstra.py

Ausgaben:
    gaussian_trace/gaussian_trace_<id>.json
    gaussian_trace/gaussian_trace_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import dijkstra

H = 0.1
TRACE_SCALE = 1000
DIRECTION_NAMES = ["west", "east", "south", "north"]
KNOWLEDGE_EFFECT = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}
Sigma = Tuple[float, float, float]


def round_to_grid(value: float, h: float) -> float:
    if value >= 0:
        k = math.floor(value / h + 0.5)
    else:
        k = math.ceil(value / h - 0.5)
    return k * h


def quantize_covariance(var_x: float, var_y: float, cov_xy: float, h: float) -> Sigma:
    qx = max(0.0, round_to_grid(var_x, h))
    qy = max(0.0, round_to_grid(var_y, h))
    qc = round_to_grid(cov_xy, h)

    if qx == 0.0 or qy == 0.0:
        qc = 0.0
    else:
        max_abs_cov = math.sqrt(qx * qy)
        max_grid_abs = math.floor(max_abs_cov / h + 1e-12) * h
        qc = min(max(qc, -max_grid_abs), max_grid_abs)

    return (round(qx, 12), round(qy, 12), round(qc, 12))


def add_sigma(a: Sigma, b: Sigma) -> Sigma:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def trace_int(sigma: Sigma, scale: int) -> int:
    return int(round((sigma[0] + sigma[1]) * scale))


def load_map(path: Path) -> List[List[str]]:
    with path.open("r", newline="") as f:
        raw = list(csv.reader(f))
    transposed = list(zip(*raw))
    return [list(row[::-1]) for row in transposed]


def is_obstacle(map_data, x, y):
    return int(map_data[x][y]) > 9


def build_policy(map_data, target_x, target_y):
    raw = dijkstra.compute_directions(map_data, (target_x, target_y))
    directions = list(zip(*raw))
    policy = {}
    size = len(map_data)
    for x in range(size):
        for y in range(size):
            d = int(directions[y][x])
            if d < 4:
                policy[(x, y)] = DIRECTION_NAMES[d]
    return policy


def apply_action(x, y, action, n):
    dx, dy = KNOWLEDGE_EFFECT[action]
    return min(max(x + dx, 0), n), min(max(y + dy, 0), n)


def robot_outcomes(action, p):
    intended = 1.0 - 3.0 * p
    if action == "east":
        return [(intended,1,0),(p,0,1),(p,0,-1),(p,-1,0)]
    if action == "west":
        return [(p,1,0),(p,0,1),(p,0,-1),(intended,-1,0)]
    if action == "north":
        return [(p,1,0),(intended,0,1),(p,0,-1),(p,-1,0)]
    if action == "south":
        return [(p,1,0),(p,0,1),(intended,0,-1),(p,-1,0)]
    raise ValueError(action)


def motion_covariance(x, y, action, n, p):
    samples = []
    for prob, dx, dy in robot_outcomes(action, p):
        nx = min(max(x + dx, 0), n)
        ny = min(max(y + dy, 0), n)
        samples.append((prob, float(nx-x), float(ny-y)))

    mx = sum(prob*dx for prob,dx,_ in samples)
    my = sum(prob*dy for prob,_,dy in samples)
    vx = sum(prob*(dx-mx)**2 for prob,dx,_ in samples)
    vy = sum(prob*(dy-my)**2 for prob,_,dy in samples)
    cxy = sum(prob*(dx-mx)*(dy-my) for prob,dx,dy in samples)
    return (vx, vy, cxy)


def analyse_map(map_id, maps_dir, refined_dir, output_dir, target_x, target_y, p, h, max_steps, scale):
    map_data = load_map(maps_dir / f"map_{map_id}.csv")
    policy = build_policy(map_data, target_x, target_y)
    n = len(map_data) - 1

    refined_path = refined_dir / f"gaussian_refined_{map_id}.json"
    with refined_path.open("r") as f:
        refined = json.load(f)

    # Map quantized Sigma -> exact gvar ID used by refined model.
    sigma_to_gvar = {}
    gvar_trace = {}
    for row in refined["gvars"]:
        sigma = (
            round(float(row["var_x"]), 12),
            round(float(row["var_y"]), 12),
            round(float(row.get("cov_xy", 0.0)), 12),
        )
        gvar = int(row["gvar"])
        sigma_to_gvar[sigma] = gvar
        gvar_trace[gvar] = trace_int(sigma, scale)

    traces_by_step = {k: [] for k in range(1, max_steps + 1)}

    for sx in range(len(map_data)):
        for sy in range(len(map_data)):
            if is_obstacle(map_data, sx, sy):
                continue

            x, y = sx, sy
            sigma_raw = (0.0, 0.0, 0.0)

            for step in range(1, max_steps + 1):
                action = policy.get((x, y))
                if action is None:
                    break

                q = motion_covariance(x, y, action, n, p)
                sigma_raw = add_sigma(sigma_raw, q)
                x, y = apply_action(x, y, action, n)

                sigma_q = quantize_covariance(*sigma_raw, h)
                if sigma_q not in sigma_to_gvar:
                    raise ValueError(
                        f"Map {map_id}: quantized Sigma {sigma_q} at step {step} "
                        "not found in gaussian_refined gvars."
                    )

                traces_by_step[step].append(trace_int(sigma_q, scale))

    # Threshold k = maximum quantized trace after exactly k steps.
    # Cumulative max ensures monotonicity.
    thresholds = []
    running = 0
    per_step = {}
    for step in range(1, max_steps + 1):
        values = traces_by_step[step]
        if not values:
            exact_max = running
        else:
            exact_max = max(values)
        running = max(running, exact_max)
        thresholds.append(running)
        per_step[str(step)] = {
            "min_trace": min(values) if values else None,
            "mean_trace": (sum(values)/len(values)) if values else None,
            "max_trace": exact_max,
            "threshold": running,
            "samples": len(values),
        }

    # If two consecutive steps have same threshold, keep them. This mirrors
    # the 10 decision choices, although two choices may be behaviorally equal.
    # Group gvars by their actual quantized trace for PRISM formulas.
    trace_groups = {}
    for gvar, value in sorted(gvar_trace.items()):
        trace_groups.setdefault(str(value), []).append(gvar)

    output = {
        "map": map_id,
        "h": h,
        "trace_scale": scale,
        "metric": "trace(Sigma)=var_x+var_y",
        "threshold_method": "cumulative maximum quantized trace across all update-start positions at exact steps 1..10",
        "thresholds": {str(i+1): thresholds[i] for i in range(max_steps)},
        "per_step_statistics": per_step,
        "gvar_trace": {str(k): v for k,v in sorted(gvar_trace.items())},
        "trace_groups": trace_groups,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"gaussian_trace_{map_id}.json").open("w") as f:
        json.dump(output, f, indent=2)

    return {
        "map": map_id,
        **{f"threshold_{i+1}": thresholds[i] for i in range(max_steps)},
        "unique_trace_values": len(trace_groups),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--maps-dir", type=Path, default=Path("maps"))
    p.add_argument("--refined-dir", type=Path, default=Path("gaussian_refined"))
    p.add_argument("--output-dir", type=Path, default=Path("gaussian_trace"))
    p.add_argument("--start-map", type=int, default=10)
    p.add_argument("--end-map", type=int, default=99)
    p.add_argument("--target-x", type=int, default=9)
    p.add_argument("--target-y", type=int, default=9)
    p.add_argument("--p", type=float, default=0.01)
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--trace-scale", type=int, default=TRACE_SCALE)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for map_id in range(args.start_map, args.end_map + 1):
        map_path = args.maps_dir / f"map_{map_id}.csv"
        refined_path = args.refined_dir / f"gaussian_refined_{map_id}.json"
        if not map_path.exists() or not refined_path.exists():
            print(f"[skip] map {map_id}: missing input")
            continue
        row = analyse_map(
            map_id, args.maps_dir, args.refined_dir, args.output_dir,
            args.target_x, args.target_y, args.p, args.h,
            args.max_steps, args.trace_scale
        )
        rows.append(row)
        print(
            f"[map {map_id}] thresholds="
            + ",".join(str(row[f'threshold_{i}']) for i in range(1,11))
        )

    if rows:
        with (args.output_dir / "gaussian_trace_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
