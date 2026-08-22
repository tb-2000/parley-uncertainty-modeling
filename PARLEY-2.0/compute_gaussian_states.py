#!/usr/bin/env python3
"""
compute_gaussian_states.py

Berechnet pro Map die tatsächlich erreichbaren diskretisierten Gaussian-
Kovarianzzustände (gvars) unter dem bestehenden Dijkstra/MAPE-Controller.

Grundidee
---------
Nach einem Knowledge-Update gilt Sigma = 0 (bzw. SENSOR_VAR, falls gesetzt).
Von jeder begehbaren Knowledge-Position wird der MAPE-Controller bis zu
MAX_STEPS Schritte verfolgt. Für jede ausgeführte Aktion wird die Kovarianz
des Bewegungsfehlers Q bestimmt und

    Sigma_{t+1} = Sigma_t + Q(xhat_t, yhat_t, action_t)

verwendet.

Standardmäßig wird Q positionsabhängig berechnet. Dadurch werden die min/max-
Clipping-Effekte an den Grenzen des 10x10-Grids berücksichtigt. Mit
--q-mode action kann stattdessen nur die aktionsabhängige Innenraum-Kovarianz
Q(action) verwendet werden.

Für jede Map entstehen:
    gaussian_states/gaussian_states_<map>.json
    gaussian_states/gaussian_states_summary.csv
    gaussian_states/gaussian_states_summary.json

Das Skript erzeugt noch KEIN PRISM-Knowledge-Modul. Es dient zunächst dazu,
empirisch zu bestimmen, wie viele verschiedene Sigma-Zustände/gvars auf den
Maps 10..99 benötigt werden.
"""

# NOTE: Dieses Skript berechnet unquantisierte Roh-Sigma-Zustaende.
# Die Produktions-Quantisierung h=0.05 wird erst in den nachgelagerten
# Quantisierungs-/Refinement-Skripten angewendet.

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import dijkstra


DIRECTION_NAMES = ["west", "east", "south", "north"]

# Deterministische Änderung des Knowledge-Mittelwerts xhat/yhat.
KNOWLEDGE_EFFECT = {
    "west":  (-1, 0),
    "east":  (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}


@dataclass(frozen=True)
class Covariance:
    var_x: float
    var_y: float
    cov_xy: float = 0.0

    def add(self, other: "Covariance") -> "Covariance":
        return Covariance(
            self.var_x + other.var_x,
            self.var_y + other.var_y,
            self.cov_xy + other.cov_xy,
        )


def covariance_key(sigma: Covariance, digits: int) -> Tuple[float, float, float]:
    """Numerisch äquivalente Sigma-Matrizen zusammenfassen."""
    def clean(v: float) -> float:
        r = round(v, digits)
        return 0.0 if abs(r) < 10 ** (-digits) else r

    return (clean(sigma.var_x), clean(sigma.var_y), clean(sigma.cov_xy))


def load_map_for_generator(path: Path) -> List[List[str]]:
    """
    Gleiche Koordinatentransformation wie prism_model_generator.build_map():
    CSV lesen -> transponieren -> jede transponierte Zeile umdrehen.
    """
    with path.open("r", newline="") as f:
        raw = list(csv.reader(f))

    transposed = list(zip(*raw))
    return [list(row[::-1]) for row in transposed]


def is_obstacle(map_data: Sequence[Sequence[str]], x: int, y: int) -> bool:
    return int(map_data[x][y]) > 9


def build_mape_policy(
    map_data: Sequence[Sequence[str]],
    target_x: int,
    target_y: int,
) -> Dict[Tuple[int, int], str]:
    """
    Rekonstruiert genau die Policy, die adaptation_mape_controller() schreibt.
    """
    target_pos = (target_x, target_y)
    raw_directions = dijkstra.compute_directions(map_data, target_pos)

    # prism_model_generator transponiert das Ergebnis anschließend.
    controller_directions = list(zip(*raw_directions))

    size = len(map_data)
    policy: Dict[Tuple[int, int], str] = {}

    for x in range(size):
        for y in range(size):
            direction = int(controller_directions[y][x])
            if direction < 4:
                policy[(x, y)] = DIRECTION_NAMES[direction]

    return policy


def apply_knowledge_action(
    x: int, y: int, action: str, n: int
) -> Tuple[int, int]:
    """Entspricht den min/max-Updates von xhat/yhat im Knowledge-Modul."""
    dx, dy = KNOWLEDGE_EFFECT[action]
    return (
        min(max(x + dx, 0), n),
        min(max(y + dy, 0), n),
    )


def robot_outcomes(action: str, p: float) -> List[Tuple[float, int, int]]:
    """
    Bewegungs-Outcomes entsprechend dem Robot-Modul.

    Rückgabe: [(probability, dx, dy), ...]
    """
    intended = 1.0 - 3.0 * p

    if action == "east":
        return [
            (intended, 1, 0),
            (p, 0, 1),
            (p, 0, -1),
            (p, -1, 0),
        ]
    if action == "west":
        return [
            (p, 1, 0),
            (p, 0, 1),
            (p, 0, -1),
            (intended, -1, 0),
        ]
    if action == "north":
        return [
            (p, 1, 0),
            (intended, 0, 1),
            (p, 0, -1),
            (p, -1, 0),
        ]
    if action == "south":
        return [
            (p, 1, 0),
            (p, 0, 1),
            (intended, 0, -1),
            (p, -1, 0),
        ]
    raise ValueError(f"Unknown action: {action}")


def motion_covariance(
    x: int,
    y: int,
    action: str,
    n: int,
    p: float,
    q_mode: str,
) -> Covariance:
    """
    Berechnet Q aus den vier probabilistischen Robot-Outcomes.

    q_mode='position':
        min/max-Clipping am Grid-Rand wird für die aktuelle Position
        berücksichtigt.

    q_mode='action':
        Q wird so berechnet, als läge die Position im Grid-Innenraum.
        Dadurch hängt Q nur von der Aktion ab.
    """
    samples: List[Tuple[float, float, float]] = []

    for prob, dx, dy in robot_outcomes(action, p):
        if q_mode == "position":
            nx = min(max(x + dx, 0), n)
            ny = min(max(y + dy, 0), n)
            actual_dx = float(nx - x)
            actual_dy = float(ny - y)
        else:
            actual_dx = float(dx)
            actual_dy = float(dy)

        samples.append((prob, actual_dx, actual_dy))

    mean_dx = sum(prob * dx for prob, dx, _ in samples)
    mean_dy = sum(prob * dy for prob, _, dy in samples)

    var_x = sum(prob * (dx - mean_dx) ** 2 for prob, dx, _ in samples)
    var_y = sum(prob * (dy - mean_dy) ** 2 for prob, _, dy in samples)
    cov_xy = sum(
        prob * (dx - mean_dx) * (dy - mean_dy)
        for prob, dx, dy in samples
    )

    return Covariance(var_x, var_y, cov_xy)


def enumerate_gaussian_states(
    map_data: Sequence[Sequence[str]],
    policy: Dict[Tuple[int, int], str],
    p: float,
    max_steps: int,
    digits: int,
    q_mode: str,
    sensor_var_x: float = 0.0,
    sensor_var_y: float = 0.0,
    sensor_cov_xy: float = 0.0,
) -> Tuple[
    Dict[Tuple[float, float, float], dict],
    List[dict],
    Dict[Tuple[int, int], List[Tuple[float, float, float]]],
]:
    """
    Enumeriert alle Sigma-Zustände, die auftreten können, wenn ein Knowledge-
    Update an irgendeiner begehbaren Position erfolgt und anschließend bis zu
    max_steps MAPE-Aktionen ohne weiteres Update ausgeführt werden.

    Weil der MAPE-Controller xhat/yhat deterministisch verändert, gibt es pro
    Startposition genau eine Controller-Trajektorie.
    """
    size = len(map_data)
    n = size - 1

    sigma0 = Covariance(sensor_var_x, sensor_var_y, sensor_cov_xy)

    states: Dict[Tuple[float, float, float], dict] = {}
    trajectories: List[dict] = []
    per_start_keys: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}

    def register(
        sigma: Covariance,
        start: Tuple[int, int],
        pos: Tuple[int, int],
        step: int,
        action: Optional[str],
    ) -> Tuple[float, float, float]:
        key = covariance_key(sigma, digits)
        entry = states.setdefault(
            key,
            {
                "sigma": {
                    "var_x": key[0],
                    "var_y": key[1],
                    "cov_xy": key[2],
                },
                "occurrences": 0,
                "first_example": {
                    "start": list(start),
                    "position": list(pos),
                    "step": step,
                    "action": action,
                },
            },
        )
        entry["occurrences"] += 1
        return key

    for start_x in range(size):
        for start_y in range(size):
            if is_obstacle(map_data, start_x, start_y):
                continue

            start = (start_x, start_y)
            x, y = start
            sigma = sigma0

            keys_for_start: List[Tuple[float, float, float]] = []
            keys_for_start.append(register(sigma, start, (x, y), 0, None))

            trace = {
                "start": [start_x, start_y],
                "steps": [],
                "stop_reason": None,
            }

            for step in range(1, max_steps + 1):
                action = policy.get((x, y))
                if action is None:
                    trace["stop_reason"] = "no_mape_action"
                    break

                q = motion_covariance(x, y, action, n, p, q_mode)
                sigma = sigma.add(q)

                old_x, old_y = x, y
                x, y = apply_knowledge_action(x, y, action, n)

                key = register(sigma, start, (x, y), step, action)
                keys_for_start.append(key)

                trace["steps"].append(
                    {
                        "step": step,
                        "from": [old_x, old_y],
                        "action": action,
                        "to": [x, y],
                        "Q": {
                            "var_x": round(q.var_x, digits),
                            "var_y": round(q.var_y, digits),
                            "cov_xy": round(q.cov_xy, digits),
                        },
                        "Sigma": {
                            "var_x": key[0],
                            "var_y": key[1],
                            "cov_xy": key[2],
                        },
                    }
                )

                # Ziel besitzt typischerweise keine weitere MAPE-Aktion.
                if (x, y) not in policy:
                    trace["stop_reason"] = "target_or_no_mape_action"
                    break
            else:
                trace["stop_reason"] = "max_steps"

            per_start_keys[start] = keys_for_start
            trajectories.append(trace)

    return states, trajectories, per_start_keys


def assign_gvars(
    states: Dict[Tuple[float, float, float], dict]
) -> List[dict]:
    """
    Deterministische gvar-Nummerierung:
    zuerst kleine Gesamtvarianz, dann var_x, var_y, cov_xy.
    """
    ordered_keys = sorted(
        states.keys(),
        key=lambda k: (k[0] + k[1], k[0], k[1], k[2]),
    )

    result: List[dict] = []
    for gvar, key in enumerate(ordered_keys):
        src = states[key]
        result.append(
            {
                "gvar": gvar,
                "var_x": key[0],
                "var_y": key[1],
                "cov_xy": key[2],
                "trace": key[0] + key[1],
                "occurrences": src["occurrences"],
                "first_example": src["first_example"],
            }
        )
    return result


def analyse_map(
    map_id: int,
    map_path: Path,
    output_dir: Path,
    target_x: int,
    target_y: int,
    p: float,
    max_steps: int,
    digits: int,
    q_mode: str,
    sensor_var_x: float,
    sensor_var_y: float,
    sensor_cov_xy: float,
    save_trajectories: bool,
) -> dict:
    map_data = load_map_for_generator(map_path)
    size = len(map_data)

    if not (0 <= target_x < size and 0 <= target_y < size):
        raise ValueError(
            f"Target ({target_x},{target_y}) liegt außerhalb von map {map_id} "
            f"mit Größe {size}."
        )

    policy = build_mape_policy(map_data, target_x, target_y)

    states, trajectories, per_start = enumerate_gaussian_states(
        map_data=map_data,
        policy=policy,
        p=p,
        max_steps=max_steps,
        digits=digits,
        q_mode=q_mode,
        sensor_var_x=sensor_var_x,
        sensor_var_y=sensor_var_y,
        sensor_cov_xy=sensor_cov_xy,
    )

    gvars = assign_gvars(states)

    key_to_gvar = {
        (g["var_x"], g["var_y"], g["cov_xy"]): g["gvar"]
        for g in gvars
    }

    start_sequences = {}
    for start, keys in per_start.items():
        start_sequences[f"{start[0]},{start[1]}"] = [
            key_to_gvar[key] for key in keys
        ]

    out = {
        "map": map_id,
        "map_file": str(map_path),
        "map_size": size,
        "target": [target_x, target_y],
        "p": p,
        "max_steps_without_update": max_steps,
        "q_mode": q_mode,
        "round_digits": digits,
        "sensor_covariance": {
            "var_x": sensor_var_x,
            "var_y": sensor_var_y,
            "cov_xy": sensor_cov_xy,
        },
        "number_of_mape_states": len(policy),
        "number_of_update_start_positions": len(per_start),
        "number_of_unique_sigma_states": len(gvars),
        "gvar_max": len(gvars) - 1,
        "gaussian_states": gvars,
        "gvar_sequences_per_update_position": start_sequences,
    }

    if save_trajectories:
        out["mape_trajectories"] = trajectories

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gaussian_states_{map_id}.json"
    with output_file.open("w") as f:
        json.dump(out, f, indent=2)

    return out


def write_summary(output_dir: Path, results: List[dict], skipped: List[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "gaussian_states_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "map",
                "unique_sigma_states",
                "gvar_max",
                "mape_states",
                "update_start_positions",
                "q_mode",
                "p",
                "max_steps",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["map"],
                    r["number_of_unique_sigma_states"],
                    r["gvar_max"],
                    r["number_of_mape_states"],
                    r["number_of_update_start_positions"],
                    r["q_mode"],
                    r["p"],
                    r["max_steps_without_update"],
                ]
            )

    counts = [r["number_of_unique_sigma_states"] for r in results]

    summary = {
        "analysed_maps": len(results),
        "skipped_missing_maps": skipped,
        "min_unique_sigma_states": min(counts) if counts else None,
        "max_unique_sigma_states": max(counts) if counts else None,
        "mean_unique_sigma_states": (
            sum(counts) / len(counts) if counts else None
        ),
        "maps": [
            {
                "map": r["map"],
                "unique_sigma_states": r["number_of_unique_sigma_states"],
                "gvar_max": r["gvar_max"],
            }
            for r in results
        ],
    }

    json_path = output_dir / "gaussian_states_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet map-spezifische diskretisierte Gaussian Sigma-Zustände "
            "aus dem Dijkstra/MAPE-Controller."
        )
    )
    parser.add_argument("--maps-dir", type=Path, default=Path("maps"))
    parser.add_argument("--output-dir", type=Path, default=Path("gaussian_states"))
    parser.add_argument("--start-map", type=int, default=10)
    parser.add_argument("--end-map", type=int, default=99)

    # Für die 10x10-Modelle aus model_10.prism.
    parser.add_argument("--target-x", type=int, default=9)
    parser.add_argument("--target-y", type=int, default=9)

    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--digits", type=int, default=10)

    parser.add_argument(
        "--q-mode",
        choices=("position", "action"),
        default="position",
        help=(
            "'position' berücksichtigt Grid-Ränder; 'action' verwendet nur "
            "aktionsabhängiges Q wie im Grid-Innenraum."
        ),
    )

    parser.add_argument("--sensor-var-x", type=float, default=0.0)
    parser.add_argument("--sensor-var-y", type=float, default=0.0)
    parser.add_argument("--sensor-cov-xy", type=float, default=0.0)

    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Speichert zusätzlich die vollständigen MAPE-Trajektorien im JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Bei fehlender Map abbrechen statt sie zu überspringen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_map > args.end_map:
        raise ValueError("--start-map darf nicht größer als --end-map sein.")
    if not (0.0 <= args.p <= 1.0 / 3.0):
        raise ValueError("p muss zwischen 0 und 1/3 liegen.")
    if args.max_steps < 0:
        raise ValueError("--max-steps muss >= 0 sein.")

    results: List[dict] = []
    skipped: List[int] = []

    for map_id in range(args.start_map, args.end_map + 1):
        map_path = args.maps_dir / f"map_{map_id}.csv"

        if not map_path.exists():
            if args.strict:
                raise FileNotFoundError(map_path)
            print(f"[skip] map {map_id}: {map_path} fehlt")
            skipped.append(map_id)
            continue

        result = analyse_map(
            map_id=map_id,
            map_path=map_path,
            output_dir=args.output_dir,
            target_x=args.target_x,
            target_y=args.target_y,
            p=args.p,
            max_steps=args.max_steps,
            digits=args.digits,
            q_mode=args.q_mode,
            sensor_var_x=args.sensor_var_x,
            sensor_var_y=args.sensor_var_y,
            sensor_cov_xy=args.sensor_cov_xy,
            save_trajectories=args.save_trajectories,
        )
        results.append(result)

        print(
            f"[map {map_id}] "
            f"MAPE states={result['number_of_mape_states']}, "
            f"unique Sigma states={result['number_of_unique_sigma_states']}, "
            f"gvar=0..{result['gvar_max']}"
        )

    write_summary(args.output_dir, results, skipped)

    if results:
        counts = [r["number_of_unique_sigma_states"] for r in results]
        print()
        print(f"Analysierte Maps: {len(results)}")
        print(f"Sigma-Zustände min: {min(counts)}")
        print(f"Sigma-Zustände max: {max(counts)}")
        print(f"Sigma-Zustände mean: {sum(counts)/len(counts):.2f}")
        print(f"Ergebnisse: {args.output_dir}")
    else:
        print("Keine Maps analysiert.")


if __name__ == "__main__":
    main()
