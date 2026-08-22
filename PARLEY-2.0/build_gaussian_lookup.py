#!/usr/bin/env python3
"""
build_gaussian_lookup.py

Erzeugt für Maps 10..99 eine Lookup-Tabelle für das spätere
GaussianKnowledge-PRISM-Modul mit fester Grid-Quantisierung h=0.05.

Lookup-Semantik
---------------
Für jeden erreichbaren quantisierten Gaussian-State gvar und jede Position
(xhat, yhat), an der der bestehende Dijkstra/MAPE-Controller eine Aktion
vorgibt, wird berechnet:

    (xhat, yhat, gvar, action)
        -> (xhat_next, yhat_next, gvar_next)

mit

    Sigma_raw_next = Sigma_raw + Q(xhat, yhat, action)
    Sigma_next_q   = Q_h(Sigma_raw_next),   h = 0.05

Sigma_raw wird intern unquantisiert fortgeführt; gvar ist nur die
diskrete Repräsentation.

Die Bewegungs-Kovarianz Q wird positionsabhängig aus exakt denselben vier
Robot-Outcomes wie im bestehenden PRISM-Robot-Modul bestimmt. Dadurch werden
min/max-Clipping-Effekte an den Grid-Grenzen berücksichtigt.

Eingaben
--------
- maps/map_<id>.csv
- dijkstra.py

Das Skript benötigt NICHT zwingend die zuvor erzeugten
gaussian_states_<id>.json-Dateien. Es konstruiert die quantisierten Gaussian-
Zustände und Lookup-Transitions direkt aus Map + MAPE-Controller.

Ausgaben pro Map
----------------
gaussian_lookup/
    gaussian_lookup_<id>.json
    gaussian_lookup_<id>.csv

Zusätzlich:
    gaussian_lookup_summary.csv
    gaussian_lookup_summary.json

WICHTIG
-------
Es werden nur Zustände aufgenommen, die innerhalb von maximal 10 Schritten
seit einem Knowledge-Update erreichbar sind. Das passt zum URC-Intervall
c in [1..10].

Nach einem [update] ist der Gaussian-State:
    Sigma = 0
und damit immer gvar 0, sofern SENSOR_VAR_* = 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import dijkstra


H = 0.05
MAX_STEPS = 10
P = 0.01

DIRECTION_NAMES = ["west", "east", "south", "north"]

KNOWLEDGE_EFFECT = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}

SigmaKey = Tuple[float, float, float]
StateKey = Tuple[int, int, SigmaKey]


def round_to_grid(value: float, h: float = H) -> float:
    """Symmetrische Rundung auf das nächste Vielfache von h."""
    if value >= 0.0:
        k = math.floor(value / h + 0.5)
    else:
        k = math.ceil(value / h - 0.5)
    result = k * h
    return 0.0 if abs(result) < 1e-15 else result


def quantize_covariance(
    var_x: float,
    var_y: float,
    cov_xy: float,
    h: float = H,
) -> SigmaKey:
    """
    Uniforme Grid-Quantisierung von (var_x, var_y, cov_xy).

    Die resultierende Kovarianzmatrix bleibt positiv semidefinit:
        |cov_xy| <= sqrt(var_x * var_y)
    """
    qx = max(0.0, round_to_grid(var_x, h))
    qy = max(0.0, round_to_grid(var_y, h))
    qc = round_to_grid(cov_xy, h)

    if qx == 0.0 or qy == 0.0:
        qc = 0.0
    else:
        max_abs_cov = math.sqrt(qx * qy)
        max_grid_abs = math.floor(max_abs_cov / h + 1e-12) * h
        qc = min(max(qc, -max_grid_abs), max_grid_abs)

    return (
        round(qx, 12),
        round(qy, 12),
        round(qc, 12),
    )


def add_sigma(a: SigmaKey, b: SigmaKey) -> SigmaKey:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def load_map_for_generator(path: Path) -> List[List[str]]:
    """
    Gleiche Transformation wie prism_model_generator.build_map():
    CSV -> transponieren -> jede transponierte Zeile umdrehen.
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
    Rekonstruiert den vorhandenen Adaptation_MAPE_controller.
    """
    raw_directions = dijkstra.compute_directions(
        map_data, (target_x, target_y)
    )

    # Wie in prism_model_generator.generate_model():
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
    x: int,
    y: int,
    action: str,
    n: int,
) -> Tuple[int, int]:
    """Deterministisches xhat/yhat-Update aus dem Knowledge-Modul."""
    dx, dy = KNOWLEDGE_EFFECT[action]
    return (
        min(max(x + dx, 0), n),
        min(max(y + dy, 0), n),
    )


def robot_outcomes(action: str, p: float) -> List[Tuple[float, int, int]]:
    """Vier Bewegungs-Outcomes des vorhandenen Robot-Moduls."""
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
) -> SigmaKey:
    """
    Positionsabhängiges Q(x,y,a), inklusive Grid-Clipping.
    """
    samples: List[Tuple[float, float, float]] = []

    for prob, dx, dy in robot_outcomes(action, p):
        nx = min(max(x + dx, 0), n)
        ny = min(max(y + dy, 0), n)

        actual_dx = float(nx - x)
        actual_dy = float(ny - y)

        samples.append((prob, actual_dx, actual_dy))

    mean_dx = sum(prob * dx for prob, dx, _ in samples)
    mean_dy = sum(prob * dy for prob, _, dy in samples)

    var_x = sum(
        prob * (dx - mean_dx) ** 2
        for prob, dx, _ in samples
    )
    var_y = sum(
        prob * (dy - mean_dy) ** 2
        for prob, _, dy in samples
    )
    cov_xy = sum(
        prob * (dx - mean_dx) * (dy - mean_dy)
        for prob, dx, dy in samples
    )

    return (var_x, var_y, cov_xy)


def enumerate_reachable_lookup(
    map_data: Sequence[Sequence[str]],
    policy: Dict[Tuple[int, int], str],
    p: float,
    h: float,
    max_steps: int,
) -> Tuple[Set[SigmaKey], List[dict]]:
    """
    Enumeriert alle innerhalb von max_steps seit einem Update erreichbaren
    Kombinationen aus Position und quantisierter Kovarianz.

    Ein Update kann an jeder nicht blockierten Position stattfinden.
    Von dort startet Sigma=0 und der MAPE-Controller wird maximal max_steps
    Schritte verfolgt.

    Mehrfach auftretende Lookup-Zeilen werden dedupliziert.
    """
    size = len(map_data)
    n = size - 1

    zero_sigma = quantize_covariance(0.0, 0.0, 0.0, h)

    sigma_states: Set[SigmaKey] = {zero_sigma}

    # Lookup zuerst mit Sigma-Werten, gvar-IDs werden danach vergeben.
    transitions_set = set()

    for start_x in range(size):
        for start_y in range(size):
            if is_obstacle(map_data, start_x, start_y):
                continue

            x, y = start_x, start_y

            # WICHTIG:
            # sigma_raw wird unquantisiert fortgeführt.
            # sigma_q ist nur die diskrete PRISM-Repräsentation.
            sigma_raw = (0.0, 0.0, 0.0)
            sigma_q = zero_sigma

            for step_since_update in range(1, max_steps + 1):
                action = policy.get((x, y))
                if action is None:
                    break

                q_motion = motion_covariance(
                    x=x,
                    y=y,
                    action=action,
                    n=n,
                    p=p,
                )

                # Exakte Gaussian-Fortschreibung im Rohraum.
                sigma_raw_next = add_sigma(sigma_raw, q_motion)

                # Erst danach auf das h-Raster projizieren.
                sigma_q_next = quantize_covariance(
                    sigma_raw_next[0],
                    sigma_raw_next[1],
                    sigma_raw_next[2],
                    h,
                )

                next_x, next_y = apply_knowledge_action(
                    x, y, action, n
                )

                sigma_states.add(sigma_q_next)

                transitions_set.add(
                    (
                        x,
                        y,
                        sigma_q,
                        action,
                        next_x,
                        next_y,
                        sigma_q_next,
                    )
                )

                x, y = next_x, next_y
                sigma_raw = sigma_raw_next
                sigma_q = sigma_q_next

    transitions = []
    for (
        x,
        y,
        sigma,
        action,
        next_x,
        next_y,
        next_sigma,
    ) in transitions_set:
        transitions.append(
            {
                "xhat": x,
                "yhat": y,
                "sigma": sigma,
                "action": action,
                "xhat_next": next_x,
                "yhat_next": next_y,
                "sigma_next": next_sigma,
            }
        )

    return sigma_states, transitions


def assign_gvars(
    sigma_states: Set[SigmaKey],
) -> Tuple[List[dict], Dict[SigmaKey, int]]:
    """
    Deterministische Nummerierung der quantisierten Gaussian States.

    Sigma=(0,0,0) wird explizit gvar 0.
    Danach Sortierung nach trace, var_x, var_y, cov_xy.
    """
    zero = (0.0, 0.0, 0.0)

    ordered = sorted(
        sigma_states,
        key=lambda s: (
            0 if s == zero else 1,
            s[0] + s[1],
            s[0],
            s[1],
            s[2],
        ),
    )

    mapping = {sigma: i for i, sigma in enumerate(ordered)}

    table = [
        {
            "gvar": mapping[sigma],
            "var_x": sigma[0],
            "var_y": sigma[1],
            "cov_xy": sigma[2],
            "trace": sigma[0] + sigma[1],
        }
        for sigma in ordered
    ]

    return table, mapping


def convert_transitions_to_gvars(
    transitions: List[dict],
    sigma_to_gvar: Dict[SigmaKey, int],
) -> List[dict]:
    rows = []

    for t in transitions:
        gvar = sigma_to_gvar[t["sigma"]]
        next_gvar = sigma_to_gvar[t["sigma_next"]]

        rows.append(
            {
                "xhat": t["xhat"],
                "yhat": t["yhat"],
                "gvar": gvar,
                "action": t["action"],
                "xhat_next": t["xhat_next"],
                "yhat_next": t["yhat_next"],
                "gvar_next": next_gvar,
                "var_x": t["sigma"][0],
                "var_y": t["sigma"][1],
                "cov_xy": t["sigma"][2],
                "var_x_next": t["sigma_next"][0],
                "var_y_next": t["sigma_next"][1],
                "cov_xy_next": t["sigma_next"][2],
            }
        )

    rows.sort(
        key=lambda r: (
            r["xhat"],
            r["yhat"],
            r["gvar"],
            r["action"],
        )
    )
    return rows


def write_map_outputs(
    output_dir: Path,
    map_id: int,
    h: float,
    p: float,
    max_steps: int,
    policy: Dict[Tuple[int, int], str],
    gaussian_states: List[dict],
    lookup_rows: List[dict],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"gaussian_lookup_{map_id}.json"
    csv_path = output_dir / f"gaussian_lookup_{map_id}.csv"

    json_data = {
        "map": map_id,
        "h": h,
        "p": p,
        "max_steps_since_update": max_steps,
        "number_of_gaussian_states": len(gaussian_states),
        "gvar_max": len(gaussian_states) - 1,
        "number_of_lookup_transitions": len(lookup_rows),
        "gaussian_states": gaussian_states,
        "lookup": lookup_rows,
    }

    with json_path.open("w") as f:
        json.dump(json_data, f, indent=2)

    fieldnames = [
        "xhat",
        "yhat",
        "gvar",
        "action",
        "xhat_next",
        "yhat_next",
        "gvar_next",
        "var_x",
        "var_y",
        "cov_xy",
        "var_x_next",
        "var_y_next",
        "cov_xy_next",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lookup_rows)

    return {
        "map": map_id,
        "gaussian_states": len(gaussian_states),
        "gvar_max": len(gaussian_states) - 1,
        "lookup_transitions": len(lookup_rows),
        "mape_positions": len(policy),
    }


def write_summary(output_dir: Path, rows: List[dict], h: float) -> None:
    csv_path = output_dir / "gaussian_lookup_summary.csv"
    json_path = output_dir / "gaussian_lookup_summary.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "map",
                "gaussian_states",
                "gvar_max",
                "lookup_transitions",
                "mape_positions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    state_counts = [r["gaussian_states"] for r in rows]
    transition_counts = [r["lookup_transitions"] for r in rows]

    summary = {
        "h": h,
        "analysed_maps": len(rows),
        "gaussian_states": {
            "min": min(state_counts) if state_counts else None,
            "max": max(state_counts) if state_counts else None,
            "mean": (
                sum(state_counts) / len(state_counts)
                if state_counts else None
            ),
        },
        "lookup_transitions": {
            "min": min(transition_counts) if transition_counts else None,
            "max": max(transition_counts) if transition_counts else None,
            "mean": (
                sum(transition_counts) / len(transition_counts)
                if transition_counts else None
            ),
        },
        "maps": rows,
    }

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt map-spezifische GaussianKnowledge-Lookup-Tabellen "
            "mit h=0.05."
        )
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("maps"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gaussian_lookup"),
    )
    parser.add_argument(
        "--start-map",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--end-map",
        type=int,
        default=99,
    )
    parser.add_argument(
        "--target-x",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--target-y",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--p",
        type=float,
        default=P,
    )
    parser.add_argument(
        "--h",
        type=float,
        default=H,
        help="Grid-Breite. Standard für die Gaussian-Implementierung: 0.05.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Maximale Schritte seit Update; passend zu URC c in [1..10].",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Bei fehlender Map abbrechen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.h <= 0:
        raise ValueError("h muss > 0 sein.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps muss > 0 sein.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []

    for map_id in range(args.start_map, args.end_map + 1):
        map_path = args.maps_dir / f"map_{map_id}.csv"

        if not map_path.exists():
            if args.strict:
                raise FileNotFoundError(map_path)
            print(f"[skip] map {map_id}: {map_path} fehlt")
            continue

        map_data = load_map_for_generator(map_path)

        policy = build_mape_policy(
            map_data=map_data,
            target_x=args.target_x,
            target_y=args.target_y,
        )

        sigma_states, transitions = enumerate_reachable_lookup(
            map_data=map_data,
            policy=policy,
            p=args.p,
            h=args.h,
            max_steps=args.max_steps,
        )

        gaussian_states, sigma_to_gvar = assign_gvars(sigma_states)

        lookup_rows = convert_transitions_to_gvars(
            transitions=transitions,
            sigma_to_gvar=sigma_to_gvar,
        )

        result = write_map_outputs(
            output_dir=args.output_dir,
            map_id=map_id,
            h=args.h,
            p=args.p,
            max_steps=args.max_steps,
            policy=policy,
            gaussian_states=gaussian_states,
            lookup_rows=lookup_rows,
        )

        summary_rows.append(result)

        print(
            f"[map {map_id}] "
            f"gvars={result['gaussian_states']}, "
            f"lookup transitions={result['lookup_transitions']}"
        )

    write_summary(args.output_dir, summary_rows, args.h)

    if summary_rows:
        counts = [r["gaussian_states"] for r in summary_rows]
        transitions = [r["lookup_transitions"] for r in summary_rows]

        print()
        print(f"Analysierte Maps: {len(summary_rows)}")
        print(
            f"gvars min/mean/max: "
            f"{min(counts)}/"
            f"{sum(counts)/len(counts):.2f}/"
            f"{max(counts)}"
        )
        print(
            f"lookup transitions min/mean/max: "
            f"{min(transitions)}/"
            f"{sum(transitions)/len(transitions):.2f}/"
            f"{max(transitions)}"
        )
        print(f"Ausgabe: {args.output_dir}")


if __name__ == "__main__":
    main()
