#!/usr/bin/env python3
"""
count_reachable_gaussian_urc_triples.py

Zaehlt fuer Maps 10..99 die tatsaechlich erreichbaren URC-Tripel

    (xhat, yhat, gvar)

fuer zwei Gaussian-Quantisierungen:

    h = 0.05
    h = 0.10

Die Roh-Kovarianz wird wie in den bisherigen Gaussian-Skripten entlang des
Dijkstra/MAPE-Controllers bis maximal 10 Schritte seit einem Knowledge-Update
fortgeschrieben:

    Sigma_raw(t+1) = Sigma_raw(t) + Q(xhat_t, yhat_t, action_t)

Erst danach wird fuer das jeweilige h quantisiert:

    gvar = Q_h(Sigma_raw)

Fuer den spaeteren positionssensitiven URC

    pi(xhat, yhat, gvar) -> c

entspricht die Anzahl eindeutiger erreichbarer Tripel direkt der Anzahl
benoetigter evolve decision_* Parameter.

Ausgaben
--------
gaussian_urc_triples/
    gaussian_urc_triples_summary.csv
    gaussian_urc_triples_per_map.csv
    gaussian_urc_triples.json

Optional mit --save-triples:
    gaussian_urc_triples_map_<id>.json

Standard:
    maps 10..99
    target (9,9)
    p=0.01
    max_steps=10
    h={0.05, 0.10}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import dijkstra


DIRECTION_NAMES = ["west", "east", "south", "north"]

KNOWLEDGE_EFFECT = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}

Sigma = Tuple[float, float, float]


def round_to_grid(value: float, h: float) -> float:
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
    h: float,
) -> Sigma:
    """
    Uniforme Grid-Quantisierung mit PSD-Erhaltung.
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


def add_sigma(a: Sigma, b: Sigma) -> Sigma:
    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    )


def load_map_for_generator(path: Path) -> List[List[str]]:
    """
    Gleiche Transformation wie prism_model_generator.build_map().
    """
    with path.open("r", newline="") as f:
        raw = list(csv.reader(f))

    transposed = list(zip(*raw))
    return [list(row[::-1]) for row in transposed]


def is_obstacle(
    map_data: Sequence[Sequence[str]],
    x: int,
    y: int,
) -> bool:
    return int(map_data[x][y]) > 9


def build_mape_policy(
    map_data: Sequence[Sequence[str]],
    target_x: int,
    target_y: int,
) -> Dict[Tuple[int, int], str]:
    raw_directions = dijkstra.compute_directions(
        map_data,
        (target_x, target_y),
    )

    controller_directions = list(zip(*raw_directions))

    size = len(map_data)
    policy = {}

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
    dx, dy = KNOWLEDGE_EFFECT[action]

    return (
        min(max(x + dx, 0), n),
        min(max(y + dy, 0), n),
    )


def robot_outcomes(
    action: str,
    p: float,
) -> List[Tuple[float, int, int]]:
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
) -> Sigma:
    """
    Positionsabhaengiges Q(x,y,a), inklusive Grid-Clipping.
    """
    samples = []

    for probability, dx, dy in robot_outcomes(action, p):
        nx = min(max(x + dx, 0), n)
        ny = min(max(y + dy, 0), n)

        actual_dx = float(nx - x)
        actual_dy = float(ny - y)

        samples.append(
            (probability, actual_dx, actual_dy)
        )

    mean_dx = sum(
        probability * dx
        for probability, dx, _ in samples
    )
    mean_dy = sum(
        probability * dy
        for probability, _, dy in samples
    )

    var_x = sum(
        probability * (dx - mean_dx) ** 2
        for probability, dx, _ in samples
    )
    var_y = sum(
        probability * (dy - mean_dy) ** 2
        for probability, _, dy in samples
    )
    cov_xy = sum(
        probability
        * (dx - mean_dx)
        * (dy - mean_dy)
        for probability, dx, dy in samples
    )

    return (var_x, var_y, cov_xy)


def enumerate_raw_states(
    map_data: Sequence[Sequence[str]],
    policy: Dict[Tuple[int, int], str],
    p: float,
    max_steps: int,
) -> List[Tuple[int, int, Sigma]]:
    """
    Liefert alle beobachteten Roh-Zustaende:
        (xhat, yhat, Sigma_raw)

    Ein Update darf an jeder nicht blockierten Position stattfinden.
    Von dort werden maximal max_steps MAPE-Schritte verfolgt.
    """
    size = len(map_data)
    n = size - 1

    raw_states = []

    for start_x in range(size):
        for start_y in range(size):
            if is_obstacle(map_data, start_x, start_y):
                continue

            x, y = start_x, start_y
            sigma_raw: Sigma = (0.0, 0.0, 0.0)

            # Zustand direkt nach Update.
            raw_states.append((x, y, sigma_raw))

            for _step in range(1, max_steps + 1):
                action = policy.get((x, y))

                if action is None:
                    break

                q_motion = motion_covariance(
                    x,
                    y,
                    action,
                    n,
                    p,
                )

                sigma_raw = add_sigma(
                    sigma_raw,
                    q_motion,
                )

                x, y = apply_knowledge_action(
                    x,
                    y,
                    action,
                    n,
                )

                raw_states.append(
                    (x, y, sigma_raw)
                )

    return raw_states


def assign_gvars_and_count_triples(
    raw_states: List[Tuple[int, int, Sigma]],
    h: float,
) -> Tuple[
    int,
    int,
    List[Tuple[int, int, int]],
]:
    """
    Bestimmt map-lokale gvar-IDs und zaehlt eindeutige
    (xhat,yhat,gvar)-Tripel.
    """
    quantized_states = []

    for xhat, yhat, sigma_raw in raw_states:
        sigma_q = quantize_covariance(
            sigma_raw[0],
            sigma_raw[1],
            sigma_raw[2],
            h,
        )

        quantized_states.append(
            (xhat, yhat, sigma_q)
        )

    unique_sigmas = {
        sigma_q
        for _, _, sigma_q in quantized_states
    }

    zero = (0.0, 0.0, 0.0)

    ordered_sigmas = sorted(
        unique_sigmas,
        key=lambda s: (
            0 if s == zero else 1,
            s[0] + s[1],
            s[0],
            s[1],
            s[2],
        ),
    )

    sigma_to_gvar = {
        sigma: gvar
        for gvar, sigma in enumerate(ordered_sigmas)
    }

    triples = {
        (
            xhat,
            yhat,
            sigma_to_gvar[sigma_q],
        )
        for xhat, yhat, sigma_q in quantized_states
    }

    ordered_triples = sorted(triples)

    return (
        len(ordered_sigmas),
        len(ordered_triples),
        ordered_triples,
    )


def analyse_map(
    map_id: int,
    map_path: Path,
    target_x: int,
    target_y: int,
    p: float,
    max_steps: int,
    h_values: List[float],
) -> Tuple[List[dict], Dict[str, dict]]:
    map_data = load_map_for_generator(map_path)

    policy = build_mape_policy(
        map_data,
        target_x,
        target_y,
    )

    raw_states = enumerate_raw_states(
        map_data,
        policy,
        p,
        max_steps,
    )

    rows = []
    details = {}

    for h in h_values:
        gvar_count, triple_count, triples = (
            assign_gvars_and_count_triples(
                raw_states,
                h,
            )
        )

        row = {
            "map": map_id,
            "h": h,
            "reachable_gvars": gvar_count,
            "reachable_urc_triples": triple_count,
            "raw_state_records": len(raw_states),
            "mape_positions": len(policy),
        }

        rows.append(row)

        details[str(h)] = {
            "reachable_gvars": gvar_count,
            "reachable_urc_triples": triple_count,
            "triples": [
                {
                    "xhat": xhat,
                    "yhat": yhat,
                    "gvar": gvar,
                }
                for xhat, yhat, gvar in triples
            ],
        }

    return rows, details


def aggregate_h(
    h: float,
    rows: List[dict],
) -> dict:
    matching = [
        row
        for row in rows
        if abs(row["h"] - h) < 1e-12
    ]

    counts = [
        row["reachable_urc_triples"]
        for row in matching
    ]

    gvars = [
        row["reachable_gvars"]
        for row in matching
    ]

    return {
        "h": h,
        "maps": len(matching),
        "min_reachable_urc_triples": min(counts),
        "mean_reachable_urc_triples": (
            sum(counts) / len(counts)
        ),
        "max_reachable_urc_triples": max(counts),
        "min_reachable_gvars": min(gvars),
        "mean_reachable_gvars": (
            sum(gvars) / len(gvars)
        ),
        "max_reachable_gvars": max(gvars),
    }


def write_csv(
    path: Path,
    rows: List[dict],
) -> None:
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Counts reachable (xhat,yhat,gvar) URC triples "
            "for h=0.05 and h=0.10."
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
        default=Path("gaussian_urc_triples"),
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
        default=0.01,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--h",
        nargs="+",
        type=float,
        default=[0.05, 0.10],
    )
    parser.add_argument(
        "--save-triples",
        action="store_true",
        help=(
            "Speichert die konkreten Tripel pro Map zusaetzlich "
            "als JSON."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if any(h <= 0.0 for h in args.h):
        raise ValueError("All h values must be > 0.")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_map_rows = []
    details_by_map = {}

    for map_id in range(
        args.start_map,
        args.end_map + 1,
    ):
        map_path = (
            args.maps_dir
            / f"map_{map_id}.csv"
        )

        if not map_path.exists():
            if args.strict:
                raise FileNotFoundError(map_path)

            print(
                f"[skip] map {map_id}: "
                f"{map_path} fehlt"
            )
            continue

        rows, details = analyse_map(
            map_id=map_id,
            map_path=map_path,
            target_x=args.target_x,
            target_y=args.target_y,
            p=args.p,
            max_steps=args.max_steps,
            h_values=args.h,
        )

        per_map_rows.extend(rows)
        details_by_map[str(map_id)] = details

        summary_text = " | ".join(
            (
                f"h={row['h']:g}: "
                f"gvars={row['reachable_gvars']}, "
                f"triples={row['reachable_urc_triples']}"
            )
            for row in rows
        )

        print(
            f"[map {map_id}] {summary_text}"
        )

        if args.save_triples:
            path = (
                args.output_dir
                / f"gaussian_urc_triples_map_{map_id}.json"
            )

            with path.open("w") as f:
                json.dump(
                    {
                        "map": map_id,
                        "results": details,
                    },
                    f,
                    indent=2,
                )

    global_rows = [
        aggregate_h(h, per_map_rows)
        for h in args.h
    ]

    write_csv(
        args.output_dir
        / "gaussian_urc_triples_per_map.csv",
        per_map_rows,
    )

    write_csv(
        args.output_dir
        / "gaussian_urc_triples_summary.csv",
        global_rows,
    )

    with (
        args.output_dir
        / "gaussian_urc_triples.json"
    ).open("w") as f:
        json.dump(
            {
                "settings": {
                    "h_values": args.h,
                    "max_steps_since_update": args.max_steps,
                    "p": args.p,
                    "target": [
                        args.target_x,
                        args.target_y,
                    ],
                },
                "summary": global_rows,
                "per_map": per_map_rows,
                "details": (
                    details_by_map
                    if args.save_triples
                    else None
                ),
            },
            f,
            indent=2,
        )

    print()
    print("Global comparison:")

    for row in global_rows:
        print(
            f"h={row['h']:g} | "
            f"URC triples min/mean/max="
            f"{row['min_reachable_urc_triples']}/"
            f"{row['mean_reachable_urc_triples']:.2f}/"
            f"{row['max_reachable_urc_triples']} | "
            f"gvars mean="
            f"{row['mean_reachable_gvars']:.2f}"
        )

    print()
    print(
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
