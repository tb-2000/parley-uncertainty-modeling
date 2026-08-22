#!/usr/bin/env python3
"""
compare_gaussian_quantization_refinement.py

Vergleicht mehrere Quantisierungsbreiten h gemeinsam anhand von:

1. Quantisierungsfehler der Kovarianzmatrizen
2. Anzahl quantisierter Gaussian-Klassen gvar
3. Anzahl finaler Markov-kompatibler Gaussian-Zustände gstate

Das Skript rekonstruiert pro Map direkt aus Map + Dijkstra/MAPE-Controller
die Roh-Kovarianztrajektorien bis maximal 10 Schritte seit einem Update.

Für jedes h:
    Sigma_raw
        -> Q_h(Sigma_raw) = gvar
        -> konfliktgetriebenes Partition Refinement
        -> gstate

Fehlermetrik
------------
Für jede Roh-Kovarianz Sigma_i:

    ||Sigma_i - Q_h(Sigma_i)||_F

Für Sigma=[[a,c],[c,b]] gilt:

    ||Sigma-Q||_F
      = sqrt((a-aq)^2 + (b-bq)^2 + 2*(c-cq)^2)

Berichtet werden:
- mean_frobenius_error
- rmse_frobenius_error
- max_frobenius_error

Zusätzlich:
- gvars pro Map
- gstates pro Map
- extra_gstates = gstates - gvars
- split_gvars
- max_gstates_per_gvar
- lookup_transitions

Ausgaben
--------
gaussian_h_comparison/
    gaussian_h_comparison_global.csv
    gaussian_h_comparison_per_map.csv
    gaussian_h_comparison.json

Standard-h-Werte:
    0.05 0.075 0.10 0.15 0.20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import dijkstra


DIRECTION_NAMES = ["west", "east", "south", "north"]

KNOWLEDGE_EFFECT = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}

Sigma = Tuple[float, float, float]


@dataclass(frozen=True)
class AtomId:
    start_x: int
    start_y: int
    step: int


@dataclass
class Atom:
    atom_id: AtomId
    xhat: int
    yhat: int
    raw_sigma: Sigma
    gvar: int = -1
    action: Optional[str] = None
    next_atom: Optional[AtomId] = None


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


def frobenius_error(raw: Sigma, quantized: Sigma) -> float:
    dx = raw[0] - quantized[0]
    dy = raw[1] - quantized[1]
    dc = raw[2] - quantized[2]

    return math.sqrt(
        dx * dx
        + dy * dy
        + 2.0 * dc * dc
    )


def add_sigma(a: Sigma, b: Sigma) -> Sigma:
    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    )


def load_map_for_generator(path: Path) -> List[List[str]]:
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


def enumerate_raw_atoms(
    map_data: Sequence[Sequence[str]],
    policy: Dict[Tuple[int, int], str],
    p: float,
    max_steps: int,
) -> Dict[AtomId, Atom]:
    """
    Erzeugt die Roh-Gaussian-Trajektorien einmalig pro Map.

    Quantisierung erfolgt erst später separat für jedes h.
    """
    size = len(map_data)
    n = size - 1

    atoms: Dict[AtomId, Atom] = {}

    for start_x in range(size):
        for start_y in range(size):
            if is_obstacle(map_data, start_x, start_y):
                continue

            x, y = start_x, start_y
            raw_sigma: Sigma = (0.0, 0.0, 0.0)

            for step in range(max_steps + 1):
                atom_id = AtomId(
                    start_x,
                    start_y,
                    step,
                )

                atom = Atom(
                    atom_id=atom_id,
                    xhat=x,
                    yhat=y,
                    raw_sigma=raw_sigma,
                )

                action = policy.get((x, y))

                if (
                    step < max_steps
                    and action is not None
                ):
                    atom.action = action

                    q_motion = motion_covariance(
                        x,
                        y,
                        action,
                        n,
                        p,
                    )

                    nx, ny = apply_knowledge_action(
                        x,
                        y,
                        action,
                        n,
                    )

                    atom.next_atom = AtomId(
                        start_x,
                        start_y,
                        step + 1,
                    )

                    raw_sigma = add_sigma(
                        raw_sigma,
                        q_motion,
                    )

                    x, y = nx, ny

                atoms[atom_id] = atom

                if action is None:
                    break

    return atoms


def assign_gvars(
    atoms: Dict[AtomId, Atom],
    h: float,
) -> Tuple[
    Dict[int, Sigma],
    Dict[AtomId, int],
    List[float],
]:
    """
    Quantisiert alle Roh-Sigma-Zustände für ein h.
    """
    quantized_by_atom = {}
    unique_sigmas: Set[Sigma] = set()
    errors = []

    for atom_id, atom in atoms.items():
        q_sigma = quantize_covariance(
            atom.raw_sigma[0],
            atom.raw_sigma[1],
            atom.raw_sigma[2],
            h,
        )

        quantized_by_atom[atom_id] = q_sigma
        unique_sigmas.add(q_sigma)

        errors.append(
            frobenius_error(
                atom.raw_sigma,
                q_sigma,
            )
        )

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
        sigma: idx
        for idx, sigma in enumerate(ordered_sigmas)
    }

    gvar_to_sigma = {
        idx: sigma
        for sigma, idx in sigma_to_gvar.items()
    }

    atom_to_gvar = {
        atom_id: sigma_to_gvar[q_sigma]
        for atom_id, q_sigma in quantized_by_atom.items()
    }

    return (
        gvar_to_sigma,
        atom_to_gvar,
        errors,
    )


def initial_blocks(
    atom_to_gvar: Dict[AtomId, int],
) -> Dict[AtomId, int]:
    gvars = sorted(set(atom_to_gvar.values()))

    gvar_to_block = {
        gvar: idx
        for idx, gvar in enumerate(gvars)
    }

    return {
        atom_id: gvar_to_block[gvar]
        for atom_id, gvar in atom_to_gvar.items()
    }


def build_conflict_graph_for_block(
    members: List[AtomId],
    atoms: Dict[AtomId, Atom],
    block_of: Dict[AtomId, int],
) -> Dict[AtomId, Set[AtomId]]:
    graph = {
        atom_id: set()
        for atom_id in members
    }

    by_context = defaultdict(list)

    for atom_id in members:
        atom = atoms[atom_id]

        context = (
            atom.xhat,
            atom.yhat,
            atom.action,
        )
        by_context[context].append(atom_id)

    for context_members in by_context.values():
        for i, a_id in enumerate(context_members):
            a = atoms[a_id]

            a_next_block = (
                None
                if a.next_atom is None
                else block_of[a.next_atom]
            )

            for b_id in context_members[i + 1:]:
                b = atoms[b_id]

                b_next_block = (
                    None
                    if b.next_atom is None
                    else block_of[b.next_atom]
                )

                if a_next_block != b_next_block:
                    graph[a_id].add(b_id)
                    graph[b_id].add(a_id)

    return graph


def greedy_color(
    graph: Dict[AtomId, Set[AtomId]],
) -> Dict[AtomId, int]:
    order = sorted(
        graph.keys(),
        key=lambda node: (
            -len(graph[node]),
            node.start_x,
            node.start_y,
            node.step,
        ),
    )

    color_of = {}

    for node in order:
        used = {
            color_of[neighbor]
            for neighbor in graph[node]
            if neighbor in color_of
        }

        color = 0
        while color in used:
            color += 1

        color_of[node] = color

    return color_of


def refine_blocks(
    atoms: Dict[AtomId, Atom],
    atom_to_gvar: Dict[AtomId, int],
) -> Dict[AtomId, int]:
    block_of = initial_blocks(atom_to_gvar)

    while True:
        members_by_block = defaultdict(list)

        for atom_id, block in block_of.items():
            members_by_block[block].append(atom_id)

        changed = False
        new_block_of = {}
        next_block_id = 0

        for old_block in sorted(members_by_block):
            members = members_by_block[old_block]

            graph = build_conflict_graph_for_block(
                members,
                atoms,
                block_of,
            )

            colors = greedy_color(graph)
            used_colors = sorted(set(colors.values()))

            if len(used_colors) > 1:
                changed = True

            color_to_new_block = {
                color: next_block_id + idx
                for idx, color in enumerate(used_colors)
            }

            for atom_id in members:
                new_block_of[atom_id] = (
                    color_to_new_block[colors[atom_id]]
                )

            next_block_id += len(used_colors)

        block_of = new_block_of

        if not changed:
            return block_of


def canonicalize_gstates(
    atom_to_gvar: Dict[AtomId, int],
    block_of: Dict[AtomId, int],
) -> Tuple[
    Dict[AtomId, int],
    Dict[int, int],
]:
    block_members = defaultdict(list)

    for atom_id, block in block_of.items():
        block_members[block].append(atom_id)

    sortable = []

    for old_block, members in block_members.items():
        gvars = {
            atom_to_gvar[atom_id]
            for atom_id in members
        }

        if len(gvars) != 1:
            raise ValueError(
                f"Block {old_block} contains multiple gvars: {gvars}"
            )

        gvar = next(iter(gvars))

        representative = min(
            members,
            key=lambda a: (
                a.start_x,
                a.start_y,
                a.step,
            ),
        )

        sortable.append(
            (
                gvar,
                representative.start_x,
                representative.start_y,
                representative.step,
                old_block,
            )
        )

    sortable.sort()

    old_to_gstate = {
        old_block: gstate
        for gstate, (*_, old_block) in enumerate(sortable)
    }

    atom_to_gstate = {
        atom_id: old_to_gstate[block_of[atom_id]]
        for atom_id in atom_to_gvar
    }

    gstate_to_gvar = {}

    for atom_id, gstate in atom_to_gstate.items():
        gvar = atom_to_gvar[atom_id]

        if (
            gstate in gstate_to_gvar
            and gstate_to_gvar[gstate] != gvar
        ):
            raise ValueError(
                "Internal gstate/gvar inconsistency."
            )

        gstate_to_gvar[gstate] = gvar

    return (
        atom_to_gstate,
        gstate_to_gvar,
    )


def validate_refined_markov_property(
    atoms: Dict[AtomId, Atom],
    atom_to_gstate: Dict[AtomId, int],
) -> int:
    """
    Validiert eindeutige Transitionen und liefert deren Anzahl.
    """
    transitions = {}

    for atom_id, atom in atoms.items():
        if (
            atom.action is None
            or atom.next_atom is None
            or atom.next_atom not in atoms
        ):
            continue

        next_atom = atoms[atom.next_atom]

        source = (
            atom.xhat,
            atom.yhat,
            atom_to_gstate[atom_id],
            atom.action,
        )

        successor = (
            next_atom.xhat,
            next_atom.yhat,
            atom_to_gstate[atom.next_atom],
        )

        if (
            source in transitions
            and transitions[source] != successor
        ):
            raise ValueError(
                "Refinement failed: "
                f"{source} -> "
                f"{transitions[source]} / {successor}"
            )

        transitions[source] = successor

    return len(transitions)


def split_statistics(
    gstate_to_gvar: Dict[int, int],
) -> Tuple[int, int]:
    by_gvar = defaultdict(list)

    for gstate, gvar in gstate_to_gvar.items():
        by_gvar[gvar].append(gstate)

    split_gvars = sum(
        1
        for states in by_gvar.values()
        if len(states) > 1
    )

    max_gstates_per_gvar = max(
        (
            len(states)
            for states in by_gvar.values()
        ),
        default=0,
    )

    return (
        split_gvars,
        max_gstates_per_gvar,
    )


def analyse_map_for_h(
    map_id: int,
    atoms: Dict[AtomId, Atom],
    h: float,
) -> dict:
    (
        gvar_to_sigma,
        atom_to_gvar,
        errors,
    ) = assign_gvars(
        atoms,
        h,
    )

    block_of = refine_blocks(
        atoms,
        atom_to_gvar,
    )

    (
        atom_to_gstate,
        gstate_to_gvar,
    ) = canonicalize_gstates(
        atom_to_gvar,
        block_of,
    )

    transition_count = validate_refined_markov_property(
        atoms,
        atom_to_gstate,
    )

    split_gvars, max_gstates_per_gvar = split_statistics(
        gstate_to_gvar
    )

    n = len(errors)

    mean_error = (
        sum(errors) / n
        if n
        else 0.0
    )
    rmse_error = (
        math.sqrt(
            sum(e * e for e in errors) / n
        )
        if n
        else 0.0
    )
    max_error = (
        max(errors)
        if errors
        else 0.0
    )

    return {
        "h": h,
        "map": map_id,
        "raw_atom_count": len(atoms),
        "mean_frobenius_error": mean_error,
        "rmse_frobenius_error": rmse_error,
        "max_frobenius_error": max_error,
        "gvars": len(gvar_to_sigma),
        "gstates": len(gstate_to_gvar),
        "extra_gstates": (
            len(gstate_to_gvar)
            - len(gvar_to_sigma)
        ),
        "split_gvars": split_gvars,
        "max_gstates_per_gvar": max_gstates_per_gvar,
        "lookup_transitions": transition_count,
    }


def aggregate_h(
    h: float,
    rows: List[dict],
) -> dict:
    def mean(key):
        return sum(r[key] for r in rows) / len(rows)

    return {
        "h": h,
        "maps": len(rows),

        "mean_frobenius_error": mean(
            "mean_frobenius_error"
        ),
        "mean_rmse_frobenius_error": mean(
            "rmse_frobenius_error"
        ),
        "max_frobenius_error_overall": max(
            r["max_frobenius_error"]
            for r in rows
        ),

        "mean_gvars": mean("gvars"),
        "min_gvars": min(r["gvars"] for r in rows),
        "max_gvars": max(r["gvars"] for r in rows),

        "mean_gstates": mean("gstates"),
        "min_gstates": min(r["gstates"] for r in rows),
        "max_gstates": max(r["gstates"] for r in rows),

        "mean_extra_gstates": mean("extra_gstates"),
        "min_extra_gstates": min(
            r["extra_gstates"] for r in rows
        ),
        "max_extra_gstates": max(
            r["extra_gstates"] for r in rows
        ),

        "mean_split_gvars": mean("split_gvars"),
        "mean_max_gstates_per_gvar": mean(
            "max_gstates_per_gvar"
        ),
        "max_gstates_per_gvar_overall": max(
            r["max_gstates_per_gvar"]
            for r in rows
        ),

        "mean_lookup_transitions": mean(
            "lookup_transitions"
        ),
        "min_lookup_transitions": min(
            r["lookup_transitions"]
            for r in rows
        ),
        "max_lookup_transitions": max(
            r["lookup_transitions"]
            for r in rows
        ),
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
            "Compares quantization error, gvar count and refined "
            "gstate count for multiple h values."
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
        default=Path("gaussian_h_comparison"),
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
        default=[
            0.01,
            0.025,
            0.05,
            0.075,
            0.10,
        ],
    )
    parser.add_argument(
        "--strict",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if any(h <= 0.0 for h in args.h):
        raise ValueError(
            "All h values must be > 0."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reconstruct raw trajectories once per map.
    atoms_by_map = {}

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

        map_data = load_map_for_generator(
            map_path
        )

        policy = build_mape_policy(
            map_data,
            args.target_x,
            args.target_y,
        )

        atoms_by_map[map_id] = enumerate_raw_atoms(
            map_data,
            policy,
            args.p,
            args.max_steps,
        )

    per_map_rows = []
    global_rows = []

    comparison_json = {
        "settings": {
            "h_values": args.h,
            "p": args.p,
            "max_steps_since_update": args.max_steps,
            "target": [
                args.target_x,
                args.target_y,
            ],
            "error_metric": (
                "mean_i ||Sigma_i - Q_h(Sigma_i)||_F"
            ),
        },
        "results": {},
    }

    for h in args.h:
        rows_for_h = []

        print()
        print(f"=== h={h:g} ===")

        for map_id in sorted(atoms_by_map):
            row = analyse_map_for_h(
                map_id,
                atoms_by_map[map_id],
                h,
            )

            rows_for_h.append(row)
            per_map_rows.append(row)

            print(
                f"[map {map_id}] "
                f"error={row['mean_frobenius_error']:.6f}, "
                f"gvars={row['gvars']}, "
                f"gstates={row['gstates']}, "
                f"extra={row['extra_gstates']}"
            )

        global_row = aggregate_h(
            h,
            rows_for_h,
        )

        global_rows.append(global_row)

        comparison_json["results"][str(h)] = {
            "global": global_row,
            "per_map": rows_for_h,
        }

        print(
            f"h={h:g}: "
            f"mean error="
            f"{global_row['mean_frobenius_error']:.6f}, "
            f"mean gvars="
            f"{global_row['mean_gvars']:.2f}, "
            f"mean gstates="
            f"{global_row['mean_gstates']:.2f}"
        )

    write_csv(
        args.output_dir
        / "gaussian_h_comparison_global.csv",
        global_rows,
    )

    write_csv(
        args.output_dir
        / "gaussian_h_comparison_per_map.csv",
        per_map_rows,
    )

    with (
        args.output_dir
        / "gaussian_h_comparison.json"
    ).open("w") as f:
        json.dump(
            comparison_json,
            f,
            indent=2,
        )

    print()
    print(
        "Global comparison:"
    )

    for row in global_rows:
        print(
            f"h={row['h']:g} | "
            f"error={row['mean_frobenius_error']:.6f} | "
            f"gvars={row['mean_gvars']:.2f} | "
            f"gstates={row['mean_gstates']:.2f} | "
            f"extra={row['mean_extra_gstates']:.2f}"
        )

    print()
    print(
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
