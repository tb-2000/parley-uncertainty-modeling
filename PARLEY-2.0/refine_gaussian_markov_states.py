#!/usr/bin/env python3
"""
refine_gaussian_markov_states.py

Erzeugt für Maps 10..99 aus dem bestehenden Dijkstra/MAPE-Controller:

1. die h=0.05-quantisierten Gaussian-Klassen ``gvar`` und
2. eine möglichst kleine technische Verfeinerung ``gstate``,

so dass die Kombination

    (xhat, yhat, gstate)

eine eindeutige Markov-Transition besitzt.

Idee
----
Mehrere unterschiedliche Roh-Kovarianzen können auf denselben gvar fallen:

    Q_h(Sigma_A) = Q_h(Sigma_B) = gvar_k

aber bei gleicher Position und gleicher MAPE-Aktion unterschiedliche
quantisierte Nachfolger erzeugen. Dann ist gvar allein nicht Markov.

Das Skript splittet deshalb nur solche gvar-Klassen, bei denen tatsächlich
ein Konflikt auftritt. Unterschiedliche Positionen dürfen denselben gstate
weiterverwenden, weil xhat/yhat ohnehin separate PRISM-Zustandsvariablen
sind. Dadurch wird nicht unnötig nach Position aufgesplittet.

Die Verfeinerung erfolgt iterativ:
- Initial: ein Block pro gvar.
- Konflikt: zwei Atome im selben Block, an derselben Position und mit
  derselben Aktion, deren Nachfolger in verschiedenen aktuellen Blöcken
  liegen.
- Nur konfliktbehaftete Blöcke werden mittels Greedy-Graph-Coloring gesplittet.
- Wiederholen, bis jede
      (xhat, yhat, gstate, action)
  genau einen Nachfolger besitzt.

gvar bleibt die semantische Gaussian-Uncertainty-Klasse und kann später
vom URC als Decision Variable verwendet werden.

gstate dient nur dazu, die Markov-Eigenschaft der DTMC-Repräsentation
wiederherzustellen.

Eingaben
--------
maps/map_<id>.csv
dijkstra.py

Ausgaben
--------
gaussian_refined/
    gaussian_refined_<id>.json
    gaussian_refined_<id>.csv
    gaussian_refinement_summary.csv
    gaussian_refinement_summary.json

Standardparameter
-----------------
h = 0.05
p = 0.01
max_steps_since_update = 10
target = (9, 9)
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
    """
    Eindeutiger beobachteter Gaussian-Historienzustand.

    start_x/start_y:
        Position direkt nach dem letzten Knowledge-Update.

    step:
        Anzahl der Bewegungen seit diesem Update.
    """
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


def add_sigma(a: Sigma, b: Sigma) -> Sigma:
    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    )


def load_map_for_generator(path: Path) -> List[List[str]]:
    """
    Gleiche Koordinatentransformation wie prism_model_generator.build_map().
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

    # Wie prism_model_generator.generate_model()
    directions = list(zip(*raw_directions))

    size = len(map_data)
    policy = {}

    for x in range(size):
        for y in range(size):
            direction = int(directions[y][x])

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
    Positionsabhängiges Q(x,y,a), inklusive min/max-Clipping am Grid-Rand.
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


def enumerate_atoms(
    map_data: Sequence[Sequence[str]],
    policy: Dict[Tuple[int, int], str],
    p: float,
    h: float,
    max_steps: int,
) -> Tuple[
    Dict[AtomId, Atom],
    Dict[int, Sigma],
]:
    """
    Rekonstruiert alle Roh-Sigma-Trajektorien direkt aus Map + MAPE.

    Sigma_raw wird unquantisiert fortgeführt.
    gvar wird erst danach aus Q_h(Sigma_raw) bestimmt.
    """
    size = len(map_data)
    n = size - 1

    atoms: Dict[AtomId, Atom] = {}
    quantized_sigmas: Set[Sigma] = set()

    for start_x in range(size):
        for start_y in range(size):
            if is_obstacle(map_data, start_x, start_y):
                continue

            x, y = start_x, start_y
            raw_sigma: Sigma = (0.0, 0.0, 0.0)

            for step in range(max_steps + 1):
                atom_id = AtomId(start_x, start_y, step)

                atom = Atom(
                    atom_id=atom_id,
                    xhat=x,
                    yhat=y,
                    raw_sigma=raw_sigma,
                )

                q_sigma = quantize_covariance(
                    raw_sigma[0],
                    raw_sigma[1],
                    raw_sigma[2],
                    h,
                )
                quantized_sigmas.add(q_sigma)

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

                    next_raw = add_sigma(
                        raw_sigma,
                        q_motion,
                    )

                    nx, ny = apply_knowledge_action(
                        x,
                        y,
                        action,
                        n,
                    )

                    next_id = AtomId(
                        start_x,
                        start_y,
                        step + 1,
                    )
                    atom.next_atom = next_id

                    x, y = nx, ny
                    raw_sigma = next_raw

                atoms[atom_id] = atom

                if action is None:
                    break

    # Deterministische gvar-IDs aus quantisierten Sigma-Werten.
    zero = (0.0, 0.0, 0.0)

    ordered_sigmas = sorted(
        quantized_sigmas,
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

    gvar_to_sigma = {
        gvar: sigma
        for sigma, gvar in sigma_to_gvar.items()
    }

    for atom in atoms.values():
        q_sigma = quantize_covariance(
            atom.raw_sigma[0],
            atom.raw_sigma[1],
            atom.raw_sigma[2],
            h,
        )
        atom.gvar = sigma_to_gvar[q_sigma]

    return atoms, gvar_to_sigma


def initial_blocks(
    atoms: Dict[AtomId, Atom],
) -> Dict[AtomId, int]:
    """
    Startpartition: alle Atome mit gleichem gvar zusammen.
    """
    gvars = sorted(
        {atom.gvar for atom in atoms.values()}
    )
    gvar_to_block = {
        gvar: block
        for block, gvar in enumerate(gvars)
    }

    return {
        atom_id: gvar_to_block[atom.gvar]
        for atom_id, atom in atoms.items()
    }


def build_conflict_graph_for_block(
    members: List[AtomId],
    atoms: Dict[AtomId, Atom],
    block_of: Dict[AtomId, int],
) -> Dict[AtomId, Set[AtomId]]:
    """
    Zwei Atome sind inkompatibel, wenn sie:
      - im selben aktuellen gstate-Block liegen,
      - dieselbe externe Position xhat/yhat besitzen,
      - dieselbe Aktion besitzen,
      - aber in verschiedene successor blocks übergehen.

    Atome an unterschiedlichen Positionen dürfen im selben gstate bleiben,
    weil xhat/yhat ohnehin Teil des PRISM-Gesamtzustands sind.
    """
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

            if a.next_atom is None:
                a_next_block = None
            else:
                a_next_block = block_of[a.next_atom]

            for b_id in context_members[i + 1:]:
                b = atoms[b_id]

                if b.next_atom is None:
                    b_next_block = None
                else:
                    b_next_block = block_of[b.next_atom]

                if a_next_block != b_next_block:
                    graph[a_id].add(b_id)
                    graph[b_id].add(a_id)

    return graph


def greedy_color(
    graph: Dict[AtomId, Set[AtomId]],
) -> Dict[AtomId, int]:
    """
    Deterministisches greedy graph coloring.

    Es minimiert nicht garantiert global die Farbzahl, hält die zusätzliche
    Zustandszahl aber typischerweise deutlich kleiner als eine Aufsplittung
    nach kompletter Historie oder Position.
    """
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
) -> Dict[AtomId, int]:
    """
    Iterative, konfliktgetriebene Partition Refinement.

    Es wird nur gesplittet, nie wieder zusammengeführt.
    """
    block_of = initial_blocks(atoms)

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
    atoms: Dict[AtomId, Atom],
    block_of: Dict[AtomId, int],
) -> Tuple[
    Dict[AtomId, int],
    Dict[int, int],
    Dict[int, List[AtomId]],
]:
    """
    Nummeriert finale Blocks deterministisch als gstate=0..K.
    Alle gstates eines gvar liegen zusammen.
    """
    block_members = defaultdict(list)

    for atom_id, block in block_of.items():
        block_members[block].append(atom_id)

    sortable = []

    for old_block, members in block_members.items():
        gvars = {
            atoms[atom_id].gvar
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
        for atom_id in atoms
    }

    gstate_to_gvar = {}
    gstate_members = defaultdict(list)

    for atom_id, gstate in atom_to_gstate.items():
        gvar = atoms[atom_id].gvar

        if (
            gstate in gstate_to_gvar
            and gstate_to_gvar[gstate] != gvar
        ):
            raise ValueError(
                "Internal gstate/gvar inconsistency."
            )

        gstate_to_gvar[gstate] = gvar
        gstate_members[gstate].append(atom_id)

    return (
        atom_to_gstate,
        gstate_to_gvar,
        gstate_members,
    )


def build_refined_lookup(
    atoms: Dict[AtomId, Atom],
    atom_to_gstate: Dict[AtomId, int],
    gstate_to_gvar: Dict[int, int],
) -> List[dict]:
    """
    Erzeugt und validiert:
      (xhat,yhat,gstate,action)
         -> (xhat_next,yhat_next,gstate_next)
    """
    transitions = {}

    for atom_id, atom in atoms.items():
        if (
            atom.action is None
            or atom.next_atom is None
        ):
            continue

        if atom.next_atom not in atoms:
            continue

        next_atom = atoms[atom.next_atom]

        gstate = atom_to_gstate[atom_id]
        next_gstate = atom_to_gstate[atom.next_atom]

        source = (
            atom.xhat,
            atom.yhat,
            gstate,
            atom.action,
        )

        successor = (
            next_atom.xhat,
            next_atom.yhat,
            next_gstate,
        )

        if (
            source in transitions
            and transitions[source] != successor
        ):
            raise ValueError(
                "Refinement failed: non-deterministic source "
                f"{source} -> {transitions[source]} / {successor}"
            )

        transitions[source] = successor

    rows = []

    for source, successor in transitions.items():
        x, y, gstate, action = source
        nx, ny, next_gstate = successor

        rows.append(
            {
                "xhat": x,
                "yhat": y,
                "gstate": gstate,
                "gvar": gstate_to_gvar[gstate],
                "action": action,
                "xhat_next": nx,
                "yhat_next": ny,
                "gstate_next": next_gstate,
                "gvar_next": gstate_to_gvar[next_gstate],
            }
        )

    rows.sort(
        key=lambda row: (
            row["xhat"],
            row["yhat"],
            row["gstate"],
            row["action"],
        )
    )

    return rows


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


def write_map_outputs(
    map_id: int,
    h: float,
    output_dir: Path,
    gvar_to_sigma: Dict[int, Sigma],
    atoms: Dict[AtomId, Atom],
    atom_to_gstate: Dict[AtomId, int],
    gstate_to_gvar: Dict[int, int],
    gstate_members: Dict[int, List[AtomId]],
    lookup_rows: List[dict],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_gvars, max_gstates_per_gvar = split_statistics(
        gstate_to_gvar
    )

    gstates = []

    for gstate in sorted(gstate_to_gvar):
        gvar = gstate_to_gvar[gstate]
        sigma = gvar_to_sigma[gvar]
        members = sorted(
            gstate_members[gstate],
            key=lambda a: (
                a.start_x,
                a.start_y,
                a.step,
            ),
        )

        gstates.append(
            {
                "gstate": gstate,
                "gvar": gvar,
                "quantized_sigma": {
                    "var_x": sigma[0],
                    "var_y": sigma[1],
                    "cov_xy": sigma[2],
                },
                "number_of_raw_atoms": len(members),
                "example_histories": [
                    {
                        "update_position": [
                            atom_id.start_x,
                            atom_id.start_y,
                        ],
                        "step_since_update": atom_id.step,
                        "raw_sigma": {
                            "var_x": atoms[atom_id].raw_sigma[0],
                            "var_y": atoms[atom_id].raw_sigma[1],
                            "cov_xy": atoms[atom_id].raw_sigma[2],
                        },
                    }
                    for atom_id in members[:5]
                ],
            }
        )

    gvars = [
        {
            "gvar": gvar,
            "var_x": sigma[0],
            "var_y": sigma[1],
            "cov_xy": sigma[2],
            "gstates": sorted(
                state
                for state, gv in gstate_to_gvar.items()
                if gv == gvar
            ),
        }
        for gvar, sigma in sorted(gvar_to_sigma.items())
    ]

    # After a perfect update Sigma=0 (thus gvar=0), but the refined
    # Markov state may still depend on the current robot position.
    # Record the correct reset gstate for every possible update position.
    reset_gstate_by_position = {}

    for atom_id, gstate in atom_to_gstate.items():
        if atom_id.step != 0:
            continue

        key = f"{atom_id.start_x},{atom_id.start_y}"

        if key in reset_gstate_by_position:
            if reset_gstate_by_position[key] != gstate:
                raise ValueError(
                    f"Multiple reset gstates for position {key}: "
                    f"{reset_gstate_by_position[key]} and {gstate}"
                )
        else:
            reset_gstate_by_position[key] = gstate

    json_data = {
        "map": map_id,
        "h": h,
        "number_of_gvars": len(gvar_to_sigma),
        "number_of_gstates": len(gstate_to_gvar),
        "extra_gstates": (
            len(gstate_to_gvar)
            - len(gvar_to_sigma)
        ),
        "split_gvar_count": split_gvars,
        "max_gstates_per_gvar": max_gstates_per_gvar,
        "reset_gstate_by_position": reset_gstate_by_position,
        "number_of_lookup_transitions": len(lookup_rows),
        "gvars": gvars,
        "gstates": gstates,
        "lookup": lookup_rows,
    }

    json_path = (
        output_dir
        / f"gaussian_refined_{map_id}.json"
    )

    with json_path.open("w") as f:
        json.dump(
            json_data,
            f,
            indent=2,
        )

    csv_path = (
        output_dir
        / f"gaussian_refined_{map_id}.csv"
    )

    fieldnames = [
        "xhat",
        "yhat",
        "gstate",
        "gvar",
        "action",
        "xhat_next",
        "yhat_next",
        "gstate_next",
        "gvar_next",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(lookup_rows)

    return {
        "map": map_id,
        "gvars": len(gvar_to_sigma),
        "gstates": len(gstate_to_gvar),
        "extra_gstates": (
            len(gstate_to_gvar)
            - len(gvar_to_sigma)
        ),
        "split_gvars": split_gvars,
        "max_gstates_per_gvar": max_gstates_per_gvar,
        "lookup_transitions": len(lookup_rows),
    }


def analyse_map(
    map_id: int,
    map_path: Path,
    output_dir: Path,
    target_x: int,
    target_y: int,
    p: float,
    h: float,
    max_steps: int,
) -> dict:
    map_data = load_map_for_generator(map_path)

    policy = build_mape_policy(
        map_data,
        target_x,
        target_y,
    )

    atoms, gvar_to_sigma = enumerate_atoms(
        map_data,
        policy,
        p,
        h,
        max_steps,
    )

    block_of = refine_blocks(atoms)

    (
        atom_to_gstate,
        gstate_to_gvar,
        gstate_members,
    ) = canonicalize_gstates(
        atoms,
        block_of,
    )

    lookup_rows = build_refined_lookup(
        atoms,
        atom_to_gstate,
        gstate_to_gvar,
    )

    return write_map_outputs(
        map_id,
        h,
        output_dir,
        gvar_to_sigma,
        atoms,
        atom_to_gstate,
        gstate_to_gvar,
        gstate_members,
        lookup_rows,
    )


def write_summary(
    output_dir: Path,
    rows: List[dict],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "map",
        "gvars",
        "gstates",
        "extra_gstates",
        "split_gvars",
        "max_gstates_per_gvar",
        "lookup_transitions",
    ]

    csv_path = (
        output_dir
        / "gaussian_refinement_summary.csv"
    )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    def stats(values):
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": (
                sum(values) / len(values)
                if values
                else None
            ),
        }

    summary = {
        "analysed_maps": len(rows),
        "gvars": stats(
            [r["gvars"] for r in rows]
        ),
        "gstates": stats(
            [r["gstates"] for r in rows]
        ),
        "extra_gstates": stats(
            [r["extra_gstates"] for r in rows]
        ),
        "split_gvars": stats(
            [r["split_gvars"] for r in rows]
        ),
        "lookup_transitions": stats(
            [r["lookup_transitions"] for r in rows]
        ),
        "maps": rows,
    }

    json_path = (
        output_dir
        / "gaussian_refinement_summary.json"
    )

    with json_path.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refines h=0.05 Gaussian gvar classes into "
            "Markov-compatible gstates."
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
        default=Path("gaussian_refined"),
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
        "--h",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.h <= 0.0:
        raise ValueError("--h must be > 0.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0.")

    rows = []

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

        result = analyse_map(
            map_id=map_id,
            map_path=map_path,
            output_dir=args.output_dir,
            target_x=args.target_x,
            target_y=args.target_y,
            p=args.p,
            h=args.h,
            max_steps=args.max_steps,
        )

        rows.append(result)

        print(
            f"[map {map_id}] "
            f"gvars={result['gvars']}, "
            f"gstates={result['gstates']}, "
            f"extra={result['extra_gstates']}, "
            f"split_gvars={result['split_gvars']}, "
            f"transitions={result['lookup_transitions']}"
        )

    write_summary(
        args.output_dir,
        rows,
    )

    if rows:
        mean_gvars = (
            sum(r["gvars"] for r in rows)
            / len(rows)
        )
        mean_gstates = (
            sum(r["gstates"] for r in rows)
            / len(rows)
        )
        mean_extra = (
            sum(r["extra_gstates"] for r in rows)
            / len(rows)
        )

        print()
        print(
            f"Analysierte Maps: {len(rows)}"
        )
        print(
            f"Mean gvars: {mean_gvars:.2f}"
        )
        print(
            f"Mean refined gstates: "
            f"{mean_gstates:.2f}"
        )
        print(
            f"Mean extra gstates: "
            f"{mean_extra:.2f}"
        )
        print(
            f"Output: {args.output_dir}"
        )


if __name__ == "__main__":
    main()
