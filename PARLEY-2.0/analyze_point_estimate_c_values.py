#!/usr/bin/env python3
"""
Analyse der c-Werte von Point-Estimate-Pareto-Policies.

Das Skript liest:
1. ein Point-Estimate-PRISM-Modell (model_<map>.prism),
2. die zugehörige EvoChecker *_Set-Datei,
3. optional die zugehörige *_Front-Datei.

Es wertet die decision_x_y-Werte auf drei Ebenen aus:

A) ALL
   Alle in der Set-Datei vorhandenen decision_x_y-Spalten.

B) MAPE
   Nur decision_x_y für Positionen, für die der
   Adaptation_MAPE_controller im PRISM-Modell tatsächlich eine Aktion besitzt.
   Das ist die wichtigste Auswertung für den Vergleich mit dem Intervallmodell.

C) NOMINAL_PATH
   Nur Positionen auf der nominalen MAPE-Route vom xstart/ystart bis zum Ziel.
   Diese Auswertung ist noch strenger, berücksichtigt aber NICHT die
   stochastischen Abweichungen des Roboters und ist daher nur ergänzend.

Wichtig:
Auch die MAPE-Auswertung zeigt noch nicht, welche Zustände unter einer
konkreten Policy probabilistisch tatsächlich besucht werden. Dafür wäre eine
Visitation-/Reachability-Analyse des vollständigen DTMC nötig. Sie ist aber
deutlich aussagekräftiger als das ungefilterte Zählen aller 100 Entscheidungen.

Beispiel:
    python analyze_point_estimate_c_values.py \
        model_14.prism \
        ROBOT14_REP0_NSGAII_175809_020826_Set \
        --front ROBOT14_REP0_NSGAII_175809_020826_Front \
        --output-dir c_analysis_map14_rep0
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


DECISION_RE = re.compile(r"decision_(\d+)_(\d+)$")


def parse_int_constant(text: str, name: str) -> int:
    match = re.search(
        rf"\bconst\s+int\s+{re.escape(name)}\s*=\s*(\d+)\s*;",
        text,
    )
    if not match:
        raise ValueError(f"Konstante {name!r} wurde im PRISM-Modell nicht gefunden.")
    return int(match.group(1))


def parse_controller(text: str) -> dict[tuple[int, int], str]:
    """Liest (xhat,yhat) -> MAPE-Aktion aus Adaptation_MAPE_controller."""
    module_match = re.search(
        r"module\s+Adaptation_MAPE_controller\s*(.*?)\s*endmodule",
        text,
        flags=re.DOTALL,
    )
    if not module_match:
        raise ValueError("Adaptation_MAPE_controller wurde nicht gefunden.")

    command_pattern = re.compile(
        r"\[(west|east|south|north)\]\s*"
        r"\(xhat\s*=\s*(\d+)\)\s*&\s*"
        r"\(yhat\s*=\s*(\d+)\)\s*->\s*true\s*;"
    )

    controller = {}
    for direction, x_value, y_value in command_pattern.findall(
        module_match.group(1)
    ):
        controller[(int(x_value), int(y_value))] = direction

    if not controller:
        raise ValueError("Keine MAPE-Aktionen im Controller erkannt.")

    return controller


def nominal_successor(
    position: tuple[int, int],
    direction: str,
    n: int,
) -> tuple[int, int]:
    x, y = position

    if direction == "east":
        return min(x + 1, n), y
    if direction == "west":
        return max(x - 1, 0), y
    if direction == "north":
        return x, min(y + 1, n)
    if direction == "south":
        return x, max(y - 1, 0)

    raise ValueError(f"Unbekannte Richtung: {direction}")


def build_nominal_path(
    controller: dict[tuple[int, int], str],
    start: tuple[int, int],
    target: tuple[int, int],
    n: int,
    max_steps: int = 1000,
) -> list[tuple[int, int]]:
    """Folgt nur der nominalen MAPE-Bewegung vom Start bis zum Ziel."""
    path = []
    current = start
    seen = set()

    for _ in range(max_steps):
        if current == target:
            break

        if current in seen:
            # Schutz gegen unerwartete Schleifen.
            break
        seen.add(current)

        direction = controller.get(current)
        if direction is None:
            break

        path.append(current)
        current = nominal_successor(current, direction, n)

    return path


def read_set_file(path: Path):
    """Liest Header und Policy-Zeilen einer EvoChecker Set-Datei."""
    with path.open("r", encoding="utf-8") as file:
        nonempty = [line.strip() for line in file if line.strip()]

    if not nonempty:
        raise ValueError("Set-Datei ist leer.")

    header = nonempty[0].split()

    policies = []
    for line_number, line in enumerate(nonempty[1:], start=2):
        values = line.split()

        if len(values) != len(header):
            raise ValueError(
                f"Set-Datei Zeile {line_number}: {len(values)} Werte, "
                f"aber {len(header)} Header-Spalten."
            )

        policies.append([int(value) for value in values])

    if not policies:
        raise ValueError("Set-Datei enthält keine Policies.")

    return header, policies


def read_front_file(path: Path, expected_policies: int):
    """Liest optional die Zielfunktionswerte zeilenweise passend zur Set-Datei."""
    with path.open("r", encoding="utf-8") as file:
        nonempty = [line.strip() for line in file if line.strip()]

    if not nonempty:
        raise ValueError("Front-Datei ist leer.")

    objective_names = nonempty[0].split("\t")
    rows = []

    for line in nonempty[1:]:
        values = line.split()
        rows.append(values)

    if len(rows) != expected_policies:
        raise ValueError(
            f"Front enthält {len(rows)} Zeilen, Set aber {expected_policies} Policies."
        )

    return objective_names, rows


def columns_for_positions(
    header: list[str],
    positions: set[tuple[int, int]],
) -> list[int]:
    indices = []

    for index, name in enumerate(header):
        match = DECISION_RE.fullmatch(name)
        if not match:
            continue

        position = (int(match.group(1)), int(match.group(2)))
        if position in positions:
            indices.append(index)

    return indices


def decision_columns(header: list[str]) -> list[int]:
    return [
        index
        for index, name in enumerate(header)
        if DECISION_RE.fullmatch(name)
    ]


def summarize_values(
    policies: list[list[int]],
    indices: list[int],
) -> dict[int, int]:
    counter = Counter()

    for policy in policies:
        for index in indices:
            counter[policy[index]] += 1

    return dict(sorted(counter.items()))


def write_distribution_csv(
    path: Path,
    scope: str,
    counts: dict[int, int],
):
    total = sum(counts.values())

    rows = []
    for c in range(1, 11):
        count = counts.get(c, 0)
        rows.append(
            {
                "scope": scope,
                "c": c,
                "count": count,
                "fraction": count / total if total else 0.0,
                "percent": 100.0 * count / total if total else 0.0,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_per_policy_csv(
    path: Path,
    policies: list[list[int]],
    indices: list[int],
    scope: str,
    front_rows=None,
    objective_names=None,
):
    fieldnames = [
        "policy_index",
        "scope",
        "num_decisions",
    ] + [f"c_{c}_count" for c in range(1, 11)] + [
        "c_7_10_count",
        "c_7_10_fraction",
        "mean_c",
    ]

    if front_rows is not None:
        fieldnames.extend(
            [f"objective_{i+1}" for i in range(len(front_rows[0]))]
        )

    rows = []

    for policy_index, policy in enumerate(policies):
        values = [policy[index] for index in indices]
        counts = Counter(values)
        long_count = sum(counts.get(c, 0) for c in range(7, 11))

        row = {
            "policy_index": policy_index,
            "scope": scope,
            "num_decisions": len(values),
        }

        for c in range(1, 11):
            row[f"c_{c}_count"] = counts.get(c, 0)

        row["c_7_10_count"] = long_count
        row["c_7_10_fraction"] = (
            long_count / len(values) if values else 0.0
        )
        row["mean_c"] = (
            sum(values) / len(values) if values else ""
        )

        if front_rows is not None:
            for i, value in enumerate(front_rows[policy_index]):
                row[f"objective_{i+1}"] = value

        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_distribution(title: str, counts: dict[int, int], num_columns: int):
    total = sum(counts.values())

    print()
    print(title)
    print("-" * len(title))
    print(f"berücksichtigte decision-Spalten pro Policy: {num_columns}")
    print(f"Policies: {total // num_columns if num_columns else 0}")
    print()
    print(" c    Anzahl     Anteil")

    for c in range(1, 11):
        count = counts.get(c, 0)
        fraction = count / total if total else 0.0
        print(f"{c:2d}  {count:8d}   {100.0*fraction:6.2f}%")

    long_count = sum(counts.get(c, 0) for c in range(7, 11))
    long_fraction = long_count / total if total else 0.0

    print()
    print(
        f"c=7..10 zusammen: {long_count} / {total} "
        f"= {100.0*long_fraction:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analysiert c=1..10 in EvoChecker-Point-Estimate-Policies "
            "und filtert auf relevante MAPE-Positionen."
        )
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Point-Estimate PRISM-Basismodell, z.B. model_14.prism",
    )
    parser.add_argument(
        "set_file",
        type=Path,
        help="EvoChecker *_Set-Datei",
    )
    parser.add_argument(
        "--front",
        type=Path,
        default=None,
        help="Optional zugehörige *_Front-Datei",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("point_estimate_c_analysis"),
    )

    args = parser.parse_args()

    model_text = args.model.read_text(encoding="utf-8")
    controller = parse_controller(model_text)

    n = parse_int_constant(model_text, "N")
    xstart = parse_int_constant(model_text, "xstart")
    ystart = parse_int_constant(model_text, "ystart")
    xtarget = parse_int_constant(model_text, "xtarget")
    ytarget = parse_int_constant(model_text, "ytarget")

    header, policies = read_set_file(args.set_file)

    front_rows = None
    objective_names = None

    if args.front is not None:
        objective_names, front_rows = read_front_file(
            args.front,
            expected_policies=len(policies),
        )

    all_indices = decision_columns(header)

    mape_positions = set(controller)
    mape_indices = columns_for_positions(header, mape_positions)

    nominal_path = build_nominal_path(
        controller=controller,
        start=(xstart, ystart),
        target=(xtarget, ytarget),
        n=n,
    )
    nominal_indices = columns_for_positions(
        header,
        set(nominal_path),
    )

    if not all_indices:
        raise ValueError("Keine decision_x_y-Spalten in der Set-Datei gefunden.")

    if not mape_indices:
        raise ValueError(
            "Keine Set-Spalten passen zu den MAPE-Positionen des Modells."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scopes = {
        "all_decisions": all_indices,
        "mape_positions": mape_indices,
        "nominal_path": nominal_indices,
    }

    for scope, indices in scopes.items():
        if not indices:
            continue

        counts = summarize_values(policies, indices)

        print_distribution(
            title=scope,
            counts=counts,
            num_columns=len(indices),
        )

        write_distribution_csv(
            args.output_dir / f"distribution_{scope}.csv",
            scope=scope,
            counts=counts,
        )

        write_per_policy_csv(
            args.output_dir / f"per_policy_{scope}.csv",
            policies=policies,
            indices=indices,
            scope=scope,
            front_rows=front_rows,
            objective_names=objective_names,
        )

    # Schreibe die tatsächlich verwendeten Positionen zur Kontrolle.
    with (
        args.output_dir / "mape_positions.csv"
    ).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "direction"])

        for (x, y), direction in sorted(controller.items()):
            writer.writerow([x, y, direction])

    with (
        args.output_dir / "nominal_path.csv"
    ).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["order", "x", "y", "direction"])

        for order, position in enumerate(nominal_path, start=1):
            writer.writerow(
                [
                    order,
                    position[0],
                    position[1],
                    controller[position],
                ]
            )

    print()
    print("Zusammenfassung")
    print("---------------")
    print(f"Set-Policies: {len(policies)}")
    print(f"decision-Spalten insgesamt: {len(all_indices)}")
    print(f"MAPE-definierte decision-Spalten: {len(mape_indices)}")
    print(f"Positionen auf nominaler Start-Ziel-Route: {len(nominal_indices)}")
    print(f"Ausgabeordner: {args.output_dir.resolve()}")
    print()
    print(
        "Für deinen Vergleich mit dem Intervallmodell ist "
        "`distribution_mape_positions.csv` die wichtigste erste Auswertung."
    )
    print(
        "Die nominal_path-Auswertung ist nur ergänzend, weil stochastische "
        "Roboterbewegungen weitere Knowledge-Positionen ermöglichen können."
    )


if __name__ == "__main__":
    main()
