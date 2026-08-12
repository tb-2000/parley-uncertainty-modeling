#!/usr/bin/env python3
"""
Empirische Kalibrierung der Intervallschwellen für das symmetrische
PARLEY-Intervallmodell.

Es werden ausschließlich folgende Dateien berücksichtigt:

    model_10.prism
    model_11.prism
    ...
    model_99.prism

Andere Dateien wie model_10_umc.prism oder model_10_old.prism werden ignoriert.

Für jede Schwelle von 1 bis 2*N+1 wird über alle Modelle und alle im
Adaptation_MAPE_controller definierten Startzustände ermittelt:

- erste Schwellenüberschreitung,
- Minimum,
- Q1,
- Median,
- Mittelwert,
- Q3,
- Maximum,
- Standardabweichung,
- Anteil der Startzustände, welche die Schwelle innerhalb von max_steps
  erreichen.

Das Skript wählt bewusst nicht automatisch die endgültigen zehn Schwellen.
Stattdessen erzeugt es eine transparente Tabelle, aus der die Schwellen
wissenschaftlich nachvollziehbar ausgewählt werden können.

Beispiel:

    python analyze_interval_thresholds_90_models.py \
        Applications/EvoChecker-master/models \
        --max-steps 20 \
        --output-dir interval_threshold_analysis
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


MODEL_FILENAME_PATTERN = re.compile(r"model_([1-9][0-9])\.prism$")


@dataclass(frozen=True)
class ModelData:
    path: Path
    number: int
    n: int
    controller: dict[tuple[int, int], str]


def parse_int_constant(text: str, name: str) -> int:
    match = re.search(
        rf"\bconst\s+int\s+{re.escape(name)}\s*=\s*(\d+)\s*;",
        text,
    )
    if not match:
        raise ValueError(f"Konstante {name!r} wurde nicht gefunden.")
    return int(match.group(1))


def parse_controller(text: str) -> dict[tuple[int, int], str]:
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

    controller: dict[tuple[int, int], str] = {}
    for direction, x_value, y_value in command_pattern.findall(
        module_match.group(1)
    ):
        controller[(int(x_value), int(y_value))] = direction

    if not controller:
        raise ValueError(
            "Im Adaptation_MAPE_controller wurden keine Richtungen erkannt."
        )

    return controller


def parse_model(path: Path) -> ModelData:
    filename_match = MODEL_FILENAME_PATTERN.fullmatch(path.name)
    if not filename_match:
        raise ValueError("Dateiname entspricht nicht model_10.prism bis model_99.prism.")

    text = path.read_text(encoding="utf-8")

    return ModelData(
        path=path,
        number=int(filename_match.group(1)),
        n=parse_int_constant(text, "N"),
        controller=parse_controller(text),
    )


def discover_models(models_dir: Path) -> list[Path]:
    """Liest ausschließlich model_10.prism bis model_99.prism ein."""
    model_paths = []

    for path in models_dir.iterdir():
        match = MODEL_FILENAME_PATTERN.fullmatch(path.name)
        if not path.is_file() or not match:
            continue

        number = int(match.group(1))
        if 10 <= number <= 99:
            model_paths.append(path)

    return sorted(
        model_paths,
        key=lambda path: int(MODEL_FILENAME_PATTERN.fullmatch(path.name).group(1)),
    )


def advance_symmetric(
    state: tuple[int, int, int, int],
    direction: str,
    n: int,
) -> tuple[int, int, int, int]:
    xhat, yhat, xradius, yradius = state

    if direction == "east":
        return (
            min(xhat + 1, n),
            yhat,
            min(xradius + 2, n),
            min(yradius + 1, n),
        )

    if direction == "west":
        return (
            max(xhat - 1, 0),
            yhat,
            min(xradius + 2, n),
            min(yradius + 1, n),
        )

    if direction == "north":
        return (
            xhat,
            min(yhat + 1, n),
            min(xradius + 1, n),
            min(yradius + 2, n),
        )

    if direction == "south":
        return (
            xhat,
            max(yhat - 1, 0),
            min(xradius + 1, n),
            min(yradius + 2, n),
        )

    raise ValueError(f"Unbekannte Richtung: {direction}")


def calculate_interval_width(
    state: tuple[int, int, int, int],
    n: int,
) -> int:
    xhat, yhat, xradius, yradius = state

    xlow = max(xhat - xradius, 0)
    xhigh = min(xhat + xradius, n)
    ylow = max(yhat - yradius, 0)
    yhigh = min(yhat + yradius, n)

    return (xhigh - xlow) + (yhigh - ylow)


def linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Quantil einer leeren Folge ist nicht definiert.")

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def simulate_model(
    model: ModelData,
    max_steps: int,
    max_threshold: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for start_x, start_y in sorted(model.controller):
        state = (start_x, start_y, 0, 0)
        first_passage: dict[int, int] = {}

        for step in range(1, max_steps + 1):
            direction = model.controller.get((state[0], state[1]))
            if direction is None:
                break

            state = advance_symmetric(state, direction, model.n)
            width = calculate_interval_width(state, model.n)

            for threshold in range(1, max_threshold + 1):
                if threshold not in first_passage and width >= threshold:
                    first_passage[threshold] = step

        for threshold in range(1, max_threshold + 1):
            first_step = first_passage.get(threshold)

            rows.append(
                {
                    "model": model.number,
                    "start_x": start_x,
                    "start_y": start_y,
                    "threshold": threshold,
                    "reached": int(first_step is not None),
                    "first_passage_step": (
                        first_step if first_step is not None else ""
                    ),
                }
            )

    return rows


def summarize(
    rows: Sequence[Mapping[str, object]],
    max_threshold: int,
) -> list[dict[str, object]]:
    starts = {
        (int(row["model"]), int(row["start_x"]), int(row["start_y"]))
        for row in rows
    }
    total_starts = len(starts)

    grouped: dict[int, list[int]] = defaultdict(list)

    for row in rows:
        if int(row["reached"]) == 1:
            grouped[int(row["threshold"])].append(
                int(row["first_passage_step"])
            )

    result: list[dict[str, object]] = []

    for threshold in range(1, max_threshold + 1):
        values = grouped.get(threshold, [])
        reached_count = len(values)
        reached_fraction = reached_count / total_starts if total_starts else 0.0

        summary: dict[str, object] = {
            "threshold": threshold,
            "total_starts": total_starts,
            "reached_count": reached_count,
            "not_reached_count": total_starts - reached_count,
            "reached_fraction": reached_fraction,
            "not_reached_fraction": 1.0 - reached_fraction,
        }

        if values:
            summary.update(
                {
                    "first_step_min": min(values),
                    "first_step_q1": linear_quantile(values, 0.25),
                    "first_step_median": linear_quantile(values, 0.50),
                    "first_step_mean": statistics.fmean(values),
                    "first_step_q3": linear_quantile(values, 0.75),
                    "first_step_max": max(values),
                    "first_step_stddev": (
                        statistics.pstdev(values) if len(values) > 1 else 0.0
                    ),
                }
            )
        else:
            summary.update(
                {
                    "first_step_min": "",
                    "first_step_q1": "",
                    "first_step_median": "",
                    "first_step_mean": "",
                    "first_step_q3": "",
                    "first_step_max": "",
                    "first_step_stddev": "",
                }
            )

        result.append(summary)

    return result


def build_behavior_groups(
    summary: Sequence[Mapping[str, object]],
    not_reached_tolerance: float = 0.02,
) -> list[dict[str, object]]:
    """Gruppiert Schwellen anhand ihres empirischen Updateverhaltens.

    Gleiches Gruppenverhalten erfordert:
    - identischen Median,
    - identisches Q3,
    - höchstens ``not_reached_tolerance`` Unterschied beim Anteil
      nicht erreichter Zustände.

    Standard: 0.02 = zwei Prozentpunkte.
    """
    groups: list[dict[str, object]] = []

    for row in summary:
        threshold = int(row["threshold"])
        reached_count = int(row["reached_count"])

        if reached_count == 0:
            median: object = "nie"
            q3: object = "nie"
            not_reached_fraction = 1.0
        else:
            median = float(row["first_step_median"])
            q3 = float(row["first_step_q3"])
            not_reached_fraction = float(row["not_reached_fraction"])

        matching_group = None
        for group in groups:
            same_passage = (
                group["median_first_passage"] == median
                and group["q3_first_passage"] == q3
            )
            similar_nonreach = abs(
                float(group["representative_not_reached_fraction"])
                - not_reached_fraction
            ) <= not_reached_tolerance

            if same_passage and similar_nonreach:
                matching_group = group
                break

        if matching_group is None:
            groups.append(
                {
                    "median_first_passage": median,
                    "q3_first_passage": q3,
                    "representative_not_reached_fraction": not_reached_fraction,
                    "threshold_values": [threshold],
                    "not_reached_values": [not_reached_fraction],
                }
            )
        else:
            matching_group["threshold_values"].append(threshold)
            matching_group["not_reached_values"].append(not_reached_fraction)
            matching_group["representative_not_reached_fraction"] = (
                statistics.fmean(matching_group["not_reached_values"])
            )

    groups.sort(
        key=lambda group: (
            math.inf
            if group["median_first_passage"] == "nie"
            else float(group["median_first_passage"]),
            math.inf
            if group["q3_first_passage"] == "nie"
            else float(group["q3_first_passage"]),
            float(group["representative_not_reached_fraction"]),
        )
    )

    output: list[dict[str, object]] = []

    for group_number, group in enumerate(groups, start=1):
        thresholds = sorted(group["threshold_values"])
        nonreach = group["not_reached_values"]

        output.append(
            {
                "group": group_number,
                "median_first_passage": group["median_first_passage"],
                "q3_first_passage": group["q3_first_passage"],
                "mean_not_reached_fraction": statistics.fmean(nonreach),
                "minimum_not_reached_fraction": min(nonreach),
                "maximum_not_reached_fraction": max(nonreach),
                "not_reached_tolerance": not_reached_tolerance,
                "thresholds": ",".join(str(value) for value in thresholds),
                "minimum_threshold": min(thresholds),
                "maximum_threshold": max(thresholds),
            }
        )

    return output


def build_manual_selection_table(
    summary: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Erzeugt eine kompakte Tabelle für die manuelle Schwellenwahl."""
    rows = []

    for row in summary:
        rows.append(
            {
                "threshold": row["threshold"],
                "median_first_passage": row["first_step_median"],
                "mean_first_passage": row["first_step_mean"],
                "q1_first_passage": row["first_step_q1"],
                "q3_first_passage": row["first_step_q3"],
                "stddev_first_passage": row["first_step_stddev"],
                "reached_fraction": row["reached_fraction"],
                "not_reached_fraction": row["not_reached_fraction"],
            }
        )

    return rows



def choose_group_representative(
    group: Mapping[str, object],
    summary_by_threshold: Mapping[int, Mapping[str, object]],
) -> int:
    """Wählt eine repräsentative Schwelle aus einer Verhaltensgruppe.

    Gewählt wird die Schwelle, deren Nichterreichungsanteil dem Gruppenmittel
    am nächsten liegt. Bei Gleichstand wird die größere Schwelle genommen,
    damit nicht unnötig aggressive Updates bevorzugt werden.
    """
    thresholds = [
        int(value)
        for value in str(group["thresholds"]).split(",")
        if str(value).strip()
    ]
    group_nonreach = float(group["mean_not_reached_fraction"])

    return min(
        thresholds,
        key=lambda threshold: (
            abs(
                float(summary_by_threshold[threshold]["not_reached_fraction"])
                - group_nonreach
            ),
            -threshold,
        ),
    )


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Liefert `count` möglichst gleichmäßig verteilte Indizes."""
    if count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length - 1]

    raw = [
        round(index * (length - 1) / (count - 1))
        for index in range(count)
    ]

    # round() kann in seltenen Fällen Duplikate erzeugen.
    indices: list[int] = []
    for value in raw:
        if value not in indices:
            indices.append(value)

    # Fehlende Plätze mit noch nicht verwendeten Indizes auffüllen.
    if len(indices) < count:
        for value in range(length):
            if value not in indices:
                indices.append(value)
            if len(indices) == count:
                break

    return sorted(indices[:count])


def choose_thresholds_from_groups(
    summary: Sequence[Mapping[str, object]],
    behavior_groups: Sequence[Mapping[str, object]],
    number_thresholds: int = 10,
) -> list[int]:
    """Wählt map-spezifisch zehn möglichst unterschiedliche Schwellen.

    Prinzip:
    1. `2*N+1` (bei N=9 also 19) wird als "kein Update" fest aufgenommen.
    2. Die übrigen Schwellen werden aus unterschiedlichen empirischen
       Verhaltensgruppen gewählt.
    3. Gibt es mehr Gruppen als Plätze, werden die Gruppen gleichmäßig über
       das gesamte beobachtete Verhalten verteilt ausgewählt.
    4. Gibt es weniger Gruppen als Plätze, werden zusätzliche Schwellen aus
       Gruppen mit mehreren Kandidaten ergänzt.

    Dadurch werden keine künstlichen Zielschritte 1..10 erzwungen.
    Stattdessen repräsentieren die zehn Entscheidungen zehn möglichst
    unterschiedliche Unsicherheitstoleranzen der jeweiligen Map.
    """
    if number_thresholds < 2:
        raise ValueError("number_thresholds muss mindestens 2 sein.")

    summary_by_threshold = {
        int(row["threshold"]): row
        for row in summary
    }

    unreachable_groups = [
        group
        for group in behavior_groups
        if str(group["median_first_passage"]) == "nie"
    ]
    reachable_groups = [
        group
        for group in behavior_groups
        if str(group["median_first_passage"]) != "nie"
    ]

    if not unreachable_groups:
        raise ValueError(
            "Keine Nie-Update-Gruppe gefunden. "
            "Bei N=9 sollte Schwelle 19 nicht erreichbar sein."
        )

    # In der Praxis ist das bei N=9 Schwelle 19.
    unreachable_thresholds = []
    for group in unreachable_groups:
        unreachable_thresholds.extend(
            int(value)
            for value in str(group["thresholds"]).split(",")
            if str(value).strip()
        )
    never_update_threshold = max(unreachable_thresholds)

    reachable_slots = number_thresholds - 1

    selected: list[int] = []

    if len(reachable_groups) >= reachable_slots:
        group_indices = evenly_spaced_indices(
            len(reachable_groups),
            reachable_slots,
        )
        chosen_groups = [
            reachable_groups[index]
            for index in group_indices
        ]
        selected.extend(
            choose_group_representative(
                group,
                summary_by_threshold,
            )
            for group in chosen_groups
        )

    else:
        # Erst jede vorhandene Verhaltensgruppe einmal repräsentieren.
        for group in reachable_groups:
            selected.append(
                choose_group_representative(
                    group,
                    summary_by_threshold,
                )
            )

        # Falls noch Plätze fehlen, zusätzliche Schwellen aus Gruppen mit
        # mehreren Kandidaten hinzufügen. Bevorzugt werden höhere Schwellen,
        # weil sie innerhalb derselben Gruppe weniger aggressive Updates
        # repräsentieren.
        extra_candidates: list[int] = []

        for group in reversed(reachable_groups):
            thresholds = sorted(
                (
                    int(value)
                    for value in str(group["thresholds"]).split(",")
                    if str(value).strip()
                ),
                reverse=True,
            )
            representative = choose_group_representative(
                group,
                summary_by_threshold,
            )
            for threshold in thresholds:
                if threshold != representative:
                    extra_candidates.append(threshold)

        for threshold in extra_candidates:
            if threshold not in selected:
                selected.append(threshold)
            if len(selected) == reachable_slots:
                break

    selected = sorted(set(selected))

    # Falls set() wegen Überschneidungen Plätze entfernt hat, mit noch
    # unbenutzten erreichbaren Schwellen auffüllen.
    if len(selected) < reachable_slots:
        all_reachable = sorted(
            threshold
            for threshold, row in summary_by_threshold.items()
            if int(row["reached_count"]) > 0
            and threshold != never_update_threshold
        )
        for threshold in reversed(all_reachable):
            if threshold not in selected:
                selected.append(threshold)
            if len(selected) == reachable_slots:
                break
        selected.sort()

    if len(selected) > reachable_slots:
        # Gleichmäßig auf die gewünschte Anzahl reduzieren.
        indices = evenly_spaced_indices(
            len(selected),
            reachable_slots,
        )
        selected = [selected[index] for index in indices]

    selected.append(never_update_threshold)
    return selected


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet für model_10.prism bis model_99.prism "
            "map-spezifische Intervallschwellen anhand unterschiedlicher "
            "First-Passage-Verhaltensklassen."
        )
    )
    parser.add_argument(
        "models_dir",
        type=Path,
        help="Ordner mit model_10.prism bis model_99.prism",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximal simulierte Schritte, Standard: 20",
    )
    parser.add_argument(
        "--number-thresholds",
        type=int,
        default=10,
        help="Anzahl Schwellen pro Map, Standard: 10",
    )
    parser.add_argument(
        "--not-reached-tolerance",
        type=float,
        default=0.02,
        help=(
            "Maximale Differenz des Anteils nicht erreichter Zustände "
            "innerhalb einer Verhaltensgruppe. Standard: 0.02."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("interval_thresholds_per_map"),
        help="Ausgabeordner",
    )
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        parser.error(f"Kein Verzeichnis: {args.models_dir}")

    if args.max_steps < 1:
        parser.error("--max-steps muss mindestens 1 sein.")

    if args.number_thresholds < 2:
        parser.error("--number-thresholds muss mindestens 2 sein.")

    if not 0.0 <= args.not_reached_tolerance <= 1.0:
        parser.error(
            "--not-reached-tolerance muss zwischen 0 und 1 liegen."
        )

    model_paths = discover_models(args.models_dir)

    if not model_paths:
        parser.error(
            "Keine Dateien model_10.prism bis model_99.prism gefunden."
        )

    models: list[ModelData] = []
    skipped_models: list[dict[str, object]] = []

    for path in model_paths:
        try:
            models.append(parse_model(path))
        except (ValueError, OSError, UnicodeError) as error:
            skipped_models.append(
                {
                    "model": path.name,
                    "reason": str(error),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_selected_rows: list[dict[str, object]] = []
    all_statistics_rows: list[dict[str, object]] = []
    all_group_rows: list[dict[str, object]] = []
    thresholds_per_map: dict[int, list[int]] = {}

    for model in models:
        max_threshold = 2 * model.n + 1

        passage_rows = simulate_model(
            model=model,
            max_steps=args.max_steps,
            max_threshold=max_threshold,
        )

        map_summary = summarize(
            passage_rows,
            max_threshold=max_threshold,
        )

        map_groups = build_behavior_groups(
            map_summary,
            not_reached_tolerance=args.not_reached_tolerance,
        )

        selected_thresholds = choose_thresholds_from_groups(
            summary=map_summary,
            behavior_groups=map_groups,
            number_thresholds=args.number_thresholds,
        )

        thresholds_per_map[model.number] = selected_thresholds

        summary_by_threshold = {
            int(row["threshold"]): row
            for row in map_summary
        }

        for row in map_summary:
            all_statistics_rows.append(
                {
                    "model": model.number,
                    **row,
                }
            )

        for row in map_groups:
            all_group_rows.append(
                {
                    "model": model.number,
                    **row,
                }
            )

        for decision, threshold in enumerate(
            selected_thresholds,
            start=1,
        ):
            stats = summary_by_threshold[threshold]

            all_selected_rows.append(
                {
                    "model": model.number,
                    "decision": decision,
                    "threshold": threshold,
                    "first_step_q1": stats["first_step_q1"],
                    "first_step_median": stats["first_step_median"],
                    "first_step_mean": stats["first_step_mean"],
                    "first_step_q3": stats["first_step_q3"],
                    "reached_fraction": stats["reached_fraction"],
                    "not_reached_fraction": stats["not_reached_fraction"],
                }
            )

        print(
            f"Map {model.number}: thresholds = "
            f"{selected_thresholds}"
        )

    write_csv(
        args.output_dir / "selected_thresholds_per_map.csv",
        all_selected_rows,
    )
    write_csv(
        args.output_dir / "threshold_statistics_per_map.csv",
        all_statistics_rows,
    )
    write_csv(
        args.output_dir / "behavior_groups_per_map.csv",
        all_group_rows,
    )
    write_csv(
        args.output_dir / "skipped_models.csv",
        skipped_models,
    )

    mapping_path = args.output_dir / "thresholds_per_map.py"
    with mapping_path.open("w", encoding="utf-8") as file:
        file.write("THRESHOLDS_PER_MAP = {\n")
        for model_number in sorted(thresholds_per_map):
            file.write(
                f"    {model_number}: "
                f"{thresholds_per_map[model_number]},\n"
            )
        file.write("}\n")

    print()
    print(f"Gefundene Modell-Dateien: {len(model_paths)}")
    print(f"Erfolgreich ausgewertet:  {len(models)}")
    print(f"Übersprungen:              {len(skipped_models)}")
    print()
    print(f"Ergebnisse: {args.output_dir.resolve()}")
    print("Wichtigste Dateien:")
    print("  thresholds_per_map.py")
    print("  selected_thresholds_per_map.csv")
    print("  behavior_groups_per_map.csv")
    print("  threshold_statistics_per_map.csv")


if __name__ == "__main__":
    main()
