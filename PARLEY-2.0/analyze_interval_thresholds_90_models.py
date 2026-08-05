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
) -> list[dict[str, object]]:
    """Gruppiert Schwellen nach Median und Q3 der First-Passage-Zeit.

    Dadurch wird sichtbar, welche Schwellen praktisch sehr ähnliches
    Updateverhalten erzeugen.
    """
    groups: dict[tuple[object, object], list[int]] = defaultdict(list)

    for row in summary:
        if int(row["reached_count"]) == 0:
            key = ("nie", "nie")
        else:
            key = (
                row["first_step_median"],
                row["first_step_q3"],
            )

        groups[key].append(int(row["threshold"]))

    output = []

    for group_number, (key, thresholds) in enumerate(
        sorted(
            groups.items(),
            key=lambda item: (
                math.inf if item[0][0] == "nie" else float(item[0][0]),
                math.inf if item[0][1] == "nie" else float(item[0][1]),
            ),
        ),
        start=1,
    ):
        median, q3 = key

        output.append(
            {
                "group": group_number,
                "median_first_passage": median,
                "q3_first_passage": q3,
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
            "Analysiert model_10.prism bis model_99.prism und berechnet "
            "First-Passage-Statistiken für Intervallschwellen."
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
        "--output-dir",
        type=Path,
        default=Path("interval_threshold_analysis"),
        help="Ausgabeordner",
    )

    args = parser.parse_args()

    if not args.models_dir.is_dir():
        parser.error(f"Kein Verzeichnis: {args.models_dir}")

    if args.max_steps < 1:
        parser.error("--max-steps muss mindestens 1 sein.")

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

    if not models:
        raise RuntimeError("Keines der gefundenen Modelle konnte gelesen werden.")

    n_values = {model.n for model in models}
    if len(n_values) != 1:
        raise ValueError(
            "Die Modelle haben unterschiedliche N-Werte. "
            "Bitte getrennt nach Kartengröße auswerten."
        )

    n = next(iter(n_values))
    max_threshold = 2 * n + 1

    all_rows: list[dict[str, object]] = []

    for model in models:
        all_rows.extend(
            simulate_model(
                model=model,
                max_steps=args.max_steps,
                max_threshold=max_threshold,
            )
        )

    combined_summary = summarize(
        all_rows,
        max_threshold=max_threshold,
    )

    behavior_groups = build_behavior_groups(combined_summary)
    selection_table = build_manual_selection_table(combined_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        args.output_dir / "first_passage_by_start.csv",
        all_rows,
    )
    write_csv(
        args.output_dir / "threshold_statistics.csv",
        combined_summary,
    )
    write_csv(
        args.output_dir / "behavior_groups.csv",
        behavior_groups,
    )
    write_csv(
        args.output_dir / "threshold_selection_table.csv",
        selection_table,
    )
    write_csv(
        args.output_dir / "skipped_models.csv",
        skipped_models,
    )

    print(f"Gefundene Modell-Dateien: {len(model_paths)}")
    print(f"Erfolgreich ausgewertet:  {len(models)}")
    print(f"Übersprungen:              {len(skipped_models)}")
    print(
        f"Analysierte Startzustände: "
        f"{combined_summary[0]['total_starts']}"
    )
    print(f"Maximale simulierte Schritte: {args.max_steps}")
    print(f"Maximale Intervallbreite:     {2 * n}")
    print(f"Nie-Update-Schwelle:          {max_threshold}")
    print()
    print("Schwelle  erreicht  Q1  Median  Mittel  Q3  Std.abw.")
    for row in selection_table:
        reached_percentage = 100.0 * (
            1.0 - float(row["not_reached_fraction"])
        )

        if row["median_first_passage"] == "":
            print(
                f"{int(row['threshold']):>8}  "
                f"{reached_percentage:>7.1f}%  nie"
            )
            continue

        print(
            f"{int(row['threshold']):>8}  "
            f"{reached_percentage:>7.1f}%  "
            f"{float(row['q1_first_passage']):>3.1f}  "
            f"{float(row['median_first_passage']):>6.1f}  "
            f"{float(row['mean_first_passage']):>6.2f}  "
            f"{float(row['q3_first_passage']):>3.1f}  "
            f"{float(row['stddev_first_passage']):>7.2f}"
        )

    print()
    print("Schwellen mit ähnlichem Verhalten:")
    for row in behavior_groups:
        print(
            f"Gruppe {row['group']}: "
            f"Median={row['median_first_passage']}, "
            f"Q3={row['q3_first_passage']}, "
            f"Schwellen=[{row['thresholds']}]"
        )

    print()
    print(f"Ergebnisse gespeichert in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
