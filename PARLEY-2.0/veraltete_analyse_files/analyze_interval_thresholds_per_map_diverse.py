#!/usr/bin/env python3
"""
Berechnung map-spezifischer Intervallschwellen für das symmetrische
PARLEY-Intervallmodell.

Ziel
----
Für jede Map (model_10.prism bis model_99.prism) sollen 10 Schwellen für
`max_interval_width` bestimmt werden. Diese 10 Schwellen sollen nicht
künstlich Updateabstände von 1 bis 10 Schritten imitieren. Stattdessen sollen
sie möglichst unterschiedliche Unsicherheits- bzw. Updateverhalten der
jeweiligen Map repräsentieren.

Grundidee
---------
Jede Map besitzt einen eigenen `Adaptation_MAPE_controller`. Dadurch folgt der
Roboter abhängig von der Map einer anderen Folge aus east/west/north/south.
Diese Bewegungsfolge beeinflusst wiederum, wie schnell `xradius`, `yradius`
und damit die Intervallbreite wachsen.

Deshalb wird jede Map getrennt analysiert:

1. Der MAPE-Controller wird aus der PRISM-Datei gelesen.
2. Für jede im MAPE-Controller vorkommende Position wird eine Simulation mit
   xradius = 0 und yradius = 0 gestartet.
3. Der MAPE-Route wird höchstens `max_steps` Schritte gefolgt.
4. Nach jedem Schritt wird die aktuelle Intervallbreite berechnet.
5. Für jede mögliche Schwelle wird gespeichert, nach wie vielen Schritten sie
   zum ersten Mal erreicht wird ("First Passage").
6. Über alle Startpositionen derselben Map werden daraus Median, Q1, Q3,
   Mittelwert und der Anteil nicht erreichter Zustände berechnet.
7. Schwellen mit sehr ähnlichem Verhalten werden zu Gruppen zusammengefasst.
8. Aus diesen Gruppen werden 9 möglichst unterschiedliche erreichbare
   Schwellen gewählt.
9. Zusätzlich wird die nicht erreichbare Schwelle 2*N+1 aufgenommen
   (bei N=9 also 19). Sie repräsentiert den Fall, dass die Intervallbreite
   selbst kein Update auslöst.

Dadurch entstehen pro Map 10 Schwellen, die verschiedene Stufen tolerierter
Positionsunsicherheit abdecken.

Berücksichtigte Dateien
-----------------------
Nur Dateien mit genau diesem Namensschema werden analysiert:

    model_10.prism
    model_11.prism
    ...
    model_99.prism

Dateien wie model_10_umc.prism, model_10_old.prism usw. werden ignoriert.

Beispielaufruf
--------------
    python analyze_interval_thresholds_per_map_diverse.py \
        Applications/EvoChecker-master/models \
        --max-steps 10 \
        --number-thresholds 10 \
        --not-reached-tolerance 0.02 \
        --output-dir interval_thresholds_per_map
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
    """Liest den map-spezifischen MAPE-Controller aus einer PRISM-Datei.

    Rückgabewert:
        {(xhat, yhat): richtung}

    Beispiel:
        (2, 3) -> "east"

    So kann die spätere Simulation für jede geschätzte Position dieselbe
    Bewegungsentscheidung verwenden wie der echte MAPE-Controller.
    """
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
    """Sucht ausschließlich die 90 Basismodelle model_10.prism bis model_99.prism.

    UMC-Dateien oder andere Varianten werden absichtlich nicht berücksichtigt,
    damit die Schwellen nur aus den ursprünglichen map-spezifischen
    MAPE-Controllern abgeleitet werden.
    """
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
    """Simuliert eine Bewegung des symmetrischen Knowledge-Modells.

    Es wird nur der Knowledge-Zustand (xhat, yhat, xradius, yradius)
    fortgeschrieben, nicht die stochastische reale Roboterposition.
    """
    # Zustand des symmetrischen Knowledge-Modells:
    # xhat/yhat      = aktuell geschätzte Position
    # xradius/yradius = Unsicherheitsradius um diese Schätzung
    xhat, yhat, xradius, yradius = state

    # Die Radiusregeln entsprechen dem symmetrischen PRISM-Intervallmodell.
    #
    # Bei horizontalen Bewegungen wächst die Unsicherheit in x stärker:
    #   xradius += 2
    #   yradius += 1
    #
    # Bei vertikalen Bewegungen entsprechend umgekehrt.
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
    """Berechnet die auf die Karte abgeschnittene Intervallbreite.

    Verwendet wird dieselbe Größe wie im PRISM-Modell:
        (xhigh - xlow) + (yhigh - ylow)
    """
    xhat, yhat, xradius, yradius = state

    # Die theoretische Bounding Box wird an den Kartengrenzen abgeschnitten.
    # Dadurch kann dieselbe Radiusgröße je nach Position auf der Map zu einer
    # unterschiedlichen tatsächlichen Intervallbreite führen.
    xlow = max(xhat - xradius, 0)
    xhigh = min(xhat + xradius, n)
    ylow = max(yhat - yradius, 0)
    yhigh = min(yhat + yradius, n)

    return (xhigh - xlow) + (yhigh - ylow)


def linear_quantile(values: Sequence[float], probability: float) -> float:
    """Berechnet ein Quantil mit linearer Interpolation.

    Wird für Q1 (0.25), Median (0.50) und Q3 (0.75) der
    First-Passage-Verteilungen verwendet.
    """
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
    """Berechnet die First-Passage-Zeit jeder Schwelle auf einer Map.

    Für jede MAPE-Startposition wird ohne Update simuliert, wie sich die
    Intervallbreite entlang der MAPE-Route entwickelt.
    """
    rows: list[dict[str, object]] = []

    # Jede Position, für die der MAPE-Controller eine Aktion definiert,
    # wird einmal als möglicher Startpunkt betrachtet.
    #
    # Die Unsicherheit startet dort immer bei 0, also direkt nach einem
    # perfekten Knowledge-Update.
    for start_x, start_y in sorted(model.controller):
        state = (start_x, start_y, 0, 0)

        # first_passage[threshold] speichert den ERSTEN Schritt, bei dem
        # interval_width >= threshold gilt.
        #
        # Beispiel:
        #   Breitenfolge: 4, 7, 11, 14, ...
        #   Schwelle 10 -> first_passage = 3
        first_passage: dict[int, int] = {}

        # Folge dem MAPE-Controller höchstens max_steps Bewegungen.
        # Es wird KEIN Update simuliert, weil wir wissen wollen, wie die
        # Unsicherheit ohne Knowledge-Korrektur anwächst.
        for step in range(1, max_steps + 1):
            direction = model.controller.get((state[0], state[1]))
            if direction is None:
                # Für diese Position gibt es keine weitere MAPE-Aktion,
                # z.B. weil das Ziel erreicht wurde.
                break

            state = advance_symmetric(state, direction, model.n)
            width = calculate_interval_width(state, model.n)

            # Für jede mögliche Schwelle merken wir nur die erste
            # Überschreitung. Genau dieser Zeitpunkt entspricht dem Moment,
            # an dem im Intervallmodell ein Update ausgelöst würde.
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
    """Fasst die First-Passage-Zeiten einer Map pro Schwelle zusammen.

    Das Ergebnis beschreibt, wie früh/spät und wie zuverlässig jede Schwelle
    innerhalb des Analysehorizonts erreicht wird.
    """
    # Fasse die First-Passage-Ergebnisse aller Startpositionen DIESER Map
    # für jede Schwelle statistisch zusammen.
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
            # Diese Statistiken beschreiben, wann die Schwelle typischerweise
            # erreicht wird. Besonders wichtig sind Median und Q3.
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
    """Fasst Schwellen mit nahezu gleichem Updateverhalten zusammen.

    Zwei Schwellen gehören zur selben Verhaltensgruppe, wenn:

    1. ihre mediane First-Passage-Zeit gleich ist,
    2. ihr Q3 gleich ist,
    3. sich der Anteil der Startpositionen, an denen die Schwelle innerhalb
       von `max_steps` NICHT erreicht wird, höchstens um die angegebene
       Toleranz unterscheidet.

    Beispiel:
        Schwelle 4: Median=1, Q3=1, nicht erreicht=0 %
        Schwelle 5: Median=1, Q3=1, nicht erreicht=1 %

    Bei einer Toleranz von 0.02 (= 2 Prozentpunkte) würden beide Schwellen
    derselben Gruppe zugeordnet.

    Die Gruppierung verhindert, dass mehrere praktisch identische Schwellen
    unnötig mehrere der 10 URC-Entscheidungen belegen.
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
    """Erzeugt eine kompakte Statistik-Tabelle für die spätere Kontrolle.

    Diese Funktion entscheidet NICHT über die Schwellen. Sie bereitet nur
    Median, Mittelwert, Q1, Q3, Standardabweichung und Erreichungsquoten
    übersichtlich auf.
    """
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
    """Wählt genau eine Schwelle als Repräsentanten einer Verhaltensgruppe.

    Innerhalb einer Gruppe verhalten sich mehrere Schwellen bereits sehr
    ähnlich. Als Repräsentant wird deshalb die Schwelle gewählt, deren Anteil
    nicht erreichter Startpositionen am nächsten am Durchschnitt der Gruppe
    liegt.

    Bei Gleichstand wird die größere Schwelle genommen. Dadurch wird innerhalb
    derselben Verhaltensklasse nicht unnötig die aggressivere, früher
    aktualisierende Schwelle bevorzugt.
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
    """Wählt Positionen möglichst gleichmäßig über eine geordnete Liste.

    Beispiel:
        Es existieren 13 Verhaltensgruppen, aber nur 9 reguläre Plätze.
        Dann werden nicht einfach die ersten 9 Gruppen genommen, sondern
        Gruppen aus dem gesamten Bereich von "frühes Update" bis
        "spätes Update" verteilt ausgewählt.
    """
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
    """Bestimmt die endgültigen 10 Schwellen für eine einzelne Map.

    Die Auswahl verfolgt bewusst NICHT das Ziel, Updateabstände 1..10 exakt
    nachzuahmen. Das symmetrische Intervall wächst anfangs schnell und sättigt
    später, weshalb eine solche Imitation viele kurze Updateabstände
    bevorzugen würde.

    Stattdessen werden 10 möglichst unterschiedliche Unsicherheitsstufen
    angeboten:

    1. Die höchste nicht erreichbare Schwelle wird fest aufgenommen.
       Bei N=9 ist die maximale Intervallbreite 18, daher ist 19 die
       "kein Intervall-Trigger"-Schwelle.

    2. Es bleiben 9 Plätze für regulär erreichbare Schwellen.

    3. Existieren mindestens 9 unterschiedliche Verhaltensgruppen, werden
       9 Gruppen möglichst gleichmäßig über das gesamte Spektrum ausgewählt.

    4. Existieren weniger als 9 Gruppen, bekommt zunächst jede Gruppe einen
       Repräsentanten. Übrige Plätze werden mit weiteren Schwellen aus
       vorhandenen Gruppen gefüllt.

    Ergebnis:
        Eine Liste aus 10 aufsteigenden Schwellen, die von geringer bis hoher
        tolerierter Intervallunsicherheit möglichst unterschiedliche
        Updateverhalten der jeweiligen Map repräsentieren.
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

    # Die nicht erreichbare Schwelle dient als Extremfall:
    # Die Intervallbreite selbst würde damit nie ein Update auslösen.
    # Bei N=9 ist das typischerweise Schwelle 19, weil maximal 18 erreichbar ist.
    unreachable_thresholds = []
    for group in unreachable_groups:
        unreachable_thresholds.extend(
            int(value)
            for value in str(group["thresholds"]).split(",")
            if str(value).strip()
        )
    never_update_threshold = max(unreachable_thresholds)

    # Ein Platz ist bereits für den Extremfall "nicht erreichbar" reserviert.
    # Bei 10 gewünschten Schwellen bleiben also 9 reguläre Plätze.
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
        # Wenn es weniger als 9 erreichbare Gruppen gibt:
        # zuerst jede unterschiedliche Verhaltensklasse einmal aufnehmen.
        for group in reachable_groups:
            selected.append(
                choose_group_representative(
                    group,
                    summary_by_threshold,
                )
            )

        # Sind danach noch Plätze frei, werden zusätzliche Schwellen aus
        # Gruppen verwendet, die mehrere ähnliche Kandidaten enthalten.
        # Höhere Schwellen werden zuerst betrachtet, damit der Suchraum nicht
        # unnötig stark auf sehr frühe Updates konzentriert wird.
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

    # Sicherheitsfall: Falls durch das Entfernen von Duplikaten weniger als
    # 9 reguläre Schwellen übrig sind, werden noch unbenutzte erreichbare
    # Schwellen ergänzt.
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
        # Sicherheitsfall: Falls zu viele Schwellen vorhanden sind, wieder
        # gleichmäßig über den gesamten ausgewählten Bereich reduzieren.
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

    # Jede Map wird vollständig unabhängig von den anderen Maps analysiert.
    # Dadurch können unterschiedliche MAPE-Routen zu unterschiedlichen
    # Schwellenmengen führen.
    for model in models:
        # Für eine N=9-Karte kann die abgeschnittene Breite höchstens 18 sein.
        # 2*N+1 = 19 wird zusätzlich als nicht erreichbarer Extremwert geprüft.
        max_threshold = 2 * model.n + 1

        # Schritt 1: First-Passage-Zeit jeder möglichen Schwelle für alle
        # MAPE-Startpositionen dieser Map bestimmen.
        passage_rows = simulate_model(
            model=model,
            max_steps=args.max_steps,
            max_threshold=max_threshold,
        )

        # Schritt 2: First-Passage-Zeiten pro Schwelle statistisch
        # zusammenfassen.
        map_summary = summarize(
            passage_rows,
            max_threshold=max_threshold,
        )

        # Schritt 3: Schwellen mit nahezu gleichem Updateverhalten
        # zu Verhaltensgruppen zusammenfassen.
        map_groups = build_behavior_groups(
            map_summary,
            not_reached_tolerance=args.not_reached_tolerance,
        )

        # Schritt 4: Aus den Gruppen 10 möglichst unterschiedliche
        # map-spezifische Schwellen auswählen.
        selected_thresholds = choose_thresholds_from_groups(
            summary=map_summary,
            behavior_groups=map_groups,
            number_thresholds=args.number_thresholds,
        )

        # Endergebnis dieser Map, z.B.:
        #   14: [3, 5, 9, 12, 13, 14, 16, 17, 18, 19]
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

    # Zusätzlich wird eine direkt importierbare Python-Datei erzeugt.
    # Diese kann anschließend von prism_model_generator_interval.py und
    # urc_synthesis_interval.py verwendet werden.
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
