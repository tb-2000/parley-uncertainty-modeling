#!/usr/bin/env python3
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
    m = re.search(rf"\bconst\s+int\s+{re.escape(name)}\s*=\s*(\d+)\s*;", text)
    if not m:
        raise ValueError(f"Konstante {name!r} wurde nicht gefunden.")
    return int(m.group(1))


def parse_controller(text: str) -> dict[tuple[int, int], str]:
    m = re.search(r"module\s+Adaptation_MAPE_controller\s*(.*?)\s*endmodule", text, re.S)
    if not m:
        raise ValueError("Adaptation_MAPE_controller wurde nicht gefunden.")
    pat = re.compile(
        r"\[(west|east|south|north)\]\s*"
        r"\(xhat\s*=\s*(\d+)\)\s*&\s*"
        r"\(yhat\s*=\s*(\d+)\)\s*->\s*true\s*;"
    )
    controller = {(int(x), int(y)): d for d, x, y in pat.findall(m.group(1))}
    if not controller:
        raise ValueError("Im Adaptation_MAPE_controller wurden keine Richtungen erkannt.")
    return controller


def parse_model(path: Path) -> ModelData:
    fm = MODEL_FILENAME_PATTERN.fullmatch(path.name)
    if not fm:
        raise ValueError("Dateiname entspricht nicht model_10.prism bis model_99.prism.")
    text = path.read_text(encoding="utf-8")
    return ModelData(path, int(fm.group(1)), parse_int_constant(text, "N"), parse_controller(text))


def discover_models(models_dir: Path) -> list[Path]:
    paths = []
    for p in models_dir.iterdir():
        m = MODEL_FILENAME_PATTERN.fullmatch(p.name)
        if p.is_file() and m and 10 <= int(m.group(1)) <= 99:
            paths.append(p)
    return sorted(paths, key=lambda p: int(MODEL_FILENAME_PATTERN.fullmatch(p.name).group(1)))


def advance_symmetric(state: tuple[int, int, int, int], direction: str, n: int) -> tuple[int, int, int, int]:
    xhat, yhat, xradius, yradius = state
    if direction == "east":
        return min(xhat + 1, n), yhat, min(xradius + 2, n), min(yradius + 1, n)
    if direction == "west":
        return max(xhat - 1, 0), yhat, min(xradius + 2, n), min(yradius + 1, n)
    if direction == "north":
        return xhat, min(yhat + 1, n), min(xradius + 1, n), min(yradius + 2, n)
    if direction == "south":
        return xhat, max(yhat - 1, 0), min(xradius + 1, n), min(yradius + 2, n)
    raise ValueError(direction)


def interval_width(state: tuple[int, int, int, int], n: int) -> int:
    xhat, yhat, xradius, yradius = state
    xlow = max(xhat - xradius, 0)
    xhigh = min(xhat + xradius, n)
    ylow = max(yhat - yradius, 0)
    yhigh = min(yhat + yradius, n)
    return (xhigh - xlow) + (yhigh - ylow)


def quantile(values: Sequence[float], p: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("Leere Folge")
    if len(vals) == 1:
        return vals[0]
    pos = p * (len(vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    f = pos - lo
    return vals[lo] * (1 - f) + vals[hi] * f


def first_passages(model: ModelData, max_steps: int, max_threshold: int) -> list[dict[str, object]]:
    rows = []
    for sx, sy in sorted(model.controller):
        state = (sx, sy, 0, 0)
        first: dict[int, int] = {}
        for step in range(1, max_steps + 1):
            direction = model.controller.get((state[0], state[1]))
            if direction is None:
                break
            state = advance_symmetric(state, direction, model.n)
            width = interval_width(state, model.n)
            for threshold in range(1, max_threshold + 1):
                if threshold not in first and width >= threshold:
                    first[threshold] = step
        for threshold in range(1, max_threshold + 1):
            s = first.get(threshold)
            rows.append({
                "model": model.number,
                "start_x": sx,
                "start_y": sy,
                "threshold": threshold,
                "reached": int(s is not None),
                "first_passage_step": s if s is not None else "",
            })
    return rows


def candidate_scores(rows: Sequence[Mapping[str, object]], max_threshold: int, max_steps: int) -> list[dict[str, object]]:
    starts = sorted({(int(r["start_x"]), int(r["start_y"])) for r in rows})
    total = len(starts)
    by_threshold: dict[int, dict[tuple[int, int], int | None]] = defaultdict(dict)
    for r in rows:
        key = (int(r["start_x"]), int(r["start_y"]))
        by_threshold[int(r["threshold"])][key] = int(r["first_passage_step"]) if int(r["reached"]) else None

    out = []
    for t in range(1, max_threshold + 1):
        vals = by_threshold[t]
        reached = [v for v in vals.values() if v is not None]
        reached_fraction = len(reached) / total if total else 0.0
        stats = {
            "q1": quantile(reached, .25) if reached else "",
            "median": quantile(reached, .5) if reached else "",
            "mean": statistics.fmean(reached) if reached else "",
            "q3": quantile(reached, .75) if reached else "",
        }
        for target in range(1, 11):
            # Not reached means the threshold failed to emulate an update within the horizon.
            # Treat it as max_steps+1, which strongly penalizes it for target steps 1..10.
            errors = []
            for start in starts:
                v = vals.get(start)
                effective = v if v is not None else max_steps + 1
                errors.append((effective - target) ** 2)
            rmse = math.sqrt(statistics.fmean(errors)) if errors else math.inf
            out.append({
                "threshold": t,
                "target_step": target,
                "rmse": rmse,
                "total_starts": total,
                "reached_fraction": reached_fraction,
                "not_reached_fraction": 1.0 - reached_fraction,
                "first_step_q1": stats["q1"],
                "first_step_median": stats["median"],
                "first_step_mean": stats["mean"],
                "first_step_q3": stats["q3"],
            })
    return out


def choose_thresholds(score_rows: Sequence[Mapping[str, object]], max_reachable_threshold: int) -> list[dict[str, object]]:
    """Choose 10 distinct increasing thresholds for target steps 1..10 by DP."""
    targets = list(range(1, 11))
    candidates = list(range(1, max_reachable_threshold + 1))
    score = {(int(r["target_step"]), int(r["threshold"])): float(r["rmse"]) for r in score_rows}
    row_lookup = {(int(r["target_step"]), int(r["threshold"])): r for r in score_rows}

    if len(candidates) < len(targets):
        raise ValueError("Zu wenige erreichbare Schwellen für 10 verschiedene Entscheidungen.")

    dp: list[dict[int, tuple[float, int | None]]] = []
    first = {}
    max_first_index = len(candidates) - len(targets)
    for idx in range(0, max_first_index + 1):
        t = candidates[idx]
        first[t] = (score[(1, t)], None)
    dp.append(first)

    for i in range(1, len(targets)):
        target = targets[i]
        layer = {}
        min_idx = i
        max_idx = len(candidates) - (len(targets) - i)
        for idx in range(min_idx, max_idx + 1):
            t = candidates[idx]
            prevs = [p for p in dp[i-1] if p < t]
            if not prevs:
                continue
            best_prev = min(prevs, key=lambda p: dp[i-1][p][0] + score[(target, t)])
            layer[t] = (dp[i-1][best_prev][0] + score[(target, t)], best_prev)
        dp.append(layer)

    last = min(dp[-1], key=lambda t: dp[-1][t][0])
    selected = [last]
    for i in range(len(targets)-1, 0, -1):
        prev = dp[i][selected[-1]][1]
        if prev is None:
            raise RuntimeError("DP-Rekonstruktion fehlgeschlagen")
        selected.append(prev)
    selected.reverse()

    result = []
    for decision, (target, threshold) in enumerate(zip(targets, selected), 1):
        r = row_lookup[(target, threshold)]
        result.append({
            "decision": decision,
            "target_step": target,
            "threshold": threshold,
            "rmse": r["rmse"],
            "reached_fraction": r["reached_fraction"],
            "not_reached_fraction": r["not_reached_fraction"],
            "first_step_q1": r["first_step_q1"],
            "first_step_median": r["first_step_median"],
            "first_step_mean": r["first_step_mean"],
            "first_step_q3": r["first_step_q3"],
        })
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Map-spezifische Intervallschwellen für Ziel-Updateabstände 1..10")
    ap.add_argument("models_dir", type=Path)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--output-dir", type=Path, default=Path("interval_thresholds_per_map"))
    args = ap.parse_args()
    if not args.models_dir.is_dir():
        ap.error(f"Kein Verzeichnis: {args.models_dir}")
    if args.max_steps < 10:
        ap.error("--max-steps muss mindestens 10 sein")

    paths = discover_models(args.models_dir)
    if not paths:
        ap.error("Keine model_10.prism bis model_99.prism gefunden")

    models, skipped = [], []
    for p in paths:
        try:
            models.append(parse_model(p))
        except Exception as e:
            skipped.append({"model": p.name, "reason": str(e)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_selected, all_scores = [], []
    threshold_map: dict[int, list[int]] = {}

    for model in models:
        max_reachable = 2 * model.n
        rows = first_passages(model, args.max_steps, max_reachable)
        scores = candidate_scores(rows, max_reachable, args.max_steps)
        selected = choose_thresholds(scores, max_reachable)
        threshold_map[model.number] = [int(r["threshold"]) for r in selected]
        for r in selected:
            all_selected.append({"model": model.number, **r})
        for r in scores:
            all_scores.append({"model": model.number, **r})
        print(f"Map {model.number}: {threshold_map[model.number]}")

    write_csv(args.output_dir / "optimal_thresholds_per_map.csv", all_selected)
    write_csv(args.output_dir / "candidate_scores_per_map.csv", all_scores)
    write_csv(args.output_dir / "skipped_models.csv", skipped)

    with (args.output_dir / "thresholds_per_map.py").open("w", encoding="utf-8") as f:
        f.write("THRESHOLDS_PER_MAP = {\n")
        for m in sorted(threshold_map):
            f.write(f"    {m}: {threshold_map[m]},\n")
        f.write("}\n")

    print(f"\nAusgewertet: {len(models)} Modelle; übersprungen: {len(skipped)}")
    print(f"Ergebnisse: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
