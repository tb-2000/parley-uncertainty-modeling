#!/usr/bin/env python3
"""
Benchmark old/exact/behavioral PARLEY belief models directly with PRISM.

Purpose
-------
Separate:
  1. model-construction cost,
  2. model-checking cost,
  3. reachable states/transitions,
  4. symbolic-vs-explicit engine behaviour,

without running EvoChecker's evolutionary search.

The script accepts EvoChecker UMC models containing declarations such as

    evolve int decision_3_5 [1..10];

and creates temporary ordinary PRISM models by replacing every evolve
decision with a fixed constant, e.g.

    const int decision_3_5 = 5;

This keeps the SAME fixed URC controller in all compared models.

Examples
--------
Basic comparison with one fixed controller:

    python benchmark_belief_models.py \
        --old model_83_umc_old.prism \
        --exact model_83_umc_exact.prism \
        --behavioral model_83_umc_behavioral.prism \
        --prism prism \
        --decision-value 5 \
        --repeats 3

On the HU server PRISM may instead be e.g.:

    --prism /path/to/prism/bin/prism

Test several PRISM engines:

    python benchmark_belief_models.py \
        --old old.prism \
        --exact exact.prism \
        --behavioral behavioral.prism \
        --engines hybrid sparse mtbdd explicit \
        --repeats 3

Use a custom property:

    --property 'P=? [ F (x=xtarget & y=ytarget & crashed=0) ]'

Or benchmark construction only:

    --build-only

Outputs
-------
benchmark_belief/
    fixed_models/
    logs/
    benchmark_runs.csv
    benchmark_summary.csv
    benchmark_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PROPERTY = 'P=? [ F (x=xtarget & y=ytarget & crashed=0) ]'

EVOLVE_RE = re.compile(
    r'^(?P<indent>\s*)evolve\s+int\s+'
    r'(?P<name>decision(?:_[A-Za-z0-9]+)+)\s*'
    r'\[\s*(?P<lo>-?\d+)\s*\.\.\s*(?P<hi>-?\d+)\s*\]\s*;\s*$'
)

# Some EvoChecker versions / generated models may use a less specific name.
GENERIC_EVOLVE_RE = re.compile(
    r'^(?P<indent>\s*)evolve\s+int\s+'
    r'(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*'
    r'\[\s*(?P<lo>-?\d+)\s*\.\.\s*(?P<hi>-?\d+)\s*\]\s*;\s*$'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PARLEY belief PRISM models without EvoChecker."
    )

    parser.add_argument("--old", type=Path, help="Old/K belief UMC model")
    parser.add_argument("--exact", type=Path, help="Exact belief UMC model")
    parser.add_argument(
        "--behavioral",
        type=Path,
        help="Behaviorally reduced belief UMC model",
    )

    parser.add_argument(
        "--prism",
        default="prism",
        help="PRISM executable (default: prism)",
    )
    parser.add_argument(
        "--decision-value",
        type=int,
        default=5,
        help="Use this fixed value for all evolve decision_* variables (default: 5)",
    )
    parser.add_argument(
        "--decision-json",
        type=Path,
        help=(
            "Optional JSON mapping decision variable names to values. "
            "Missing variables use --decision-value."
        ),
    )
    parser.add_argument(
        "--property",
        default=DEFAULT_PROPERTY,
        help=f"PRISM property passed via -pf (default: {DEFAULT_PROPERTY})",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only construct each model; do not model-check a property.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["hybrid", "explicit"],
        choices=["hybrid", "sparse", "mtbdd", "explicit"],
        help="PRISM engines to test (default: hybrid explicit)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per model/engine (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up runs per model/engine, excluded from CSV (default: 1)",
    )
    parser.add_argument(
        "--maxiters",
        type=int,
        default=50000,
        help="PRISM max iterations (default: 50000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for one PRISM invocation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_belief"),
        help="Output directory (default: benchmark_belief)",
    )
    parser.add_argument(
        "--extra-prism-arg",
        action="append",
        default=[],
        help="Additional PRISM argument; can be supplied multiple times",
    )

    return parser.parse_args()


def load_decision_map(path: Path | None):
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("--decision-json must contain a JSON object.")

    return {str(k): int(v) for k, v in data.items()}


def discover_models(args):
    models = []

    if args.old:
        models.append(("old", args.old))
    if args.exact:
        models.append(("exact", args.exact))
    if args.behavioral:
        models.append(("behavioral", args.behavioral))

    if not models:
        raise SystemExit(
            "Give at least one of --old, --exact or --behavioral."
        )

    for label, path in models:
        if not path.exists():
            raise FileNotFoundError(f"{label}: model not found: {path}")

    return models


def fix_evolve_decisions(
    input_path: Path,
    output_path: Path,
    default_value: int,
    decision_map: dict[str, int],
):
    """
    Convert EvoChecker evolve declarations to normal PRISM constants.
    """
    source = input_path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    replaced = []
    names = []

    for line in lines:
        stripped = line.rstrip("\r\n")
        ending = line[len(stripped):]

        match = EVOLVE_RE.match(stripped)
        if match is None:
            generic = GENERIC_EVOLVE_RE.match(stripped)
            if generic and generic.group("name").startswith("decision"):
                match = generic

        if match is None:
            replaced.append(line)
            continue

        name = match.group("name")
        lo = int(match.group("lo"))
        hi = int(match.group("hi"))
        value = decision_map.get(name, default_value)

        if not lo <= value <= hi:
            raise ValueError(
                f"{name}={value} is outside [{lo}..{hi}] "
                f"in {input_path}"
            )

        replaced.append(
            f'{match.group("indent")}const int {name} = {value};{ending}'
        )
        names.append(name)

    remaining_evolve = [
        line.strip()
        for line in replaced
        if re.search(r'\bevolve\b', line)
    ]
    if remaining_evolve:
        raise ValueError(
            "The fixed model still contains evolve declarations. "
            "Unsupported syntax encountered, e.g.:\n"
            + "\n".join(remaining_evolve[:5])
        )

    if not names:
        print(
            f"WARNING: no evolve decision variables found in {input_path}. "
            "The model may already be fixed."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(replaced), encoding="utf-8")

    return len(names)


def engine_switch(engine):
    return {
        "hybrid": "-hybrid",
        "sparse": "-sparse",
        "mtbdd": "-mtbdd",
        "explicit": "-explicit",
    }[engine]


def count_source_commands(path: Path):
    """
    Counts PRISM commands in source text only. This is NOT the number of
    reachable transition-matrix entries.
    """
    text = path.read_text(encoding="utf-8")

    total = text.count("->")

    knowledge = None
    match = re.search(
        r'\bmodule\s+Knowledge\b(?P<body>.*?)\bendmodule\b',
        text,
        re.DOTALL,
    )
    if match:
        knowledge = match.group("body").count("->")

    return total, knowledge


def _number(patterns, text, cast=int):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            token = match.group(1).replace(",", "")
            try:
                return cast(token)
            except ValueError:
                continue
    return None


def parse_prism_output(text):
    """
    PRISM wording differs slightly by version/engine.
    Keep the parser deliberately permissive.
    """
    stats = {}

    stats["states"] = _number(
        [
            r'^\s*States\s*:\s*([\d,]+)',
            r'^\s*Number of states\s*:\s*([\d,]+)',
        ],
        text,
    )

    stats["transitions"] = _number(
        [
            r'^\s*Transitions\s*:\s*([\d,]+)',
            r'^\s*Number of transitions\s*:\s*([\d,]+)',
        ],
        text,
    )

    stats["choices"] = _number(
        [
            r'^\s*Choices\s*:\s*([\d,]+)',
            r'^\s*Number of choices\s*:\s*([\d,]+)',
        ],
        text,
    )

    stats["model_construction_s"] = _number(
        [
            r'Time for model construction\s*:\s*'
            r'([0-9]+(?:\.[0-9]+)?)',
            r'Model construction.*?([0-9]+(?:\.[0-9]+)?)\s*seconds',
        ],
        text,
        float,
    )

    stats["model_checking_s"] = _number(
        [
            r'Time for model checking\s*:\s*'
            r'([0-9]+(?:\.[0-9]+)?)',
            r'Model checking.*?([0-9]+(?:\.[0-9]+)?)\s*seconds',
        ],
        text,
        float,
    )

    stats["total_time_s_reported"] = _number(
        [
            r'Total time\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        ],
        text,
        float,
    )

    # Common symbolic statistics. Names vary across PRISM versions.
    stats["mtbdd_nodes"] = _number(
        [
            r'MTBDD.*?nodes?\s*[:=]\s*([\d,]+)',
            r'Nodes.*?MTBDD.*?[:=]\s*([\d,]+)',
        ],
        text,
    )

    stats["bdd_nodes"] = _number(
        [
            r'BDD.*?nodes?\s*[:=]\s*([\d,]+)',
        ],
        text,
    )

    result_match = re.search(
        r'^\s*Result(?:\s*\([^)]*\))?\s*:\s*(.+?)\s*$',
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    stats["result"] = (
        result_match.group(1).strip()
        if result_match
        else None
    )

    return stats


def run_prism(
    prism_executable,
    model_path,
    engine,
    prop,
    build_only,
    maxiters,
    timeout,
    extra_args,
    log_path,
):
    command = [
        prism_executable,
        str(model_path),
        engine_switch(engine),
        "-maxiters",
        str(maxiters),
        "-verbose",
    ]

    # Symbolic diagnostics are useful for the hypothesis that kstate harms
    # BDD/MTBDD compactness. They are harmless if a given engine ignores
    # information that does not apply to it.
    if engine in {"hybrid", "sparse", "mtbdd"}:
        command.extend(["-extraddinfo", "-extrareachinfo"])

    if not build_only:
        command.extend(["-pf", prop])

    command.extend(extra_args)

    start = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        timed_out = False
        return_code = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = -1
        output = (
            (exc.stdout or "")
            + "\nTIMEOUT\n"
        )

    wall_s = time.perf_counter() - start

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8", errors="replace")

    stats = parse_prism_output(output)
    stats.update({
        "wall_time_s": wall_s,
        "return_code": return_code,
        "timed_out": timed_out,
        "command": " ".join(command),
    })

    return stats


def median_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def fmt(value, decimals=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def write_summary(runs, output_dir):
    groups = {}

    for row in runs:
        key = (row["model"], row["engine"])
        groups.setdefault(key, []).append(row)

    summary_rows = []

    for (model, engine), rows in groups.items():
        successful = [
            row
            for row in rows
            if row["return_code"] == 0 and not row["timed_out"]
        ]

        row0 = rows[0]

        summary_rows.append({
            "model": model,
            "engine": engine,
            "successful_runs": len(successful),
            "runs": len(rows),
            "source_commands": row0["source_commands"],
            "source_knowledge_commands":
                row0["source_knowledge_commands"],
            "states": median_or_none(
                [r["states"] for r in successful]
            ),
            "transitions": median_or_none(
                [r["transitions"] for r in successful]
            ),
            "median_wall_s": median_or_none(
                [r["wall_time_s"] for r in successful]
            ),
            "mean_wall_s": mean_or_none(
                [r["wall_time_s"] for r in successful]
            ),
            "median_construction_s": median_or_none(
                [r["model_construction_s"] for r in successful]
            ),
            "median_checking_s": median_or_none(
                [r["model_checking_s"] for r in successful]
            ),
            "median_mtbdd_nodes": median_or_none(
                [r["mtbdd_nodes"] for r in successful]
            ),
            "median_bdd_nodes": median_or_none(
                [r["bdd_nodes"] for r in successful]
            ),
            "result": next(
                (
                    r["result"]
                    for r in successful
                    if r["result"] is not None
                ),
                None,
            ),
        })

    summary_csv = output_dir / "benchmark_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = []
    lines.append("PARLEY belief benchmark summary")
    lines.append("=" * 100)
    lines.append(
        f"{'model':<12} {'engine':<9} {'src cmd':>8} {'K cmd':>8} "
        f"{'states':>12} {'trans':>12} {'wall s':>10} "
        f"{'build s':>10} {'check s':>10}"
    )
    lines.append("-" * 100)

    for row in summary_rows:
        lines.append(
            f"{row['model']:<12} "
            f"{row['engine']:<9} "
            f"{fmt(row['source_commands']):>8} "
            f"{fmt(row['source_knowledge_commands']):>8} "
            f"{fmt(row['states'], 0):>12} "
            f"{fmt(row['transitions'], 0):>12} "
            f"{fmt(row['median_wall_s']):>10} "
            f"{fmt(row['median_construction_s']):>10} "
            f"{fmt(row['median_checking_s']):>10}"
        )

    lines.append("")
    lines.append(
        "Interpretation hint: if exact/behavioral are slow mainly for "
        "hybrid/MTBDD but not explicit, symbolic encoding/BDD structure "
        "is a strong suspect."
    )
    lines.append(
        "If all engines are similarly slower, inspect reachable state count, "
        "transition count and numerical model-checking iterations instead."
    )

    summary_txt = output_dir / "benchmark_summary.txt"
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary_rows, summary_csv, summary_txt


def main():
    args = parse_args()
    models = discover_models(args)
    decision_map = load_decision_map(args.decision_json)

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    output_dir = args.output_dir
    fixed_dir = output_dir / "fixed_models"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Fail early if PRISM is not callable.
    prism_found = (
        shutil.which(args.prism)
        if os.path.sep not in args.prism
        else args.prism
    )
    if not prism_found or (
        os.path.sep in args.prism and not Path(args.prism).exists()
    ):
        raise SystemExit(
            f"PRISM executable not found: {args.prism}\n"
            "Pass it explicitly with --prism /path/to/prism."
        )

    prepared = []

    for label, original_path in models:
        fixed_path = fixed_dir / f"{label}_fixed.prism"

        decision_count = fix_evolve_decisions(
            original_path,
            fixed_path,
            args.decision_value,
            decision_map,
        )

        source_commands, knowledge_commands = count_source_commands(
            fixed_path
        )

        prepared.append({
            "label": label,
            "original": original_path,
            "fixed": fixed_path,
            "decision_count": decision_count,
            "source_commands": source_commands,
            "source_knowledge_commands": knowledge_commands,
        })

        print(
            f"{label:>10}: fixed {decision_count} decisions; "
            f"source commands={source_commands}, "
            f"Knowledge commands={knowledge_commands}"
        )

    all_runs = []

    for item in prepared:
        for engine in args.engines:
            total_invocations = args.warmup + args.repeats

            print(
                f"\n[{item['label']} / {engine}] "
                f"{args.warmup} warm-up + {args.repeats} measured runs"
            )

            for index in range(total_invocations):
                is_warmup = index < args.warmup
                measured_index = index - args.warmup + 1

                if is_warmup:
                    run_name = f"warmup_{index + 1}"
                else:
                    run_name = f"run_{measured_index}"

                log_path = (
                    logs_dir
                    / f"{item['label']}_{engine}_{run_name}.log"
                )

                stats = run_prism(
                    prism_executable=args.prism,
                    model_path=item["fixed"],
                    engine=engine,
                    prop=args.property,
                    build_only=args.build_only,
                    maxiters=args.maxiters,
                    timeout=args.timeout,
                    extra_args=args.extra_prism_arg,
                    log_path=log_path,
                )

                status = (
                    "TIMEOUT"
                    if stats["timed_out"]
                    else f"rc={stats['return_code']}"
                )

                print(
                    f"  {run_name:<9} {status:<10} "
                    f"wall={stats['wall_time_s']:.3f}s "
                    f"states={fmt(stats['states'], 0)} "
                    f"trans={fmt(stats['transitions'], 0)} "
                    f"build={fmt(stats['model_construction_s'])} "
                    f"check={fmt(stats['model_checking_s'])}"
                )

                if is_warmup:
                    continue

                all_runs.append({
                    "model": item["label"],
                    "engine": engine,
                    "run": measured_index,
                    "fixed_decision_value": args.decision_value,
                    "source_commands": item["source_commands"],
                    "source_knowledge_commands":
                        item["source_knowledge_commands"],
                    "states": stats["states"],
                    "transitions": stats["transitions"],
                    "choices": stats["choices"],
                    "wall_time_s": stats["wall_time_s"],
                    "model_construction_s":
                        stats["model_construction_s"],
                    "model_checking_s":
                        stats["model_checking_s"],
                    "total_time_s_reported":
                        stats["total_time_s_reported"],
                    "mtbdd_nodes": stats["mtbdd_nodes"],
                    "bdd_nodes": stats["bdd_nodes"],
                    "result": stats["result"],
                    "return_code": stats["return_code"],
                    "timed_out": stats["timed_out"],
                    "log": str(log_path),
                })

    runs_csv = output_dir / "benchmark_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(all_runs[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_runs)

    summary_rows, summary_csv, summary_txt = write_summary(
        all_runs,
        output_dir,
    )

    print("\n" + summary_txt.read_text(encoding="utf-8"))
    print(f"Detailed runs : {runs_csv}")
    print(f"Summary CSV   : {summary_csv}")
    print(f"Summary text  : {summary_txt}")
    print(f"PRISM logs    : {logs_dir}")


if __name__ == "__main__":
    main()
