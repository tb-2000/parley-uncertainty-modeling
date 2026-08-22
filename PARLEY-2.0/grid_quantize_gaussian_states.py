#!/usr/bin/env python3
"""
grid_quantize_gaussian_states.py

Liest bereits berechnete gaussian_states_<map>.json-Dateien ein und vergleicht
mehrere gleichmäßige Grid-Quantisierungen der Kovarianzmatrizen.

Die Quantisierung orientiert sich am Grundprinzip von Zheng et al.:
jede kontinuierliche Dimension wird auf ein gleichmäßiges Raster mit einer
vorgegebenen Quantisierungsbreite h abgebildet.

Für unsere 2x2-Kovarianzmatrizen wird der Zustandsvektor

    z = (var_x, var_y, cov_xy)

quantisiert.

WICHTIG:
Zheng et al. analysieren den Fehler gelernter HMM-Übergangsparameter, nicht
den Frobenius-Abstand zwischen Kovarianzmatrizen. Da hier keine HMM-Matrix
aus Stichproben gelernt wird, berichten wir als problemangepasste Diagnose:

    mean_F_error(h) = (1/N) sum_i ||Sigma_i - Q_h(Sigma_i)||_F

sowie RMSE, Maximum und gewichtete Varianten.

Für Sigma = [[a,c],[c,b]] gilt:

    ||Sigma-Q||_F
      = sqrt((a-aq)^2 + (b-bq)^2 + 2*(c-cq)^2)

Zusätzlich wird auf positive Semidefinitheit (PSD) geachtet. Die beiden
Varianzen werden auf nichtnegative Rasterwerte quantisiert; cov_xy wird auf
einen Rasterwert begrenzt, der |cov_xy| <= sqrt(var_x*var_y) erfüllt.

Ausgaben:
    grid_quantization_comparison.csv
    grid_quantization_comparison.json
    grid_quantization_per_map.csv

Optional:
    quantized_states_h_<h>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_H_VALUES = [0.01, 0.025, 0.05, 0.075, 0.10]


def h_label(h: float) -> str:
    """Dateisicheres Label, z.B. 0.025 -> 0p025."""
    return f"{h:g}".replace(".", "p")


def round_to_grid(value: float, h: float) -> float:
    """
    Quantisierung auf den nächsten ganzzahligen Rasterpunkt k*h.

    Python round() verwendet bankers rounding; deshalb wird hier bewusst eine
    symmetrische half-away-from-zero-Regel verwendet.
    """
    if value >= 0.0:
        k = math.floor(value / h + 0.5)
    else:
        k = math.ceil(value / h - 0.5)
    return k * h


def quantize_covariance(
    var_x: float,
    var_y: float,
    cov_xy: float,
    h: float,
) -> Tuple[float, float, float]:
    """
    Gleichmäßige Grid-Quantisierung von (var_x, var_y, cov_xy).

    Die resultierende 2x2-Matrix wird auf dem Grid PSD gehalten:
        var_x >= 0, var_y >= 0,
        |cov_xy| <= sqrt(var_x*var_y).

    cov_xy bleibt dabei ebenfalls ein Vielfaches von h.
    """
    qx = max(0.0, round_to_grid(var_x, h))
    qy = max(0.0, round_to_grid(var_y, h))
    qc = round_to_grid(cov_xy, h)

    if qx == 0.0 or qy == 0.0:
        qc = 0.0
    else:
        max_abs_cov = math.sqrt(qx * qy)

        # Größter zulässiger Betrag, der weiterhin exakt auf dem h-Grid liegt.
        max_grid_abs = math.floor(max_abs_cov / h + 1e-12) * h

        if qc > max_grid_abs:
            qc = max_grid_abs
        elif qc < -max_grid_abs:
            qc = -max_grid_abs

    # Numerisches -0.0 entfernen.
    qx = 0.0 if abs(qx) < 1e-15 else qx
    qy = 0.0 if abs(qy) < 1e-15 else qy
    qc = 0.0 if abs(qc) < 1e-15 else qc

    return (qx, qy, qc)


def frobenius_error(
    raw: Tuple[float, float, float],
    quantized: Tuple[float, float, float],
) -> float:
    """
    Frobenius-Abstand zwischen symmetrischen 2x2-Matrizen.

    Sigma = [[var_x, cov_xy],
             [cov_xy, var_y]]
    """
    dx = raw[0] - quantized[0]
    dy = raw[1] - quantized[1]
    dc = raw[2] - quantized[2]
    return math.sqrt(dx * dx + dy * dy + 2.0 * dc * dc)


def determinant(state: Tuple[float, float, float]) -> float:
    vx, vy, c = state
    return vx * vy - c * c


def is_psd(state: Tuple[float, float, float], tol: float = 1e-12) -> bool:
    vx, vy, _ = state
    return vx >= -tol and vy >= -tol and determinant(state) >= -tol


def load_gaussian_files(input_dir: Path) -> List[dict]:
    files = sorted(
        input_dir.glob("gaussian_states_*.json"),
        key=lambda p: int(p.stem.split("_")[-1])
        if p.stem.split("_")[-1].isdigit()
        else 10**9,
    )

    # Summary-Dateien explizit ausschließen.
    files = [
        p for p in files
        if p.name not in {
            "gaussian_states_summary.json",
        }
        and p.stem.split("_")[-1].isdigit()
    ]

    if not files:
        raise FileNotFoundError(
            f"Keine gaussian_states_<map>.json in {input_dir} gefunden."
        )

    result = []
    for path in files:
        with path.open("r") as f:
            data = json.load(f)

        if "gaussian_states" not in data:
            raise ValueError(
                f"{path} enthält kein Feld 'gaussian_states'."
            )

        map_id = int(data.get("map", path.stem.split("_")[-1]))
        result.append(
            {
                "map": map_id,
                "path": path,
                "data": data,
            }
        )

    return result


def extract_states(file_entry: dict) -> List[dict]:
    """
    Extrahiert die Roh-Sigma-Zustände.

    occurrences wird übernommen, sodass neben einer ungewichteten Analyse
    auch eine nach tatsächlichen Auftretenshäufigkeiten gewichtete Analyse
    möglich ist.
    """
    states = []
    for s in file_entry["data"]["gaussian_states"]:
        states.append(
            {
                "map": file_entry["map"],
                "gvar": s.get("gvar"),
                "raw": (
                    float(s["var_x"]),
                    float(s["var_y"]),
                    float(s.get("cov_xy", 0.0)),
                ),
                "occurrences": int(s.get("occurrences", 1)),
            }
        )
    return states


def analyse_h(
    all_states: Sequence[dict],
    per_map_states: Dict[int, List[dict]],
    h: float,
) -> Tuple[dict, List[dict], dict]:
    """
    Analysiert eine Rasterweite h global und pro Map.
    """
    global_unique_raw = {s["raw"] for s in all_states}
    global_unique_quantized = set()

    errors = []
    weighted_error_sum = 0.0
    weighted_squared_error_sum = 0.0
    total_weight = 0

    invalid_psd_raw = 0
    invalid_psd_quantized = 0

    # Mapping jedes beobachteten Rohzustands auf seinen quantisierten Zustand.
    raw_to_quantized = {}

    for s in all_states:
        raw = s["raw"]
        if not is_psd(raw):
            invalid_psd_raw += 1

        q = quantize_covariance(*raw, h)
        raw_to_quantized[raw] = q
        global_unique_quantized.add(q)

        if not is_psd(q):
            invalid_psd_quantized += 1

        err = frobenius_error(raw, q)
        errors.append(err)

        w = s["occurrences"]
        weighted_error_sum += w * err
        weighted_squared_error_sum += w * err * err
        total_weight += w

    n = len(errors)
    mean_error = sum(errors) / n if n else 0.0
    rmse = math.sqrt(sum(e * e for e in errors) / n) if n else 0.0
    max_error = max(errors) if errors else 0.0

    weighted_mean = (
        weighted_error_sum / total_weight if total_weight else 0.0
    )
    weighted_rmse = (
        math.sqrt(weighted_squared_error_sum / total_weight)
        if total_weight else 0.0
    )

    raw_count = len(global_unique_raw)
    q_count = len(global_unique_quantized)

    global_row = {
        "h": h,
        "input_state_records": len(all_states),
        "global_unique_raw_sigma": raw_count,
        "global_unique_quantized_sigma": q_count,
        "reduction_absolute": raw_count - q_count,
        "reduction_percent": (
            100.0 * (raw_count - q_count) / raw_count
            if raw_count else 0.0
        ),
        # Eigene problemangepasste Quantisierungsmetriken.
        "mean_frobenius_error": mean_error,
        "rmse_frobenius_error": rmse,
        "max_frobenius_error": max_error,
        "weighted_mean_frobenius_error": weighted_mean,
        "weighted_rmse_frobenius_error": weighted_rmse,
        "invalid_psd_raw_records": invalid_psd_raw,
        "invalid_psd_quantized_records": invalid_psd_quantized,
    }

    per_map_rows = []
    quantized_states_by_map = {}

    for map_id, states in sorted(per_map_states.items()):
        raw_unique = {s["raw"] for s in states}
        q_unique = set()
        map_errors = []
        weighted_sum = 0.0
        weighted_sq_sum = 0.0
        map_weight = 0

        mappings = []

        for s in states:
            raw = s["raw"]
            q = quantize_covariance(*raw, h)
            q_unique.add(q)

            err = frobenius_error(raw, q)
            map_errors.append(err)

            w = s["occurrences"]
            weighted_sum += w * err
            weighted_sq_sum += w * err * err
            map_weight += w

            mappings.append(
                {
                    "raw_gvar": s["gvar"],
                    "raw_sigma": {
                        "var_x": raw[0],
                        "var_y": raw[1],
                        "cov_xy": raw[2],
                    },
                    "quantized_sigma": {
                        "var_x": q[0],
                        "var_y": q[1],
                        "cov_xy": q[2],
                    },
                    "frobenius_error": err,
                    "occurrences": w,
                }
            )

        # Map-lokale gvar-IDs für die quantisierten Zustände.
        ordered_q = sorted(
            q_unique,
            key=lambda q: (q[0] + q[1], q[0], q[1], q[2]),
        )
        q_to_gvar = {q: i for i, q in enumerate(ordered_q)}

        for m in mappings:
            q = (
                m["quantized_sigma"]["var_x"],
                m["quantized_sigma"]["var_y"],
                m["quantized_sigma"]["cov_xy"],
            )
            m["quantized_gvar"] = q_to_gvar[q]

        raw_n = len(raw_unique)
        quant_n = len(q_unique)

        per_map_rows.append(
            {
                "h": h,
                "map": map_id,
                "raw_sigma_states": raw_n,
                "quantized_sigma_states": quant_n,
                "gvar_max": quant_n - 1,
                "reduction_absolute": raw_n - quant_n,
                "reduction_percent": (
                    100.0 * (raw_n - quant_n) / raw_n
                    if raw_n else 0.0
                ),
                "mean_frobenius_error": (
                    sum(map_errors) / len(map_errors)
                    if map_errors else 0.0
                ),
                "rmse_frobenius_error": (
                    math.sqrt(
                        sum(e * e for e in map_errors) / len(map_errors)
                    )
                    if map_errors else 0.0
                ),
                "max_frobenius_error": (
                    max(map_errors) if map_errors else 0.0
                ),
                "weighted_mean_frobenius_error": (
                    weighted_sum / map_weight if map_weight else 0.0
                ),
                "weighted_rmse_frobenius_error": (
                    math.sqrt(weighted_sq_sum / map_weight)
                    if map_weight else 0.0
                ),
            }
        )

        quantized_states_by_map[str(map_id)] = {
            "number_of_quantized_states": quant_n,
            "gvar_max": quant_n - 1,
            "states": [
                {
                    "gvar": q_to_gvar[q],
                    "var_x": q[0],
                    "var_y": q[1],
                    "cov_xy": q[2],
                }
                for q in ordered_q
            ],
            "raw_to_quantized_mapping": mappings,
        }

    return global_row, per_map_rows, quantized_states_by_map


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vergleicht gleichmäßige Grid-Quantisierungen bereits berechneter "
            "Gaussian covariance states."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("gaussian_states"),
        help="Ordner mit gaussian_states_<map>.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gaussian_quantization"),
    )
    parser.add_argument(
        "--h",
        nargs="+",
        type=float,
        default=DEFAULT_H_VALUES,
        help="Rasterweiten für Sensitivitätsanalyse; Produktionskandidat ist h=0.05.",
    )
    parser.add_argument(
        "--save-mappings",
        action="store_true",
        help=(
            "Speichert für jedes h zusätzlich alle quantisierten Zustände und "
            "Raw->Quantized-Mappings in einer JSON-Datei."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if any(h <= 0.0 for h in args.h):
        raise ValueError("Alle h-Werte müssen > 0 sein.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = load_gaussian_files(args.input_dir)

    per_map_states: Dict[int, List[dict]] = {}
    all_states: List[dict] = []

    for entry in files:
        states = extract_states(entry)
        per_map_states[entry["map"]] = states
        all_states.extend(states)

    comparison_rows = []
    all_per_map_rows = []
    detailed_json = {}

    for h in args.h:
        global_row, per_map_rows, quantized_maps = analyse_h(
            all_states=all_states,
            per_map_states=per_map_states,
            h=h,
        )

        comparison_rows.append(global_row)
        all_per_map_rows.extend(per_map_rows)

        detailed_json[str(h)] = {
            "global": global_row,
            "per_map_summary": per_map_rows,
        }

        if args.save_mappings:
            mapping_path = (
                args.output_dir
                / f"quantized_states_h_{h_label(h)}.json"
            )
            with mapping_path.open("w") as f:
                json.dump(
                    {
                        "h": h,
                        "method": "uniform_grid_nearest_with_psd_constraint",
                        "maps": quantized_maps,
                    },
                    f,
                    indent=2,
                )

        print(
            f"h={h:g}: "
            f"{global_row['global_unique_raw_sigma']} raw unique -> "
            f"{global_row['global_unique_quantized_sigma']} quantized "
            f"({global_row['reduction_percent']:.2f}% reduction), "
            f"mean Frobenius error="
            f"{global_row['mean_frobenius_error']:.6f}"
        )

    write_csv(
        args.output_dir / "grid_quantization_comparison.csv",
        comparison_rows,
    )
    write_csv(
        args.output_dir / "grid_quantization_per_map.csv",
        all_per_map_rows,
    )

    output_json = {
        "method": {
            "name": "uniform grid quantization of covariance parameters",
            "state_vector": ["var_x", "var_y", "cov_xy"],
            "quantizer": "nearest multiple of h",
            "psd_handling": (
                "cov_xy is clipped to the largest admissible h-grid value "
                "satisfying |cov_xy| <= sqrt(var_x*var_y)"
            ),
            "literature_relation": (
                "The equal-width grid follows the quantization principle used "
                "by Zheng et al. for continuous Gaussian state dimensions. "
                "The Frobenius covariance error is an adaptation for this "
                "PARLEY covariance-state representation; it is not the HMM "
                "transition-parameter error analysed by Zheng et al."
            ),
            "frobenius_metric": (
                "E(h)=(1/N) sum_i ||Sigma_i-Q_h(Sigma_i)||_F"
            ),
        },
        "number_of_maps": len(per_map_states),
        "h_values": args.h,
        "comparison": detailed_json,
    }

    with (
        args.output_dir / "grid_quantization_comparison.json"
    ).open("w") as f:
        json.dump(output_json, f, indent=2)

    print()
    print(f"Analysierte Maps: {len(per_map_states)}")
    print(f"Ausgabeordner: {args.output_dir}")
    print("  grid_quantization_comparison.csv")
    print("  grid_quantization_comparison.json")
    print("  grid_quantization_per_map.csv")
    if args.save_mappings:
        print("  quantized_states_h_<h>.json")


if __name__ == "__main__":
    main()
