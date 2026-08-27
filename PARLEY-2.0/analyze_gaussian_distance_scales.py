import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import full_gaussian_representatives_bias as gm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_map(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


def percentile(values, q):
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    alpha = pos - lo
    return xs[lo] * (1.0 - alpha) + xs[hi] * alpha


def summarize(values):
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def parse_float_list(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0.0:
            raise argparse.ArgumentTypeError("lambda values must be >= 0")
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one lambda value is required")
    return values


def lambda_tag(value):
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"lambda_{text}"


# ---------------------------------------------------------------------------
# Raw reachable states
# ---------------------------------------------------------------------------


def iter_raw_occurrences(map_data, target, p, max_steps):
    size = len(map_data)
    n = size - 1
    controller = gm._controller(map_data, target)

    for start_x in range(size):
        for start_y in range(size):
            if int(map_data[start_x][start_y]) > 9:
                continue

            xhat, yhat = start_x, start_y
            state = gm.ZERO_STATE

            for age in range(max_steps + 1):
                state = gm._state_key(state)
                terminal = age >= max_steps or (xhat, yhat) == target
                action = None if terminal else gm._direction(controller, xhat, yhat)

                yield {
                    "xhat": xhat,
                    "yhat": yhat,
                    "age": age,
                    "state": state,
                    "action": action,
                }

                if terminal or action is None:
                    break

                state = gm._predict_state(state, xhat, yhat, action, n, p)
                xhat, yhat = gm._move(xhat, yhat, action, n)


def unique_candidates(map_data, target, p, max_steps):
    unique = {}
    mse_by_age = defaultdict(list)

    for rec in iter_raw_occurrences(map_data, target, p, max_steps):
        state = rec["state"]
        mse_by_age[rec["age"]].append(gm._mse(state))
        if state not in unique:
            unique[state] = {"state": state, "weight": 1}
        else:
            unique[state]["weight"] += 1

    candidates = list(unique.values())
    candidates.sort(
        key=lambda item: (
            0 if item["state"] == gm.ZERO_STATE else 1,
            gm._mse(item["state"]),
            item["state"],
        )
    )
    return candidates, mse_by_age


# ---------------------------------------------------------------------------
# W2 + MSE distance
# ---------------------------------------------------------------------------


def compute_component_matrices(states):
    """Compute pairwise W2 and |Delta MSE| once per map."""
    count = len(states)
    w2 = [[0.0] * count for _ in range(count)]
    mse = [[0.0] * count for _ in range(count)]
    w2_values = []
    mse_values = []
    mses = [gm._mse(state) for state in states]

    for i in range(count):
        for j in range(i + 1, count):
            dw = gm._wasserstein(states[i], states[j])
            dm = abs(mses[i] - mses[j])
            w2[i][j] = w2[j][i] = dw
            mse[i][j] = mse[j][i] = dm
            if dw > 0.0:
                w2_values.append(dw)
            if dm > 0.0:
                mse_values.append(dm)

    return w2, mse, w2_values, mse_values


def robust_scale(values, mode):
    if mode == "none":
        return 1.0
    if not values:
        return 1.0
    if mode == "p95":
        value = percentile(values, 0.95)
    elif mode == "median":
        value = percentile(values, 0.50)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    return value if value > 1e-15 else 1.0


def composite_matrix(w2_matrix, mse_matrix, lambda_mse, w2_scale, mse_scale):
    count = len(w2_matrix)
    result = [[0.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            value = (
                w2_matrix[i][j] / w2_scale
                + lambda_mse * (mse_matrix[i][j] / mse_scale)
            )
            result[i][j] = result[j][i] = value
    return result


def composite_distance(a, b, lambda_mse, w2_scale, mse_scale):
    return (
        gm._wasserstein(a, b) / w2_scale
        + lambda_mse * abs(gm._mse(a) - gm._mse(b)) / mse_scale
    )


# ---------------------------------------------------------------------------
# Weighted k-medoids
# ---------------------------------------------------------------------------


def cluster_with_matrix(candidates, k, matrix, max_iter=100):
    states = [item["state"] for item in candidates]
    weights = [item["weight"] for item in candidates]
    if not states:
        raise ValueError("No reachable Gaussian states")

    zero_index = states.index(gm.ZERO_STATE)
    if len(states) <= k:
        medoids = list(range(len(states)))
    else:
        medoids = [zero_index]
        nearest = [matrix[i][zero_index] for i in range(len(states))]

        while len(medoids) < k:
            next_medoid = max(
                (i for i in range(len(states)) if i not in medoids),
                key=lambda i: (nearest[i], -i),
            )
            medoids.append(next_medoid)
            for i in range(len(states)):
                nearest[i] = min(nearest[i], matrix[i][next_medoid])

        def objective(current):
            return sum(
                weights[i] * min(matrix[i][m] for m in current)
                for i in range(len(states))
            )

        best_medoids = list(medoids)
        best_objective = objective(medoids)
        seen = set()

        for _ in range(max_iter):
            key = tuple(sorted(medoids))
            if key in seen:
                break
            seen.add(key)

            clusters = {m: [] for m in medoids}
            for i in range(len(states)):
                m = min(medoids, key=lambda candidate: (matrix[i][candidate], candidate))
                clusters[m].append(i)

            refined = []
            for medoid in medoids:
                members = clusters[medoid]
                if medoid == zero_index:
                    refined.append(zero_index)
                    continue
                best = min(
                    members,
                    key=lambda candidate: (
                        sum(weights[m] * matrix[candidate][m] for m in members),
                        candidate,
                    ),
                )
                refined.append(best)

            current_objective = objective(refined)
            if current_objective < best_objective:
                best_objective = current_objective
                best_medoids = list(refined)

            if set(refined) == set(medoids):
                break
            medoids = refined

        medoids = best_medoids

    medoids = [zero_index] + sorted(m for m in medoids if m != zero_index)
    return [states[index] for index in medoids]


def nearest(state, representatives, lambda_mse, w2_scale, mse_scale):
    state = gm._state_key(state)
    if state == gm.ZERO_STATE:
        return 0
    if len(representatives) <= 1:
        raise ValueError("Need at least one non-zero representative")
    return min(
        range(1, len(representatives)),
        key=lambda i: (
            composite_distance(
                state, representatives[i], lambda_mse, w2_scale, mse_scale
            ),
            i,
        ),
    )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model_for_lambda(
    map_id,
    map_data,
    target,
    p,
    k,
    max_steps,
    lambda_mse,
    normalization,
    output_dir,
):
    candidates, mse_by_age = unique_candidates(map_data, target, p, max_steps)
    states = [item["state"] for item in candidates]

    w2_matrix, mse_matrix, w2_values, mse_values = compute_component_matrices(states)
    w2_scale = robust_scale(w2_values, normalization)
    mse_scale = robust_scale(mse_values, normalization)

    matrix = composite_matrix(
        w2_matrix, mse_matrix, lambda_mse, w2_scale, mse_scale
    )
    representatives = cluster_with_matrix(
        candidates, min(k, len(candidates)), matrix, max_iter=100
    )

    thresholds = gm._thresholds(mse_by_age, max_steps, scale=gm.MSE_SCALE)
    controller = gm._controller(map_data, target)
    n = len(map_data) - 1

    transitions = {}
    for x in range(len(map_data)):
        for y in range(len(map_data)):
            action = gm._direction(controller, x, y)
            if action is None:
                continue
            for state_id, state in enumerate(representatives):
                successor = gm._predict_state(state, x, y, action, n, p)
                next_state = nearest(
                    successor,
                    representatives,
                    lambda_mse,
                    w2_scale,
                    mse_scale,
                )
                transitions[f"{x},{y},{state_id}"] = {
                    "action": action,
                    "next_state": next_state,
                }

    model = {
        "map_id": map_id,
        "k": k,
        "state_count": len(representatives),
        "max_steps": max_steps,
        "p": p,
        "metric": "wasserstein_plus_mse",
        "uncertainty_metric": "mse",
        "lambda_mse": lambda_mse,
        "normalization": normalization,
        "w2_scale": w2_scale,
        "mse_difference_scale": mse_scale,
        "mse_scale": gm.MSE_SCALE,
        "thresholds": thresholds,
        "uncertainties": [
            int(round(gm._mse(state) * gm.MSE_SCALE))
            for state in representatives
        ],
        "representatives": [],
        "transitions": transitions,
    }

    for state_id, state in enumerate(representatives):
        (bx, by), sigma = gm._split_state(state)
        model["representatives"].append({
            "state_id": state_id,
            "bias_x": bx,
            "bias_y": by,
            "var_x": sigma[0],
            "var_y": sigma[1],
            "cov_xy": sigma[2],
            "trace": gm._trace(sigma),
            "bias_squared": bx * bx + by * by,
            "mse": gm._mse(state),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"map_{map_id}.json"
    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    return model, representatives


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def threshold_vector(mse_value, thresholds):
    scaled = int(round(mse_value * gm.MSE_SCALE))
    return tuple(scaled >= int(t) for t in thresholds)


def uncertainty_level(mse_value, thresholds):
    """0..10 = number of thresholds reached by this MSE."""
    scaled = int(round(mse_value * gm.MSE_SCALE))
    level = 0
    for idx, threshold in enumerate(thresholds, start=1):
        if scaled >= int(threshold):
            level = idx
        else:
            break
    return level


def analyze_map(
    map_id,
    map_data,
    model,
    representatives,
    target,
):
    p = float(model["p"])
    max_steps = int(model["max_steps"])
    thresholds = [int(v) for v in model["thresholds"]]
    lambda_mse = float(model["lambda_mse"])
    w2_scale = float(model["w2_scale"])
    mse_scale = float(model["mse_difference_scale"])
    n = len(map_data) - 1

    total_states = 0
    threshold_decision_mismatches = 0
    threshold_decision_total = 0
    states_with_any_threshold_mismatch = 0

    state_w2_errors = []
    state_mse_abs_errors = []
    state_mse_rel_errors = []

    transition_total = 0
    transition_id_mismatches = 0
    successor_level_mismatches = 0
    successor_level_abs_differences = []
    transition_endpoint_w2 = []
    transition_endpoint_mse_abs = []

    successor_level_by_age = defaultdict(lambda: [0, 0])
    transition_id_by_age = defaultdict(lambda: [0, 0])

    for rec in iter_raw_occurrences(map_data, target, p, max_steps):
        state = rec["state"]
        xhat = rec["xhat"]
        yhat = rec["yhat"]
        age = rec["age"]
        action = rec["action"]

        total_states += 1
        rep_id = nearest(
            state, representatives, lambda_mse, w2_scale, mse_scale
        )
        rep = representatives[rep_id]

        raw_mse = gm._mse(state)
        rep_mse = gm._mse(rep)
        state_w2_errors.append(gm._wasserstein(state, rep))
        mse_abs = abs(raw_mse - rep_mse)
        state_mse_abs_errors.append(mse_abs)
        if raw_mse > 1e-15:
            state_mse_rel_errors.append(mse_abs / raw_mse)

        raw_vector = threshold_vector(raw_mse, thresholds)
        rep_vector = threshold_vector(rep_mse, thresholds)
        mismatches = sum(a != b for a, b in zip(raw_vector, rep_vector))
        threshold_decision_mismatches += mismatches
        threshold_decision_total += len(thresholds)
        if mismatches:
            states_with_any_threshold_mismatch += 1

        if action is None:
            continue

        transition_total += 1
        transition_id_by_age[age][1] += 1
        successor_level_by_age[age][1] += 1

        raw_successor = gm._predict_state(state, xhat, yhat, action, n, p)
        exact_successor_rep_id = nearest(
            raw_successor,
            representatives,
            lambda_mse,
            w2_scale,
            mse_scale,
        )

        key = f"{xhat},{yhat},{rep_id}"
        abstract_next_id = int(model["transitions"][key]["next_state"])
        abstract_next = representatives[abstract_next_id]

        if exact_successor_rep_id != abstract_next_id:
            transition_id_mismatches += 1
            transition_id_by_age[age][0] += 1

        exact_level = uncertainty_level(gm._mse(raw_successor), thresholds)
        abstract_level = uncertainty_level(gm._mse(abstract_next), thresholds)
        level_diff = abs(exact_level - abstract_level)
        successor_level_abs_differences.append(level_diff)
        if level_diff != 0:
            successor_level_mismatches += 1
            successor_level_by_age[age][0] += 1

        transition_endpoint_w2.append(
            gm._wasserstein(raw_successor, abstract_next)
        )
        transition_endpoint_mse_abs.append(
            abs(gm._mse(raw_successor) - gm._mse(abstract_next))
        )

    return {
        "map_id": map_id,
        "state_count": len(representatives),
        "raw_state_occurrences": total_states,
        "state_wasserstein_error": summarize(state_w2_errors),
        "state_mse_abs_error": summarize(state_mse_abs_errors),
        "state_mse_rel_error": summarize(state_mse_rel_errors),
        "states_with_any_urc_threshold_mismatch": states_with_any_threshold_mismatch,
        "state_urc_threshold_mismatch_rate": (
            states_with_any_threshold_mismatch / total_states if total_states else 0.0
        ),
        "urc_threshold_decision_mismatches": threshold_decision_mismatches,
        "urc_threshold_decision_total": threshold_decision_total,
        "urc_threshold_decision_mismatch_rate": (
            threshold_decision_mismatches / threshold_decision_total
            if threshold_decision_total else 0.0
        ),
        "transition_occurrences": transition_total,
        "transition_id_mismatches": transition_id_mismatches,
        "transition_id_mismatch_rate": (
            transition_id_mismatches / transition_total if transition_total else 0.0
        ),
        "successor_urc_level_mismatches": successor_level_mismatches,
        "successor_urc_level_mismatch_rate": (
            successor_level_mismatches / transition_total if transition_total else 0.0
        ),
        "successor_urc_level_abs_difference": summarize(
            successor_level_abs_differences
        ),
        "transition_endpoint_wasserstein_error": summarize(transition_endpoint_w2),
        "transition_endpoint_mse_abs_error": summarize(transition_endpoint_mse_abs),
        "transition_id_mismatch_by_age": {
            str(age): {
                "mismatches": mis,
                "total": total,
                "rate": mis / total if total else 0.0,
            }
            for age, (mis, total) in sorted(transition_id_by_age.items())
        },
        "successor_urc_level_mismatch_by_age": {
            str(age): {
                "mismatches": mis,
                "total": total,
                "rate": mis / total if total else 0.0,
            }
            for age, (mis, total) in sorted(successor_level_by_age.items())
        },
    }


def weighted_mean(results, metric_name, weight_name, field="mean"):
    num = 0.0
    den = 0
    for result in results:
        value = result[metric_name].get(field)
        if value is None:
            continue
        weight = result[weight_name]
        num += float(value) * weight
        den += weight
    return num / den if den else None


def aggregate(results):
    state_total = sum(r["raw_state_occurrences"] for r in results)
    transition_total = sum(r["transition_occurrences"] for r in results)
    urc_mis = sum(r["urc_threshold_decision_mismatches"] for r in results)
    urc_total = sum(r["urc_threshold_decision_total"] for r in results)
    transition_id_mis = sum(r["transition_id_mismatches"] for r in results)
    successor_level_mis = sum(r["successor_urc_level_mismatches"] for r in results)

    return {
        "maps": len(results),
        "raw_state_occurrences": state_total,
        "transition_occurrences": transition_total,
        "state_wasserstein_mean_weighted": weighted_mean(
            results, "state_wasserstein_error", "raw_state_occurrences"
        ),
        "state_mse_abs_mean_weighted": weighted_mean(
            results, "state_mse_abs_error", "raw_state_occurrences"
        ),
        "state_mse_rel_mean_weighted": weighted_mean(
            results, "state_mse_rel_error", "raw_state_occurrences"
        ),
        "urc_threshold_decision_mismatch_rate": urc_mis / urc_total if urc_total else 0.0,
        "transition_id_mismatch_rate": (
            transition_id_mis / transition_total if transition_total else 0.0
        ),
        "successor_urc_level_mismatch_rate": (
            successor_level_mis / transition_total if transition_total else 0.0
        ),
        "transition_endpoint_wasserstein_mean_weighted": weighted_mean(
            results,
            "transition_endpoint_wasserstein_error",
            "transition_occurrences",
        ),
        "transition_endpoint_mse_abs_mean_weighted": weighted_mean(
            results,
            "transition_endpoint_mse_abs_error",
            "transition_occurrences",
        ),
        "successor_urc_level_abs_difference_mean_weighted": weighted_mean(
            results,
            "successor_urc_level_abs_difference",
            "transition_occurrences",
        ),
    }


def write_csv(path, sweep_results):
    fields = [
        "lambda_mse",
        "normalization",
        "k",
        "maps",
        "state_wasserstein_mean_weighted",
        "state_mse_abs_mean_weighted",
        "state_mse_rel_mean_weighted",
        "urc_threshold_decision_mismatch_rate",
        "transition_id_mismatch_rate",
        "successor_urc_level_mismatch_rate",
        "successor_urc_level_abs_difference_mean_weighted",
        "transition_endpoint_wasserstein_mean_weighted",
        "transition_endpoint_mse_abs_mean_weighted",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in sweep_results:
            row = {
                "lambda_mse": entry["lambda_mse"],
                "normalization": entry["normalization"],
                "k": entry["k"],
            }
            row.update({name: entry["aggregate"].get(name) for name in fields if name not in row})
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare K=100 Gaussian clustering distances of the form "
            "W2/scale_W2 + lambda * |Delta MSE|/scale_MSE."
        )
    )
    parser.add_argument("--first-map", type=int, default=10)
    parser.add_argument("--last-map", type=int, default=99)
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument("--target-x", type=int, default=9)
    parser.add_argument("--target-y", type=int, default=9)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument(
        "--lambdas",
        type=parse_float_list,
        default=parse_float_list("0,0.25,0.5,1,2"),
        help="Comma-separated MSE weights. lambda=0 is pure W2.",
    )
    parser.add_argument(
        "--normalization",
        choices=("p95", "median", "none"),
        default="p95",
        help=(
            "How W2 and |Delta MSE| are put on comparable scales. "
            "p95 is recommended; none uses raw units."
        ),
    )
    parser.add_argument(
        "--models-dir",
        default="gaussian_models_distance_sweep",
    )
    parser.add_argument(
        "--output",
        default="gaussian_distance_sweep.json",
    )
    parser.add_argument(
        "--csv-output",
        default="gaussian_distance_sweep.csv",
    )
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("K must be >= 1")

    target = (args.target_x, args.target_y)
    sweep_results = []

    for lambda_mse in args.lambdas:
        print("\n" + "=" * 88)
        print(
            f"K={args.k}, lambda_MSE={lambda_mse:g}, "
            f"normalization={args.normalization}"
        )
        print("=" * 88)

        setting_dir = Path(args.models_dir) / f"k_{args.k}_{lambda_tag(lambda_mse)}_{args.normalization}"
        map_results = []

        for map_id in range(args.first_map, args.last_map + 1):
            map_path = Path(args.maps_dir) / f"map_{map_id}.csv"
            if not map_path.exists():
                print(f"skip map {map_id}: {map_path} missing")
                continue

            map_data = load_map(map_path)
            model, reps = build_model_for_lambda(
                map_id=map_id,
                map_data=map_data,
                target=target,
                p=args.p,
                k=args.k,
                max_steps=args.max_steps,
                lambda_mse=lambda_mse,
                normalization=args.normalization,
                output_dir=setting_dir,
            )
            result = analyze_map(map_id, map_data, model, reps, target)
            map_results.append(result)

            print(
                f"map {map_id}: "
                f"URC={100*result['urc_threshold_decision_mismatch_rate']:.3f}%  "
                f"Trans-ID={100*result['transition_id_mismatch_rate']:.3f}%  "
                f"Succ-URC={100*result['successor_urc_level_mismatch_rate']:.3f}%"
            )

        agg = aggregate(map_results)
        sweep_results.append({
            "lambda_mse": lambda_mse,
            "normalization": args.normalization,
            "k": args.k,
            "aggregate": agg,
            "maps": map_results,
        })

    payload = {
        "config": {
            "first_map": args.first_map,
            "last_map": args.last_map,
            "k": args.k,
            "lambdas": args.lambdas,
            "normalization": args.normalization,
            "distance": "W2 / scale_W2 + lambda_MSE * abs(delta_MSE) / scale_MSE",
            "target": list(target),
            "p": args.p,
            "max_steps": args.max_steps,
        },
        "results": sweep_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(args.csv_output, sweep_results)

    print("\n" + "=" * 100)
    print("Distance-scale comparison")
    print("=" * 100)
    print(
        f"{'lambda':>8} {'W2 mean':>11} {'MSE abs':>11} {'MSE rel':>10} "
        f"{'URC dec.':>10} {'Trans-ID':>10} {'Succ-URC':>10}"
    )
    for entry in sweep_results:
        a = entry["aggregate"]
        print(
            f"{entry['lambda_mse']:>8g} "
            f"{a['state_wasserstein_mean_weighted']:>11.6f} "
            f"{a['state_mse_abs_mean_weighted']:>11.6f} "
            f"{100*a['state_mse_rel_mean_weighted']:>9.3f}% "
            f"{100*a['urc_threshold_decision_mismatch_rate']:>9.3f}% "
            f"{100*a['transition_id_mismatch_rate']:>9.3f}% "
            f"{100*a['successor_urc_level_mismatch_rate']:>9.3f}%"
        )

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
