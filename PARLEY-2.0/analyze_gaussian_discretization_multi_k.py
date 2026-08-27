import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import full_gaussian_representatives_bias as gm


def load_map(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


def load_model(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def representatives_from_model(model):
    reps = []
    for r in model["representatives"]:
        reps.append((
            float(r["bias_x"]),
            float(r["bias_y"]),
            float(r["var_x"]),
            float(r["var_y"]),
            float(r["cov_xy"]),
        ))
    return reps


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
    return xs[lo] * (1 - alpha) + xs[hi] * alpha


def summarize(values):
    if not values:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def state_threshold_vector(mse_value, thresholds, scale):
    scaled = int(round(mse_value * scale))
    return tuple(scaled >= int(t) for t in thresholds)


def iter_raw_occurrences(map_data, target, p, max_steps):
    """Yield all raw reachable Gaussian states with their position and action context."""
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
                    "start_x": start_x,
                    "start_y": start_y,
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


def analyze_map(map_id, map_path, model_path, target):
    map_data = load_map(map_path)
    model = load_model(model_path)
    reps = representatives_from_model(model)

    if model.get("metric") != "wasserstein":
        raise ValueError(f"map {map_id}: expected metric='wasserstein', got {model.get('metric')!r}")
    if model.get("uncertainty_metric") != "mse":
        raise ValueError(f"map {map_id}: expected uncertainty_metric='mse', got {model.get('uncertainty_metric')!r}")

    p = float(model["p"])
    max_steps = int(model["max_steps"])
    thresholds = [int(v) for v in model["thresholds"]]
    scale = int(model.get("mse_scale", gm.MSE_SCALE))
    n = len(map_data) - 1

    total_states = 0
    non_medoid_states = 0
    threshold_mismatch_states = 0
    threshold_decision_mismatches = 0
    total_threshold_decisions = 0

    wasserstein_errors = []
    mse_abs_errors = []
    mse_rel_errors = []

    transition_total = 0
    transition_mismatches = 0
    transition_endpoint_w2 = []
    transition_mse_abs_errors = []
    transition_by_age = defaultdict(lambda: [0, 0])
    state_threshold_by_age = defaultdict(lambda: [0, 0])

    unique_raw_states = set()
    unique_non_medoid_states = set()
    unique_transition_pairs = set()
    unique_transition_mismatch_pairs = set()

    for rec in iter_raw_occurrences(map_data, target, p, max_steps):
        state = rec["state"]
        xhat = rec["xhat"]
        yhat = rec["yhat"]
        age = rec["age"]
        action = rec["action"]

        total_states += 1
        unique_raw_states.add(state)

        rep_id = gm._nearest(state, reps, "wasserstein")
        rep = reps[rep_id]

        exact = gm._state_key(state) == gm._state_key(rep)
        if not exact:
            non_medoid_states += 1
            unique_non_medoid_states.add(state)

        w2 = gm._distance(state, rep, "wasserstein")
        wasserstein_errors.append(w2)

        raw_mse = gm._mse(state)
        rep_mse = gm._mse(rep)
        mse_abs = abs(raw_mse - rep_mse)
        mse_abs_errors.append(mse_abs)
        if raw_mse > 1e-15:
            mse_rel_errors.append(mse_abs / raw_mse)

        raw_decisions = state_threshold_vector(raw_mse, thresholds, scale)
        rep_decisions = state_threshold_vector(rep_mse, thresholds, scale)
        mismatches_here = sum(a != b for a, b in zip(raw_decisions, rep_decisions))
        threshold_decision_mismatches += mismatches_here
        total_threshold_decisions += len(thresholds)
        state_threshold_by_age[age][1] += 1
        if mismatches_here:
            threshold_mismatch_states += 1
            state_threshold_by_age[age][0] += 1

        if action is None:
            continue

        transition_total += 1
        transition_by_age[age][1] += 1

        raw_successor = gm._predict_state(state, xhat, yhat, action, n, p)
        raw_successor_rep_id = gm._nearest(raw_successor, reps, "wasserstein")

        key = f"{xhat},{yhat},{rep_id}"
        if key not in model["transitions"]:
            raise KeyError(f"map {map_id}: missing abstract transition {key}")
        abstract_next_id = int(model["transitions"][key]["next_state"])

        unique_transition_pairs.add((xhat, yhat, state, action))

        if raw_successor_rep_id != abstract_next_id:
            transition_mismatches += 1
            transition_by_age[age][0] += 1
            unique_transition_mismatch_pairs.add((xhat, yhat, state, action))

        abstract_next = reps[abstract_next_id]
        transition_endpoint_w2.append(
            gm._distance(raw_successor, abstract_next, "wasserstein")
        )
        transition_mse_abs_errors.append(
            abs(gm._mse(raw_successor) - gm._mse(abstract_next))
        )

    result = {
        "map_id": map_id,
        "state_count": int(model["state_count"]),
        "raw_state_occurrences": total_states,
        "unique_raw_states": len(unique_raw_states),
        "non_medoid_state_occurrences": non_medoid_states,
        "non_medoid_state_occurrence_rate": non_medoid_states / total_states if total_states else 0.0,
        "unique_non_medoid_states": len(unique_non_medoid_states),
        "state_wasserstein_error": summarize(wasserstein_errors),
        "state_mse_abs_error": summarize(mse_abs_errors),
        "state_mse_rel_error": summarize(mse_rel_errors),
        "states_with_any_urc_threshold_mismatch": threshold_mismatch_states,
        "state_urc_threshold_mismatch_rate": threshold_mismatch_states / total_states if total_states else 0.0,
        "urc_threshold_decision_mismatches": threshold_decision_mismatches,
        "urc_threshold_decision_total": total_threshold_decisions,
        "urc_threshold_decision_mismatch_rate": threshold_decision_mismatches / total_threshold_decisions if total_threshold_decisions else 0.0,
        "transition_occurrences": transition_total,
        "transition_mismatches": transition_mismatches,
        "transition_mismatch_rate": transition_mismatches / transition_total if transition_total else 0.0,
        "unique_transition_cases": len(unique_transition_pairs),
        "unique_transition_mismatch_cases": len(unique_transition_mismatch_pairs),
        "transition_endpoint_wasserstein_error": summarize(transition_endpoint_w2),
        "transition_endpoint_mse_abs_error": summarize(transition_mse_abs_errors),
        "transition_mismatch_by_age": {
            str(age): {
                "mismatches": mis,
                "total": total,
                "rate": mis / total if total else 0.0,
            }
            for age, (mis, total) in sorted(transition_by_age.items())
        },
        "urc_threshold_mismatch_by_age": {
            str(age): {
                "states_with_any_mismatch": mis,
                "total": total,
                "rate": mis / total if total else 0.0,
            }
            for age, (mis, total) in sorted(state_threshold_by_age.items())
        },
    }
    return result


def weighted_map_mean(results, metric_name, field="mean"):
    """Occurrence-weighted mean of a per-map error mean."""
    total_weight = 0
    weighted_sum = 0.0
    for r in results:
        value = r[metric_name].get(field)
        if value is None:
            continue
        weight = r["raw_state_occurrences"] if metric_name.startswith("state_") else r["transition_occurrences"]
        weighted_sum += float(value) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else None


def aggregate(results):
    state_occ = sum(r["raw_state_occurrences"] for r in results)
    non_med = sum(r["non_medoid_state_occurrences"] for r in results)
    threshold_states = sum(r["states_with_any_urc_threshold_mismatch"] for r in results)
    threshold_dec_mis = sum(r["urc_threshold_decision_mismatches"] for r in results)
    threshold_dec_total = sum(r["urc_threshold_decision_total"] for r in results)
    trans_total = sum(r["transition_occurrences"] for r in results)
    trans_mis = sum(r["transition_mismatches"] for r in results)

    return {
        "maps": len(results),
        "raw_state_occurrences": state_occ,
        "non_medoid_state_occurrences": non_med,
        "non_medoid_state_occurrence_rate": non_med / state_occ if state_occ else 0.0,
        "states_with_any_urc_threshold_mismatch": threshold_states,
        "state_urc_threshold_mismatch_rate": threshold_states / state_occ if state_occ else 0.0,
        "urc_threshold_decision_mismatches": threshold_dec_mis,
        "urc_threshold_decision_total": threshold_dec_total,
        "urc_threshold_decision_mismatch_rate": threshold_dec_mis / threshold_dec_total if threshold_dec_total else 0.0,
        "transition_occurrences": trans_total,
        "transition_mismatches": trans_mis,
        "transition_mismatch_rate": trans_mis / trans_total if trans_total else 0.0,
        "state_wasserstein_mean_weighted": weighted_map_mean(results, "state_wasserstein_error"),
        "state_mse_abs_mean_weighted": weighted_map_mean(results, "state_mse_abs_error"),
        "state_mse_rel_mean_weighted": weighted_map_mean(results, "state_mse_rel_error"),
        "transition_endpoint_wasserstein_mean_weighted": weighted_map_mean(
            results, "transition_endpoint_wasserstein_error"
        ),
        "transition_endpoint_mse_abs_mean_weighted": weighted_map_mean(
            results, "transition_endpoint_mse_abs_error"
        ),
    }


def parse_ks(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 1:
            raise argparse.ArgumentTypeError("K values must be >= 1")
        if k not in values:
            values.append(k)
    if not values:
        raise argparse.ArgumentTypeError("At least one K value is required")
    return values


def write_comparison_csv(path, sweep_results):
    fields = [
        "k",
        "maps",
        "raw_state_occurrences",
        "non_medoid_state_occurrence_rate",
        "state_wasserstein_mean_weighted",
        "state_mse_abs_mean_weighted",
        "state_mse_rel_mean_weighted",
        "state_urc_threshold_mismatch_rate",
        "urc_threshold_decision_mismatch_rate",
        "transition_mismatch_rate",
        "transition_endpoint_wasserstein_mean_weighted",
        "transition_endpoint_mse_abs_mean_weighted",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in sweep_results:
            row = {"k": entry["k"]}
            row.update({name: entry["aggregate"].get(name) for name in fields if name != "k"})
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Gaussian PARLEY discretisation for several K values."
    )
    parser.add_argument("--first-map", type=int, default=10)
    parser.add_argument("--last-map", type=int, default=99)
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument(
        "--models-dir",
        default="gaussian_models_k_sweep",
        help="Root directory for generated models. Each K is stored in k_<K>/.",
    )
    parser.add_argument(
        "--ks",
        type=parse_ks,
        default=parse_ks("50,100,150,200,300"),
        help="Comma-separated K values, e.g. 50,100,150,200,300.",
    )
    parser.add_argument("--target-x", type=int, default=9)
    parser.add_argument("--target-y", type=int, default=9)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--output",
        default="gaussian_discretization_k_sweep.json",
        help="JSON containing aggregate and per-map results for all K values.",
    )
    parser.add_argument(
        "--csv-output",
        default="gaussian_discretization_k_sweep.csv",
        help="Compact aggregate comparison CSV.",
    )
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="Reuse existing k_<K>/map_<id>.json files instead of rebuilding them.",
    )
    args = parser.parse_args()

    target = (args.target_x, args.target_y)
    sweep_results = []

    for k in args.ks:
        print(f"\n{'=' * 72}\nK = {k}\n{'=' * 72}")
        k_models_dir = Path(args.models_dir) / f"k_{k}"
        k_models_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for map_id in range(args.first_map, args.last_map + 1):
            map_path = Path(args.maps_dir) / f"map_{map_id}.csv"
            if not map_path.exists():
                print(f"skip map {map_id}: {map_path} missing")
                continue

            model_path = k_models_dir / f"map_{map_id}.json"

            if not args.reuse_models or not model_path.exists():
                map_data = load_map(map_path)
                gm.build_gaussian_model(
                    map_id=map_id,
                    map_data=map_data,
                    target=target,
                    p=args.p,
                    k=k,
                    max_steps=args.max_steps,
                    metric="wasserstein",
                    cache_dir=k_models_dir,
                )

            result = analyze_map(map_id, map_path, model_path, target)
            results.append(result)

            print(
                f"K={k:>3} map {map_id}: states={result['state_count']}, "
                f"URC-decision mismatch={100*result['urc_threshold_decision_mismatch_rate']:.3f}%, "
                f"transition mismatch={result['transition_mismatches']}/{result['transition_occurrences']} "
                f"({100*result['transition_mismatch_rate']:.3f}%)"
            )

        agg = aggregate(results)
        sweep_results.append({
            "k": k,
            "aggregate": agg,
            "maps": results,
        })

        print("\nAggregate for K =", k)
        print(json.dumps(agg, indent=2))

    payload = {
        "config": {
            "first_map": args.first_map,
            "last_map": args.last_map,
            "ks": args.ks,
            "target": list(target),
            "p": args.p,
            "max_steps": args.max_steps,
            "metric": "wasserstein",
            "uncertainty_metric": "mse",
        },
        "results": sweep_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_comparison_csv(args.csv_output, sweep_results)

    print("\n" + "=" * 72)
    print("K comparison")
    print("=" * 72)
    print(
        f"{'K':>5} {'W2 mean':>12} {'MSE abs':>12} {'MSE rel':>12} "
        f"{'URC dec.':>12} {'Trans.':>12}"
    )
    for entry in sweep_results:
        a = entry["aggregate"]
        print(
            f"{entry['k']:>5} "
            f"{a['state_wasserstein_mean_weighted']:>12.6f} "
            f"{a['state_mse_abs_mean_weighted']:>12.6f} "
            f"{100*a['state_mse_rel_mean_weighted']:>11.3f}% "
            f"{100*a['urc_threshold_decision_mismatch_rate']:>11.3f}% "
            f"{100*a['transition_mismatch_rate']:>11.3f}%"
        )

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
