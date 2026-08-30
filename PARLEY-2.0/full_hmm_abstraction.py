"""
full_hmm_abstraction.py

Offline construction of a finite HMM abstraction for the PARLEY 10x10 robot.

Model layers
------------
Physical position:
    X_t = (x_t, y_t)

Point estimate:
    Xhat_t = (xhat_t, yhat_t)

Hidden error:
    e_t = X_t - Xhat_t

Hidden HMM state:
    S_t = Q_e(e_t)

Observation:
    z_t = X_t + v_t
    r_t = z_t - Xhat_t = e_t + v_t
    O_t = Q_o(r_t)

For the current 10x10 integer grid, the exact error lattice is already finite:
    e_x, e_y in {-9, ..., 9}
so the default Q_e is the exact lattice quantizer (361 hidden states).

The transition model is position/action dependent:
    A^{xhat,yhat,a}_{ij}
      = P(S_{t+1}=j | S_t=i, Xhat_t=(xhat,yhat), a_t=a)

The observation model is:
    B_{io} = P(O_t=o | S_t=i)

For sigma_obs = 0, B is the identity matrix (perfect observation).
For sigma_obs > 0, independent Gaussian measurement noise is assumed:
    v_x, v_y ~ N(0, sigma_obs^2)
and B is computed analytically by integrating the Gaussian density over
the quantization cells.

The script writes one JSON file per map to hmm_models/map_<id>.json.
No PRISM model is generated here; this script deliberately separates
offline HMM construction from PARLEY/EvoChecker synthesis.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import dijkstra


DIRECTIONS = ("west", "east", "south", "north")
MOVE = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}

# Same movement semantics as the existing PARLEY/Gaussian model.
DEFAULT_P = 0.01

# 10x10 grid -> coordinate values 0..9 -> exact error values -9..9.
DEFAULT_GRID_SIZE = 10

# Observation model. 0.0 preserves the current perfect-localization semantics.
# Set e.g. 0.5 or 1.0 only for a deliberate noisy-sensor HMM experiment.
DEFAULT_SIGMA_OBS = 0.0

# Sparse JSON threshold for B. Rows are renormalized after truncation.
EMISSION_EPS = 1e-15


def _clip(v, n):
    return min(max(v, 0), n)


def _move(x, y, action, n):
    dx, dy = MOVE[action]
    return _clip(x + dx, n), _clip(y + dy, n)


def _robot_outcomes(action, p):
    """
    Return (probability, dx, dy) for the physical robot movement.

    Intended direction has probability 1-3p; the other three directions p.
    This exactly mirrors the current Gaussian/PARLEY robot semantics.
    """
    intended = 1.0 - 3.0 * p

    if action == "east":
        return (
            (intended, 1, 0),
            (p, 0, 1),
            (p, 0, -1),
            (p, -1, 0),
        )
    if action == "west":
        return (
            (p, 1, 0),
            (p, 0, 1),
            (p, 0, -1),
            (intended, -1, 0),
        )
    if action == "north":
        return (
            (p, 1, 0),
            (intended, 0, 1),
            (p, 0, -1),
            (p, -1, 0),
        )
    if action == "south":
        return (
            (p, 1, 0),
            (p, 0, 1),
            (intended, 0, -1),
            (p, -1, 0),
        )

    raise ValueError(f"Unknown action: {action}")


def _controller(map_data, target):
    """Same fixed MAPE/Dijkstra policy convention as the Gaussian script."""
    return list(zip(*dijkstra.compute_directions(map_data, target)))


def _direction(controller, x, y):
    value = int(controller[y][x])
    return DIRECTIONS[value] if 0 <= value < 4 else None


def load_map(path):
    """
    Load a PARLEY map using the same orientation convention as the existing
    Gaussian model generator.
    """
    with Path(path).open("r", newline="") as file:
        rows = list(csv.reader(file))

    transposed = list(zip(*rows))
    return [row[::-1] for row in transposed]


# ---------------------------------------------------------------------------
# 1. Hidden-state quantization Q_e
# ---------------------------------------------------------------------------

def build_error_lattice(n):
    """
    Exact quantization of the physical position error.

    Because x,xhat,y,yhat are integer grid coordinates in [0,n],
        e_x = x-xhat, e_y = y-yhat
    are already discrete integers in [-n,n].

    Hence:
        Q_e(e_x,e_y) = state_id(e_x,e_y)

    This is the lossless baseline abstraction. A later experiment may cluster
    these exact states, but that should be evaluated separately.
    """
    errors = []
    state_id = {}

    # Put zero error first so pi and perfect reset have state 0.
    ordered = [(0, 0)]
    ordered.extend(
        (ex, ey)
        for ex in range(-n, n + 1)
        for ey in range(-n, n + 1)
        if (ex, ey) != (0, 0)
    )

    for idx, error in enumerate(ordered):
        state_id[error] = idx
        errors.append({
            "state_id": idx,
            "error_x": error[0],
            "error_y": error[1],
            "squared_error": error[0] ** 2 + error[1] ** 2,
        })

    return errors, state_id


def q_e(error_x, error_y, state_id, n):
    """
    Exact lattice quantizer Q_e.

    Values should already lie in [-n,n]. Clipping is intentionally NOT used:
    an out-of-range error indicates a bug in the transition construction.
    """
    if not (-n <= error_x <= n and -n <= error_y <= n):
        raise ValueError(
            f"Error ({error_x},{error_y}) outside valid lattice [-{n},{n}]^2."
        )
    return state_id[(int(error_x), int(error_y))]


def valid_hidden_state_at_estimate(error_x, error_y, xhat, yhat, n):
    """
    A global error state is physically possible at a given point estimate iff
        X = Xhat + e
    is still inside the grid.
    """
    x = xhat + error_x
    y = yhat + error_y
    return 0 <= x <= n and 0 <= y <= n


# ---------------------------------------------------------------------------
# 2. Transition model A
# ---------------------------------------------------------------------------

def transition_row(error_x, error_y, xhat, yhat, action, n, p, state_id):
    r"""
    Compute one exact row of A^{xhat,yhat,a}.

    Given hidden error e_t:
        X_t = Xhat_t + e_t

    The point estimate follows the commanded move deterministically:
        Xhat_{t+1} = clip(Xhat_t + u_t)

    The physical robot realizes delta with the four PARLEY probabilities:
        X_{t+1} = clip(X_t + delta)

    Therefore:
        e_{t+1} = X_{t+1} - Xhat_{t+1}

    and:
        A^{xhat,yhat,a}_{ij}
        = sum_{delta : Q_e(e_{t+1})=j} P(delta | a)

    Duplicate successor states caused by boundary clipping are automatically
    aggregated.
    """
    if not valid_hidden_state_at_estimate(
        error_x, error_y, xhat, yhat, n
    ):
        return None

    x = xhat + error_x
    y = yhat + error_y

    next_xhat, next_yhat = _move(xhat, yhat, action, n)

    probabilities = defaultdict(float)

    for probability, dx, dy in _robot_outcomes(action, p):
        next_x = _clip(x + dx, n)
        next_y = _clip(y + dy, n)

        next_error_x = next_x - next_xhat
        next_error_y = next_y - next_yhat

        next_state = q_e(
            next_error_x,
            next_error_y,
            state_id,
            n,
        )
        probabilities[next_state] += probability

    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Transition row does not sum to 1: {total} "
            f"for xhat={xhat}, yhat={yhat}, e=({error_x},{error_y}), "
            f"action={action}"
        )

    return [
        {"next_state": sid, "probability": prob}
        for sid, prob in sorted(probabilities.items())
    ]


def build_transition_model(
    map_data,
    target,
    p,
    errors,
    state_id,
    policy_only=True,
):
    """
    Construct sparse position-dependent transition matrices.

    policy_only=True:
        stores only the MAPE-selected action for every (xhat,yhat), matching
        the current Gaussian abstraction.

    policy_only=False:
        stores all four actions and can be useful for diagnostics.
    """
    n = len(map_data) - 1
    controller = _controller(map_data, target)

    transitions = {}
    valid_states = {}

    for xhat in range(n + 1):
        for yhat in range(n + 1):
            valid = [
                item["state_id"]
                for item in errors
                if valid_hidden_state_at_estimate(
                    item["error_x"],
                    item["error_y"],
                    xhat,
                    yhat,
                    n,
                )
            ]
            valid_states[f"{xhat},{yhat}"] = valid

            if policy_only:
                selected = _direction(controller, xhat, yhat)
                actions = () if selected is None else (selected,)
            else:
                actions = DIRECTIONS

            for action in actions:
                for item in errors:
                    row = transition_row(
                        item["error_x"],
                        item["error_y"],
                        xhat,
                        yhat,
                        action,
                        n,
                        p,
                        state_id,
                    )
                    if row is None:
                        continue

                    key = f"{xhat},{yhat},{action},{item['state_id']}"
                    transitions[key] = row

    return transitions, valid_states, controller


# ---------------------------------------------------------------------------
# 3. Observation quantization Q_o and emission model B
# ---------------------------------------------------------------------------

def _normal_cdf(x, mean, sigma):
    if sigma <= 0.0:
        if x < mean:
            return 0.0
        return 1.0
    z = (x - mean) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _quantization_interval(value, low, high):
    """
    Cell of a nearest-integer quantizer with saturation at the boundaries.

    Interior observation k:
        [k-0.5, k+0.5)

    Lowest bin:
        (-inf, low+0.5)

    Highest bin:
        [high-0.5, +inf)

    The infinite boundary cells guarantee that every B row sums to 1 even
    when Gaussian observation noise would otherwise leave [-n,n].
    """
    if value == low:
        return -math.inf, low + 0.5
    if value == high:
        return high - 0.5, math.inf
    return value - 0.5, value + 0.5


def _gaussian_bin_probability(observed_value, hidden_value, sigma, low, high):
    lower, upper = _quantization_interval(observed_value, low, high)

    lower_cdf = (
        0.0
        if lower == -math.inf
        else _normal_cdf(lower, hidden_value, sigma)
    )
    upper_cdf = (
        1.0
        if upper == math.inf
        else _normal_cdf(upper, hidden_value, sigma)
    )
    return max(0.0, upper_cdf - lower_cdf)


def build_observation_model(errors, sigma_obs, n, eps=EMISSION_EPS):
    r"""
    Construct the HMM emission matrix B.

    Measurement model:
        z_t = X_t + v_t,
        v_t ~ N(0, sigma_obs^2 I)

    Relative measurement:
        r_t = z_t - Xhat_t
            = e_t + v_t

    Observation quantizer:
        O_t = Q_o(r_t)

    We use the same {-n,...,n}^2 lattice for observations. Therefore:
        B_{io} = P(O_t=o | S_t=i)

    With independent x/y noise:
        B_{i,(ox,oy)}
          = P(Q( e_x + v_x ) = ox)
            * P(Q( e_y + v_y ) = oy)

    For sigma_obs=0 this becomes exactly:
        B_{io} = 1  if o corresponds to the same error as state i,
                 0  otherwise.

    B is deliberately independent of (xhat,yhat) because the observation is
    represented as the residual z-Xhat. If a future sensor is clipped in
    absolute world coordinates, B must instead become position-dependent.
    """
    observations = [
        {
            "observation_id": item["state_id"],
            "residual_x": item["error_x"],
            "residual_y": item["error_y"],
        }
        for item in errors
    ]

    emissions = {}

    if sigma_obs == 0.0:
        for item in errors:
            sid = item["state_id"]
            emissions[str(sid)] = [
                {"observation": sid, "probability": 1.0}
            ]
        return observations, emissions

    low = -n
    high = n

    # Precompute 1-D probabilities because B factorizes in x and y.
    one_d = {}
    for hidden in range(low, high + 1):
        probs = {}
        for observed in range(low, high + 1):
            probs[observed] = _gaussian_bin_probability(
                observed,
                hidden,
                sigma_obs,
                low,
                high,
            )

        total = sum(probs.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"1D emission probabilities do not sum to 1: "
                f"hidden={hidden}, total={total}"
            )
        one_d[hidden] = probs

    obs_lookup = {
        (item["residual_x"], item["residual_y"]): item["observation_id"]
        for item in observations
    }

    for hidden in errors:
        ex = hidden["error_x"]
        ey = hidden["error_y"]

        row = []
        for ox in range(low, high + 1):
            px = one_d[ex][ox]
            if px <= 0.0:
                continue

            for oy in range(low, high + 1):
                probability = px * one_d[ey][oy]
                if probability >= eps:
                    row.append({
                        "observation": obs_lookup[(ox, oy)],
                        "probability": probability,
                    })

        # Sparse truncation can remove tiny tails. Renormalize explicitly.
        total = sum(item["probability"] for item in row)
        if total <= 0.0:
            raise ValueError(
                f"Emission row became empty for hidden state "
                f"{hidden['state_id']}."
            )

        for item in row:
            item["probability"] /= total

        final_total = sum(item["probability"] for item in row)
        if not math.isclose(
            final_total, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Emission row does not sum to 1 after normalization: "
                f"state={hidden['state_id']}, total={final_total}"
            )

        emissions[str(hidden["state_id"])] = row

    return observations, emissions


# ---------------------------------------------------------------------------
# 4. Initial distribution pi and complete model
# ---------------------------------------------------------------------------

def build_initial_distribution(errors):
    """
    Perfect localization/reset implies:
        e_0 = (0,0)

    Because zero error is state 0:
        pi = delta_0.
    """
    zero = next(
        item for item in errors
        if item["error_x"] == 0 and item["error_y"] == 0
    )
    return {
        "type": "delta",
        "state": zero["state_id"],
        "probability": 1.0,
    }


def _validate_model(model):
    if model["hidden_states"][0]["error_x"] != 0:
        raise ValueError("Hidden state 0 must be zero-error state.")
    if model["hidden_states"][0]["error_y"] != 0:
        raise ValueError("Hidden state 0 must be zero-error state.")

    # Validate all A rows.
    for key, row in model["A"].items():
        total = sum(entry["probability"] for entry in row)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"A row {key} sums to {total}.")

    # Validate all B rows.
    for key, row in model["B"].items():
        total = sum(entry["probability"] for entry in row)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"B row {key} sums to {total}.")


def build_hmm_model(
    map_id,
    map_data,
    target=(9, 9),
    p=DEFAULT_P,
    sigma_obs=DEFAULT_SIGMA_OBS,
    policy_only=True,
    cache_dir="hmm_models",
):
    """
    Build one map-specific HMM abstraction.

    Important:
    This is the *model definition* (Q_e, A, B, pi), not yet the finite
    discretization of the continuous HMM belief vector beta_t. That second
    abstraction should be implemented and evaluated separately.
    """
    map_size = len(map_data)
    n = map_size - 1

    errors, state_id = build_error_lattice(n)

    transitions, valid_states, controller = build_transition_model(
        map_data=map_data,
        target=target,
        p=p,
        errors=errors,
        state_id=state_id,
        policy_only=policy_only,
    )

    observations, emissions = build_observation_model(
        errors=errors,
        sigma_obs=sigma_obs,
        n=n,
    )

    pi = build_initial_distribution(errors)

    model = {
        "map_id": int(map_id),
        "grid_size": map_size,
        "n": n,
        "p": float(p),
        "target": [int(target[0]), int(target[1])],

        "hidden_state_definition": "S_t = Q_e(X_t - Xhat_t)",
        "quantization": {
            "type": "exact_integer_error_lattice",
            "error_min": -n,
            "error_max": n,
            "state_count": len(errors),
            "zero_state": 0,
        },

        "transition_definition": (
            "A[xhat,yhat,action][i,j] = "
            "P(S_{t+1}=j | S_t=i, Xhat_t=(xhat,yhat), action)"
        ),
        "policy_only": bool(policy_only),
        "A": transitions,
        "valid_hidden_states_by_estimate": valid_states,

        "observation_definition": "O_t = Q_o(z_t - Xhat_t)",
        "sensor_model": {
            "measurement": "z_t = X_t + v_t",
            "noise": "v_t ~ N(0, sigma_obs^2 I)",
            "sigma_obs": float(sigma_obs),
            "quantizer": "nearest integer residual with saturated boundary bins",
            "perfect_observation": bool(sigma_obs == 0.0),
        },
        "B": emissions,

        "pi": pi,
        "hidden_states": errors,
        "observations": observations,

        "notes": [
            "The exact 10x10 error lattice has 19*19 = 361 hidden states.",
            "A is computed analytically from the PARLEY movement probabilities; no Monte Carlo estimation is required.",
            "A is position-dependent because grid clipping changes the error dynamics.",
            "Obstacle/crash information is not conditioned into A, matching the current Gaussian uncertainty propagation.",
            "B is identity when sigma_obs=0, which matches the current perfect localization/update semantics.",
            "For sigma_obs>0, B is an analytic discretized Gaussian observation model.",
            "The HMM belief beta_t is not yet discretized here; a later script should compute reachable beliefs and representative belief states for PRISM.",
        ],
    }

    _validate_model(model)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"map_{map_id}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)

    return model, output_path


def precompute_maps(
    first_map=10,
    last_map=99,
    maps_dir="maps",
    target=(9, 9),
    p=DEFAULT_P,
    sigma_obs=DEFAULT_SIGMA_OBS,
    policy_only=True,
    cache_dir="hmm_models",
):
    for map_id in range(first_map, last_map + 1):
        path = Path(maps_dir) / f"map_{map_id}.csv"
        if not path.exists():
            print(f"skip map {map_id}: {path} missing")
            continue

        map_data = load_map(path)

        model, output_path = build_hmm_model(
            map_id=map_id,
            map_data=map_data,
            target=target,
            p=p,
            sigma_obs=sigma_obs,
            policy_only=policy_only,
            cache_dir=cache_dir,
        )

        print(
            f"map {map_id}: "
            f"{model['quantization']['state_count']} hidden states, "
            f"{len(model['observations'])} observations, "
            f"{len(model['A'])} sparse A rows, "
            f"sigma_obs={sigma_obs}, "
            f"output={output_path}"
        )


if __name__ == "__main__":
    # Reference/fair-comparison configuration:
    # sigma_obs=0.0 keeps the existing perfect-localization semantics.
    #
    # For a genuine noisy-emission HMM experiment, deliberately change e.g.:
    # sigma_obs=0.5
    precompute_maps(
        first_map=10,
        last_map=99,
        maps_dir="maps",
        target=(9, 9),
        p=0.01,
        sigma_obs=0.0,
        policy_only=True,
        cache_dir="hmm_models",
    )
