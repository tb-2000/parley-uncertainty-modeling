import os
import re
import shutil


def manipulate_prism_model(
        input_path,
        output_path,
        possible_decisions=[1, 10],
        decision_variables=['x', 'y'],
        before_actions=['east', 'west', 'north', 'south'],
        after_actions=['update', 'skip_update'],
        module_name='Knowledge',
        baseline=False):
    """
    Gaussian trace URC, analogous to the Belief-State URC.

    The URC logic is independent of the concrete grid width h;
    for the current pipeline the thresholds come from h=0.05 models.

    The search space remains position-based:
        decision_x_y in [1..10]

    Each decision selects one of:
        gaussian_threshold_1 .. gaussian_threshold_10

    The Knowledge module then compares the current trace(Sigma) uncertainty
    with max_gaussian_uncertainty via formula update_required.
    """
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output files cannot be the same.")

    shutil.copyfile(input_path, output_path)

    variables, guard_variables = get_variables(
        input_path, decision_variables
    )
    thresholds = get_gaussian_thresholds(input_path)

    remove_counter_from_module(output_path)
    add_controller(
        output_path,
        guard_variables,
        variables,
        thresholds,
        possible_decisions,
        baseline
    )
    add_turn(output_path, before_actions, after_actions)


def _parse_int_constants(path):
    pattern = re.compile(r'const\s+int\s+(\w+)\s*=\s*(-?\d+)\s*;')
    constants = {}
    with open(path, 'r') as f:
        for line in f:
            for m in pattern.finditer(line):
                constants[m.group(1)] = int(m.group(2))
    return constants


def _parse_int_variables(path, constants):
    pattern = re.compile(
        r'(\w+)\s*:\s*\[(-?\w+)\s*\.\.\s*(-?\w+)\]\s*init\s*(-?\w+)\s*;'
    )
    result = {}
    with open(path, 'r') as f:
        for line in f:
            for m in pattern.finditer(line):
                name = m.group(1)
                result[name] = [
                    name,
                    _resolve(m.group(2), constants),
                    _resolve(m.group(3), constants),
                ]
    return result


def _resolve(value, constants):
    if value.lstrip("-").isdigit():
        return int(value)
    if value not in constants:
        raise ValueError(f"Unknown integer bound: {value}")
    return constants[value]


def get_variables(path, decision_variables):
    constants = _parse_int_constants(path)
    declared = _parse_int_variables(path, constants)
    variables = []
    guards = []

    for name in decision_variables:
        if name not in declared:
            raise ValueError(f"Decision variable '{name}' not declared.")
        variables.append(declared[name])
        estimate = name + "hat"
        guards.append(estimate if estimate in declared else name)

    return variables, guards


def get_gaussian_thresholds(path):
    pattern = re.compile(
        r'const\s+int\s+gaussian_threshold_(\d+)\s*=\s*(\d+)\s*;'
    )
    thresholds = {}
    with open(path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                thresholds[int(m.group(1))] = int(m.group(2))

    missing = [i for i in range(1, 11) if i not in thresholds]
    if missing:
        raise ValueError(
            f"Missing gaussian thresholds {missing} in {path}. "
            "Generate the Gaussian trace model first."
        )

    return [thresholds[i] for i in range(1, 11)]


def remove_counter_from_module(output_path):
    # Point-estimate models have const int c=...; trace models do not need it.
    pattern = re.compile(r"^\s*const\s+int\s+c\d*\s*=\s*\d+\s*;")
    with open(output_path, 'r') as f:
        lines = f.readlines()
    with open(output_path, 'w') as f:
        f.writelines(line for line in lines if not pattern.match(line))


def generate_combinations_list(variables):
    result = []

    def rec(current, remaining):
        if not remaining:
            result.append(tuple(current))
            return
        var = remaining[0]
        for value in range(var[1], var[2] + 1):
            rec(current + [value], remaining[1:])

    rec([], variables)
    return result


def add_controller(
        file_path,
        guard_variables,
        variables,
        thresholds,
        possible_decisions,
        baseline):
    combinations = generate_combinations_list(variables)

    with open(file_path, 'a') as f:
        # 100 position-based evolvables on a 10x10 grid.
        for combination in combinations:
            suffix = "".join(f"_{v}" for v in combination)
            if baseline:
                f.write(f"\nconst int decision{suffix}=1;")
            else:
                f.write(
                    f"\nevolve int decision{suffix} "
                    f"[{possible_decisions[0]}..{possible_decisions[1]}];"
                )

        f.write("\nmodule URC\n")

        low = min(thresholds)
        high = max(thresholds)
        f.write(
            f"  max_gaussian_uncertainty : [{low}..{high}] "
            f"init {low};\n"
        )

        for combination in combinations:
            suffix = "".join(f"_{v}" for v in combination)
            guard = "true"
            for value, guard_variable in zip(combination, guard_variables):
                guard += f" & {guard_variable}={value}"

            for decision in range(1, 11):
                threshold = thresholds[decision - 1]
                f.write(
                    f"  [URC] {guard} & decision{suffix}={decision} "
                    f"-> (max_gaussian_uncertainty'={threshold});\n"
                )

        f.write("endmodule\n")


def add_turn(file_path, before_actions, after_actions):
    with open(file_path, 'a') as f:
        f.write("module Turn\n")
        f.write("  t : [0..2] init 0;\n")
        for action in before_actions:
            f.write(f"  [{action}] (t=0) -> (t'=1);\n")
        f.write("\n  [URC] (t=1) -> (t'=2);\n\n")
        for action in after_actions:
            f.write(f"  [{action}] (t=2) -> (t'=0);\n")
        if not after_actions:
            f.write("  [] (t=2) -> (t'=0);\n")
        f.write("endmodule\n")
