import os
import re
import shutil


# Ten threshold choices, analogous to the ten interval-model choices.
#
# These thresholds refer to:
#     belief_uncertainty = 100-b1
#
# The exact list can later be calibrated empirically from the generated
# belief-state catalogue.
BELIEF_THRESHOLDS = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]


def manipulate_prism_model(
    input_path,
    output_path,
    possible_decisions=[1, 10],
    decision_variables=None,
    before_actions=['east', 'west', 'north', 'south'],
    after_actions=['update', 'skip_update'],
    module_name='Knowledge',
    baseline=False,
):
    """
    Position-specific threshold policy.

    One evolved decision_x_y is generated per MAP position.
    That decision selects a belief-uncertainty threshold.

    The full abstract Top-4 belief remains in the PRISM model:
        belief_state
        b1,b2,b3,b4,other

    However, unlike B2 with decision_x_y_belief_state, the number of
    evolved parameters stays the same order as in the point-estimate
    model: one decision per (xhat,yhat).
    """
    if decision_variables is None:
        decision_variables = ['xhat', 'yhat']

    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output files cannot be the same.")

    shutil.copyfile(input_path, output_path)

    variables = get_policy_variables(
        input_path,
        decision_variables,
    )

    remove_default_threshold_constant(output_path)
    add_controller(
        output_path,
        variables,
        possible_decisions,
        baseline=baseline,
    )
    add_turn(
        output_path,
        before_actions,
        after_actions,
    )


def _read_int_constants(prism_model_path):
    pattern = re.compile(
        r'const\s+int\s+(\w+)\s*=\s*(-?\s*\d+)\s*;'
    )
    constants = {}

    with open(prism_model_path, 'r') as prism_model_file:
        for line in prism_model_file:
            for match in pattern.finditer(line):
                constants[match.group(1)] = int(
                    match.group(2).replace(" ", "")
                )

    return constants


def _resolve_limit(token, constants):
    token = token.replace(" ", "")

    if token.lstrip("-").isdigit():
        return int(token)

    return constants[token]


def get_policy_variables(prism_model_path, decision_variables):
    constants = _read_int_constants(prism_model_path)

    declaration_pattern = re.compile(
        r'(\w+)\s*:\s*\[(-?\s*\w+)\s*\.\.\s*(-?\s*\w+)\]\s*'
        r'init\s*(-?\s*\w+)\s*;'
    )

    found = {}

    with open(prism_model_path, 'r') as prism_model_file:
        for line in prism_model_file:
            for match in declaration_pattern.finditer(line):
                name = match.group(1)

                if name not in decision_variables:
                    continue

                lower = _resolve_limit(
                    match.group(2),
                    constants,
                )
                upper = _resolve_limit(
                    match.group(3),
                    constants,
                )

                found[name] = [
                    name,
                    lower,
                    upper,
                ]

    missing = [
        name
        for name in decision_variables
        if name not in found
    ]

    if missing:
        raise ValueError(
            "Could not find URC decision variables in PRISM model: "
            + ", ".join(missing)
        )

    return [
        found[name]
        for name in decision_variables
    ]


def remove_default_threshold_constant(output_path):
    pattern = re.compile(
        r"^\s*const\s+int\s+max_belief_uncertainty\s*=\s*\d+\s*;"
    )

    with open(output_path, 'r') as file:
        lines = file.readlines()

    new_lines = [
        line
        for line in lines
        if not pattern.match(line)
    ]

    with open(output_path, 'w') as file:
        file.writelines(new_lines)


def _decision_name(combination):
    return (
        'decision_'
        + '_'.join(
            str(value)
            for value in combination
        )
    )


def _threshold_expression(decision_name):
    if len(BELIEF_THRESHOLDS) != 10:
        raise ValueError(
            "BELIEF_THRESHOLDS must contain exactly 10 values."
        )

    expression = ""

    for index, threshold in enumerate(
        BELIEF_THRESHOLDS[:-1],
        start=1,
    ):
        expression += (
            f"{decision_name}={index} ? {threshold} : "
        )

    expression += str(BELIEF_THRESHOLDS[-1])

    return expression


def add_controller(
    file_path,
    variables,
    possible_decisions,
    baseline,
):
    combinations = generate_combinations_list(variables)

    with open(file_path, 'a') as file:
        file.write('\n')

        for combination in combinations:
            name = _decision_name(combination)

            if baseline:
                file.write(
                    f'const int {name}=1;\n'
                )
            else:
                file.write(
                    f'evolve int {name} '
                    f'[{possible_decisions[0]}..'
                    f'{possible_decisions[1]}];\n'
                )

        file.write('\nmodule URC\n')

        min_threshold = min(BELIEF_THRESHOLDS)
        max_threshold = max(BELIEF_THRESHOLDS)

        file.write(
            f'  max_belief_uncertainty : '
            f'[{min_threshold}..{max_threshold}] '
            f'init {min_threshold};\n'
        )

        for combination in combinations:
            decision_name = _decision_name(combination)

            guard = '  [URC] true'

            for value, variable in zip(
                combination,
                variables,
            ):
                guard += (
                    f' & {variable[0]}={value}'
                )

            threshold_expression = _threshold_expression(
                decision_name
            )

            file.write(
                guard
                + " -> "
                + "(max_belief_uncertainty'="
                + threshold_expression
                + ");\n"
            )

        file.write('endmodule\n')


def add_turn(
    file_path,
    before_actions,
    after_actions,
):
    with open(file_path, 'a') as file:
        file.write('module Turn\n')
        file.write('  t : [0..2] init 0;\n')

        for action in before_actions:
            file.write(
                f'  [{action}] (t=0) -> (t\'=1);\n'
            )

        file.write('\n')
        file.write(
            '  [URC] (t=1) -> (t\'=2);\n'
        )
        file.write('\n')

        for action in after_actions:
            file.write(
                f'  [{action}] (t=2) -> (t\'=0);\n'
            )

        if len(after_actions) == 0:
            file.write(
                '  [] (t=2) -> (t\'=0);\n'
            )

        file.write('endmodule\n')


def generate_combinations_list(variables):
    result = []

    def generate_combinations_recursive(
        current_combination,
        remaining_variables,
    ):
        if not remaining_variables:
            result.append(
                tuple(current_combination)
            )
            return

        current_variable = remaining_variables[0]

        for value in range(
            current_variable[1],
            current_variable[2] + 1,
        ):
            generate_combinations_recursive(
                current_combination + [value],
                remaining_variables[1:],
            )

    generate_combinations_recursive(
        [],
        variables,
    )

    return result


if __name__ == "__main__":
    # Minimal local test. Adjust these paths when testing inside PARLEY.
    i=10
    infile = f'Applications/EvoChecker-master/models/model_{i}.prism'
    outfile = f'Applications/EvoChecker-master/models/model_{i}_umc.prism'

    if os.path.exists(infile):
        manipulate_prism_model(
            input_path=infile,
            output_path=outfile,
            possible_decisions=[1, 10],
            decision_variables=['xhat', 'yhat'],
            baseline=False,
        )
        print(f"generated: {outfile}")
    else:
        print(
            "Test input not found. "
            "Generate a base belief PRISM model first."
        )
