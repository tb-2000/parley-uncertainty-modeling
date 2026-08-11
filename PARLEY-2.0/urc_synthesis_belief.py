import os
import re
import shutil


def manipulate_prism_model(
    input_path,
    output_path,
    possible_decisions=[0, 1],
    decision_variables=None,
    before_actions=['east', 'west', 'north', 'south'],
    after_actions=['update', 'skip_update'],
    module_name='Knowledge',
    baseline=False,
):
    """
    B2 belief-state-specific URC.

    For every (xhat, yhat, belief_state) combination EvoChecker evolves
    one binary decision:
        0 -> skip_update
        1 -> update

    b1,b2,b3,b4,other are represented by belief_state and are therefore
    all taken into account by the policy without becoming five separate
    cartesian policy dimensions.
    """
    if decision_variables is None:
        decision_variables = ['xhat', 'yhat', 'belief_state']

    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output files cannot be the same.")

    shutil.copyfile(input_path, output_path)

    variables = get_policy_variables(input_path, decision_variables)

    remove_default_update_constant(output_path)
    add_controller(
        output_path,
        variables,
        possible_decisions,
        baseline=baseline,
    )
    add_turn(output_path, before_actions, after_actions)


def _read_int_constants(prism_model_path):
    pattern = re.compile(r'const\s+int\s+(\w+)\s*=\s*(-?\s*\d+)\s*;')
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
    """
    Read only the variables explicitly requested as URC policy inputs.

    This deliberately excludes cnt_e/cnt_w/cnt_n/cnt_s, belief_age and
    the individual b-values.  The abstract belief_state already encodes
    the complete (b1,b2,b3,b4,other) distribution class.
    """
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

                lower = _resolve_limit(match.group(2), constants)
                upper = _resolve_limit(match.group(3), constants)
                found[name] = [name, lower, upper]

    missing = [name for name in decision_variables if name not in found]
    if missing:
        raise ValueError(
            "Could not find URC decision variables in PRISM model: "
            + ", ".join(missing)
        )

    return [found[name] for name in decision_variables]


def remove_default_update_constant(output_path):
    # The generated base model contains:
    #     const int urc_update = 0;
    # This is replaced by the state variable in module URC.
    pattern = re.compile(
        r"^\s*const\s+int\s+urc_update\s*=\s*[01]\s*;"
    )

    with open(output_path, 'r') as file:
        lines = file.readlines()

    new_lines = [line for line in lines if not pattern.match(line)]

    with open(output_path, 'w') as file:
        file.writelines(new_lines)


def _decision_name(combination):
    return 'decision_' + '_'.join(str(value) for value in combination)


def add_controller(file_path, variables, possible_decisions, baseline):
    combinations = generate_combinations_list(variables)

    with open(file_path, 'a') as file:
        file.write('\n')

        for combination in combinations:
            name = _decision_name(combination)

            if baseline:
                # Baseline default: do not request an update.
                file.write(f'const int {name}=0;\n')
            else:
                file.write(
                    f'evolve int {name} '
                    f'[{possible_decisions[0]}..{possible_decisions[1]}];\n'
                )

        file.write('\nmodule URC\n')
        file.write('  urc_update : [0..1] init 0;\n')

        for combination in combinations:
            name = _decision_name(combination)

            guard = '  [URC] true'
            for value, variable in zip(combination, variables):
                guard += f' & {variable[0]}={value}'

            file.write(
                guard
                + f" -> (urc_update'={name});\n"
            )

        file.write('endmodule\n')


def add_turn(file_path, before_actions, after_actions):
    with open(file_path, 'a') as file:
        file.write('module Turn\n')
        file.write('  t : [0..2] init 0;\n')

        for action in before_actions:
            file.write(f'  [{action}] (t=0) -> (t\'=1);\n')

        file.write('\n')
        file.write('  [URC] (t=1) -> (t\'=2);\n')
        file.write('\n')

        for action in after_actions:
            file.write(f'  [{action}] (t=2) -> (t\'=0);\n')

        if len(after_actions) == 0:
            file.write('  [] (t=2) -> (t\'=0);\n')

        file.write('endmodule\n')


def generate_combinations_list(variables):
    result = []

    def generate_combinations_recursive(current_combination, remaining_variables):
        if not remaining_variables:
            result.append(tuple(current_combination))
            return

        current_variable = remaining_variables[0]

        for value in range(current_variable[1], current_variable[2] + 1):
            generate_combinations_recursive(
                current_combination + [value],
                remaining_variables[1:],
            )

    generate_combinations_recursive([], variables)
    return result
