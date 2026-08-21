import os
import re
import shutil


def manipulate_prism_model(
        input_path,
        output_path,
        possible_decisions=[1, 10],
        decision_variables=None,
        before_actions=['east', 'west', 'north', 'south'],
        after_actions=['update', 'skip_update'],
        module_name='Knowledge',
        baseline=False):
    """
    Adds the URC controller to a PRISM model.

    Gaussian refined model:
      - gstate is the technical Markov state.
      - gvar is the quantized Gaussian uncertainty class.
      - the URC synthesizes one update interval c in [1..10] per gvar.

    If gvar exists and no explicit decision_variables are supplied,
    gvar is automatically used.
    """
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output files cannot be the same.")

    shutil.copyfile(input_path, output_path)

    variables, guard_variables = get_variables(
        input_path,
        decision_variables=decision_variables
    )

    remove_counter_from_module(output_path)

    add_controller(
        output_path,
        guard_variables,
        variables,
        possible_decisions,
        baseline=baseline
    )

    add_turn(output_path, before_actions, after_actions)


def _parse_int_constants(prism_model_path):
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


def _parse_int_variables(prism_model_path, int_constants):
    pattern = re.compile(
        r'(\w+)\s*:\s*\[(-?\s*\w+)\s*\.\.\s*(-?\s*\w+)\]'
        r'\s*init\s*(-?\s*\w+)\s*;'
    )

    variables = {}

    with open(prism_model_path, 'r') as prism_model_file:
        for line in prism_model_file:
            for match in pattern.finditer(line):
                name = match.group(1)
                lower = __get_limit(
                    match.group(2).replace(" ", ""),
                    int_constants
                )
                upper = __get_limit(
                    match.group(3).replace(" ", ""),
                    int_constants
                )
                variables[name] = [name, lower, upper]

    return variables


def get_variables(prism_model_path, decision_variables=None):
    """
    Returns:
      variables:
        value ranges used to create evolve decision_* parameters
      guard_variables:
        model variables used in [URC] guards

    Refined Gaussian model:
        variables       = [['gvar', 0, GVAR_MAX]]
        guard_variables = ['gvar']

    Thus the URC learns:
        gvar -> c in [1..10]

    gstate is intentionally NOT used as a decision variable.
    """
    int_constants = _parse_int_constants(prism_model_path)
    declared = _parse_int_variables(
        prism_model_path,
        int_constants
    )

    if decision_variables is None:
        decision_variables = []
    else:
        decision_variables = list(decision_variables)

    # Gaussian default.
    if len(decision_variables) == 0 and 'gvar' in declared:
        decision_variables = ['gvar']

    variables = []
    guard_variables = []

    for name in decision_variables:
        if name not in declared:
            raise ValueError(
                f"Decision variable '{name}' is not declared in "
                f"{prism_model_path}."
            )

        variables.append(declared[name])

        # Legacy behavior for x/y:
        # x -> xhat, y -> yhat.
        estimated_name = name + 'hat'

        if estimated_name in declared:
            guard_variables.append(estimated_name)
        else:
            guard_variables.append(name)

    return variables, guard_variables


def __get_limit(string, constants):
    if not string.lstrip("-").isdigit():
        if string not in constants:
            raise ValueError(
                f"Unknown integer bound/constant '{string}'."
            )
        return constants[string]

    return int(string)


def remove_counter_from_module(output_path):
    pattern = re.compile(
        r"^\s*const\s+int\s+c\d*\s*=\s*\d+\s*;"
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


def add_controller(
        file_path,
        guard_variables,
        variables,
        possible_decisions,
        baseline):
    combinations = generate_combinations_list(variables)

    __add_controller_prefix(
        file_path,
        possible_decisions,
        combinations,
        variables,
        baseline
    )

    with open(file_path, 'a') as file:
        file.write('  c : [1..10] init 1;\n')

        for combination in combinations:
            new_line = '  [URC] true'

            for value, guard_variable in zip(
                    combination,
                    guard_variables):
                new_line += (
                    f' & {guard_variable}={value}'
                )

            new_line += ' -> (c\'=decision'

            for value in combination:
                new_line += f'_{value}'

            new_line += ');\n'
            file.write(new_line)

        file.write('endmodule\n')


def __add_controller_prefix(
        file_path,
        possible_decisions,
        combinations,
        variables,
        baseline):
    with open(file_path, 'a') as file:
        for combination in combinations:
            if baseline:
                new_line = 'const int decision'
            else:
                new_line = 'evolve int decision'

            for var in range(0, len(variables)):
                new_line += (
                    '_'
                    + str(combination[var])
                )

            if baseline:
                new_line += '=1;'
            else:
                new_line += (
                    f' [{possible_decisions[0]}..'
                    f'{possible_decisions[1]}];'
                )

            file.write('\n' + new_line)

        file.write('\nmodule URC\n')


def add_turn(
        file_path,
        before_actions,
        after_actions):
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
            remaining_variables):
        if not remaining_variables:
            result.append(
                tuple(current_combination)
            )
            return

        current_variable = remaining_variables[0]

        for value in range(
                current_variable[1],
                current_variable[2] + 1):
            generate_combinations_recursive(
                current_combination + [value],
                remaining_variables[1:]
            )

    generate_combinations_recursive(
        [],
        variables
    )

    return result
