import os
import re
import shutil


def manipulate_prism_model(input_path, output_path, possible_decisions=[1, 10], decision_variables=[],
                           before_actions=['east', 'west', 'north', 'south'], after_actions=['update', 'skip_update'], module_name='Knowledge', baseline=False):
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output files cannot be the same.")

    shutil.copyfile(input_path, output_path)

    variables, estimates = get_variables(input_path, decision_variables)

    remove_counter_from_module(output_path)

    add_controller(output_path, estimates, variables, possible_decisions, baseline=baseline)

    add_turn(output_path, before_actions, after_actions)


def get_variables(prism_model_path, decision_variables):
    # get all int constants
    int_constants_pattern = re.compile(r'const\s+int\s+(\w+)\s*=\s*(-?\s*\d+)\s*;')
    int_constants = {}

    with open(prism_model_path, 'r') as prism_model_file:
        # Process the file line by line
        for line in prism_model_file:
            # Match constants in each line
            matches = int_constants_pattern.finditer(line)
            for match in matches:
                int_constants[match.group(1)] = int(match.group(2).replace(" ", ""))

    int_variable_declaration_pattern = re.compile(
        r'(\w+)\s*:\s*\[(-?\s*\w+)\s*\.\.\s*(-?\s*\w+)\]\s*init\s*(-?\s*\w+)\s*;')
    _vars = []
    _bel = []

    with open(prism_model_path, 'r') as prism_model_file:
        # Process the file line by line again
        for line in prism_model_file:
            # Match variables in each line
            matches = int_variable_declaration_pattern.finditer(line)
            for match in matches:
                if match.group(1)[-3:] == 'hat':
                    _bel.append(match.group(1))
                elif match.group(1) not in decision_variables:
                    continue
                lower_limit = __get_limit(match.group(2).replace(" ", ""), int_constants)
                upper_limit = __get_limit(match.group(3).replace(" ", ""), int_constants)
                _vars.append([match.group(1), lower_limit, upper_limit])

    return _vars, _bel


def __get_limit(string, constants):
    if not string.lstrip("-").isdigit():
        return constants[string]
    else:
        return int(string)


def remove_counter_from_module(output_path):
    pattern = re.compile(
        r"^\s*const\s+int\s+max_belief_uncertainty\s*=\s*\d+\s*;"
    )
    with open(output_path, 'r') as file:
        lines = file.readlines()

    new_lines = [
        line for line in lines
        if not pattern.match(line)
    ]

    with open(output_path, 'w') as file:
        file.writelines(new_lines)


def get_belief_thresholds(file_path):
    pattern = re.compile(
        r"^\s*const\s+int\s+belief_threshold_(\d+)\s*=\s*(\d+)\s*;"
    )
    thresholds = {}

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.match(line)
            if match:
                thresholds[int(match.group(1))] = int(match.group(2))

    if len(thresholds) != 10:
        raise ValueError(
            "Expected belief_threshold_1..belief_threshold_10 in PRISM model."
        )

    return [thresholds[i] for i in range(1, 11)]


def add_controller(
    file_path,
    estimates,
    variables,
    possible_decisions,
    baseline,
):
    combinations = generate_combinations_list(variables)

    __add_controller_prefix(
        file_path,
        possible_decisions,
        combinations,
        variables,
        baseline,
    )

    thresholds = get_belief_thresholds(file_path)

    with open(file_path, 'a') as file:
        file.write(
            f'  max_belief_uncertainty : '
            f'[{min(thresholds)}..{max(thresholds)}] '
            f'init {thresholds[0]};\n'
        )

        # Helpful map-specific documentation.
        file.write(
            '  // decision value -> map-specific '
            'max_belief_uncertainty\n'
        )
        for index, threshold in enumerate(
            thresholds,
            start=1,
        ):
            file.write(
                f'  // {index} -> {threshold}\n'
            )

        for combination in combinations:
            decision_name = 'decision'
            for value in combination:
                decision_name += f'_{value}'

            position_guard = ''
            for value, estimate in zip(
                combination,
                estimates,
            ):
                position_guard += (
                    f' & {estimate}={value}'
                )

            # One compact URC command per position.
            #
            # Example:
            # decision=1 ? threshold_1 :
            # decision=2 ? threshold_2 :
            # ...
            # threshold_10
            #
            # This keeps the decision parameter on the RHS, as in the
            # point-estimate/interval PARLEY encoding, while still mapping
            # decision 1..10 to map-specific belief thresholds.
            ternary_parts = []

            for index, threshold in enumerate(
                thresholds[:-1],
                start=1,
            ):
                ternary_parts.append(
                    f'{decision_name}={index} ? '
                    f'{threshold} : '
                )

            ternary_expression = (
                ''.join(ternary_parts)
                + str(thresholds[-1])
            )

            file.write(
                f'  [URC] true{position_guard} -> '
                f"(max_belief_uncertainty'="
                f"{ternary_expression});\n"
            )

        file.write('endmodule\n')



def __add_controller_prefix(file_path, possible_decisions, combinations, variables, baseline):
    # write decision variables
    with open(file_path, 'a') as file:
        for combination in combinations:
            if baseline:
                new_line = 'const int decision'
            else:
                new_line = 'evolve int decision'
            for var in range(0, len(variables)):
                new_line += '_' + str(combination[var])
            if baseline:
                new_line += '=1;'
            else:
                new_line += f' [{possible_decisions[0]}..{possible_decisions[1]}];'
            file.write('\n' + new_line)
        file.write('\nmodule URC\n')


def add_turn(file_path, before_actions, after_actions):
    with open(file_path, 'a') as file:
        file.write('module Turn\n')
        file.write('  t : [0..2] init 0;\n')
        # actions that precede
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
                remaining_variables[1:]
            )

    generate_combinations_recursive([], variables)
    return result
