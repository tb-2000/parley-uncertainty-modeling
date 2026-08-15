import os
import re
import shutil
from interval_thresholds_per_map import THRESHOLDS_PER_MAP


def manipulate_prism_model(input_path, output_path, possible_decisions=[1, 10], decision_variables=[],
                           before_actions=['east', 'west', 'north', 'south'], after_actions=['update', 'skip_update'], module_name='Knowledge', baseline=False):
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output files cannot be the same.")

    shutil.copyfile(input_path, output_path)

    variables, estimates = get_variables(input_path, decision_variables)

    remove_counter_from_module(output_path)
    add_max_update_distance(output_path)

    map_match = re.search(r'model_(\d+)', os.path.basename(input_path))
    if not map_match:
        raise ValueError(f'Could not determine map number from {input_path}')
    thresholds = THRESHOLDS_PER_MAP[int(map_match.group(1))]

    add_controller(output_path, estimates, variables, possible_decisions, thresholds, baseline=baseline)

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
    # The URC replaces the fixed interval-width threshold.
    pattern = re.compile(r"^\s*const\s+int\s+max_interval_width\s*=\s*\d+\s*;")
    with open(output_path, 'r') as file:
        lines = file.readlines()

    # Filter out lines that match the regex pattern
    new_lines = [line for line in lines if not pattern.match(line)]

    # Write the modified content back to the file
    with open(output_path, 'w') as file:
        file.writelines(new_lines)



def add_max_update_distance(output_path):
    # Interval width remains the primary trigger. The Knowledge module also
    # stores how many movements occurred since the last update and forces an
    # update after at most 10 movements.
    update_formula_pattern = re.compile(
        r"^\s*formula\s+update_required\s*=\s*interval_width\s*>=\s*max_interval_width\s*;\s*$"
    )
    ready_declaration_pattern = re.compile(
        r"^\s*ready\s*:\s*\[0\.\.1\]\s*init\s*1\s*;\s*$"
    )
    movement_pattern = re.compile(
        r"^\s*\[(west|east|south|north)\]\s+ready=1\s*->"
    )
    update_pattern = re.compile(
        r"^\s*\[update\]\s+update_required\s*&\s*ready=0\s*->"
    )

    with open(output_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    in_knowledge = False
    movement_command = None
    in_update_command = False
    inserted_counter = False
    replaced_formula = False

    for line in lines:
        stripped = line.strip()

        if stripped == "module Knowledge":
            in_knowledge = True

        if update_formula_pattern.match(line):
            new_lines.append(
                "formula update_required = "
                "interval_width>=max_interval_width | steps_since_update>=10;\n"
            )
            replaced_formula = True
            continue

        if in_knowledge and ready_declaration_pattern.match(line):
            new_lines.append(line)
            new_lines.append(
                "  steps_since_update : [0..10] init 0;\n"
            )
            inserted_counter = True
            continue

        if in_knowledge and movement_pattern.match(line):
            movement_command = True
            new_lines.append(line)
            continue

        if in_knowledge and movement_command:
            # Insert before the final (ready'=0); line of each movement command.
            if "(ready'=0);" in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(
                    indent
                    + "(steps_since_update'=min(steps_since_update+1,10)) &\n"
                )
                new_lines.append(line)
                movement_command = None
            else:
                new_lines.append(line)
            continue

        if in_knowledge and update_pattern.match(line):
            in_update_command = True
            new_lines.append(line)
            continue

        if in_knowledge and in_update_command:
            # Insert reset before the final (ready'=1); line of [update].
            if "(ready'=1);" in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(
                    indent + "(steps_since_update'=0) &\n"
                )
                new_lines.append(line)
                in_update_command = False
            else:
                new_lines.append(line)
            continue

        new_lines.append(line)

        if in_knowledge and stripped == "endmodule":
            in_knowledge = False

    if not replaced_formula:
        raise ValueError(
            "Could not find the expected update_required formula in "
            f"{output_path}"
        )
    if not inserted_counter:
        raise ValueError(
            "Could not insert steps_since_update into Knowledge module in "
            f"{output_path}"
        )

    with open(output_path, 'w') as file:
        file.writelines(new_lines)



def add_controller(file_path, estimates, variables, possible_decisions, thresholds, baseline):
    combinations = generate_combinations_list(variables)
    __add_controller_prefix(file_path, possible_decisions, combinations, variables, baseline)
    with open(file_path, 'a') as file:
        # Ten decisions map to the thresholds calibrated for this map.
        file.write(
            f'  max_interval_width : [{min(thresholds)}..{max(thresholds)}] '
            f'init {thresholds[0]};\n'
        )

        for combination in combinations:
            # combination describes a tuple of values, e.g., for ^x and ^y, such as (0, 0)
            #new_line = '  [URC] 1=1' ##Fehler?
            new_line = '  [URC] true'
            for c, estimate in zip(combination, estimates):
                # estimate describes the variable's name
                new_line += f' & {estimate}={c}'
            decision_name = 'decision'
            for c in combination:
                decision_name += f'_{c}'
            threshold_expression = " : ".join(
                f"{decision_name}={decision} ? {threshold}"
                for decision, threshold in enumerate(thresholds[:-1], start=1)
            ) + f" : {thresholds[-1]}"
            new_line += (
                f" -> (max_interval_width'={threshold_expression});\n"
            )
            file.write(new_line)
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
