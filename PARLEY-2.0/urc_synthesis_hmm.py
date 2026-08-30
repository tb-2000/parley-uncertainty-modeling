import os
import re
import shutil


def manipulate_prism_model(
    input_path,
    output_path,
    possible_decisions=[1, 10],
    decision_variables=[],
    before_actions=['east', 'west', 'north', 'south'],
    after_actions=['update', 'skip_update'],
    module_name='Knowledge',
    baseline=False,
):
    """
    Add the PARLEY/EvoChecker URC to a PRISM model.

    HMM semantics
    -------------
    If the input model contains an exact reachable-belief variable

        hstate : [0..HMM_STATES-1] ...

    hstate is deliberately NOT made a decision dimension.  The URC still
    synthesizes one threshold decision per estimated robot position
    (decision_x_y in the usual setup).  The Knowledge module itself compares
    the selected threshold c against the HMM uncertainty level derived from
    hstate.

    Thus:

        decision_x_y = k

    means "at estimated position (x,y), localize when the exact HMM belief
    reaches uncertainty level k".

    This keeps the policy comparable with the point-estimate / Gaussian URCs
    and avoids a combinatorial decision variable decision_x_y_hstate.
    """
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Input and output files cannot be the same.")

    shutil.copyfile(input_path, output_path)

    hmm_mode = has_hmm_state(input_path)

    # hstate is the knowledge state used to evaluate uncertainty, not a
    # synthesis dimension.  Ignore it even if a caller accidentally includes
    # it in decision_variables.
    effective_decision_variables = [
        name for name in decision_variables if name != 'hstate'
    ]

    variables, estimates = get_variables(
        input_path,
        effective_decision_variables
    )

    if hmm_mode and 'hstate' in decision_variables:
        print(
            "HMM: ignoring 'hstate' as a URC decision dimension; "
            "the URC remains position-dependent and hstate determines "
            "when the selected threshold is reached."
        )

    remove_counter_from_module(output_path)

    add_controller(
        output_path,
        estimates,
        variables,
        possible_decisions,
        baseline=baseline,
    )

    add_turn(output_path, before_actions, after_actions)


def has_hmm_state(prism_model_path):
    pattern = re.compile(r'\bhstate\s*:\s*\[')
    with open(prism_model_path, 'r') as prism_model_file:
        return any(pattern.search(line) for line in prism_model_file)


def get_variables(prism_model_path, decision_variables):
    """
    Return:
        variables : domains used to generate decision_x_y combinations
        estimates : corresponding estimate variables used in URC guards

    Backward-compatible robot behavior:
    -----------------------------------
    manipulate_prism_model(...) historically works without explicitly passing
    decision_variables=['x','y'].

    Therefore, if decision_variables is empty, infer the decision dimensions
    automatically from estimate variables:
        xhat -> x
        yhat -> y

    hstate is never inferred as a decision dimension.
    """
    int_constants_pattern = re.compile(
        r'const\s+int\s+(\w+)\s*=\s*(-?\s*\d+)\s*;'
    )
    int_constants = {}

    with open(prism_model_path, 'r') as prism_model_file:
        for line in prism_model_file:
            matches = int_constants_pattern.finditer(line)
            for match in matches:
                int_constants[match.group(1)] = int(
                    match.group(2).replace(" ", "")
                )

    int_variable_declaration_pattern = re.compile(
        r'(\w+)\s*:\s*\[(-?\s*\w+)\s*\.\.\s*(-?\s*\w+)\]'
        r'\s*init\s*(-?\s*\w+)\s*;'
    )

    declarations = {}

    with open(prism_model_path, 'r') as prism_model_file:
        for line in prism_model_file:
            for match in int_variable_declaration_pattern.finditer(line):
                name = match.group(1)
                lower = __get_limit(
                    match.group(2).replace(" ", ""),
                    int_constants,
                )
                upper = __get_limit(
                    match.group(3).replace(" ", ""),
                    int_constants,
                )
                declarations[name] = [name, lower, upper]

    # Preserve estimate order from the PRISM declarations.
    estimates = [
        name for name in declarations
        if name.endswith("hat")
    ]

    if decision_variables:
        requested = [
            name for name in decision_variables
            if name != "hstate"
        ]
    else:
        # Infer x/y from xhat/yhat, etc.
        requested = []
        for estimate in estimates:
            base = estimate[:-3]  # remove "hat"
            if base in declarations:
                requested.append(base)

    variables = []
    paired_estimates = []

    for estimate in estimates:
        base = estimate[:-3]

        if base in requested:
            if base not in declarations:
                raise ValueError(
                    f"Estimate variable '{estimate}' implies decision "
                    f"dimension '{base}', but '{base}' is not declared."
                )

            variables.append(declarations[base])
            paired_estimates.append(estimate)

    if not variables:
        raise ValueError(
            "Could not infer any URC decision dimensions. "
            "Expected robot variables such as x/y together with xhat/yhat, "
            "or pass decision_variables explicitly."
        )

    if len(variables) != len(paired_estimates):
        raise ValueError(
            "URC decision-variable/estimate mismatch: "
            f"decision variables={variables}, estimates={paired_estimates}."
        )

    return variables, paired_estimates


def __get_limit(string, constants):
    if not string.lstrip("-").isdigit():
        if string not in constants:
            raise KeyError(
                f"Unknown integer bound/constant '{string}' in PRISM model."
            )
        return constants[string]
    else:
        return int(string)


def remove_counter_from_module(output_path):
    """
    Remove the fixed baseline period constant `c`.

    For synthesized models c becomes the URC variable in module URC.
    This works for the old step model as well as for the exact-HMM model.
    """
    pattern = re.compile(r"^\s*const\s+int\s+c\d*\s*=\s*\d+\s*;")
    with open(output_path, 'r') as file:
        lines = file.readlines()

    new_lines = [line for line in lines if not pattern.match(line)]

    with open(output_path, 'w') as file:
        file.writelines(new_lines)


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

    with open(file_path, 'a') as file:
        file.write('  c : [1..10] init 1;\n')

        for combination in combinations:
            new_line = '  [URC] true'
            for value, estimate in zip(combination, estimates):
                new_line += f' & {estimate}={value}'

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
    baseline,
):
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
                new_line += (
                    f' [{possible_decisions[0]}..'
                    f'{possible_decisions[1]}];'
                )

            file.write('\n' + new_line)

        file.write('\nmodule URC\n')


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

    def generate_combinations_recursive(
        current_combination,
        remaining_variables,
    ):
        if not remaining_variables:
            result.append(tuple(current_combination))
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

    generate_combinations_recursive([], variables)
    return result
