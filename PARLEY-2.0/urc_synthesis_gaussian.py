import os
import re
import shutil


def manipulate_prism_model(
    input_path,
    output_path,
    possible_decisions=[1, 10],
    decision_variables=[],
    before_actions=[
        "east",
        "west",
        "north",
        "south",
    ],
    after_actions=[
        "update",
        "skip_update",
    ],
    module_name="Knowledge",
    baseline=False,
):
    """
    Gaussian representative-state URC.

    This intentionally mirrors urc_synthesis_belief_full.py:
      * position-dependent decision_x_y in [1..10]
      * each decision selects one map-specific Gaussian trace threshold
      * one compact ternary [URC] command per position
    """
    if (
        os.path.abspath(input_path)
        == os.path.abspath(output_path)
    ):
        raise ValueError(
            "Input and output files cannot be the same."
        )

    shutil.copyfile(
        input_path,
        output_path,
    )

    variables, estimates = get_variables(
        input_path,
        decision_variables,
    )

    remove_counter_from_module(
        output_path
    )

    add_controller(
        output_path,
        estimates,
        variables,
        possible_decisions,
        baseline=baseline,
    )

    add_turn(
        output_path,
        before_actions,
        after_actions,
    )


def get_variables(
    prism_model_path,
    decision_variables,
):
    int_constants_pattern = re.compile(
        r"const\s+int\s+(\w+)\s*=\s*(-?\s*\d+)\s*;"
    )

    int_constants = {}

    with open(
        prism_model_path,
        "r",
    ) as prism_model_file:
        for line in prism_model_file:
            for match in (
                int_constants_pattern.finditer(
                    line
                )
            ):
                int_constants[
                    match.group(1)
                ] = int(
                    match.group(2).replace(
                        " ",
                        "",
                    )
                )

    variable_pattern = re.compile(
        r"(\w+)\s*:\s*\[(-?\s*\w+)\s*\.\.\s*(-?\s*\w+)\]"
        r"\s*init\s*(-?\s*\w+)\s*;"
    )

    variables = []
    estimates = []

    with open(
        prism_model_path,
        "r",
    ) as prism_model_file:
        for line in prism_model_file:
            for match in (
                variable_pattern.finditer(
                    line
                )
            ):
                name = match.group(1)

                if name.endswith(
                    "hat"
                ):
                    estimates.append(
                        name
                    )
                elif (
                    name
                    not in decision_variables
                ):
                    continue

                lower = _get_limit(
                    match.group(2).replace(
                        " ",
                        "",
                    ),
                    int_constants,
                )
                upper = _get_limit(
                    match.group(3).replace(
                        " ",
                        "",
                    ),
                    int_constants,
                )

                variables.append(
                    [
                        name,
                        lower,
                        upper,
                    ]
                )

    return (
        variables,
        estimates,
    )


def _get_limit(
    string,
    constants,
):
    if not string.lstrip(
        "-"
    ).isdigit():
        return constants[
            string
        ]

    return int(
        string
    )


def remove_counter_from_module(
    output_path,
):
    """
    Remove legacy numeric max_gaussian_uncertainty declarations if present.
    New models use only gaussian_threshold_level in [1..10].
    """
    pattern = re.compile(
        r"^\s*(?:"
        r"(?:const\s+int\s+)?max_gaussian_uncertainty"
        r"(?:\s*:\s*\[\d+\.\.\d+\]\s*init\s*\d+|\s*=\s*\d+)"
        r"|const\s+int\s+gaussian_threshold_level\s*=\s*\d+"
        r")\s*;"
    )

    with open(
        output_path,
        "r",
    ) as file:
        lines = file.readlines()

    lines = [
        line
        for line in lines
        if not pattern.match(
            line
        )
    ]

    with open(
        output_path,
        "w",
    ) as file:
        file.writelines(
            lines
        )


def get_gaussian_thresholds(
    file_path,
):
    pattern = re.compile(
        r"^\s*const\s+int\s+"
        r"gaussian_threshold_(\d+)"
        r"\s*=\s*(\d+)\s*;"
    )

    thresholds = {}

    with open(
        file_path,
        "r",
    ) as file:
        for line in file:
            match = pattern.match(
                line
            )

            if match:
                thresholds[
                    int(
                        match.group(1)
                    )
                ] = int(
                    match.group(2)
                )

    if len(thresholds) != 10:
        raise ValueError(
            "Expected gaussian_threshold_1.."
            "gaussian_threshold_10 in PRISM model."
        )

    return [
        thresholds[index]
        for index in range(
            1,
            11,
        )
    ]


def add_controller(
    file_path,
    estimates,
    variables,
    possible_decisions,
    baseline,
):
    combinations = (
        generate_combinations_list(
            variables
        )
    )

    _add_controller_prefix(
        file_path,
        possible_decisions,
        combinations,
        variables,
        baseline,
    )

    with open(
        file_path,
        "a",
    ) as file:
        file.write(
            "  gaussian_threshold_level : [1..10] init 1;\n"
        )
        file.write(
            "  // decision_x_y directly selects threshold level 1..10\n"
        )

        for combination in combinations:
            decision_name = "decision"

            for value in combination:
                decision_name += f"_{value}"

            position_guard = ""

            for (
                value,
                estimate,
            ) in zip(
                combination,
                estimates,
            ):
                position_guard += (
                    f" & {estimate}={value}"
                )

            # Keep the compact ternary form used in the other URC models.
            # The result is still only a LEVEL 1..10, so no large numeric
            # threshold range becomes part of the state space.
            level_expression = "".join(
                f"{decision_name}={level} ? {level} : "
                for level in range(1, 10)
            ) + "10"

            file.write(
                f"  [URC] true{position_guard} -> "
                f"(gaussian_threshold_level'={level_expression});\n"
            )

        file.write(
            "endmodule\n"
        )


def _add_controller_prefix(
    file_path,
    possible_decisions,
    combinations,
    variables,
    baseline,
):
    with open(
        file_path,
        "a",
    ) as file:
        for combination in combinations:
            if baseline:
                new_line = (
                    "const int decision"
                )
            else:
                new_line = (
                    "evolve int decision"
                )

            for index in range(
                len(variables)
            ):
                new_line += (
                    "_"
                    + str(
                        combination[index]
                    )
                )

            if baseline:
                new_line += "=1;"
            else:
                new_line += (
                    f" [{possible_decisions[0]}"
                    f"..{possible_decisions[1]}];"
                )

            file.write(
                "\n"
                + new_line
            )

        file.write(
            "\nmodule URC\n"
        )


def add_turn(
    file_path,
    before_actions,
    after_actions,
):
    with open(
        file_path,
        "a",
    ) as file:
        file.write(
            "module Turn\n"
        )
        file.write(
            "  t : [0..2] init 0;\n"
        )

        for action in before_actions:
            file.write(
                f"  [{action}] (t=0) -> (t'=1);\n"
            )

        file.write(
            "\n"
            "  [URC] (t=1) -> (t'=2);\n"
            "\n"
        )

        for action in after_actions:
            file.write(
                f"  [{action}] (t=2) -> (t'=0);\n"
            )

        if len(
            after_actions
        ) == 0:
            file.write(
                "  [] (t=2) -> (t'=0);\n"
            )

        file.write(
            "endmodule\n"
        )


def generate_combinations_list(
    variables,
):
    result = []

    def recurse(
        current,
        remaining,
    ):
        if not remaining:
            result.append(
                tuple(current)
            )
            return

        current_variable = (
            remaining[0]
        )

        for value in range(
            current_variable[1],
            current_variable[2] + 1,
        ):
            recurse(
                current + [value],
                remaining[1:],
            )

    recurse(
        [],
        variables,
    )

    return result
