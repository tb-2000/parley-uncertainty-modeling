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
    Exact reachable Gaussian bias+covariance-state URC using MSE thresholds.

    This intentionally mirrors urc_synthesis_belief_full.py:
      * position-dependent decision_x_y in [1..10]
      * each decision selects one map-specific Gaussian MSE threshold
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
    """
    Return the virtual xhat/yhat controller dimensions.

    In compact Gaussian models, xhat/yhat are encoded in gstate and exposed
    through formulas estimate_X_Y. We reconstruct the same 10x10 URC policy
    dimensions from const int N.
    """
    int_constants_pattern = re.compile(
        r"const\s+int\s+(\w+)\s*=\s*(-?\s*\d+)\s*;"
    )

    with open(
        prism_model_path,
        "r",
    ) as prism_model_file:
        text = prism_model_file.read()

    int_constants = {}
    for match in int_constants_pattern.finditer(text):
        int_constants[
            match.group(1)
        ] = int(
            match.group(2).replace(
                " ",
                "",
            )
        )

    # New compact representation.
    if re.search(r"\bgstate\s*:", text) and not re.search(
        r"\bxhat\s*:",
        text,
    ):
        if "N" not in int_constants:
            raise ValueError(
                "Compact Gaussian model is missing const int N."
            )

        n = int_constants["N"]

        for x in range(n + 1):
            for y in range(n + 1):
                if not re.search(
                    rf"^\s*formula\s+estimate_{x}_{y}\s*=",
                    text,
                    re.MULTILINE,
                ):
                    raise ValueError(
                        f"Missing estimate_{x}_{y} formula."
                    )

        return (
            [
                ["xhat", 0, n],
                ["yhat", 0, n],
            ],
            ["xhat", "yhat"],
        )

    # Backward-compatible old explicit-xhat/yhat representation.
    variable_pattern = re.compile(
        r"(\w+)\s*:\s*\[(-?\s*\w+)\s*\.\.\s*(-?\s*\w+)\]"
        r"\s*init\s*(-?\s*\w+)\s*;"
    )

    variables = []
    estimates = []

    for match in variable_pattern.finditer(text):
        name = match.group(1)

        if name.endswith("hat"):
            estimates.append(name)
        elif name not in decision_variables:
            continue

        lower = _get_limit(
            match.group(2).replace(" ", ""),
            int_constants,
        )
        upper = _get_limit(
            match.group(3).replace(" ", ""),
            int_constants,
        )

        variables.append([
            name,
            lower,
            upper,
        ])

    return variables, estimates


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
    pattern = re.compile(
        r"^\s*const\s+int\s+max_gaussian_uncertainty\s*=\s*\d+\s*;"
    )

    with open(
        output_path,
        "r",
    ) as file:
        lines = file.readlines()

    new_lines = [
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
            new_lines
        )


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
        # Exactly like the point-estimate period variable: the mutable URC
        # state contains only the selected level 1..10, never the raw
        # MSE value.
        file.write(
            "  max_gaussian_uncertainty : [1..10] init 1;\n"
        )
        file.write(
            "  // Levels 1..10 refer to the raw MSE thresholds "
            "documented at the top of this model.\n"
        )

        for combination in combinations:
            decision_name = "decision"

            for value in combination:
                decision_name += (
                    f"_{value}"
                )

            # Compact models encode xhat/yhat inside gstate.
            # Keep the same position-dependent decision_x_y policy by using
            # the generated estimate_X_Y formulas.
            if estimates == ["xhat", "yhat"]:
                x, y = combination
                position_guard = f"estimate_{x}_{y}"
            else:
                parts = [
                    f"{estimate}={value}"
                    for value, estimate in zip(
                        combination,
                        estimates,
                    )
                ]
                position_guard = (
                    " & ".join(parts)
                    if parts
                    else "true"
                )

            file.write(
                f"  [URC] {position_guard} -> "
                f"(max_gaussian_uncertainty'={decision_name});\n"
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
