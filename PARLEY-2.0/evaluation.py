import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from deap.tools._hypervolume.pyhv import hypervolume   //veraltet
from moocore import hypervolume
from scipy.stats import wilcoxon, anderson, mannwhitneyu
from scipy.stats import t


MAXIMUM_SPREAD_VALUE = 1.5
plt.rcParams.update({'font.size': 16})

def plot_cumulative_map_results(
        gains_data,
        first_map_number,
        ylabel,
        filename
):
    """
    Stellt dar, wie sich der mittlere Gain mit zunehmender
    Anzahl ausgewerteter Maps entwickelt.

    gains_data:
        Liste der Form:
        [
            [rep0, rep1, ..., rep9],  # Map 10
            [rep0, rep1, ..., rep9],  # Map 11
            ...
        ]

    first_map_number:
        Nummer der ersten ausgewerteten Map, z. B. 10.
    """

    gains_array = np.asarray(gains_data, dtype=float)

    if gains_array.ndim != 2:
        raise ValueError(
            "gains_data muss die Form [Maps][Repetitions] haben."
        )

    number_of_maps = gains_array.shape[0]

    # Mittelwert über die Wiederholungen jeder Map
    map_means = np.mean(gains_array, axis=1)

    cumulative_means = []
    lower_bounds = []
    upper_bounds = []

    for number_of_used_maps in range(1, number_of_maps + 1):
        current_map_means = map_means[:number_of_used_maps]

        cumulative_mean = np.mean(current_map_means)
        cumulative_means.append(cumulative_mean)

        # Für nur eine Map kann noch kein Konfidenzintervall
        # aus der Streuung zwischen Maps berechnet werden.
        if number_of_used_maps == 1:
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            continue

        standard_error = (
            np.std(current_map_means, ddof=1)
            / np.sqrt(number_of_used_maps)
        )

        critical_value = t.ppf(
            0.975,
            df=number_of_used_maps - 1
        )

        margin = critical_value * standard_error

        lower_bounds.append(cumulative_mean - margin)
        upper_bounds.append(cumulative_mean + margin)

    number_of_maps_axis = np.arange(1, number_of_maps + 1)

    plt.figure(figsize=(12, 7))

    plt.plot(
        number_of_maps_axis,
        cumulative_means,
        marker="o",
        markersize=3,
        label="Kumulativer Mittelwert"
    )

    plt.fill_between(
        number_of_maps_axis,
        lower_bounds,
        upper_bounds,
        alpha=0.2,
        label="95-%-Konfidenzintervall"
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
        label="Kein Gain"
    )

    # Orientierungslinien bei 20 und 30 Maps
    if number_of_maps >= 20:
        plt.axvline(
            x=20,
            linestyle=":",
            linewidth=1
        )

    if number_of_maps >= 30:
        plt.axvline(
            x=30,
            linestyle=":",
            linewidth=1
        )

    plt.xlabel("Anzahl ausgewerteter Maps")
    plt.ylabel(ylabel)
    plt.xticks(
        np.arange(
            0,
            number_of_maps + 1,
            5
        )
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    plt.savefig(filename)
    plt.close()

    return map_means, np.asarray(cumulative_means)

def random_subset_stability_analysis(
        map_values,
        acceptable_interval,
        sample_sizes=None,
        repetitions=2000,
        random_seed=42,
        output_directory="plots/map-stability"
):
    """
    Untersucht, wie stabil der mittlere Hypervolume-Gain bei
    zufällig ausgewählten Teilmengen der Maps ist.

    map_values:
        Ein mittlerer Hypervolume-Gain pro Map, also hv_map.

    sample_sizes:
        Untersuchte Anzahlen von Maps, z. B.
        [5, 10, 15, 20, 25, 30, ..., 90].

    repetitions:
        Anzahl zufälliger Ziehungen pro Stichprobengröße.

    Es wird ohne Zurücklegen gezogen. Innerhalb einer
    Stichprobe kann dieselbe Map daher nicht doppelt vorkommen.
    """

    values = np.asarray(map_values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "map_values muss genau einen Wert pro Map enthalten."
        )

    number_of_available_maps = len(values)

    if number_of_available_maps < 2:
        raise ValueError(
            "Für die Analyse werden mindestens zwei Maps benötigt."
        )

    if sample_sizes is None:
        sample_sizes = list(
            range(5, number_of_available_maps + 1, 5)
        )

        # Gesamtzahl ergänzen, falls sie nicht durch 5 teilbar ist
        if sample_sizes[-1] != number_of_available_maps:
            sample_sizes.append(number_of_available_maps)

    sample_sizes = [
        size
        for size in sample_sizes
        if 1 <= size <= number_of_available_maps
    ]

    if not sample_sizes:
        raise ValueError(
            "Keine gültigen Stichprobengrößen vorhanden."
        )

    rng = np.random.default_rng(random_seed)

    full_mean = np.mean(values)

    result_sizes = []
    result_means = []
    lower_bounds = []
    upper_bounds = []
    interval_widths = []
    mean_absolute_deviations = []
    standard_deviations = []

    # Optional: alle Ziehungen für spätere Auswertungen speichern
    subset_means_by_size = {}

    for sample_size in sample_sizes:

        # Wenn alle Maps verwendet werden, existiert nur eine
        # mögliche vollständige Stichprobe.
        if sample_size == number_of_available_maps:
            subset_means = np.array([full_mean])
        else:
            subset_means = np.empty(repetitions)

            for repetition in range(repetitions):
                selected_values = rng.choice(
                    values,
                    size=sample_size,
                    replace=False
                )

                subset_means[repetition] = np.mean(
                    selected_values
                )

        subset_means_by_size[sample_size] = subset_means

        mean_of_subset_means = np.mean(subset_means)

        lower = np.percentile(subset_means, 2.5)
        upper = np.percentile(subset_means, 97.5)

        mean_absolute_deviation = np.mean(
            np.abs(subset_means - full_mean)
        )

        result_sizes.append(sample_size)
        result_means.append(mean_of_subset_means)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        interval_widths.append(upper - lower)
        mean_absolute_deviations.append(
            mean_absolute_deviation
        )
        standard_deviations.append(
            np.std(subset_means, ddof=1)
            if len(subset_means) > 1
            else 0.0
        )

    result_sizes = np.asarray(result_sizes)
    result_means = np.asarray(result_means)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    interval_widths = np.asarray(interval_widths)
    mean_absolute_deviations = np.asarray(
        mean_absolute_deviations
    )
    standard_deviations = np.asarray(
        standard_deviations
    )

    os.makedirs(output_directory, exist_ok=True)

    interval_name = (
        f"{acceptable_interval[0]}-"
        f"{acceptable_interval[1]}"
    )

    # Plot 1: Verteilung der geschätzten Mittelwerte
    plt.figure(figsize=(12, 7))

    plt.plot(
        result_sizes,
        result_means,
        marker="o",
        label="Mittel der zufälligen Stichproben"
    )

    plt.fill_between(
        result_sizes,
        lower_bounds,
        upper_bounds,
        alpha=0.2,
        label="Empirisches 95-%-Intervall"
    )

    plt.axhline(
        full_mean,
        linestyle="--",
        linewidth=1,
        label=f"Mittelwert aller Maps: {full_mean:.4f}"
    )

    if 20 in result_sizes:
        plt.axvline(
            20,
            linestyle=":",
            linewidth=1
        )

    if 30 in result_sizes:
        plt.axvline(
            30,
            linestyle=":",
            linewidth=1
        )

    plt.xlabel("Anzahl zufällig ausgewählter Maps")
    plt.ylabel("Mittlerer Hypervolume-Gain")
    plt.title(
        "Stabilität zufälliger Map-Stichproben "
        f"für {interval_name}"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            f"random_subsets_hv_{interval_name}.pdf"
        )
    )

    plt.close()

    # Plot 2: Breite des empirischen Intervalls
    plt.figure(figsize=(12, 7))

    plt.plot(
        result_sizes,
        interval_widths,
        marker="o"
    )

    if 20 in result_sizes:
        plt.axvline(
            20,
            linestyle=":",
            linewidth=1
        )

    if 30 in result_sizes:
        plt.axvline(
            30,
            linestyle=":",
            linewidth=1
        )

    plt.xlabel("Anzahl zufällig ausgewählter Maps")
    plt.ylabel("Breite des empirischen 95-%-Intervalls")
    plt.title(
        "Unsicherheit der Mittelwertschätzung "
        f"für {interval_name}"
    )
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            f"random_subsets_width_{interval_name}.pdf"
        )
    )

    plt.close()

    return {
        "sample_sizes": result_sizes,
        "mean_subset_estimates": result_means,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "interval_widths": interval_widths,
        "standard_deviations": standard_deviations,
        "mean_absolute_deviations": (
            mean_absolute_deviations
        ),
        "full_mean": full_mean,
        "subset_means": subset_means_by_size
    }


def is_dominated(x, y, data):
    for other_x, other_y in data:
        if other_x <= x and other_y <= y:
            return True
        # # verhindert, dass ein Punkt als dominiert gilt, wenn er gleich ist
        # strictly_better = other_x < x and other_y < y
        # no_worse = other_x <= x and other_y <= y
        # if strictly_better and no_worse:
        #     return True
    return False


def filter_dominated_points(data):
    non_dominated_data = []
    for x, y in data:
        if is_dominated(x, y, data):
            non_dominated_data.append((x, y))
    return non_dominated_data


def compute_spread(front_data):
    # Normalize objectives
    front_data = np.array(front_data)
    if any(np.max(front_data, axis=0) - np.min(front_data, axis=0)) == 0:
        return MAXIMUM_SPREAD_VALUE

    normalized_front = (front_data - np.min(front_data, axis=0)) / \
                       (np.max(front_data, axis=0) - np.min(front_data, axis=0))

    # Sort normalized solutions based on the first objective
    sorted_front = normalized_front[np.argsort(normalized_front[:, 0])]

    # Compute Euclidean distances between adjacent solutions
    distances = np.linalg.norm(np.diff(sorted_front, axis=0), axis=1)

    # Calculate spread as the average distance
    spread = np.mean(distances)
    # test for NaN
    if spread != spread:
        return MAXIMUM_SPREAD_VALUE
    return spread


def anderson_darling(umc, baseline):
    differences = np.array(baseline) - np.array(
        umc[0])  # Assuming you are comparing with the first repetition of UMC
    # Anderson-Darling normality test
    statistic, critical_values, significance_level = anderson(differences)
    print(f'Anderson-Darling Statistic: {statistic}')
    print(f'Critical Values: {critical_values}')
    print(f'Significance Level: {significance_level}')

    chosen_significance_level = 0.05
    if statistic < critical_values[2]:  # Index 2 corresponds to the 5% significance level
        print('The differences appear to be normally distributed.')
    else:
        print('The differences do not appear to be normally distributed.')


def perform_wilcoxon_test_against_zero(gains_data, alternative='two-sided'):
    # Perform Wilcoxon signed-rank test against zero
    gains_data = list(map(list, zip(*gains_data)))

    statistic, p_value = wilcoxon(gains_data, alternative=alternative)

    # Output Wilcoxon statistic and p-value
    # print(f'Wilcoxon Statistic: {statistic}')
    # print(f'P-Value: {p_value}')

    # Count statistically significant results
    significant_count = sum(p < 0.05 for p in p_value)

    # Check for statistical significance
    if significant_count == len(p_value):
        print('All gains are statistically different from zero.')
    elif significant_count > 0:
        print(f'{significant_count} out of {len(p_value)} gains are statistically different from zero.')
    else:
        print('There is no significant difference from zero.')


def perform_mann_whitney_u_test(data, alpha=0.05):
    """
    Perform Mann-Whitney U test against zero for each map's gain data.

    Parameters:
    - data: List of lists where each inner list represents the gain data for a map.
    - alpha: Significance level.

    Returns:
    - results: Dictionary containing the results for each map, categorizing as 'better', 'worse', or 'no difference'.
    """
    results = {'higher': 0, 'lower': 0, 'no_difference': 0}

    for map_data in data:
        statistic, p_value = mannwhitneyu(map_data, np.zeros_like(map_data), alternative='two-sided')
        mean_difference = np.mean(map_data)

        if p_value < alpha:
            if mean_difference > 0:
                results['higher'] += 1
            elif mean_difference < 0:
                results['lower'] += 1
        else:
            results['no_difference'] += 1

    return results


def create_selected_box_plots(gains_data, selected_maps, ylabel, title):
    # Extract gains for the selected maps
    gains_selected = [gains_data[i] for i in selected_maps]

    # Create a single plot for the selected gains
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=gains_selected)

    # Add a dashed line at y=0
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xticks(np.arange(0, len(selected_maps), 5))

    plt.xlabel('Map')
    plt.ylabel(ylabel)
    # plt.title(title)
    # plt.legend()  # Add legend to show the zero line
    plt.savefig(f'plots/box-plots/{ylabel}_{title[:1]}_{title[2:]}.pdf')



# Specify the paths to CSV files and the file containing expected values
fronts_dir = 'Applications/EvoChecker-master/data/'

maps = 100

acceptable_intervals = [(0.8, 100), (0.8, 80), (0.8, 60),
                        (0.7, 100), (0.7, 80), (0.7, 60),
                        (0.6, 100), (0.6, 80), (0.6, 60)]


def main():
    for acceptable_interval in acceptable_intervals:
        ref_point = np.array(acceptable_interval)

        hv_map = []
        baseline_hv = []
        baseline_spread = []
        umc_hv = []
        umc_spread = []

        # for each map
        for m in range(10, maps):
            # first let's get the hypervolume for the baseline
            periodic = []
            with open(f'Applications/EvoChecker-master/data/ROBOT{m}_BASELINE/Front', 'r') as file:
                for line in file:
                    x, y = map(float, line.strip().split('	'))
                    if x > acceptable_interval[0] and y < acceptable_interval[1]:
                        periodic.append((1 - x, y))
            periodic = filter_dominated_points(periodic[0:20])
            if len(periodic) == 0:
                hv_periodic = 0
                baseline_spread.append(MAXIMUM_SPREAD_VALUE)
            else:
                hv_periodic = hypervolume(np.array(periodic), ref_point)
                baseline_spread.append(compute_spread(periodic))

            baseline_hv.append(hv_periodic)

            hv_rep = 0
            rep_hv = []
            rep_spread = []
            # for each replication
            for rep in range(0, 10):
                # Read the expected values from the external file (excluding the first line)
                pareto_data = []
                filename = ""
                for filename_ in os.listdir(fronts_dir + 'ROBOT{0}_REP{1}/NSGAII/'.format(str(m), str(rep))):
                    if "Front" in filename_:
                        filename = filename_

                # directory = (
                #     fronts_dir
                #     + f"ROBOT{m}_REP{rep}/NSGAII/"
                # )

                # front_files = [
                #     filename
                #     for filename in os.listdir(directory)
                #     if "Front" in filename
                # ]

                # if not front_files:
                #     raise FileNotFoundError(
                #         f"Keine Front-Datei in {directory} gefunden."
                #     )

                # filename = max(
                #     front_files,
                #     key=lambda name: os.path.getmtime(
                #         os.path.join(directory, name)
                #     )
                # )

                # front_path = os.path.join(directory, filename)

                # print(f"Map {m}, Rep {rep}: verwende {filename}")

                with open(fronts_dir + 'ROBOT{0}_REP{1}/NSGAII/'.format(str(m), str(rep)) + filename, 'r') as f:
                # with open(front_path, 'r') as f:
                    next(f)  # Skip the first line
                    for line in f:
                        values = line.strip().split('\t')
                        if len(values) >= 2 and float(values[0]) > acceptable_interval[0] and float(values[1]) < \
                                acceptable_interval[1]:
                            pareto_data.append((1 - float(values[0]), float(values[1])))

                    # Convert the pareto_data to a NumPy array
                    pareto_array = np.array(filter_dominated_points(pareto_data))
                    # Calculate the hypervolume
                    if len(pareto_array) == 0:
                        hv = 0
                        rep_spread.append(MAXIMUM_SPREAD_VALUE)
                    else:
                        # Sort the Pareto front based on the first objective (probability)
                        pareto_array = pareto_array[np.argsort(pareto_array[:, 0])]
                        hv = hypervolume(np.array(pareto_array), np.array(ref_point))
                        rep_spread.append(compute_spread(pareto_array))
                    hv_rep += hv - hv_periodic
                    rep_hv.append(hv)
            umc_spread.append(rep_spread)
            umc_hv.append(rep_hv)
            hv_map.append(hv_rep / 10)

        # Calculate differences for spread and hypervolume
        spread_gain = [[umc - baseline for umc, baseline in zip(repetition, baseline_spread)] for repetition in umc_spread]
        # spread_gain = [[value - baseline for value in repetition] for repetition, baseline in zip(umc_spread, baseline_spread)]
        hv_gain = [[umc - baseline for umc, baseline in zip(repetition, baseline_hv)] for repetition in umc_hv]
        # hv_gain = [[value - baseline for value in repetition] for repetition, baseline in zip(umc_hv, baseline_hv)]

        # mean_hv_gain_per_map = np.mean(
        #             np.asarray(hv_gain, dtype=float),
        #             axis=1
        #         )
        
        # stability_results = random_subset_stability_analysis(
        #     map_values=mean_hv_gain_per_map,
        #     acceptable_interval=acceptable_interval,
        #     sample_sizes=[
        #         5, 10, 15, 20, 25, 30,
        #         35, 40, 50, 60, 70, 80, 90
        #     ],
        #     repetitions=2000,
        #     random_seed=42
        # )
        
        # print(
        #     "\nZufällige Stichprobenanalyse für "
        #     f"{acceptable_interval}"
        # )

        # for index, sample_size in enumerate(
        #         stability_results["sample_sizes"]
        # ):
        #     if sample_size in [10, 20, 30, 40, 90]:
        #         mean_estimate = (
        #             stability_results[
        #                 "mean_subset_estimates"
        #             ][index]
        #         )

        #         lower = stability_results[
        #             "lower_bounds"
        #         ][index]

        #         upper = stability_results[
        #             "upper_bounds"
        #         ][index]

        #         width = stability_results[
        #             "interval_widths"
        #         ][index]

        #         deviation = stability_results[
        #             "mean_absolute_deviations"
        #         ][index]

        #         print(
        #             f"{sample_size:2d} Maps: "
        #             f"Mittel = {mean_estimate:.4f}, "
        #             f"95-%-Intervall = "
        #             f"[{lower:.4f}, {upper:.4f}], "
        #             f"Breite = {width:.4f}, "
        #             f"mittlere Abweichung vom "
        #             f"90-Map-Mittel = {deviation:.4f}"
        #         )

        # Select the maps shown in the plots (if too many maps)
        selected_maps = range(maps-10)

        # Create box plots for spread gains
        create_selected_box_plots(spread_gain, selected_maps, 'Spread-Gains',
                                  f'{acceptable_interval[0]}-{acceptable_interval[1]}')

        # Create box plots for hypervolume gains
        create_selected_box_plots(hv_gain, selected_maps, 'Hypervolume-Gains',
                                   f'{acceptable_interval[0]}-{acceptable_interval[1]}')
        # perform_wilcoxon_test_against_zero(hv_gain, alternative='greater')
        # perform_wilcoxon_test_against_zero(spread_gain, alternative='less')

        # interval_name = (
        #     f"{acceptable_interval[0]}-"
        #     f"{acceptable_interval[1]}"
        # )

        # hv_map_means, hv_cumulative_means = (
        #     plot_cumulative_map_results(
        #         gains_data=hv_gain,
        #         first_map_number=10,
        #         ylabel="Mittlerer Hypervolume-Gain",
        #         filename=(
        #             "plots/map-stability/"
        #             f"hypervolume_{interval_name}.pdf"
        #         )
        #     )
        # )

        # spread_map_means, spread_cumulative_means = (
        #     plot_cumulative_map_results(
        #         gains_data=spread_gain,
        #         first_map_number=10,
        #         ylabel="Mittlerer Spread-Gain",
        #         filename=(
        #             "plots/map-stability/"
        #             f"spread_{interval_name}.pdf"
        #         )
        #     )
        # )

        print(perform_mann_whitney_u_test(spread_gain))
        print(perform_mann_whitney_u_test(hv_gain))

if __name__ == '__main__':
    main()
