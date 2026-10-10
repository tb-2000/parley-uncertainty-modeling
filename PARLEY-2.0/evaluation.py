import os
import csv
import re
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from deap.tools._hypervolume.pyhv import hypervolume   //veraltet
from moocore import hypervolume
from scipy.stats import wilcoxon, anderson, mannwhitneyu
from scipy.stats import t


def newest_front_set_pair(directory):
    """Neueste vollständige EvoChecker-Version nach Zeitstempel im Dateinamen."""
    pairs = []
    for name in os.listdir(directory):
        if not name.endswith('_Front'):
            continue
        prefix = name[:-len('_Front')]
        set_name = prefix + '_Set'
        if not os.path.isfile(os.path.join(directory, set_name)):
            continue
        match = re.search(r'_(\d{6})_(\d{6})$', prefix)
        if not match:
            continue
        try:
            timestamp = datetime.strptime(match.group(2) + match.group(1), '%d%m%y%H%M%S')
        except ValueError:
            continue
        pairs.append((timestamp, name, set_name))
    if not pairs:
        raise FileNotFoundError(f'Kein vollständiges, datiertes Front-/Set-Paar in {directory}')
    _, front, set_file = max(pairs, key=lambda item: (item[0], item[1]))
    return front, set_file


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
    """Minimierung beider Ziele; ein anderer Punkt muss strikt besser sein."""
    return any((ox <= x and oy <= y) and (ox < x or oy < y)
               for ox, oy in data)


def filter_dominated_points(data):
    """Entfernt dominierte Punkte und Duplikate."""
    unique = list(dict.fromkeys(tuple(p) for p in data))
    return [p for p in unique if not is_dominated(p[0], p[1], unique)]


def compute_spread(front_data):
    # Normalize objectives
    front_data = np.array(front_data)
    if len(front_data) < 2 or np.any(np.ptp(front_data, axis=0) == 0):
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



def success_probability_coverage(front):
    """Maximale minus minimale Erfolgswahrscheinlichkeit der akzeptierten Front.

    Ein Punkt: Abdeckung 0; keine akzeptierten Punkte: nicht definiert (NaN).
    Die Eingabe enthält (1 - Erfolg, Kosten).
    """
    if not front:
        return np.nan
    probabilities = 1.0 - np.asarray(front, dtype=float)[:, 0]
    return float(np.max(probabilities) - np.min(probabilities))


def load_current_front(map_id, repetition, min_success, max_cost):
    directory = os.path.join(fronts_dir, f'ROBOT{map_id}_REP{repetition}', 'NSGAII')
    filename, set_filename = newest_front_set_pair(directory)
    print(f'Map {map_id}, Rep {repetition}: {filename} | {set_filename}')
    points = []
    with open(os.path.join(directory, filename), encoding='utf-8') as f:
        next(f)
        for line in f:
            values = line.split()
            if len(values) >= 2:
                probability, cost = float(values[0]), float(values[1])
                if probability > min_success and cost < max_cost:
                    points.append((1.0 - probability, cost))
    return filter_dominated_points(points)


def load_periodic_front(map_id, min_success, max_cost):
    points = []
    path = os.path.join(fronts_dir, f'ROBOT{map_id}_BASELINE', 'Front')
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            if len(values) >= 2:
                probability, cost = float(values[0]), float(values[1])
                if probability > min_success and cost < max_cost:
                    points.append((1.0 - probability, cost))
    return filter_dominated_points(points)


def per_map_diversity(periodic, repetitions):
    """Originale PARLEY-Spread-Kennzahl und Erfolgswahrscheinlichkeits-Abdeckung.

    Spread: frontweise Min-Max-Normalisierung und mittlerer euklidischer
    Abstand benachbarter Punkte. Bei <2 Punkten oder konstanter Zielgröße
    wird wie in der GitHub-Evaluation der Ersatzwert 1.5 verwendet.
    """
    fronts = [periodic] + repetitions
    spreads = [compute_spread(front) for front in fronts]
    coverages = [success_probability_coverage(front) for front in fronts]
    return spreads, coverages


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
    """gains_data ist in derselben Reihenfolge wie selected_maps angeordnet."""
    if len(gains_data) != len(selected_maps):
        raise ValueError(f"{len(gains_data)} Datensätze für {len(selected_maps)} Karten")
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=gains_data)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xticks(range(len(selected_maps)), selected_maps, rotation=90)
    plt.xlabel('Map')
    plt.ylabel(ylabel)
    plt.tight_layout()
    os.makedirs('plots/box-plots', exist_ok=True)
    plt.savefig(f'plots/box-plots/{ylabel}_{title}.pdf')
    plt.close()


# Specify the paths to CSV files and the file containing expected values
fronts_dir = 'Applications/EvoChecker-master/data/'

SELECTED_MAPS = [
    14, 21, 23, 30, 31, 32, 40, 43, 44, 46,
    47, 48, 49, 50, 54, 55, 56, 57, 63, 66,
    71, 75, 81, 82, 83, 85, 87, 89, 90, 97
]
REPETITIONS = 10

# (minimale Erfolgswahrscheinlichkeit, maximale Kosten)
acceptable_intervals = [(0.8, 100), (0.8, 80), (0.8, 60),
                        (0.7, 100), (0.7, 80), (0.7, 60),
                        (0.6, 100), (0.6, 80), (0.6, 60)]


def summarize_gains_by_map(gains, metric, min_success, max_cost):
    """Mapweise Wilcoxon-Tests (10 Repetitionen) und globaler Test (30 Map-Mittelwerte).

    Positive HV-/COV-Gains und negative PARLEY-Spread-Gains sind besser.
    Die mapweisen p-Werte sind explorativ und werden innerhalb der 30 Maps
    mit Benjamini-Hochberg korrigiert.
    """
    values = np.asarray(gains, dtype=float)
    if values.shape != (len(SELECTED_MAPS), REPETITIONS):
        raise ValueError(f'Unerwartete Gain-Form: {values.shape}')
    per_map = []
    for map_id, row in zip(SELECTED_MAPS, values):
        valid = row[np.isfinite(row)]
        p = np.nan
        if len(valid) >= 2 and np.any(valid != 0):
            p = float(wilcoxon(valid, alternative='two-sided', method='auto').pvalue)
        per_map.append({'map': map_id, 'metric': metric,
                        'min_success': min_success, 'max_cost': max_cost,
                        'n_reps': len(valid), 'mean_gain': float(np.mean(valid)) if len(valid) else np.nan,
                        'median_gain': float(np.median(valid)) if len(valid) else np.nan,
                        'p_raw': p, 'p_bh': np.nan})

    # Benjamini-Hochberg über die 30 mapweisen Tests eines Zielbereichs.
    indexed = [(i, r['p_raw']) for i, r in enumerate(per_map) if np.isfinite(r['p_raw'])]
    indexed.sort(key=lambda x: x[1])
    ntests = len(indexed)
    previous = 1.0
    for rank in range(ntests, 0, -1):
        idx, pval = indexed[rank - 1]
        previous = min(previous, pval * ntests / rank)
        per_map[idx]['p_bh'] = previous

    better = worse = 0
    for row in per_map:
        if np.isfinite(row['p_bh']) and row['p_bh'] < 0.05:
            if (row['mean_gain'] > 0 and metric in ('HV', 'COV')) or (row['mean_gain'] < 0 and metric == 'SP'):
                better += 1
            elif row['mean_gain'] != 0:
                worse += 1
    inconclusive = len(per_map) - better - worse

    map_means = np.asarray([r['mean_gain'] for r in per_map], dtype=float)
    valid_means = map_means[np.isfinite(map_means)]
    n = len(valid_means)
    mean = float(np.mean(valid_means)) if n else np.nan
    median = float(np.median(valid_means)) if n else np.nan
    if n >= 2:
        margin = float(t.ppf(0.975, n - 1) * np.std(valid_means, ddof=1) / np.sqrt(n))
        ci_low, ci_high = mean - margin, mean + margin
    else:
        ci_low = ci_high = np.nan
    global_p = float(wilcoxon(valid_means, alternative='two-sided').pvalue) if n and np.any(valid_means != 0) else np.nan
    summary = {'metric': metric, 'min_success': min_success, 'max_cost': max_cost,
               'better': better, 'worse': worse, 'not_significant': inconclusive,
               'n_maps': n, 'mean_gain': mean, 'median_gain': median,
               'ci95_low': ci_low, 'ci95_high': ci_high, 'wilcoxon_maps_p': global_p,
               'positive_map_means': int(np.sum(valid_means > 0)),
               'negative_map_means': int(np.sum(valid_means < 0))}
    return summary, per_map


def write_statistical_tables(summaries, map_rows, output_dir='plots/statistics'):
    os.makedirs(output_dir, exist_ok=True)
    for filename, rows in [('summary.csv', summaries), ('per_map_tests.csv', map_rows)]:
        with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # Tabelle 1 des Papers: Erfolgsschwelle x Kostenlimit, HV/SP,
    # Einträge signifikant besser / schlechter / nicht signifikant.
    lookup = {(r['metric'], r['min_success'], r['max_cost']): r for r in summaries}
    with open(os.path.join(output_dir, 'paper_table.md'), 'w', encoding='utf-8') as f:
        f.write('| min_success | Metrik | max_cost 60 | max_cost 80 | max_cost 100 |\n')
        f.write('|---|---|---|---|---|\n')
        for threshold in (0.6, 0.7, 0.8):
            for metric in ('HV', 'SP', 'COV'):
                cells = []
                for cost in (60, 80, 100):
                    r = lookup[(metric, threshold, cost)]
                    cells.append(f"{r['better']}/{r['worse']}/{r['not_significant']}")
                f.write(f'| {threshold:.0%} | {metric} | ' + ' | '.join(cells) + ' |\n')
        f.write('\nReihenfolge: signifikant besser / signifikant schlechter / nicht signifikant.\n')
        f.write('Mapweise Wilcoxon-Tests mit BH-Korrektur innerhalb der 30 Karten je Zielbereich.\n')
        f.write('SP = originaler PARLEY-Spread (mittlerer normalisierter Nachbarabstand; kleiner = dichter; Ersatzwert 1.5 bei ungueltiger Front); COV = Spannweite der Erfolgswahrscheinlichkeit (groesser = breiter).\n')

    with open(os.path.join(output_dir, 'paper_table.tex'), 'w', encoding='utf-8') as f:
        f.write('\\begin{tabular}{llccc}\n\\hline\n')
        f.write('min\\_success & Metrik & max\\_cost 60 & 80 & 100 \\\\ \n\\hline\n')
        for threshold in (0.6, 0.7, 0.8):
            for metric in ('HV', 'SP', 'COV'):
                cells = []
                for cost in (60, 80, 100):
                    r = lookup[(metric, threshold, cost)]
                    cells.append(f"{r['better']}/{r['worse']}/{r['not_significant']}")
                f.write(f'{threshold:.0%} & {metric} & ' + ' & '.join(cells) + ' \\\\ \n')
        f.write('\\hline\n\\end{tabular}\n')


def main():
    all_summaries = []
    all_map_rows = []
    for acceptable_interval in acceptable_intervals:
        ref_point = np.array((1 - acceptable_interval[0], acceptable_interval[1]))

        hv_map = []
        hv_gain = []
        spread_gain = []
        coverage_gain = []
        # Spread wie im Original PARLEY frontweise normalisiert; Coverage unverändert.
        for m in SELECTED_MAPS:
            periodic = load_periodic_front(m, *acceptable_interval)
            fronts = [load_current_front(m, rep, *acceptable_interval)
                      for rep in range(REPETITIONS)]
            hv_periodic = (float(hypervolume(np.asarray(periodic), ref_point))
                           if periodic else 0.0)
            hv_reps = [(float(hypervolume(np.asarray(front), ref_point))
                        if front else 0.0) for front in fronts]
            spreads, coverages = per_map_diversity(periodic, fronts)
            hv_row = [v - hv_periodic for v in hv_reps]
            sp_row = [v - spreads[0] for v in spreads[1:]]
            cov_row = [v - coverages[0] for v in coverages[1:]]
            hv_gain.append(hv_row)
            spread_gain.append(sp_row)
            coverage_gain.append(cov_row)
            hv_map.append(float(np.mean(hv_row)))

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

        fewer_maps = [14, 21, 23, 30, 31, 32, 40, 43, 44, 46, 47, 48, 49, 50, 
             54, 55, 56, 57, 63, 66, 71, 75, 81, 82, 83, 85, 87, 89, 90, 97]
        # Select the maps shown in the plots (if too many maps)
        selected_maps = SELECTED_MAPS

        # Create box plots for spread gains
        create_selected_box_plots(spread_gain, selected_maps, 'Spread-Gains',
                                  f'{acceptable_interval[0]}-{acceptable_interval[1]}')

        # Create box plots for hypervolume gains
        create_selected_box_plots(hv_gain, selected_maps, 'Hypervolume-Gains',
                                   f'{acceptable_interval[0]}-{acceptable_interval[1]}')
        create_selected_box_plots(coverage_gain, selected_maps, 'Success-Coverage-Gains',
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

        for metric, gains in [('HV', hv_gain), ('SP', spread_gain), ('COV', coverage_gain)]:
            summary, map_rows = summarize_gains_by_map(
                gains, metric, acceptable_interval[0], acceptable_interval[1]
            )
            all_summaries.append(summary)
            all_map_rows.extend(map_rows)
            print(f"{metric} {acceptable_interval}: besser/schlechter/nicht signifikant "
                  f"{summary['better']}/{summary['worse']}/{summary['not_significant']}; "
                  f"95%-KI des mittleren Map-Gains "
                  f"[{summary['ci95_low']:.4f}, {summary['ci95_high']:.4f}]")

        mean_gains = np.mean(np.asarray(hv_gain), axis=1)
        print(f"Schwellen {acceptable_interval}: {len(SELECTED_MAPS)} Maps, "
              f"mittlerer HV-Gain={np.mean(mean_gains):.6f}, "
              f"Median={np.median(mean_gains):.6f}, "
              f"positiv={np.sum(mean_gains > 0)}, negativ={np.sum(mean_gains < 0)}")
        if np.any(mean_gains != 0):
            stat, p_value = wilcoxon(mean_gains, alternative='two-sided')
            print(f"Wilcoxon (gepaarte Map-Mittelwerte): W={stat:.3f}, p={p_value:.6g}")
        else:
            print("Wilcoxon nicht definiert: alle Map-Gains sind 0")
        print("SP = originaler PARLEY-Spread (kleiner = dichter); COV = Erfolgswahrscheinlichkeits-Abdeckung (größer = breiter).")

    write_statistical_tables(all_summaries, all_map_rows)
    print('Statistische Tabellen: plots/statistics/')

if __name__ == '__main__':
    main()
