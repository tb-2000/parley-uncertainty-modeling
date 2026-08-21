import os
import matplotlib.pyplot as plt
from datetime import datetime


def pareto_front(data):
    pareto_data = []
    for x, y in data:
        if not is_dominated(x, y, pareto_data):
            pareto_data.append((x, y))
    return pareto_data


def is_dominated(x, y, data):
    for other_x, other_y in data:
        if other_x >= x and other_y <= y:
            return True
    return False


def plot_pareto_front(m=10, replication=0, header=True):
    data = []
    filename = ''
    file_path = f'Applications/EvoChecker-master/data/ROBOT{m}_REP{replication}/NSGAII/'
    # for f_name in os.listdir(file_path):
    #     if "Front" in f_name:
    #         filename = f_name
    front_files = [
        f_name for f_name in os.listdir(file_path)
        if f_name.endswith("_Front")
    ]
    if not front_files:
        raise FileNotFoundError(
            f"Keine Front-Datei für ROBOT{m}_REP{replication} gefunden."
        )
    def get_timestamp(filename):
        # z.B. ROBOT49_REP9_NSGAII_005212_130826_Front
        parts = filename.split("_")
        time_str = parts[-3]   # 005212
        date_str = parts[-2]   # 130826
        return datetime.strptime(date_str + time_str, "%d%m%y%H%M%S")

    filename = max(front_files, key=get_timestamp)
    with open(file_path + filename, 'r') as file:
        if header:
            next(file)  # Skip the header row
        for line in file:
            x, y = map(float, line.strip().split('\t'))
            # x, y = map(float, line.split())
            data.append((x, y))

    pareto_data = pareto_front(data)

    x_values = [x for x, _ in pareto_data]
    y_values = [y for _, y in pareto_data]

    # data = []
    # with open(f'Applications/EvoChecker-master/data/ROBOT{m}_BASELINE/Front', 'r') as file:
    #     for line in file:
    #         x, y = map(float, line.strip().split('	'))
    #         # x, y = map(float, line.split())
    #         data.append((x, y))

    # baseline_data = pareto_front(data[:20])

    # x_values_b = [x for x, y in baseline_data]

    # y_values_b = [y for x, y in baseline_data]

    plt.figure(figsize=(8, 6))

    plt.scatter(x_values, y_values, color='blue', label='URC')

    # # Add red crosses for the baseline
    # plt.scatter(x_values_b, y_values_b, color='red', marker='x', label='Baseline')

    plt.xlabel('Probability of mission success')
    plt.ylabel('Cost')
    output_filename = f'robot{m}_rep{replication}'
    plt.title(output_filename)
    plt.xlim(1, 0.2)
    plt.ylim(0, 200)
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig('plots/fronts/' + output_filename + '.pdf')
    plt.close()
