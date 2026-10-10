"""Plot EvoChecker URC fronts together with optional fixed-update baselines."""
from pathlib import Path
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'Applications' / 'EvoChecker-master' / 'data'


def pareto_front(points):
    unique = sorted(set(points), key=lambda v: (-v[0], v[1]))
    return [(x, y) for x, y in unique if not any(
        (a >= x and b <= y) and (a > x or b < y) for a, b in unique
    )]


def read_points(path):
    points = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        parts = line.strip().replace(',', ' ').split()
        if len(parts) < 2:
            continue
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue  # header
    if not points:
        raise ValueError(f'Keine numerischen (success, cost)-Paare in {path}')
    return points


def latest_front(folder):
    candidates = list(folder.glob('*_Front'))
    if not candidates:
        raise FileNotFoundError(f'Keine *_Front-Datei in {folder}')
    def key(path):
        parts = path.stem.split('_')
        try:
            return datetime.strptime(parts[-3] + parts[-2], '%H%M%S%d%m%y')
        except (ValueError, IndexError):
            return datetime.fromtimestamp(path.stat().st_mtime)
    return max(candidates, key=key)


def plot_pareto_front(m=10, replication=0, header=True, *, problem=None,
                      baseline_file=None, baseline_problem=None,
                      baseline_pareto=False, output_dir='plots/fronts'):
    # header retained for compatibility; read_points detects headers automatically.
    problem = problem or f'ROBOT{m}_REP{replication}'
    front = latest_front(DATA / problem / 'NSGAII')
    urc = pareto_front(read_points(front))

    baseline = None
    if baseline_file is not None:
        path = Path(baseline_file)
        baseline = read_points(path)
    else:
        base_dir = DATA / (baseline_problem or f'ROBOT{m}_BASELINE')
        direct = base_dir / 'Front'
        if direct.is_file():
            baseline = read_points(direct)
        elif (base_dir / 'NSGAII').is_dir():
            baseline = read_points(latest_front(base_dir / 'NSGAII'))

    if baseline_pareto and baseline is not None:
        baseline = pareto_front(baseline)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([x for x, _ in urc], [y for _, y in urc], label='URC', color='tab:blue', s=35)
    if baseline is not None:
        ax.scatter([x for x, _ in baseline], [y for _, y in baseline],
                   label='Baseline', color='tab:red', marker='x', s=65)
    else:
        print('Hinweis: Keine Baseline-Datei gefunden. Übergib --baseline-file PFAD.')
    ax.set_xlabel('Probability of mission success')
    ax.set_ylabel('Cost')
    ax.set_title(problem)
    ax.set_xlim(1.0, 0.0)  # Erfolgswahrscheinlichkeit: links 1.0, rechts 0.0
    ax.set_ylim(0, 200)   # Kosten: 0 bis 200
    ax.grid(True, alpha=0.3)
    ax.legend()
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / f'{problem}_front_baseline.pdf'
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(f'URC: {front}; Baseline-Punkte: {len(baseline) if baseline is not None else 0}')
    print(f'Plot: {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=int, default=10)
    parser.add_argument('--replication', type=int, default=0)
    parser.add_argument('--problem', help='EvoChecker-PROBLEM, z.B. ROBOT2_INTERVAL_RQ3_REP0')
    parser.add_argument('--baseline-file', type=Path)
    parser.add_argument('--baseline-problem')
    parser.add_argument('--baseline-pareto', action='store_true')
    args = parser.parse_args()
    plot_pareto_front(args.map, args.replication, problem=args.problem,
                      baseline_file=args.baseline_file,
                      baseline_problem=args.baseline_problem,
                      baseline_pareto=args.baseline_pareto)
