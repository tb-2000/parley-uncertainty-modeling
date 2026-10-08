"""RQ3-Karten: map_0=5x5, map_1=10x10 (unverändert), map_2=15x15, map_3=20x20.

PRISM-Generator liest CSV mit map_data[x][y] = csv[size-1-y][x].
Damit liegt PRISM-Start (0,0) links unten und Ziel (N,N) rechts oben.
Zellwerte >=10 sind Hindernisse, Werte 0..9 begehbar.
"""
from collections import deque
from pathlib import Path
import csv
import random

MAP_DIR = Path(__file__).resolve().parent / "maps"
RQ3_SIZES = {0: 5, 1: 10, 2: 15, 3: 20}
OBSTACLE_PROBABILITY = 0.25
MAX_ATTEMPTS = 10000


def has_path(map_data, start_pos, target_pos):
    """Prüft den Weg in CSV-Koordinaten (Zeile, Spalte) mit PRISM-Hindernisregel."""
    rows = len(map_data)
    cols = len(map_data[0])
    if any(not (0 <= r < rows and 0 <= c < cols) for r, c in (start_pos, target_pos)):
        return False
    if map_data[start_pos[0]][start_pos[1]] >= 10 or map_data[target_pos[0]][target_pos[1]] >= 10:
        return False
    queue = deque([start_pos])
    visited = {start_pos}
    while queue:
        r, c = queue.popleft()
        if (r, c) == target_pos:
            return True
        for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            if (0 <= nr < rows and 0 <= nc < cols and
                (nr,nc) not in visited and map_data[nr][nc] < 10):
                visited.add((nr,nc))
                queue.append((nr,nc))
    return False


def add_penalties(old_map_data):
    """Erhöht Wegkosten neben Hindernissen, ohne neue Hindernisse zu erzeugen."""
    size = len(old_map_data)
    result = [row[:] for row in old_map_data]
    for r in range(size):
        for c in range(size):
            if old_map_data[r][c] >= 10:
                continue
            adjacent = sum(
                0 <= nr < size and 0 <= nc < size and old_map_data[nr][nc] >= 10
                for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1))
            )
            result[r][c] = min(9, old_map_data[r][c] + 3 * adjacent)
    return result


def generate_one_map(size=10, rng=None):
    rng = rng or random.Random()
    grid = [[10 if rng.random() < OBSTACLE_PROBABILITY else 1
             for _ in range(size)] for _ in range(size)]
    # Start links unten, Ziel rechts oben. Die Ränder bleiben nicht pauschal frei.
    grid[size - 1][0] = 0
    grid[0][size - 1] = 0
    return grid


def generate_map(size=10, rng=None):
    rng = rng or random.Random()
    start, target = (size - 1, 0), (0, size - 1)
    for _ in range(MAX_ATTEMPTS):
        raw = generate_one_map(size, rng)
        if not has_path(raw, start, target):
            continue
        final = add_penalties(raw)
        if has_path(final, start, target):
            return final
    raise RuntimeError(f"Keine begehbare {size}x{size}-Karte nach {MAX_ATTEMPTS} Versuchen")


def _write_map(index, grid):
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    path = MAP_DIR / f"map_{index}.csv"
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(grid)
    print(f"Erzeugt: {path} ({len(grid)}x{len(grid)})")


def create_3_maps(seed=20261008):
    """Erzeugt nur map_0, map_2 und map_3. map_1 (10x10) bleibt unverändert."""
    rng = random.Random(seed)
    existing = MAP_DIR / "map_1.csv"
    if not existing.is_file():
        raise FileNotFoundError(f"10x10-Referenzkarte fehlt: {existing}")
    with existing.open(newline="") as f:
        reference = [[int(v) for v in row] for row in csv.reader(f)]
    if len(reference) != 10 or any(len(row) != 10 for row in reference):
        raise ValueError(f"{existing} ist nicht 10x10. Bitte Kartenzuordnung prüfen!")
    print(f"Unverändert: {existing} (10x10)")
    for index in (0, 2, 3):
        _write_map(index, generate_map(RQ3_SIZES[index], rng))


def create_90_maps(seed=None):
    """Optional: alte 10x10-Trainingskarten map_10..map_99 neu erzeugen."""
    rng = random.Random(seed)
    for index in range(10, 100):
        _write_map(index, generate_map(10, rng))


if __name__ == "__main__":
    create_3_maps()
