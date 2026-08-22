#!/usr/bin/env python3
"""
compute_gaussian_states_raw.py

Berechnet pro Map alle tatsächlich auftretenden Roh-Kovarianzzustände Sigma
innerhalb von maximal 10 Schritten seit einem Knowledge-Update.

Keine Grid-Quantisierung:
    kein h
    kein Q_h(Sigma)

Jeder numerisch kanonisierte Rohzustand
    Sigma = (var_x, var_y, cov_xy)
erhält eine diskrete ID raw_state.

Die Kanonisierung mit --digits (Default 12) dient nur dazu, numerisches
Floating-Point-Rauschen bei mathematisch gleichen Werten zusammenzufassen.
Sie ist keine fachliche Grid-Quantisierung.
"""

from __future__ import annotations
import argparse, csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import dijkstra

DIRECTION_NAMES = ["west", "east", "south", "north"]
KNOWLEDGE_EFFECT = {
    "west": (-1, 0), "east": (1, 0),
    "south": (0, -1), "north": (0, 1),
}

@dataclass(frozen=True)
class Covariance:
    var_x: float
    var_y: float
    cov_xy: float = 0.0

    def add(self, other: "Covariance") -> "Covariance":
        return Covariance(
            self.var_x + other.var_x,
            self.var_y + other.var_y,
            self.cov_xy + other.cov_xy,
        )

def covariance_key(sigma: Covariance, digits: int) -> Tuple[float,float,float]:
    def clean(v: float) -> float:
        r = round(v, digits)
        return 0.0 if abs(r) < 10 ** (-digits) else r
    return clean(sigma.var_x), clean(sigma.var_y), clean(sigma.cov_xy)

def load_map(path: Path) -> List[List[str]]:
    with path.open("r", newline="") as f:
        raw = list(csv.reader(f))
    transposed = list(zip(*raw))
    return [list(row[::-1]) for row in transposed]

def is_obstacle(map_data, x, y):
    return int(map_data[x][y]) > 9

def build_policy(map_data, target_x, target_y):
    raw = dijkstra.compute_directions(map_data, (target_x, target_y))
    dirs = list(zip(*raw))
    policy = {}
    for x in range(len(map_data)):
        for y in range(len(map_data)):
            d = int(dirs[y][x])
            if d < 4:
                policy[(x,y)] = DIRECTION_NAMES[d]
    return policy

def apply_action(x, y, action, n):
    dx, dy = KNOWLEDGE_EFFECT[action]
    return min(max(x+dx,0),n), min(max(y+dy,0),n)

def robot_outcomes(action, p):
    intended = 1.0 - 3.0*p
    if action == "east":
        return [(intended,1,0),(p,0,1),(p,0,-1),(p,-1,0)]
    if action == "west":
        return [(p,1,0),(p,0,1),(p,0,-1),(intended,-1,0)]
    if action == "north":
        return [(p,1,0),(intended,0,1),(p,0,-1),(p,-1,0)]
    if action == "south":
        return [(p,1,0),(p,0,1),(intended,0,-1),(p,-1,0)]
    raise ValueError(action)

def motion_covariance(x, y, action, n, p):
    samples = []
    for prob, dx, dy in robot_outcomes(action,p):
        nx = min(max(x+dx,0),n)
        ny = min(max(y+dy,0),n)
        samples.append((prob,float(nx-x),float(ny-y)))
    mx = sum(prob*dx for prob,dx,_ in samples)
    my = sum(prob*dy for prob,_,dy in samples)
    vx = sum(prob*(dx-mx)**2 for prob,dx,_ in samples)
    vy = sum(prob*(dy-my)**2 for prob,_,dy in samples)
    cxy = sum(prob*(dx-mx)*(dy-my) for prob,dx,dy in samples)
    return Covariance(vx,vy,cxy)

def analyse_map(map_id, maps_dir, out_dir, target_x, target_y, p, max_steps, digits):
    map_data = load_map(maps_dir / f"map_{map_id}.csv")
    policy = build_policy(map_data,target_x,target_y)
    n = len(map_data)-1

    states: Dict[Tuple[float,float,float], dict] = {}
    occurrences = []

    def register(sigma, start, pos, step):
        key = covariance_key(sigma,digits)
        entry = states.setdefault(key,{
            "sigma": {"var_x":key[0],"var_y":key[1],"cov_xy":key[2]},
            "occurrences":0,
            "first_example":{"start":list(start),"position":list(pos),"step":step},
        })
        entry["occurrences"] += 1
        return key

    for sx in range(len(map_data)):
        for sy in range(len(map_data)):
            if is_obstacle(map_data,sx,sy):
                continue
            x,y = sx,sy
            sigma = Covariance(0.0,0.0,0.0)
            key = register(sigma,(sx,sy),(x,y),0)
            occurrences.append({"start":[sx,sy],"step":0,"xhat":x,"yhat":y,"sigma":key})
            for step in range(1,max_steps+1):
                action = policy.get((x,y))
                if action is None:
                    break
                sigma = sigma.add(motion_covariance(x,y,action,n,p))
                x,y = apply_action(x,y,action,n)
                key = register(sigma,(sx,sy),(x,y),step)
                occurrences.append({"start":[sx,sy],"step":step,"xhat":x,"yhat":y,"sigma":key})

    zero = (0.0,0.0,0.0)
    ordered = sorted(states, key=lambda s:(0 if s==zero else 1, s[0]+s[1], s[0],s[1],s[2]))
    sigma_to_id = {s:i for i,s in enumerate(ordered)}

    gaussian_states = []
    for s in ordered:
        entry = states[s]
        gaussian_states.append({
            "raw_state": sigma_to_id[s],
            "var_x": s[0], "var_y": s[1], "cov_xy": s[2],
            "trace": round(s[0]+s[1], digits),
            "occurrences": entry["occurrences"],
            "first_example": entry["first_example"],
        })

    for row in occurrences:
        row["raw_state"] = sigma_to_id[tuple(row.pop("sigma"))]

    out = {
        "map":map_id,
        "representation":"raw_sigma",
        "digits":digits,
        "max_steps":max_steps,
        "number_of_raw_states":len(gaussian_states),
        "gaussian_states":gaussian_states,
        "occurrences":occurrences,
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/f"gaussian_raw_states_{map_id}.json").open("w") as f:
        json.dump(out,f,indent=2)
    return {"map":map_id,"raw_states":len(gaussian_states)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir",type=Path,default=Path("maps"))
    ap.add_argument("--output-dir",type=Path,default=Path("gaussian_raw_states"))
    ap.add_argument("--start-map",type=int,default=10)
    ap.add_argument("--end-map",type=int,default=99)
    ap.add_argument("--target-x",type=int,default=9)
    ap.add_argument("--target-y",type=int,default=9)
    ap.add_argument("--p",type=float,default=0.01)
    ap.add_argument("--max-steps",type=int,default=10)
    ap.add_argument("--digits",type=int,default=12)
    args = ap.parse_args()

    rows=[]
    for map_id in range(args.start_map,args.end_map+1):
        path=args.maps_dir/f"map_{map_id}.csv"
        if not path.exists():
            print(f"[skip] map {map_id}")
            continue
        row=analyse_map(map_id,args.maps_dir,args.output_dir,args.target_x,args.target_y,
                        args.p,args.max_steps,args.digits)
        rows.append(row)
        print(f"[map {map_id}] raw_states={row['raw_states']}")
    if rows:
        with (args.output_dir/"gaussian_raw_states_summary.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["map","raw_states"]); w.writeheader(); w.writerows(rows)
        print(f"mean raw states = {sum(r['raw_states'] for r in rows)/len(rows):.2f}")

if __name__=="__main__":
    main()
