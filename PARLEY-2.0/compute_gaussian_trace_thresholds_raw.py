#!/usr/bin/env python3
"""
compute_gaussian_trace_thresholds_raw.py

Berechnet die Gaussian-Unsicherheitsmetrik direkt aus den Roh-Kovarianzen:

    U_G(Sigma) = trace(Sigma) = var_x + var_y

Keine Grid-Quantisierung.

Analog zum Belief-Modell werden 10 positionsunabhängige Unsicherheits-
Schwellwerte für die Schritte 1..10 erzeugt. Der URC besitzt weiterhin
decision_x_y in [1..10] und wählt einen dieser Schwellwerte.
"""

from __future__ import annotations
import argparse,csv,json
from pathlib import Path

TRACE_SCALE=1000000

def trace_int(vx,vy,scale): return int(round((vx+vy)*scale))

def analyse(map_id,lookup_dir,out_dir,scale,max_steps):
    path=lookup_dir/f"gaussian_raw_lookup_{map_id}.json"
    with path.open() as f: data=json.load(f)

    state_trace={int(s["raw_state"]):trace_int(float(s["var_x"]),float(s["var_y"]),scale)
                 for s in data["gaussian_states"]}

    # Reconstruct step depths from every reset position/state 0 using lookup.
    index={(int(r["xhat"]),int(r["yhat"]),int(r["raw_state"]),str(r["action"])):r
           for r in data["lookup"]}
    # MAPE gives one action per x,y in lookup; derive available actions.
    action_by_pos={}
    for r in data["lookup"]:
        action_by_pos.setdefault((int(r["xhat"]),int(r["yhat"])),str(r["action"]))

    traces={k:[] for k in range(1,max_steps+1)}
    positions={ (int(r["xhat"]),int(r["yhat"])) for r in data["lookup"] if int(r["raw_state"])==0 }

    for sx,sy in positions:
        x,y,state=sx,sy,0
        for step in range(1,max_steps+1):
            a=action_by_pos.get((x,y))
            if a is None: break
            row=index.get((x,y,state,a))
            if row is None: break
            x=int(row["xhat_next"]); y=int(row["yhat_next"]); state=int(row["raw_state_next"])
            traces[step].append(state_trace[state])

    thresholds=[]; running=0; stats={}
    for step in range(1,max_steps+1):
        vals=traces[step]
        exact=max(vals) if vals else running
        running=max(running,exact)
        thresholds.append(running)
        stats[str(step)]={"min_trace":min(vals) if vals else None,
                          "mean_trace":sum(vals)/len(vals) if vals else None,
                          "max_trace":exact,"threshold":running,"samples":len(vals)}

    groups={}
    for state,val in sorted(state_trace.items()):
        groups.setdefault(str(val),[]).append(state)

    out={"map":map_id,"representation":"raw_sigma","trace_scale":scale,
         "metric":"trace(Sigma)=var_x+var_y",
         "thresholds":{str(i+1):thresholds[i] for i in range(max_steps)},
         "per_step_statistics":stats,
         "raw_state_trace":{str(k):v for k,v in sorted(state_trace.items())},
         "trace_groups":groups}
    out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/f"gaussian_raw_trace_{map_id}.json").open("w") as f: json.dump(out,f,indent=2)
    return {"map":map_id, **{f"threshold_{i+1}":thresholds[i] for i in range(max_steps)},
            "unique_trace_values":len(groups)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lookup-dir",type=Path,default=Path("gaussian_raw_lookup"))
    ap.add_argument("--output-dir",type=Path,default=Path("gaussian_raw_trace"))
    ap.add_argument("--start-map",type=int,default=10);ap.add_argument("--end-map",type=int,default=99)
    ap.add_argument("--max-steps",type=int,default=10);ap.add_argument("--trace-scale",type=int,default=TRACE_SCALE)
    args=ap.parse_args(); rows=[]
    for i in range(args.start_map,args.end_map+1):
        if not (args.lookup_dir/f"gaussian_raw_lookup_{i}.json").exists(): continue
        r=analyse(i,args.lookup_dir,args.output_dir,args.trace_scale,args.max_steps)
        rows.append(r); print(f"[map {i}] thresholds={[r[f'threshold_{k}'] for k in range(1,11)]}")
    if rows:
        with (args.output_dir/"gaussian_raw_trace_summary.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

if __name__=="__main__": main()
