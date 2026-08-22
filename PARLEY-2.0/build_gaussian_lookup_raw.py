#!/usr/bin/env python3
"""
build_gaussian_lookup_raw.py

Erzeugt direkt aus Map + MAPE eine deterministische Lookup-Tabelle für rohe
Gaussian-Kovarianzzustände, ohne Grid-Quantisierung und ohne Refinement.

State:
    raw_state <-> Sigma=(var_x,var_y,cov_xy)

Transition:
    (xhat,yhat,raw_state,action)
       -> (xhat_next,yhat_next,raw_state_next)

Da Sigma_next = Sigma + Q(xhat,yhat,action), ist diese Darstellung Markov.
"""

from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from typing import Dict,List,Tuple
import dijkstra

DIRECTION_NAMES=["west","east","south","north"]
KNOWLEDGE_EFFECT={"west":(-1,0),"east":(1,0),"south":(0,-1),"north":(0,1)}

def key(s,digits):
    def c(v):
        r=round(v,digits)
        return 0.0 if abs(r)<10**(-digits) else r
    return c(s[0]),c(s[1]),c(s[2])

def load_map(path):
    with path.open("r",newline="") as f: raw=list(csv.reader(f))
    return [list(row[::-1]) for row in zip(*raw)]

def obstacle(m,x,y): return int(m[x][y])>9

def policy(m,tx,ty):
    raw=dijkstra.compute_directions(m,(tx,ty)); dirs=list(zip(*raw)); out={}
    for x in range(len(m)):
        for y in range(len(m)):
            d=int(dirs[y][x])
            if d<4: out[(x,y)]=DIRECTION_NAMES[d]
    return out

def apply(x,y,a,n):
    dx,dy=KNOWLEDGE_EFFECT[a]
    return min(max(x+dx,0),n),min(max(y+dy,0),n)

def outcomes(a,p):
    q=1-3*p
    if a=="east": return [(q,1,0),(p,0,1),(p,0,-1),(p,-1,0)]
    if a=="west": return [(p,1,0),(p,0,1),(p,0,-1),(q,-1,0)]
    if a=="north": return [(p,1,0),(q,0,1),(p,0,-1),(p,-1,0)]
    if a=="south": return [(p,1,0),(p,0,1),(q,0,-1),(p,-1,0)]
    raise ValueError(a)

def motion_cov(x,y,a,n,p):
    samples=[]
    for pr,dx,dy in outcomes(a,p):
        nx=min(max(x+dx,0),n); ny=min(max(y+dy,0),n)
        samples.append((pr,float(nx-x),float(ny-y)))
    mx=sum(pr*dx for pr,dx,_ in samples); my=sum(pr*dy for pr,_,dy in samples)
    vx=sum(pr*(dx-mx)**2 for pr,dx,_ in samples)
    vy=sum(pr*(dy-my)**2 for pr,_,dy in samples)
    c=sum(pr*(dx-mx)*(dy-my) for pr,dx,dy in samples)
    return vx,vy,c

def add(a,b): return a[0]+b[0],a[1]+b[1],a[2]+b[2]

def analyse(map_id,maps_dir,out_dir,tx,ty,p,max_steps,digits):
    m=load_map(maps_dir/f"map_{map_id}.csv"); pol=policy(m,tx,ty); n=len(m)-1
    records=[]
    sigmas={(0.0,0.0,0.0)}
    for sx in range(len(m)):
        for sy in range(len(m)):
            if obstacle(m,sx,sy): continue
            x,y=sx,sy; sigma=(0.0,0.0,0.0)
            for _step in range(1,max_steps+1):
                a=pol.get((x,y))
                if a is None: break
                source=key(sigma,digits)
                sigma_next=key(add(sigma,motion_cov(x,y,a,n,p)),digits)
                nx,ny=apply(x,y,a,n)
                sigmas.add(source); sigmas.add(sigma_next)
                records.append((x,y,source,a,nx,ny,sigma_next))
                x,y,sigma=nx,ny,sigma_next

    zero=(0.0,0.0,0.0)
    ordered=sorted(sigmas,key=lambda s:(0 if s==zero else 1,s[0]+s[1],s[0],s[1],s[2]))
    sid={s:i for i,s in enumerate(ordered)}

    transitions={}
    for x,y,s,a,nx,ny,sn in records:
        src=(x,y,sid[s],a); dst=(nx,ny,sid[sn])
        if src in transitions and transitions[src]!=dst:
            raise ValueError(f"Non-Markov raw state: {src}: {transitions[src]} vs {dst}")
        transitions[src]=dst

    states=[{"raw_state":sid[s],"var_x":s[0],"var_y":s[1],"cov_xy":s[2],
             "trace":round(s[0]+s[1],digits)} for s in ordered]
    lookup=[{"xhat":src[0],"yhat":src[1],"raw_state":src[2],"action":src[3],
             "xhat_next":dst[0],"yhat_next":dst[1],"raw_state_next":dst[2]}
            for src,dst in sorted(transitions.items())]

    out={"map":map_id,"representation":"raw_sigma","digits":digits,
         "number_of_raw_states":len(states),"gaussian_states":states,"lookup":lookup}
    out_dir.mkdir(parents=True,exist_ok=True)
    with (out_dir/f"gaussian_raw_lookup_{map_id}.json").open("w") as f: json.dump(out,f,indent=2)
    with (out_dir/f"gaussian_raw_lookup_{map_id}.csv").open("w",newline="") as f:
        fields=["xhat","yhat","raw_state","action","xhat_next","yhat_next","raw_state_next"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(lookup)
    return {"map":map_id,"raw_states":len(states),"lookup_transitions":len(lookup)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--maps-dir",type=Path,default=Path("maps"))
    ap.add_argument("--output-dir",type=Path,default=Path("gaussian_raw_lookup"))
    ap.add_argument("--start-map",type=int,default=10); ap.add_argument("--end-map",type=int,default=99)
    ap.add_argument("--target-x",type=int,default=9); ap.add_argument("--target-y",type=int,default=9)
    ap.add_argument("--p",type=float,default=0.01); ap.add_argument("--max-steps",type=int,default=10)
    ap.add_argument("--digits",type=int,default=12)
    args=ap.parse_args(); rows=[]
    for i in range(args.start_map,args.end_map+1):
        if not (args.maps_dir/f"map_{i}.csv").exists(): continue
        r=analyse(i,args.maps_dir,args.output_dir,args.target_x,args.target_y,args.p,args.max_steps,args.digits)
        rows.append(r); print(f"[map {i}] states={r['raw_states']} transitions={r['lookup_transitions']}")
    if rows:
        with (args.output_dir/"gaussian_raw_lookup_summary.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

if __name__=="__main__": main()
