#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,math,re,statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

MODEL_FILENAME_PATTERN=re.compile(r"model_([1-9][0-9])\.prism$")

@dataclass(frozen=True)
class ModelData:
    path: Path
    number: int
    n: int
    controller: dict[tuple[int,int],str]

def parse_int_constant(text,name):
    m=re.search(rf"\bconst\s+int\s+{re.escape(name)}\s*=\s*(\d+)\s*;",text)
    if not m: raise ValueError(f"Konstante {name!r} wurde nicht gefunden.")
    return int(m.group(1))

def parse_controller(text):
    m=re.search(r"module\s+Adaptation_MAPE_controller\s*(.*?)\s*endmodule",text,re.DOTALL)
    if not m: raise ValueError("Adaptation_MAPE_controller wurde nicht gefunden.")
    p=re.compile(r"\[(west|east|south|north)\]\s*\(xhat\s*=\s*(\d+)\)\s*&\s*\(yhat\s*=\s*(\d+)\)\s*->\s*true\s*;")
    c={(int(x),int(y)):d for d,x,y in p.findall(m.group(1))}
    if not c: raise ValueError("Keine MAPE-Richtungsentscheidungen erkannt.")
    return c

def parse_model(path):
    m=MODEL_FILENAME_PATTERN.fullmatch(path.name)
    if not m: raise ValueError("Dateiname entspricht nicht model_10.prism bis model_99.prism.")
    text=path.read_text(encoding='utf-8')
    return ModelData(path,int(m.group(1)),parse_int_constant(text,'N'),parse_controller(text))

def discover_models(models_dir):
    arr=[]
    for p in models_dir.iterdir():
        m=MODEL_FILENAME_PATTERN.fullmatch(p.name)
        if p.is_file() and m and 10<=int(m.group(1))<=99: arr.append(p)
    return sorted(arr,key=lambda p:int(MODEL_FILENAME_PATTERN.fullmatch(p.name).group(1)))

def advance_symmetric(state,direction,n):
    xhat,yhat,xradius,yradius=state
    if direction=='east': return (min(xhat+1,n),yhat,min(xradius+2,n),min(yradius+1,n))
    if direction=='west': return (max(xhat-1,0),yhat,min(xradius+2,n),min(yradius+1,n))
    if direction=='north': return (xhat,min(yhat+1,n),min(xradius+1,n),min(yradius+2,n))
    if direction=='south': return (xhat,max(yhat-1,0),min(xradius+1,n),min(yradius+2,n))
    raise ValueError(direction)

def calculate_interval_width(state,n):
    xhat,yhat,xradius,yradius=state
    xlow=max(xhat-xradius,0); xhigh=min(xhat+xradius,n)
    ylow=max(yhat-yradius,0); yhigh=min(yhat+yradius,n)
    return (xhigh-xlow)+(yhigh-ylow)

def linear_quantile(values,probability):
    ordered=sorted(float(v) for v in values)
    if len(ordered)==1:return ordered[0]
    pos=probability*(len(ordered)-1); lo=math.floor(pos); hi=math.ceil(pos)
    if lo==hi:return ordered[lo]
    f=pos-lo; return ordered[lo]*(1-f)+ordered[hi]*f

def round_half_up(v): return int(math.floor(v+0.5))

def analyze_model(model,steps):
    widths={k:[] for k in range(1,steps+1)}
    total=len(model.controller)
    for sx,sy in sorted(model.controller):
        state=(sx,sy,0,0)
        for step in range(1,steps+1):
            direction=model.controller.get((state[0],state[1]))
            if direction is None: break
            state=advance_symmetric(state,direction,model.n)
            widths[step].append(calculate_interval_width(state,model.n))
    thresholds=[]; rows=[]
    for step in range(1,steps+1):
        vals=widths[step]
        if not vals: raise ValueError(f"Map {model.number}: keine auswertbaren Routen für Schritt {step}.")
        mean=statistics.fmean(vals); thr=round_half_up(mean)
        is_new_threshold = thr not in thresholds
        if is_new_threshold:
            thresholds.append(thr)
        rows.append({'model':model.number,'step':step,'threshold_rounded_mean':thr,'selected_as_threshold':int(is_new_threshold),'width_mean':mean,'width_median':statistics.median(vals),'width_q1':linear_quantile(vals,0.25),'width_q3':linear_quantile(vals,0.75),'width_min':min(vals),'width_max':max(vals),'width_stddev':statistics.pstdev(vals) if len(vals)>1 else 0.0,'total_mape_start_states':total,'contributing_start_states':len(vals),'contributing_fraction':len(vals)/total if total else 0.0})
    return thresholds,rows

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description='Berechnet pro Map bis zu 10 unterschiedliche Schwellen aus der durchschnittlichen symmetrischen Intervallbreite nach Schritt 1..10.')
    ap.add_argument('models_dir',type=Path); ap.add_argument('--steps',type=int,default=10); ap.add_argument('--output-dir',type=Path,default=Path('interval_thresholds_mean_by_step'))
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    models=discover_models(a.models_dir); thresholds_per_map={}; stats=[]; skipped=[]
    for p in models:
        try:
            m=parse_model(p); th,rows=analyze_model(m,a.steps); thresholds_per_map[m.number]=th; stats.extend(rows)
            print(f"Map {m.number}: thresholds = {th} ({len(th)} unterschiedliche Schwellen)")
        except Exception as e:
            skipped.append({'model':p.name,'reason':str(e)}); print(f"Übersprungen: {p.name}: {e}")
    write_csv(a.output_dir/'mean_interval_width_per_step.csv',stats); write_csv(a.output_dir/'skipped_models.csv',skipped)
    with (a.output_dir/'thresholds_per_map.py').open('w',encoding='utf-8') as f:
        f.write('# Bis zu 10 unterschiedliche Schwellen = gerundete mittlere Intervallbreite nach Schritt 1..10\nTHRESHOLDS_PER_MAP = {\n')
        for k in sorted(thresholds_per_map): f.write(f'    {k}: {thresholds_per_map[k]},\n')
        f.write('}\n')

if __name__=='__main__': main()
