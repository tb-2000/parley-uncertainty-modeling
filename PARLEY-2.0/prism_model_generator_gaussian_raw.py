import csv
import json
import os
import dijkstra

startX=0; startY=0; targetX=4; targetY=4; p=0.01
directions=['west','east','south','north']
obstacles=[]; updates=[5]
map_data=[]; mapSize=0; prism_file=""

gaussian_raw_lookup_dir="gaussian_raw_lookup"
gaussian_raw_trace_dir="gaussian_raw_trace"

raw_states=[]; raw_lookup=[]
raw_state_max=0
trace_thresholds=[]
trace_groups={}

def build_map(filename):
    global mapSize,map_data,obstacles
    with open(filename,'r') as f: rows=list(csv.reader(f))
    mapSize=len(rows); transposed=list(zip(*rows)); map_data=[row[::-1] for row in transposed]
    obstacles=[[x,y] for x in range(mapSize) for y in range(mapSize) if int(map_data[x][y])>9]

def load_raw_lookup(map_id):
    global raw_states,raw_lookup,raw_state_max
    path=os.path.join(gaussian_raw_lookup_dir,f"gaussian_raw_lookup_{map_id}.json")
    with open(path) as f: data=json.load(f)
    raw_states=data["gaussian_states"]; raw_lookup=data["lookup"]
    ids={int(s["raw_state"]) for s in raw_states}
    if 0 not in ids:
        raise ValueError("raw_state 0 (Sigma=0) missing")
    zero=next(s for s in raw_states if int(s["raw_state"])==0)
    if abs(float(zero["var_x"]))>1e-12 or abs(float(zero["var_y"]))>1e-12 or abs(float(zero["cov_xy"]))>1e-12:
        raise ValueError("raw_state 0 must be Sigma=0")
    raw_state_max=max(ids)

def load_trace(map_id):
    global trace_thresholds,trace_groups
    path=os.path.join(gaussian_raw_trace_dir,f"gaussian_raw_trace_{map_id}.json")
    with open(path) as f: data=json.load(f)
    trace_thresholds=[int(data["thresholds"][str(i)]) for i in range(1,11)]
    trace_groups={int(k):[int(s) for s in v] for k,v in data["trace_groups"].items()}

def preambel():
    with open(prism_file,'a') as f:
        f.write('dtmc\n')
        for i,t in enumerate(trace_thresholds,1):
            f.write(f'const int gaussian_threshold_{i} = {t};\n')
        f.write(f'const int N={mapSize-1};\n')
        f.write(f'const int xstart = {startX};\nconst int ystart = {startY};\n')
        f.write(f'const int xtarget = {targetX};\nconst int ytarget = {targetY};\n')
        f.write(f'const double p = {p};\nconst int RAW_STATE_MAX = {raw_state_max};\n\n')
        f.write('formula hasCrashed = (1=0) ')
        for x,y in obstacles: f.write(f'| (x={x} & y={y}) ')
        f.write(';\n\n')
        values=sorted(trace_groups)
        for idx,val in enumerate(values):
            expr=' | '.join(f'raw_state={s}' for s in trace_groups[val])
            f.write(f'formula gaussian_u_{idx} = {expr};\n')
        terms=[f'(gaussian_u_{idx} & max_gaussian_uncertainty<={val})' for idx,val in enumerate(values)]
        f.write('formula update_required = '+' | '.join(terms)+';\n\n')
        f.write('// Raw Gaussian covariance states: no grid quantization h.\n')
        f.write('// raw_state directly identifies Sigma; trace(Sigma)=var_x+var_y.\n\n')

def robot():
    with open(prism_file,'a') as f:
        f.write('module Robot\n  x : [0..N] init xstart;\n  y : [0..N] init ystart;\n')
        f.write('  move_ready : [0..1] init 1;\n  crashed : [0..1] init 0;\n\n')
        f.write("  [east] (move_ready=1) -> (1-3*p):(x'=min(x+1,N))&(move_ready'=0)+p:(y'=min(y+1,N))&(move_ready'=0)+p:(y'=max(y-1,0))&(move_ready'=0)+p:(x'=max(x-1,0))&(move_ready'=0);\n")
        f.write("  [west] (move_ready=1) -> p:(x'=min(x+1,N))&(move_ready'=0)+p:(y'=min(y+1,N))&(move_ready'=0)+p:(y'=max(y-1,0))&(move_ready'=0)+(1-3*p):(x'=max(x-1,0))&(move_ready'=0);\n")
        f.write("  [north] (move_ready=1) -> p:(x'=min(x+1,N))&(move_ready'=0)+(1-3*p):(y'=min(y+1,N))&(move_ready'=0)+p:(y'=max(y-1,0))&(move_ready'=0)+p:(x'=max(x-1,0))&(move_ready'=0);\n")
        f.write("  [south] (move_ready=1) -> p:(x'=min(x+1,N))&(move_ready'=0)+p:(y'=min(y+1,N))&(move_ready'=0)+(1-3*p):(y'=max(y-1,0))&(move_ready'=0)+p:(x'=max(x-1,0))&(move_ready'=0);\n")
        f.write("  [check] (move_ready=0)&hasCrashed -> (crashed'=1)&(move_ready'=1);\n")
        f.write("  [check] (move_ready=0)&!hasCrashed -> (move_ready'=1);\nendmodule\n\n")

def adaptation_mape_controller(d):
    with open(prism_file,'a') as f:
        f.write('module Adaptation_MAPE_controller\n')
        for x in range(mapSize):
            for y in range(mapSize):
                direction=int(d[y][x])
                if direction<4:
                    f.write(f'  [{directions[direction]}] (xhat={x}) & (yhat={y}) -> true;\n')
        f.write('endmodule\n\n')

def knowledge():
    with open(prism_file,'a') as f:
        f.write('module Knowledge\n')
        f.write('  xhat : [0..N] init xstart;\n  yhat : [0..N] init ystart;\n')
        f.write('  raw_state : [0..RAW_STATE_MAX] init 0;\n')
        f.write('  step : [1..10] init 1;\n  ready : [0..1] init 1;\n\n')
        for r in raw_lookup:
            f.write(
                f"  [{r['action']}] ready=1 & xhat={int(r['xhat'])} & yhat={int(r['yhat'])} "
                f"& raw_state={int(r['raw_state'])} -> "
                f"(xhat'={int(r['xhat_next'])}) & (yhat'={int(r['yhat_next'])}) & "
                f"(raw_state'={int(r['raw_state_next'])}) & (ready'=0);\n"
            )
        # Position estimate and Sigma reset exactly on update.
        f.write("\n  [update] ready=0 & (update_required | step>=10) -> "
                "(xhat'=x) & (yhat'=y) & (raw_state'=0) & (step'=1) & (ready'=1);\n")
        f.write("  [skip_update] ready=0 & !update_required & step<10 -> "
                "(step'=step+1) & (ready'=1);\nendmodule\n\n")

def rewards():
    with open(prism_file,'a') as f:
        f.write('rewards "cost"\n')
        for a in ['east','west','north','south']: f.write(f'  [{a}] true : 1;\n')
        f.write('  [update] true : 5;\nendrewards\n\n')

def read_params():
    global startX,startY,targetX,targetY,p,updates
    with open('input.json') as f: params=json.load(f)
    startX=params["startX"];startY=params["startY"];targetX=params["targetX"];targetY=params["targetY"]
    p=params["p"];updates=params["updates"]

def generate_model(i):
    global prism_file
    read_params(); build_map(f"maps/map_{i}.csv"); load_raw_lookup(i); load_trace(i)
    _d=dijkstra.compute_directions(map_data,(targetX,targetY)); d=list(zip(*_d))
    prism_file=f"Applications/EvoChecker-master/models/model_{i}.prism"
    open(prism_file,'w').close()
    preambel();robot();adaptation_mape_controller(d);knowledge();rewards()
    print(f"finished map {i}: {raw_state_max+1} raw Gaussian states, {len(raw_lookup)} transitions")
