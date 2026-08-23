import os,re,shutil

def manipulate_prism_model(
        input_path, output_path,
        possible_decisions=[1,10],
        decision_variables=['x','y'],
        before_actions=['east','west','north','south'],
        after_actions=['update','skip_update'],
        module_name='Knowledge',
        baseline=False):
    """
    Raw-Sigma Gaussian URC.

    Wie beim Belief-/Point-Estimate-Modell bleibt der Suchraum positionsbasiert:
        decision_x_y in [1..10]

    Jede Decision wählt einen der 10 raw-trace(Sigma)-Schwellwerte.
    """
    if os.path.abspath(input_path)==os.path.abspath(output_path):
        raise ValueError("Input and output files cannot be the same.")
    shutil.copyfile(input_path,output_path)
    variables,guards=get_variables(input_path,decision_variables)
    thresholds=get_thresholds(input_path)
    add_controller(output_path,guards,variables,thresholds,possible_decisions,baseline)
    add_turn(output_path,before_actions,after_actions)

def constants(path):
    pat=re.compile(r'const\s+int\s+(\w+)\s*=\s*(-?\d+)\s*;'); out={}
    with open(path) as f:
        for line in f:
            for m in pat.finditer(line): out[m.group(1)]=int(m.group(2))
    return out

def resolve(v,c):
    if v.lstrip("-").isdigit(): return int(v)
    if v not in c: raise ValueError(f"Unknown bound {v}")
    return c[v]

def variables(path,c):
    pat=re.compile(r'(\w+)\s*:\s*\[(-?\w+)\s*\.\.\s*(-?\w+)\]\s*init\s*(-?\w+)\s*;')
    out={}
    with open(path) as f:
        for line in f:
            for m in pat.finditer(line):
                out[m.group(1)]=[m.group(1),resolve(m.group(2),c),resolve(m.group(3),c)]
    return out

def get_variables(path,names):
    c=constants(path); decl=variables(path,c); vals=[];guards=[]
    for name in names:
        vals.append(decl[name]); estimate=name+"hat";guards.append(estimate if estimate in decl else name)
    return vals,guards

def get_thresholds(path):
    pat=re.compile(r'const\s+int\s+gaussian_threshold_(\d+)\s*=\s*(\d+)\s*;')
    t={}
    with open(path) as f:
        for line in f:
            m=pat.search(line)
            if m:t[int(m.group(1))]=int(m.group(2))
    missing=[i for i in range(1,11) if i not in t]
    if missing: raise ValueError(f"Missing thresholds {missing}")
    return [t[i] for i in range(1,11)]

def combos(vars_):
    out=[]
    def rec(cur,rest):
        if not rest:out.append(tuple(cur));return
        v=rest[0]
        for x in range(v[1],v[2]+1):rec(cur+[x],rest[1:])
    rec([],vars_);return out

def add_controller(path,guards,vars_,thresholds,possible,baseline):
    cs=combos(vars_)
    with open(path,'a') as f:
        for c in cs:
            suffix=''.join(f'_{v}' for v in c)
            if baseline:f.write(f'\nconst int decision{suffix}=1;')
            else:f.write(f'\nevolve int decision{suffix} [{possible[0]}..{possible[1]}];')
        f.write('\nmodule URC\n')
        low=min(thresholds);high=max(thresholds)
        f.write(f'  max_gaussian_uncertainty : [{low}..{high}] init {low};\n')

        for c in cs:
            suffix=''.join(f'_{v}' for v in c)
            guard='true'+''.join(f' & {g}={v}' for v,g in zip(c,guards))
            decision=f'decision{suffix}'

            # Nested PRISM ternary: one URC command per (xhat,yhat)
            # instead of ten commands for decision=1,...,10.
            expr=str(thresholds[-1])
            for d in range(len(thresholds)-1,0,-1):
                expr=f'({decision}={d} ? {thresholds[d-1]} : {expr})'

            f.write(
                f"  [URC] {guard} -> "
                f"(max_gaussian_uncertainty'={expr});\n"
            )
        f.write('endmodule\n')

def add_turn(path,before,after):
    with open(path,'a') as f:
        f.write('module Turn\n  t : [0..2] init 0;\n')
        for a in before:f.write(f"  [{a}] (t=0) -> (t'=1);\n")
        f.write("  [URC] (t=1) -> (t'=2);\n")
        for a in after:f.write(f"  [{a}] (t=2) -> (t'=0);\n")
        f.write('endmodule\n')
