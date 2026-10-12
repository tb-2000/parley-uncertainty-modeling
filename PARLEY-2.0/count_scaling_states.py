#!/usr/bin/env python3
"""Count reachable PRISM states/transitions for fixed URC policies.
Run: python3 count_scaling_states.py scaling_prism_models.zip --prism /path/to/prism
"""
import argparse, csv, re, shutil, subprocess, tempfile, zipfile
from pathlib import Path

DECL = re.compile(r'^\s*evolve\s+int\s+(decision_\d+_\d+)\s*\[(\d+)\.\.(\d+)\]\s*;', re.M)
COUNT = {'states': re.compile(r'(?im)^\s*States\s*:\s*([\d,]+)'),
         'transitions': re.compile(r'(?im)^\s*Transitions\s*:\s*([\d,]+)')}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--prism',default='prism')
    p.add_argument('--output',type=Path,default=Path('scaling_state_counts.csv'))
    p.add_argument('--timeout',type=int,default=1800)
    args=p.parse_args()
    if not shutil.which(args.prism) and not Path(args.prism).is_file():
        p.error('PRISM executable not found. Supply --prism /path/to/prism')
    results=[]
    with tempfile.TemporaryDirectory(prefix='prism_scaling_') as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(args.archive) as z:
            for name in z.namelist():
                if name.startswith('/') or '..' in Path(name).parts:
                    raise ValueError('Unsafe ZIP member: '+name)
            z.extractall(tmp)
        models=sorted(tmp.rglob('*_umc.prism'))
        if len(models)!=16:
            print(f'WARNING: Expected 16 UMC models, found {len(models)}')
        for source in models:
            original=source.read_text(encoding='utf-8')
            decisions=DECL.findall(original)
            if not decisions: raise ValueError('No evolve decisions in '+str(source))
            model=source.parent.name
            index=int(re.search(r'_(\d+)_umc\.prism$',source.name).group(1))
            size=(5,10,15,20)[index]
            for policy in ('min','max'):
                def replace(match):
                    name,lo,hi=match.groups()
                    value=lo if policy=='min' else hi
                    return f'const int {name} = {value};'
                instantiated=DECL.sub(replace,original)
                target=tmp/f'fixed_{model}_{size}_{policy}.prism'
                target.write_text(instantiated,encoding='utf-8')
                print(f'Counting {model:17s} {size:2d}x{size:<2d} {policy} ...',flush=True)
                try:
                    proc=subprocess.run([args.prism,str(target),'-build'],capture_output=True,text=True,timeout=args.timeout)
                    output=proc.stdout+'\n'+proc.stderr
                    values={key: (int(m.group(1).replace(',','')) if (m:=regex.search(output)) else None)
                            for key,regex in COUNT.items()}
                    if proc.returncode or None in values.values():
                        status='ERROR'
                        log=args.output.parent/f'prism_error_{model}_{size}_{policy}.log'
                        log.write_text(output,encoding='utf-8')
                        print('  ERROR, log:',log)
                    else:
                        status='OK'
                        print('  states:',values['states'],'transitions:',values['transitions'])
                except subprocess.TimeoutExpired:
                    values={'states':None,'transitions':None};status='TIMEOUT'
                results.append(dict(model=model,grid=f'{size}x{size}',policy=policy,
                    threshold=(decisions[0][1] if policy=='min' else decisions[0][2]),
                    decisions=len(decisions),states=values['states'],transitions=values['transitions'],status=status))
                with args.output.open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(results[0]))
                    writer.writeheader();writer.writerows(results)
    print('Results:',args.output)

if __name__=='__main__': main()
