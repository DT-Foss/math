from __future__ import annotations
import argparse, json, os, platform, subprocess, sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--out',default='results/local_run.json')
args=parser.parse_args()
env=os.environ.copy(); env['PYTHONPATH']=str(root/'src')
env.setdefault('OPENBLAS_NUM_THREADS','1'); env.setdefault('OMP_NUM_THREADS','1')
records=[]
for p in sorted((root/'experiments').glob('e*.py')):
    cp=subprocess.run([sys.executable,str(p)],cwd=p.parent,env=env,check=True,text=True,capture_output=True)
    line=[x for x in cp.stdout.splitlines() if x.strip()][-1]
    records.append(json.loads(line))
cpu='unknown'
try:
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.lower().startswith('model name'):
            cpu=line.split(':',1)[1].strip(); break
except OSError:
    cpu=platform.processor() or 'unknown'
payload={
  'run_id':'chatgpt-vm-2026-11-05',
  'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'cpu':cpu,
                 'OPENBLAS_NUM_THREADS':env.get('OPENBLAS_NUM_THREADS'),'OMP_NUM_THREADS':env.get('OMP_NUM_THREADS')},
  'experiments':records,
}
out=root/args.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(out)
