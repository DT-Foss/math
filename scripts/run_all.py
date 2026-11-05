from pathlib import Path
import os, subprocess, sys
root=Path(__file__).resolve().parents[1]
env=os.environ.copy(); env["PYTHONPATH"]=str(root/"src")
for p in sorted((root/"experiments").glob("e*.py")):
    print(f"== {p.name} ==", flush=True)
    subprocess.run([sys.executable,str(p)],cwd=p.parent,env=env,check=True)
