"""Reproduce the fixed upgraded protocol in a new directory."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--destination',required=True);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
src=Path(__file__).resolve().parent;dest=Path(a.destination).expanduser().resolve()
if dest.exists():raise SystemExit('Destination already exists; choose a new path.')
if dest==src or src in dest.parents:raise SystemExit('Choose a destination outside the supplied project.')
shutil.copytree(src,dest,ignore=shutil.ignore_patterns('.cache','__pycache__','.venv','.git'))
for folder in ['results','models','outputs']:
    shutil.rmtree(dest/folder);(dest/folder).mkdir()
(dest/'FILE_CHECKSUMS.json').unlink(missing_ok=True)
env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
commands=[['train.py','--jobs',str(a.jobs)],['interpretability.py','--jobs',str(a.jobs)],
 ['scripts/build_outputs.py'],['scripts/summarize_study.py'],['scripts/build_report.py'],
 ['scripts/build_notebook.py'],['scripts/execute_notebook.py'],
 ['predict.py','--csv','data/sample_employees.csv','--row','0','--output','outputs/cli_example.json'],
 ['verify.py','--with-cache','--save'],['tests/test_app.py'],['scripts/checksums.py']]
for command in commands:subprocess.run([sys.executable,*command],cwd=dest,env=env,check=True)
print('Reproduction complete:',dest)
