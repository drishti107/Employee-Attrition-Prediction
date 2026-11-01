"""Optional exact accelerator. Standard alternative: python interpretability.py.

Install FastTreeSHAP with a C++ compiler (see requirements-fast.txt), then:
python scripts/fast_interpretability.py --jobs 6
--wait-for-cv permits pipelining diagnostics behind a still-running CV job.
On this Unix runner file locks avoid duplicate work across concurrent invocations.
"""
from pathlib import Path
import argparse,hashlib,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
from interpretability import KEYS,ICACHE,analyze_fold,summarize
from train import CACHE,fingerprint
from fast_tree_backend import activate

def worker(key,fold,stamp,wait):
    import fcntl
    locks=ROOT/'.cache/explanation_locks';locks.mkdir(exist_ok=True,parents=True)
    with (locks/f'{fold:02d}_{key}.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        source=CACHE/f'{fold:02d}_{key}.joblib'
        if wait:
            while not source.exists():time.sleep(10)
            # CV serializes each checkpoint just before printing completion.
            while True:
                try:joblib.load(source);break
                except (EOFError,ValueError):time.sleep(2)
        activate()
        return analyze_fold(key,fold,stamp)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--wait-for-cv',action='store_true');p.add_argument('--start-fold',type=int,default=0)
    p.add_argument('--no-summary',action='store_true');a=p.parse_args()
    ICACHE.mkdir(parents=True,exist_ok=True)
    stamp=hashlib.sha256((ROOT/'interpretability.py').read_bytes()+fingerprint().encode()).hexdigest()
    with joblib.parallel_config(backend='multiprocessing'):
        joblib.Parallel(n_jobs=a.jobs)(joblib.delayed(worker)(key,fold,stamp,a.wait_for_cv)
            for fold in range(a.start_fold,25) for key in KEYS)
    if not a.no_summary:summarize()
