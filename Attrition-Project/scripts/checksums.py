"""Refresh checksums of distributable files; excludes caches and bytecode."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest={}
for path in sorted(ROOT.rglob('*')):
    if not path.is_file():continue
    rel=path.relative_to(ROOT)
    if any(part in {'.cache','__pycache__','.venv','.git'} for part in rel.parts):continue
    if rel.name=='FILE_CHECKSUMS.json' or rel.suffix=='.pyc':continue
    manifest[rel.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/'FILE_CHECKSUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Checksummed',len(manifest),'files')
