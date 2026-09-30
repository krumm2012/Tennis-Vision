"""Per-attempt records and hashes shared by GPU worker and local publisher."""
import hashlib,json,datetime,subprocess
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)

def revision(root):
    r=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else 'unknown'
