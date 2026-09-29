"""Resume bounded CPU SAM2 chunks and merge only verified CPU outputs."""
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    base=Path('output/sam3d_cloud/racket_work')
    rows=[]
    for start in range(0,250,32):
        end=min(start+32,250)
        output=base/f'masks_{start}_{end}.json'
        saved=json.loads(output.read_text()) if output.exists() else {}
        expected=list(range(start,end))
        if saved.get('device')!='cpu' or [r['frame'] for r in saved.get('frames',[])]!=expected:
            subprocess.run([sys.executable,str(Path(__file__).with_name('track_racket_masks.py')),
                            '--device','cpu','--start',str(start),'--end',str(end)],
                           env={**os.environ,'OMP_NUM_THREADS':'4'},check=True)
            saved=json.loads(output.read_text())
        rows.extend(saved['frames'])
        print(f'{end}/250 completed',flush=True)
    saved['frames']=rows;saved.pop('seeds',None)
    target=base.parent/'racket_tracked_masks.json'
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(saved));temporary.replace(target)

if __name__=='__main__':main()
