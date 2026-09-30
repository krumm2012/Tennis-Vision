"""Replay actual racket output to detect abrupt face changes and missing display frames."""
import argparse,json
from pathlib import Path
import numpy as np
from racket_stability import motion_metrics

def audit(path):
    data=json.loads(Path(path).read_text());rows=data['frames'];matrices=np.array([r['rotation_camera_columns'] for r in rows]);report=motion_metrics(matrices)
    report.update(frames=len(rows),hidden=sum(r['status']!='fitted' for r in rows),estimated=sum(r['status']=='fitted' and r['quality']!='silhouette_fitted' for r in rows))
    if not np.isfinite(matrices).all() or not np.allclose(np.einsum('nji,njk->nik',matrices,matrices),np.eye(3),atol=1e-6):raise ValueError('无效旋转')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('poses',type=Path);p.add_argument('--check',action='store_true');p.add_argument('--max-hidden',type=int,default=0);a=p.parse_args();r=audit(a.poses);print(json.dumps(r,ensure_ascii=False))
    if a.check and (r['step_max_deg']>45 or r['acceleration_p95_deg']>20 or r['hidden']>a.max_hidden):raise SystemExit('FAIL: 拍面突跳/高频抖动/缺拍')
