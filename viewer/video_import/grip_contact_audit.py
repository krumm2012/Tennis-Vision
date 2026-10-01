"""Audit the demo MCP/PIP grip prior, keeping contact proxies explicitly provisional."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from run_records import write,sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sam3d'))
from fit_wilson_grip import palm_frame,grip_contact


def audit(poses, joints, roots, radius=.013):
    rows=[];model=poses['model'];grip=np.array([0,model['grip_y_m'],0])
    for i,row in enumerate(poses['frames']):
        if row['status']!='fitted':continue
        wrist,hand=palm_frame(joints[i]);offset,axis=grip_contact(joints[i]);target=wrist+hand@offset+roots[i]
        r=np.array(row['rotation_camera_columns']);t=np.array(row['translation_camera_m']);anchor=r@grip+t
        proximal=joints[i,[27,31,35,39]]+roots[i];local=(proximal-t)@r
        radial=np.linalg.norm(local[:,[0,2]],axis=1);on=(local[:,1]>=0)&(local[:,1]<.16)
        rows.append({'frame':i,'palm_center_gap_mm':float(np.linalg.norm(anchor-target)*1000),
                     'shaft_vs_mcp_pip_deg':float(np.degrees(np.arccos(np.clip(r[:,1]@(hand@axis),-1,1)))),
                     'proximal_joint_radial_gap_mean_mm':float(np.mean(np.abs(radial-radius))*1000),
                     'joints_inside_assumed_grip':int(np.count_nonzero(on&(radial<radius)))})
    stats=lambda name:{'median':float(np.median([r[name] for r in rows])),'p95':float(np.percentile([r[name] for r in rows],95))} if rows else {}
    return {'method':'demo MCP/PIP palm corridor + directed shaft; same anatomical method',
            'frames':rows,'palm_center_gap_mm':stats('palm_center_gap_mm'),'shaft_vs_mcp_pip_deg':stats('shaft_vs_mcp_pip_deg'),
            'proximal_joint_radial_gap_mean_mm':stats('proximal_joint_radial_gap_mean_mm'),
            'handle_radius_assumed_m':radius,'mesh_surface_collision_checked':False,
            'physical_grip_bevel_verified':False,'grip_style_classified':False}


def run(folder):
    folder=Path(folder);out=folder/'result';record=json.loads((folder/'record.json').read_text());archive=folder/'attempts'/f"{record['attempt']:04d}"/'work/reconstruction.npz'
    poses=json.loads((out/'racket_poses.json').read_text());meta=json.loads((out/'mesh_meta.json').read_text())
    if poses['video_sha256']!=meta['video_sha256']:raise ValueError('握拍审计与当前视频不一致')
    with np.load(archive,allow_pickle=False) as d:joints=d['joints'].copy();roots=d['source_roots'].copy()
    if (out/'multiview_constraints.npz').exists():
        report=json.loads((out/'multiview_constraints_report.json').read_text())
        if report['constraints_sha256']!=sha256(out/'multiview_constraints.npz') or report['archive_sha256']!=sha256(archive):raise ValueError('手部约束来源不匹配')
        with np.load(out/'multiview_constraints.npz',allow_pickle=False) as d:joints=d['joints'].copy()
    report=audit(poses,joints,roots);report.update(video_sha256=meta['video_sha256'],archive_sha256=sha256(archive),racket_poses_sha256=sha256(out/'racket_poses.json'))
    write(out/'grip_contact_report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();r=run(a.dataset);print(json.dumps({k:v for k,v in r.items() if k!='frames'}))
