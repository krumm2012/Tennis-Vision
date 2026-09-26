"""Offline SAM pose smoothing and explicitly uncalibrated ground-relative placement.

Never substitutes the estimated camera plane for a measured court calibration.
The original 70 camera-frame joints and projection are retained for comparison.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d

FEET = [15,16,17,18,19,20]

def smooth_sequence(points, fps):
    """Symmetric Gaussian filtering with less filtering during fast motion (offline)."""
    speed = np.linalg.norm(np.gradient(points, 1/fps, axis=0), axis=-1)
    filtered = gaussian_filter1d(points, sigma=max(.5, fps*.04), axis=0, mode='nearest')
    # Preserve more of fast wrist/elbow motion; no causal-filter time lag.
    strength = .85 / (1 + (speed/2.0)**2)
    return points + strength[...,None]*(filtered-points)

def ground_basis(camera_joints):
    """Estimate up from the torso; this is NOT a surveyed ground plane."""
    torso = camera_joints[:,[5,6]].mean(1)-camera_joints[:,[9,10]].mean(1)
    up = np.median(torso,axis=0); up /= np.linalg.norm(up)
    right = np.array([1.,0.,0.]); right -= up*np.dot(right,up); right /= np.linalg.norm(right)
    forward = np.cross(right,up); forward /= np.linalg.norm(forward)
    return np.stack([right,up,forward],axis=1)

def process(raw):
    fps=float(raw['fps']); frames=raw['frames']
    if not all(f.get('result') for f in frames):
        raise ValueError('Missing frames require explicit gap handling; refusing to bridge them silently')
    j=np.array([f['result']['pred_keypoints_3d'] for f in frames],dtype=float)
    t=np.array([f['result']['pred_cam_t'] for f in frames],dtype=float)
    image=np.array([f['result']['pred_keypoints_2d'] for f in frames],dtype=float)
    if j.shape[1:]!=(70,3) or not np.isfinite(j).all() or not np.isfinite(t).all():
        raise ValueError('Expected finite MHR70 joints and camera translations')
    roots=j[:,[9,10]].mean(1); local=j-roots[:,None,:]
    smooth=smooth_sequence(local,fps); smooth-=smooth[:,[9,10]].mean(1)[:,None,:]
    camera_root=gaussian_filter1d(roots+t,max(.5,fps*.08),axis=0,mode='nearest')
    camera=smooth+camera_root[:,None,:]
    basis=ground_basis(camera); ground=camera@basis
    floor=float(np.median(ground[:,FEET,1].min(1)))
    origin=np.array([ground[0,[9,10],0].mean(),floor,ground[0,[9,10],2].mean()])
    relative=ground-origin
    # Keep real vertical changes. No per-frame forced foot-to-floor snapping.
    def accel(a):return float(np.sqrt(np.mean(np.diff(a,n=2,axis=0)**2))*fps**2)
    metrics={'raw_acceleration_rms':accel(local),'smoothed_acceleration_rms':accel(smooth),'median_joint_adjustment_model_units':float(np.median(np.linalg.norm(smooth-local,axis=-1))),'max_joint_adjustment_model_units':float(np.max(np.linalg.norm(smooth-local,axis=-1)))}
    metrics['acceleration_reduction_percent']=100*(1-metrics['smoothed_acceleration_rms']/metrics['raw_acceleration_rms'])
    out={'fps':fps,'backend':'SAM 3D Body ViT-H','coordinate_status':'estimated_relative_ground_NOT_calibrated_court','axes':'X screen-right projected on estimated plane, Y estimated up, Z forward; model scale only','ground_basis_camera_columns':basis.tolist(),'ground_origin':origin.tolist(),'metrics':metrics,'frames':[]}
    for k,f in enumerate(frames):
        out['frames'].append({'frame':f['frame'],'time':f['time'],'raw':local[k].tolist(),'smooth':smooth[k].tolist(),'ground':relative[k].tolist(),'image':image[k].tolist(),'root':relative[k,[9,10]].mean(0).tolist()})
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('output/sam3d_cloud/sam_pose.json'));p.add_argument('--output',type=Path,default=Path('output/sam3d_cloud/processed_pose.json'));a=p.parse_args()
    out=process(json.loads(a.input.read_text()));a.output.write_text(json.dumps(out,separators=(',',':')));print(json.dumps(out['metrics'],indent=2))
