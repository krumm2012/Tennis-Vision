"""Geodesic sequence smoothing and evidence-bounded gap completion."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

def smooth_vectors(values,fps):
    values=np.asarray(values,float);window=min(len(values)//2*2-1,max(3,round(.20*fps)//2*2+1))
    if window<3:return values.copy()
    robust=median_filter(values,size=(3,1),mode='nearest')
    return savgol_filter(robust,window,2,axis=0,mode='interp')

def stable_rotations(matrices,weights,fps,acceleration_strength=5,fixed_frames=()):
    """Fit on SO(3), penalizing angular acceleration, not matrix components.

    Constant angular velocity has zero acceleration. Low-evidence frames may
    move farther; this is an estimated display sequence, not new observations.
    """
    matrices=np.asarray(matrices);count=len(matrices)
    fixed=np.array(sorted(set(fixed_frames)),dtype=int)
    if len(fixed) and (fixed.min()<0 or fixed.max()>=count):raise ValueError('Fixed rotation index outside sequence')
    if count<3:return matrices.copy()
    weights=np.asarray(weights);scale=fps/25
    movable=np.array([i for i in range(count) if i not in set(fixed)],dtype=int)
    initial=Rotation.from_matrix(matrices).as_rotvec()
    if not len(movable):return matrices.copy()
    def unpack(x):
        vectors=initial.copy();vectors[movable]=x.reshape(-1,3)
        rs=Rotation.from_rotvec(vectors).as_matrix();rs[fixed]=matrices[fixed]
        return rs
    def residual(x):
        rs=unpack(x)
        anchor=Rotation.from_matrix(np.einsum('nji,njk->nik',matrices,rs)).as_rotvec()*weights[:,None]
        # Spatial (camera-frame) angular increments avoid changing local axes.
        velocity=Rotation.from_matrix(np.einsum('nij,nkj->nik',rs[1:],rs[:-1])).as_rotvec()
        return np.r_[anchor.ravel(),(.5*scale*velocity).ravel(),(acceleration_strength*scale**2*np.diff(velocity,axis=0)).ravel()]
    sparsity=lil_matrix((count*3+(count-1)*3+(count-2)*3,count*3),dtype=int)
    for i in range(count):sparsity[3*i:3*i+3,3*i:3*i+3]=1
    start=count*3
    for i in range(count-1):sparsity[start+3*i:start+3*i+3,3*i:3*i+6]=1
    start+=(count-1)*3
    for i in range(count-2):sparsity[start+3*i:start+3*i+3,3*i:3*i+9]=1
    columns=(movable[:,None]*3+np.arange(3)).ravel()
    fit=least_squares(residual,initial[movable].ravel(),jac_sparsity=sparsity.tocsr()[:,columns],max_nfev=80,ftol=1e-5)
    return unpack(fit.x)

def supported_frames(observed,fps,max_gap_seconds=.6):
    """Bounded loss can be estimated; leading/trailing/long absence stays hidden."""
    observed=np.asarray(observed,bool);supported=observed.copy();keys=np.flatnonzero(observed)
    for a,b in zip(keys,keys[1:]):
        if (b-a-1)/fps<=max_gap_seconds:supported[a:b+1]=True
    return supported

def motion_metrics(matrices):
    matrices=np.asarray(matrices)
    velocity=Rotation.from_matrix(np.einsum('nij,nkj->nik',matrices[1:],matrices[:-1])).as_rotvec()
    steps=np.rad2deg(np.linalg.norm(velocity,axis=1));acc=np.rad2deg(np.linalg.norm(np.diff(velocity,axis=0),axis=1))
    return {'step_p95_deg':float(np.percentile(steps,95)),'step_max_deg':float(steps.max()),'acceleration_p95_deg':float(np.percentile(acc,95)),'acceleration_max_deg':float(acc.max())}
