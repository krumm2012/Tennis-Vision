"""Sequence-level hand/racket candidate solver. No per-frame overrides.

SAM hand joints are model priors, not independent labelled image observations.
Contact is a soft capsule prior. This solver does NOT produce rigged hand meshes.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch


def rotation6d(x):
    a=torch.nn.functional.normalize(x[...,:3],dim=-1)
    b=x[...,3:]-a*(a*x[...,3:]).sum(-1,keepdim=True)
    b=torch.nn.functional.normalize(b,dim=-1)
    return torch.stack((a,b,torch.linalg.cross(a,b)),dim=-1)


def project(p,focal,principal):
    return p[...,:2]/p[...,2:].clamp_min(.05)*focal[:,None,None]+principal


def huber(x):
    a=x.abs();return torch.where(a<1,.5*x*x,a-.5)


def palm(j):
    w=j[:,20];forward=torch.nn.functional.normalize(j[:,[7,11,15,19]].mean(1)-w,dim=-1)
    a=j[:,7]-j[:,19];a=torch.nn.functional.normalize(a-forward*(a*forward).sum(-1,keepdim=True),dim=-1)
    return torch.stack((a,forward,torch.linalg.cross(a,forward)),dim=-1)


def shaft_distance(points,rotation,grip):
    rel=points-grip[:,None,:];axis=rotation[:,:,1]
    axial=(rel*axis[:,None,:]).sum(-1)
    radial=torch.linalg.vector_norm(rel-axial[...,None]*axis[:,None,:],dim=-1)
    return radial,axial


def solve(a,steps=1600,device='cpu'):
    torch.manual_seed(0)
    T=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    base=T(a['hand']);R0=T(a['rotation']);grip0=T(a['grip']);ring=T(a['ring']);model_grip=T(a['model_grip'])
    hweight=T(a['hand_weight']);obsweight=T(a['observation_weight']);edge=T(a['edge_weight']);dt=T(np.diff(a['time']))
    if (dt<=0).any():raise ValueError('Solver timestamps must be strictly increasing')
    if len(base)<3:raise ValueError('At least three frames are required')
    dt=dt[:,None,None]
    r=torch.nn.Parameter(torch.cat((R0[:,:,0],R0[:,:,1]),-1));g=torch.nn.Parameter(torch.zeros_like(grip0));h=torch.nn.Parameter(torch.zeros_like(base))
    sigma=T([.025]*20+[0.]);sigma[[3,7,11,15,19]]=.010
    parents=[];children=[]
    for tip in [0,4,8,12,16]:
        parents.extend([20,tip+3,tip+2,tip+1]);children.extend([tip+3,tip+2,tip+1,tip])
    lengths=torch.linalg.vector_norm(base[:,children]-base[:,parents],dim=-1)
    center=T(a['ellipse_center']);radii=T(a['ellipse_radii']);axes=T(a['ellipse_axes']);focal=T(a['focal']);principal=T(a['principal'])
    targets=T(a['hand_image']);grasp=T(a['grasp_weight'])
    opt=torch.optim.Adam([r,g,h],lr=.015);history=[]
    for step in range(steps):
        opt.zero_grad();R=rotation6d(r);grip=grip0+.035*torch.tanh(g);hand=base+sigma[None,:,None]*torch.tanh(h)
        points=torch.einsum('nij,kj->nki',R,ring-model_grip)+grip[:,None,:]
        uv=project(points,focal,principal);norm=torch.einsum('nki,nij->nkj',uv-center[:,None,:],axes)/radii[:,None,:]
        er=(torch.linalg.vector_norm(norm,dim=-1)-1)*torch.sqrt(radii.prod(-1))[:,None]
        observation=(huber(er/3).mean(-1)*obsweight).mean()+.3*(huber((uv.mean(1)-center)/4).mean(-1)*obsweight).mean()
        image=(huber((project(hand,focal,principal)-targets)/4).mean((1,2))*hweight).mean()
        prior=((hand-base)/.012).square().mean()
        bone=((torch.linalg.vector_norm(hand[:,children]-hand[:,parents],dim=-1)-lengths)/.002).square().mean()
        radial,axial=shaft_distance(hand[:,[0,4,8,12,16]],R,grip)
        # Finger-center contact includes a pad allowance beyond the 13mm handle radius.
        contact=(huber((radial-.020)/.008).mean(1)*grasp).mean()
        end=((torch.relu(-.035-axial).square()+torch.relu(axial-.095).square()).mean(1)*grasp).mean()/.015**2
        allrad,allax=shaft_distance(hand[:,:20],R,grip)
        active=((allax>-.045)&(allax<.11)).float()
        penetration=(torch.relu(.016-allrad).square()*active*grasp[:,None]).mean()/.008**2
        H=palm(hand);local=torch.einsum('nji,nj->ni',H,grip-hand[:,20]);relative=H.transpose(1,2)@R
        # No fixed shared grasp axis: only adjacent hand-relative changes are discouraged.
        stick=((local[1:]-local[:-1])/.01).square().mean(1)
        turn=huber((relative[1:]-relative[:-1])/.20).mean((1,2))
        temporal=((stick+.5*turn)*edge).mean()
        vel=(R[1:]-R[:-1])/dt
        acceleration=(huber((vel[1:]-vel[:-1])/8).mean((1,2))*torch.minimum(edge[1:],edge[:-1])).mean()
        anchor=((grip-grip0)/.025).square().mean()
        motion_gate=(dt[:,0,0]<.15).float()
        motion=(((R[1:]-R[:-1])/.18).square().mean((1,2))*motion_gate).mean()
        orientation_prior=huber((R-R0)/.5).mean()
        total=observation+.30*image+.15*prior+.4*bone+.30*contact+.1*end+.2*penetration+.20*temporal+.08*acceleration+.15*anchor+.35*motion+.12*orientation_prior
        if not torch.isfinite(total):raise FloatingPointError('Non-finite joint objective')
        total.backward();torch.nn.utils.clip_grad_norm_([r,g,h],10);opt.step()
        if step%200==0 or step==steps-1:
            metrics={k:round(float(v.detach()),5) for k,v in dict(total=total,observation=observation,contact=contact,bone=bone,temporal=temporal).items()};history.append({'step':step,**metrics});print(history[-1],flush=True)
    with torch.no_grad():
        R=rotation6d(r);grip=grip0+.035*torch.tanh(g);hand=base+sigma[None,:,None]*torch.tanh(h)
    return {k:v.cpu().numpy() for k,v in {'rotation':R,'grip':grip,'hand':hand}.items()},history


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--steps',type=int,default=1600);p.add_argument('--device',default='cpu');args=p.parse_args()
    result,history=solve(dict(np.load(args.input)),args.steps,args.device);np.savez_compressed(args.output,**result);args.output.with_suffix('.history.json').write_text(json.dumps(history,indent=2))
if __name__=='__main__':main()
