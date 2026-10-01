"""Differentiable grip proxies; never a claim of anatomical surface contact.

All inputs use camera metres. Keep directed shaft constraints separate from
symmetric face constraints. Evidence weights are fixed before optimisation.
"""
import torch
import torch.nn.functional as F


def hand_geometry(world, radius):
    bases=world[:,[28,32,36,40]];prox=world[:,[27,31,35,39]]
    forward=F.normalize(bases.mean(1)-world[:,41],dim=-1)
    across=bases[:,0]-bases[:,3]
    across=F.normalize(across-(across*forward).sum(-1,keepdim=True)*forward,dim=-1)
    normal=F.normalize(torch.cross(across,forward,dim=-1),dim=-1)
    curl=((world[:,[25,29,33,37]]-bases)*normal[:,None]).sum(-1).mean(1)
    center=(bases+prox).mean(1)*.5+normal*torch.where(curl>=0,radius,-radius)[:,None]
    axis=F.normalize((bases[:,0]+prox[:,0])-(bases[:,3]+prox[:,3]),dim=-1)
    return center,axis,prox


def grip_terms(world, rotation, translation, grip, radius, hand_weight, handle_length=.16):
    center,axis,prox=hand_geometry(world,radius)
    anchor=torch.einsum('bij,j->bi',rotation,grip)+translation
    local=torch.einsum('bji,bkj->bki',rotation,prox-translation[:,None])
    radial=torch.linalg.vector_norm(local[:,:,[0,2]],dim=-1)
    # Both separated and penetrating joints have gradients. The old one-sided
    # penetration term became zero for fingers entirely detached from the handle.
    contact=F.smooth_l1_loss(radial,torch.full_like(radial,radius),beta=.005,reduction='none').mean(1)
    axial=(F.relu(-local[:,:,1])+F.relu(local[:,:,1]-handle_length)).mean(1)
    directed=1-(rotation[:,:,1]*axis).sum(-1).clamp(-1,1)
    return {'anchor':(((anchor-center)**2).mean(-1)*(.25+hand_weight)).mean()*600,
            'finger_distance':(contact*(.25+hand_weight)).mean()*2,
            'handle_segment':(axial*(.25+hand_weight)).mean()*2,
            'hand_direction':(directed*hand_weight).mean()*.12}


def directed_image_loss(projected,observed):
    """Two ordered endpoints; do not treat an axis flipped 180 degrees as equal."""
    a=projected[1]-projected[0];b=observed[1]-observed[0]
    return 1-(F.normalize(a,dim=-1)*F.normalize(b,dim=-1)).sum().clamp(-1,1)


def reliability(joints,roots,focal,image_size,marks,train):
    """Downweight a conflicting SAM hand prior using TRAIN evidence only."""
    world=joints+roots[:,None];center,axis,_=hand_geometry(world,.013)
    weight=torch.full((len(joints),),.1,device=joints.device)
    for row in marks['frames']:
        i=row['frame'];seen=row['points']
        if i not in train or not {'handle_end','tip'}<=set(seen):continue
        p=torch.stack((center[i],center[i]+axis[i]*.2))
        if torch.any(p[:,2]<=.1):weight[i]=.03;continue
        uv=p[:,:2]/p[:,2:]*focal[i]+image_size/2
        observed=torch.as_tensor([seen['handle_end'],seen['tip']],device=joints.device,dtype=joints.dtype)
        if torch.linalg.vector_norm(observed[1]-observed[0])<4:continue
        weight[i]=.03 if directed_image_loss(uv,observed)>.1808479557 else .4 # cos(35 degrees)
    return weight


def summary(joints,roots,rotation,translation,grip,radius):
    """Full-clip diagnostics, independent of which observations train the fit."""
    world=joints+roots[:,None];center,axis,prox=hand_geometry(world,radius)
    anchor=torch.einsum('bij,j->bi',rotation,grip)+translation
    local=torch.einsum('bji,bkj->bki',rotation,prox-translation[:,None])
    angle=torch.rad2deg(torch.acos((rotation[:,:,1]*axis).sum(-1).clamp(-1,1)))
    radial=(torch.linalg.vector_norm(local[:,:,[0,2]],dim=-1)-radius).abs().mean(1)*1000
    def stats(values):return {'median':float(values.median()),'p95':float(torch.quantile(values,.95))}
    steps=rotation[1:]@rotation[:-1].transpose(-1,-2)
    angles=torch.rad2deg(torch.acos(((steps.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
    acceleration=steps[1:]@steps[:-1].transpose(-1,-2)
    accel_angles=torch.rad2deg(torch.acos(((acceleration.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
    return {'angular_acceleration_deg_frame2':stats(accel_angles) if len(accel_angles) else {'median':0.,'p95':0.},
            'palm_gap_mm':stats(torch.linalg.vector_norm(anchor-center,dim=-1)*1000),
            'finger_radial_gap_mm':stats(radial),'handle_segment_gap_mm':stats((F.relu(-local[:,:,1])+F.relu(local[:,:,1]-.16)).mean(1)*1000),'hand_shaft_deg':stats(angle),
            'rotation_step_deg':stats(angles) if len(angles) else {'median':0.,'p95':0.}}
