"""Export the provisioned MHR rig's actual UV layout for matching topology."""
import argparse,json
from pathlib import Path
import numpy as np
from run_records import sha256,write

def export(model,output):
    import torch
    rig=torch.jit.load(str(model),map_location='cpu');mesh=rig.character_torch.mesh
    array=lambda t:t.detach().cpu().numpy()
    faces=array(mesh.faces);uv=array(mesh.texcoords);uv_faces=array(mesh.texcoord_faces);rest=array(mesh.rest_vertices)/100
    if faces.shape!=uv_faces.shape or uv.shape[1:]!=(2,) or not np.isfinite(uv).all() or uv_faces.min()<0 or uv_faces.max()>=len(uv):raise ValueError('MHR UV layout invalid')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,faces=faces,uv=uv,uv_faces=uv_faces,rest_vertices=rest*np.array([1,-1,-1]),model_sha256=np.array(sha256(model)))
    write(output.with_suffix('.json'),{'model_sha256':sha256(model),'artifact_sha256':sha256(output),'faces':len(faces),'uv_vertices':len(uv),'vertices':len(rest),'units':'metres; axes x,-y,-z','source':'actual provisioned MHR UV topology'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.model,a.output)
