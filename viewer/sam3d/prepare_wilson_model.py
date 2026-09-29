"""Normalize the downloaded Wilson GLB and bake its material colors for WebGL."""
import io,json,struct
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull


def prepare(source=Path('/Users/krum5539/Downloads/wilson_blade_team_tennis_racket-2.glb'),out=Path('output/sam3d_cloud')):
    blob=source.read_bytes();size=struct.unpack_from('<I',blob,12)[0]
    doc=json.loads(blob[20:20+size]);binary=blob[28+size:]
    def view(i):
        v=doc['bufferViews'][i];start=v.get('byteOffset',0);return binary[start:start+v['byteLength']]
    def accessor(i):
        a=doc['accessors'][i];v=doc['bufferViews'][a['bufferView']];types={5126:'<f4',5125:'<u4',5123:'<u2'}
        width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];dtype=np.dtype(types[a['componentType']])
        return np.ndarray((a['count'],width),dtype,buffer=view(a['bufferView']),offset=a.get('byteOffset',0),strides=(v.get('byteStride',width*dtype.itemsize),dtype.itemsize)).copy()
    images=[np.asarray(Image.open(io.BytesIO(view(im['bufferView']))).convert('RGB')) for im in doc['images']]
    meshes=[]
    def visit(i,parent):
        node=doc['nodes'][i];m=np.array(node.get('matrix',np.eye(4).flatten(order='F'))).reshape(4,4,order='F');world=parent@m
        if 'mesh' in node:
            for p in doc['meshes'][node['mesh']]['primitives']:
                xyz=accessor(p['attributes']['POSITION']);xyz=np.c_[xyz,np.ones(len(xyz))]@world.T
                mat=doc['materials'][p['material']]['pbrMetallicRoughness'];color=np.tile(mat.get('baseColorFactor',[1,1,1,1])[:3],(len(xyz),1))
                if 'baseColorTexture' in mat:
                    img=images[doc['textures'][mat['baseColorTexture']['index']]['source']];uv=accessor(p['attributes']['TEXCOORD_0']);pixels=np.floor(uv*np.array([img.shape[1]-1,img.shape[0]-1])).astype(int);pixels=np.clip(pixels,0,[img.shape[1]-1,img.shape[0]-1]);color=color*img[pixels[:,1],pixels[:,0]]/255
                meshes.append((xyz[:,:3],accessor(p['indices']).ravel(),color,p['material']))
        for child in node.get('children',[]):visit(child,world)
    for i in doc['scenes'][doc.get('scene',0)]['nodes']:visit(i,np.eye(4))
    all_xyz=np.vstack([m[0] for m in meshes]);lo=all_xyz.min(0);hi=all_xyz.max(0);scale=.685/(hi[2]-lo[2]);packed=[];strings=[];normalized_all=[]
    for xyz,indices,color,material in meshes:
        normalized=np.c_[xyz[:,0],hi[2]-xyz[:,2],xyz[:,1]]*scale
        normalized_all.append(normalized)
        packed.append(np.c_[normalized[indices],color[indices]].astype('<f4'))
        if material==1:strings.append(normalized)
    string=np.vstack(strings);full=np.vstack(normalized_all);head=full[full[:,1]>=string[:,1].min()-.015];hull=head[ConvexHull(head[:,:2]).vertices,:2];closed=np.vstack([hull,hull[0]]);dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))];query=np.linspace(0,dist[-1],32,endpoint=False);outline=np.c_[np.interp(query,dist,closed[:,0]),np.interp(query,dist,closed[:,1]),np.zeros(32)]
    model={'length_m':.685,'head_half_width_m':float((hi[0]-lo[0])*scale/2),'head_center_y_m':float((string[:,1].min()+string[:,1].max())/2),'throat_y_m':float(string[:,1].min()),'grip_y_m':.10,'head_outline':outline.tolist(),'asset_file':'wilson_mesh.bin','dimensions_measured':False}
    out.mkdir(parents=True,exist_ok=True);np.vstack(packed).tofile(out/'wilson_mesh.bin')
    (out/'wilson_model.json').write_text(json.dumps({'model':model,'source':str(source),'asset':doc['asset'],'triangles':len(np.vstack(packed))//3,'scale':scale,'axis_mapping':'(x, z_max-z, y); uniform assumed length 0.685 m','grip_anchor_status':'provisional 0.10 m from butt; requires real grip review'},indent=2))
    return model

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
