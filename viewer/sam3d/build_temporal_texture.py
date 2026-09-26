"""Cache real observations on corresponding SAM vertices, at most +/-3 frames."""
from pathlib import Path
import json,cv2,numpy as np,argparse
from numba import njit
parser=argparse.ArgumentParser();parser.add_argument('--mask-atlas',default='person_masks_soft.png');parser.add_argument('--output-prefix',default='temporal_texture');args=parser.parse_args()
r=Path(__file__).parent;meta=json.loads((r/'mesh_meta.json').read_text());F=meta['frames'];V=meta['vertices'];vertices=np.fromfile(r/'mesh_local.bin',dtype='<f4').reshape(F,V,3);faces=np.fromfile(r/'mesh_faces.bin',dtype='<u4').reshape(-1,3);g=json.loads((r/'mirror_geometry.json').read_text());normal=np.array(g['normal_camera']);distance=g['distance_camera_m'];atlas=cv2.imread(str(r/args.mask_atlas));cap=cv2.VideoCapture(str(r/'video.mp4'));colors=np.zeros((F,V,3),np.uint8);confidence=np.zeros((F,V),np.float32);sources=np.zeros((F,V),np.uint8)
@njit
def raster(uv,z,faces):
 depth=np.full((720,1280),1e6,np.float32)
 for face in faces:
  a,b,c=face;ax,ay=uv[a];bx,by=uv[b];cx,cy=uv[c]
  if min(z[a],z[b],z[c])<=.1:continue
  den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
  if abs(den)<1e-7:continue
  xmin=max(0,int(np.floor(min(ax,bx,cx))));xmax=min(1279,int(np.ceil(max(ax,bx,cx))));ymin=max(0,int(np.floor(min(ay,by,cy))));ymax=min(719,int(np.ceil(max(ay,by,cy))))
  for y in range(ymin,ymax+1):
   for x in range(xmin,xmax+1):
    aa=((by-cy)*(x+.5-cx)+(cx-bx)*(y+.5-cy))/den;bb=((cy-ay)*(x+.5-cx)+(ax-cx)*(y+.5-cy))/den;cc=1-aa-bb
    if min(aa,bb,cc)<0:continue
    zz=1/(aa/z[a]+bb/z[b]+cc/z[c])
    if zz<depth[y,x]:depth[y,x]=zz
 return depth
for i in range(F):
 ok,im=cap.read();assert ok
 X=vertices[i]+meta['source_roots'][i];Y=X-2*(X@normal-distance)[:,None]*normal;views=[X,Y];uvs=[a[:,:2]/a[:,2:]*meta['focal'][i]+[640,360] for a in views];depths=[raster(uv.astype(np.float32),a[:,2].astype(np.float32),faces) for uv,a in zip(uvs,views)];tile=atlas[i//16*360:(i//16+1)*360,i%16*640:(i%16+1)*640]
 for k,(uv,P,dep) in enumerate(zip(uvs,views,depths)):
  x=np.clip(uv[:,0].astype(int),0,1279);y=np.clip(uv[:,1].astype(int),0,719);mask=cv2.remap(tile[:,:,2-k].astype(np.float32)/255,(uv[:,0]/2).astype(np.float32)[:,None],(uv[:,1]/2).astype(np.float32)[:,None],cv2.INTER_LINEAR).ravel();valid=(uv[:,0]>=1)&(uv[:,0]<1279)&(uv[:,1]>=1)&(uv[:,1]<719)&(P[:,2]>0)&(abs(P[:,2]-dep[y,x])<.012)&(mask>.9)
  if k:valid&=(uv[:,0]>537)&(uv[:,0]<972)&(uv[:,1]<230)&(depths[0][y,x]>1e5)
  score=np.where(valid,mask,0);replace=score>confidence[i];colors[i,replace]=im[y[replace],x[replace],::-1];confidence[i,replace]=score[replace];sources[i,replace]=k+1
 if i%50==0:print('observations',i,flush=True)
# RGB,confidence,source(1 real/2 mirror),signed frame offset encoded +3.
cache=np.zeros((F,V,6),np.uint8);source_frames=np.full((F,V),-1,np.int16)
for i in range(F):
 best=np.zeros(V);speed=np.linalg.norm(vertices[min(i+1,F-1)]-vertices[max(i-1,0)],axis=1)*12.5
 for j in range(max(0,i-3),min(F,i+4)):
  if j==i:continue
  age=abs(j-i);motion=np.linalg.norm(vertices[j]-vertices[i],axis=1);score=confidence[j]*(1-age*.15);valid=(confidence[j]>.9)&(motion<.10)&(speed<3.0);score=np.where(valid,score,0);take=score>best;best[take]=score[take];cache[i,take,:3]=colors[j,take];cache[i,take,3]=(score[take]*255).astype(np.uint8);cache[i,take,4]=sources[j,take];cache[i,take,5]=j-i+3;source_frames[i,take]=j
cache.tofile(r/(args.output_prefix+'.bin'));source_frames.tofile(r/(args.output_prefix+'_sources.bin'));assert np.all(np.abs(source_frames[source_frames>=0]-np.broadcast_to(np.arange(F)[:,None],source_frames.shape)[source_frames>=0])<=3)
(r/(args.output_prefix+'_meta.json')).write_text(json.dumps({'frames':F,'vertices':V,'stride':6,'max_frame_offset':3,'max_time_offset_seconds':.12,'source_mask_threshold':.9,'max_vertex_motion_model_m':.1,'max_speed_model_m_s':3,'source_types':{'1':'real','2':'mirror'},'format':'uint8 RGB confidence source offset_plus_3','candidate_vertices':int(np.count_nonzero(cache[:,:,3]))},indent=2));print('DONE',np.count_nonzero(cache[:,:,3]),flush=True)
