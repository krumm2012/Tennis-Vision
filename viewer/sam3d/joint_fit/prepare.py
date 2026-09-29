"""Build sequence observations from automatic tracks; no chosen/manual keyframes."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fit_racket_video import head_ellipse
from fit_wilson_grip import grip_contact,palm_frame


def continuity_edges(times,quality,grasp,max_gap=.25):
    times=np.asarray(times);quality=np.asarray(quality);grasp=np.asarray(grasp)
    if np.any(np.diff(times)<=0):raise ValueError('Non-increasing timestamps')
    edge=np.minimum(grasp[:-1],grasp[1:]).astype(float)
    valid=np.flatnonzero(quality>.25)
    if len(valid)<2:return edge*0
    edge[:valid[0]]=0;edge[valid[-1]:]=0
    for a,b in zip(valid[:-1],valid[1:]):
        if times[b]-times[a]>max_gap:edge[a:b]=0
    return edge


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=Path('output/sam3d_cloud'));ap.add_argument('--out',type=Path);args=ap.parse_args();root=args.data;out=args.out or root/'joint_fit_v4';out.mkdir(parents=True,exist_ok=True)
    read=lambda n:json.loads((root/n).read_text())
    native=np.load(root/'native_hand_sequence/sequence.npz');mapping=read('hand_probe/frame_mapping.json')['frames'];diag=read('native_hand_sequence/diagnostics.json')['frames'];old=read('racket_poses_v2.json');masks=read('racket_tracked_masks.json')['frames'];dets=read('racket_yolo_candidates.json')['frames']
    n=len(mapping);assert n==len(old['frames'])==len(masks)==len(dets)
    index=np.array([r['native_frame'] for r in mapping]);times=np.array([r['viewer_time'] for r in mapping]);j=native['joints'][index]+native['translations'][index,None,:];hand=j[:,21:42];uv=native['joints2d'][index]/2
    model=dict(old['model'],grip_y_m=.045);offsets=[];grips=[]
    for row in j:
        off,_=grip_contact(row);w,H=palm_frame(row);offsets.append(off);grips.append(w+H@off)
    centers=[];radii=[];axes=[];scores=[];grasp=[];sources=[];sharp=[]
    cap=cv2.VideoCapture(str(root/'video.mp4'))
    for i in range(n):
        ok,im=cap.read()
        if not ok:raise ValueError(f'Missing video frame {i}')
        height,width=im.shape[:2];wrist=uv[i,41];options=[]
        for c in dets[i]['candidates']:
            poly=np.array(c['polygon']);e=head_ellipse(poly)
            if e is None:continue
            dist=np.linalg.norm(poly-wrist,axis=1).min()
            if dist<65 and max(e[1])<180:options.append((c['confidence']*np.exp(-dist/30),e,dist))
        best=max(options,key=lambda x:x[0]) if options else None
        e=head_ellipse(masks[i]['polygon']) if masks[i]['mask_valid'] else None
        source='tracked_mask' if e is not None else 'missing'
        if best and best[0]>.04 and (e is None or np.linalg.norm(e[0]-best[1][0])>20):e=best[1];source='wrist_associated_detection'
        centers.append(e[0] if e else [width/2,height/2]);radii.append(e[1] if e else [1,1]);angle=e[2] if e else 0;axes.append([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        scores.append((min(1,best[0]/.2) if best else .12) if e else 0.)
        grasp.append(float(best is not None and best[2]<35));sources.append(source)
        x,y=wrist.astype(int);crop=im[max(0,y-24):min(height,y+24),max(0,x-24):min(width,x+24)];sharp.append(float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var()) if crop.size else 0.)
    cap.release();sharp=np.array(sharp);blur=np.clip(np.sqrt(sharp/max(np.median(sharp),1)),.1,1)
    mapping_weight=np.array([np.clip((r.get('runner_up_margin') or 0)/2,.1,1) for r in mapping]);duplicates=np.bincount(index,minlength=len(native['joints']))[index]
    accepted=np.array([bool(diag[k]['diagnostics']['valid_angle'][0][1][0]) for k in index]);separation=np.linalg.norm(uv[:,41]-uv[:,62],axis=1);overlap=np.clip(separation/45,.25,1)
    hw=np.where(accepted,1.,.2)*blur*overlap*mapping_weight
    ow=np.array(scores)*blur*mapping_weight/duplicates
    # Carry a possible grasp over short observation gaps only; long gaps remain unknown.
    gw=np.array(grasp)
    known=np.flatnonzero(gw)
    for a,b in zip(known[:-1],known[1:]):
        if times[b]-times[a]<=.25:gw[a:b+1]=1
    edge=continuity_edges(times,np.maximum(hw,ow),gw)
    arrays=dict(hand=hand,hand_image=uv[:,21:42],hand_weight=hw,observation_weight=ow,edge_weight=edge,grasp_weight=gw,rotation=np.array([r['rotation_camera_columns'] for r in old['frames']]),grip=np.array(grips),ring=np.array(model['head_outline']),model_grip=np.array([0,.045,0]),ellipse_center=np.array(centers),ellipse_radii=np.array(radii),ellipse_axes=np.array(axes),focal=native['focals'][index]/2,principal=np.array([width/2,height/2]),time=times,native_index=index)
    assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(out/'input.npz',**arrays)
    quality=[{'frame':i,'native_frame':int(index[i]),'time':float(times[i]),'hand_decoder_accepted':bool(accepted[i]),'hand_weight':float(hw[i]),'observation_weight':float(ow[i]),'grasp_state':'possible_grasp' if gw[i] else 'unknown','observation_source':sources[i],'blur_weight':float(blur[i]),'mapping_weight':float(mapping_weight[i]),'overlap_weight':float(overlap[i])} for i in range(n)]
    (out/'quality.json').write_text(json.dumps({'weights_are_not_probabilities':True,'frames':quality},indent=2));(out/'model.json').write_text(json.dumps(model));print('Prepared',n,'frames; no manually selected frames',flush=True)
if __name__=='__main__':main()
