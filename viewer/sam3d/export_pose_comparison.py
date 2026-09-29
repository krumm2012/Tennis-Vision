"""Export synchronized original video and a court-aligned overhead SAM 3D view."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from numba import njit


FEET = (15, 16, 17, 18, 19, 20)
PANEL = 720
SCALE = 340.0  # pixels per ground metre in the following overhead camera


@njit(cache=True)
def raster_mesh(canvas, depth, xy, height, face, color):
    h, w = depth.shape
    for k in range(len(face)):
        a, b, c = face[k]
        ax, ay = xy[a]; bx, by = xy[b]; cx, cy = xy[c]
        det = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(det) < 0.03:
            continue
        x0 = max(0, int(np.floor(min(ax,bx,cx))))
        x1 = min(w-1, int(np.ceil(max(ax,bx,cx))))
        y0 = max(0, int(np.floor(min(ay,by,cy))))
        y1 = min(h-1, int(np.ceil(max(ay,by,cy))))
        if x1 < x0 or y1 < y0:
            continue
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                u = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/det
                v = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/det
                t = 1-u-v
                if u < -0.001 or v < -0.001 or t < -0.001:
                    continue
                z = u*height[a]+v*height[b]+t*height[c]
                if z <= depth[y,x]:
                    continue
                depth[y,x] = z
                for ch in range(3):
                    canvas[y,x,ch] = np.uint8(min(255,max(0,u*color[a,ch]+v*color[b,ch]+t*color[c,ch])))


def map_ground(h, uv):
    p = h @ np.array([uv[0], uv[1], 1.])
    return p[:2]/p[2]


def overhead_xy(points, center):
    return np.column_stack((PANEL/2+(points[:,0]-center[0])*SCALE,
                            PANEL/2+(points[:,2]-center[1])*SCALE))


def racket_segments(model, pose, root, basis, shift, correction):
    m = np.asarray(pose['rotation_camera_columns'])
    t = np.asarray(pose['translation_camera_m'])
    local = []
    span = model['length_m']-model['head_center_y_m']
    for theta in np.linspace(0, 2*np.pi, 65):
        local.append([model['head_half_width_m']*np.cos(theta),
                      model['head_center_y_m']+span*np.sin(theta), 0])
    local += [[0,0,0],[0,model['throat_y_m'],0],[0,model['head_center_y_m']-span,0]]
    points = (np.asarray(local)@m.T+t-root+correction)@basis+shift
    pairs = [(j,j+1) for j in range(64)] + [(65,66),(66,67)]
    return points,pairs


def draw_depth_line(image, depth, a, b, color, thickness=4):
    p0, p1 = np.round(a[:2]).astype(int), np.round(b[:2]).astype(int)
    count = max(1,int(np.linalg.norm(p1-p0)*1.3))
    for u in np.linspace(0,1,count):
        p = (a*(1-u)+b*u)
        x,y = int(round(p[0])),int(round(p[1]))
        # The racket is intentionally drawn as a diagnostic overlay. Its
        # monocular depth is uncertain, so mesh occlusion would conceal it.
        if 0<=x<PANEL and 0<=y<PANEL:
            cv2.circle(image,(x,y),thickness//2,color,-1,cv2.LINE_AA)


def draw_panel(frame_index, body, faces, colors, pose_frame, racket_frame,
               racket_model, meta, calibration, wilson=None, grasp=None):
    basis = np.asarray(calibration['basis'])
    h = np.asarray(calibration['H']).reshape(3,3)
    q = np.asarray(pose_frame['smooth'])@basis
    foot = min(FEET,key=lambda j:q[j,1])
    ground = map_ground(h,pose_frame['image'][foot])
    shift = np.array([ground[0],0,ground[1]])-q[foot]
    pelvis = (q[9]+q[10])*.5+shift
    center = pelvis[[0,2]]
    panel = np.full((PANEL,PANEL,3),(36,31,24),np.uint8)
    # A person-following camera; grid remains aligned to calibrated court axes.
    x_min,x_max=center[0]-PANEL/(2*SCALE),center[0]+PANEL/(2*SCALE)
    z_min,z_max=center[1]-PANEL/(2*SCALE),center[1]+PANEL/(2*SCALE)
    for x in np.arange(np.floor(x_min*2)/2,x_max+.5,.5):
        px=int(round(PANEL/2+(x-center[0])*SCALE))
        cv2.line(panel,(px,0),(px,PANEL),(51,54,48),1,cv2.LINE_AA)
    for z in np.arange(np.floor(z_min*2)/2,z_max+.5,.5):
        py=int(round(PANEL/2+(z-center[1])*SCALE))
        cv2.line(panel,(0,py),(PANEL,py),(51,54,48),1,cv2.LINE_AA)
    court=np.array([[0,0],[calibration['width'],0],
                    [calibration['width'],calibration['length']],[0,calibration['length']]])
    court_px=np.round(np.column_stack((PANEL/2+(court[:,0]-center[0])*SCALE,
                                       PANEL/2+(court[:,1]-center[1])*SCALE))).astype(np.int32)
    cv2.polylines(panel,[court_px],True,(104,172,180),2,cv2.LINE_AA)
    xyz=body@basis+shift
    xy=overhead_xy(xyz,center)
    depth=np.full((PANEL,PANEL),-999.,np.float32)
    raster_mesh(panel,depth,xy,xyz[:,1],faces,colors)
    if wilson is not None:
        if grasp is not None:
            from fit_wilson_sequence import palm_frame
            rw,rh=palm_frame(pose_frame['raw']);sw,sh=palm_frame(pose_frame['smooth'])
            rotation=sh@rh.T@np.array(racket_frame['rotation_camera_columns'])
            target=np.array(meta['source_roots'][frame_index])+sw+sh@np.array(racket_frame.get('palm_offset_m',grasp['palm_offset_m']))
            translation=target-rotation@np.array(grasp['grip_local_m'])
        else:
            rotation=np.array(racket_frame['rotation_camera_columns'])
            translation=np.array(racket_frame['translation_camera_m'])+np.array(pose_frame['smooth'][41])-np.array(pose_frame['raw'][41])
        points=(wilson[:,:3]@rotation.T+translation-np.array(meta['source_roots'][frame_index]))@basis+shift
        wf=np.arange(len(points),dtype=np.uint32).reshape(-1,3)
        raster_mesh(panel,depth,overhead_xy(points,center),points[:,1],wf,(wilson[:,3:6][:,::-1]*255).astype(np.uint8))
    if wilson is None and racket_frame['status']=='fitted':
        root=np.asarray(meta['source_roots'][frame_index])
        correction=np.asarray(pose_frame['smooth'][41])-np.asarray(pose_frame['raw'][41])
        points,pairs=racket_segments(racket_model,racket_frame,root,basis,shift,correction)
        projected=overhead_xy(points,center)
        projected=np.column_stack((projected,points[:,1]))
        color=(70,222,113) if racket_frame['quality']=='silhouette_fitted' else (67,177,255)
        for a,b in pairs:
            draw_depth_line(panel,depth,projected[a],projected[b],color,5)
        grip=projected[65]
        cv2.circle(panel,tuple(np.round(grip[:2]).astype(int)),5,(255,255,255),-1,cv2.LINE_AA)
    cv2.rectangle(panel,(0,0),(PANEL,76),(20,24,29),-1)
    cv2.putText(panel,'3D POSE  |  OVERHEAD',(24,34),cv2.FONT_HERSHEY_SIMPLEX,.75,(241,241,235),2,cv2.LINE_AA)
    quality='SILHOUETTE FIT' if racket_frame['quality']=='silhouette_fitted' else ('CONSTRAINED ESTIMATE' if racket_frame['quality']=='temporal_estimate' else 'INTERPOLATED')
    if racket_frame['quality']=='joint_candidate':quality='JOINT CANDIDATE'
    cv2.putText(panel,f'RACKET: {quality}  |  FRAME {frame_index+1:03d}',
                (24,63),cv2.FONT_HERSHEY_SIMPLEX,.49,(119,210,240),1,cv2.LINE_AA)
    cv2.putText(panel,'0.5 m GRID  |  CAMERA FOLLOWS PLAYER',(22,PANEL-22),
                cv2.FONT_HERSHEY_SIMPLEX,.48,(205,207,202),1,cv2.LINE_AA)
    if racket_frame.get('joint_hand_camera') is not None:
        hp=(np.array(racket_frame['joint_hand_camera'])-np.array(meta['source_roots'][frame_index]))@basis+shift
        pixels=overhead_xy(hp,center)
        for tip in [0,4,8,12,16]:
            chain=[20,tip+3,tip+2,tip+1,tip]
            cv2.polylines(panel,[np.round(pixels[chain]).astype(np.int32)],False,(70,170,255),1,cv2.LINE_AA)
    return panel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('output/sam3d_cloud'))
    parser.add_argument('--out',type=Path,default=Path('output/sam3d_cloud/real_vs_3d_overhead.mp4'))
    parser.add_argument('--poses',default='racket_poses.json')
    parser.add_argument('--first',type=int,default=0)
    parser.add_argument('--last',type=int,default=250)
    args=parser.parse_args()
    load=lambda name:json.loads((args.data/name).read_text())
    meta=load('mesh_meta.json');pose=load('temporal_pose.json');racket=load(args.poses)
    calibration=load('ground_calibration.json')
    wilson=np.fromfile(args.data/'wilson_mesh.bin',dtype='<f4').reshape(-1,6) if racket['model'].get('asset_file') else None
    assert len(pose['frames'])==len(racket['frames'])==meta['frames']==250
    assert args.last<=250 and 0<=args.first<args.last
    vertices=np.memmap(args.data/'mesh_smooth.bin',dtype='<f4',mode='r',shape=(250,meta['vertices'],3))
    faces=np.fromfile(args.data/'mesh_faces.bin',dtype='<u4').reshape(-1,3)
    texture=np.memmap(args.data/'temporal_texture_sam2.bin',dtype='u1',mode='r',shape=(250,meta['vertices'],6))
    source=cv2.VideoCapture(str(args.data/'video.mp4'))
    source.set(cv2.CAP_PROP_POS_FRAMES,args.first)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    ffmpeg=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
                             '-s','2000x720','-r','25','-i','-','-an','-c:v','libx264',
                             '-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',
                             str(args.out)],stdin=subprocess.PIPE)
    try:
        for i in range(args.first,args.last):
            ok,real=source.read()
            if not ok:raise RuntimeError(f'cannot decode source frame {i}')
            tex=np.asarray(texture[i])
            # Cached video color, falling back to a neutral mesh where no source is trusted.
            confidence=tex[:,3:4].astype(np.float32)/255
            color=tex[:,:3][:,::-1].astype(np.float32)*confidence+np.array([150,141,128])*(1-confidence)
            if not np.any(confidence):
                body=np.asarray(vertices[i]);tri=body[faces];norm=np.zeros_like(body);fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                for k in range(3):np.add.at(norm,faces[:,k],fn)
                norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-8)
                light=np.array([.2,-.8,-.5]);light/=np.linalg.norm(light)
                color=np.array([180,200,215])*(.45+.55*np.abs(norm@light))[:,None]
            panel=draw_panel(i,np.asarray(vertices[i]),faces,color.astype(np.uint8),
                             pose['frames'][i],racket['frames'][i],racket['model'],meta,calibration,wilson,racket.get('grasp_calibration'))
            cv2.rectangle(real,(0,0),(1280,76),(20,24,29),-1)
            cv2.putText(real,'ORIGINAL VIDEO',(24,48),cv2.FONT_HERSHEY_SIMPLEX,.95,(241,241,235),2,cv2.LINE_AA)
            ffmpeg.stdin.write(np.ascontiguousarray(np.hstack((real,panel))).tobytes())
            if (i+1)%25==0:print(f'{i+1}/{args.last} frames',flush=True)
    finally:
        source.release()
        ffmpeg.stdin.close()
        if ffmpeg.wait()!=0:raise RuntimeError('ffmpeg encode failed')
    print(args.out)


if __name__=='__main__':main()
