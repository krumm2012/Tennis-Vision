"""Select diverse clear observations, never label selections as reviewed landmarks."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from run_records import write, sha256


def crop_box(points, grip, size):
    values = [p for p in points.values()] + [grip]
    a = np.asarray(values, float)
    span = np.maximum(np.ptp(a, axis=0), 80)
    center = (a.min(0)+a.max(0))/2
    span = np.maximum(span*1.9, [240, 180])
    lo = np.maximum(0, np.floor(center-span/2)).astype(int)
    hi = np.minimum(size, np.ceil(center+span/2)).astype(int)
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def choose(rows, fps, count=10):
    """One clear observation per time band; no bias toward static opening frames."""
    if not rows:
        return []
    duration = (max(r['frame'] for r in rows)+1)/fps
    selected = []
    for band in np.array_split(np.arange(max(r['frame'] for r in rows)+1), min(count, len(rows))):
        options = [r for r in rows if band[0] <= r['frame'] <= band[-1]]
        if options:
            selected.append(max(options, key=lambda r: (r['score'], -r['frame'])))
    return selected


def build(folder, result=None):
    folder = Path(folder)
    out = Path(result) if result else folder/'result'
    meta = json.loads((out/'mesh_meta.json').read_text())
    keypoints = json.loads((out/'racket_keypoints.json').read_text())
    poses = json.loads((out/'racket_poses.json').read_text())
    tracking=json.loads((out/'multiview_manifest.json').read_text()).get('tracking',[]) if (out/'multiview_manifest.json').exists() else []
    if any(d['video_sha256'] != meta['video_sha256'] for d in (keypoints, poses)) or sha256(folder/'source.mp4') != meta['video_sha256']:
        raise ValueError('清晰帧来源与当前视频不一致')
    mirror = json.loads((out/'mirror_geometry.json').read_text())
    n = np.asarray(mirror['normal_camera']); n /= np.linalg.norm(n)
    size = np.asarray(meta['image_size']); cap = cv2.VideoCapture(str(folder/'source.mp4'))
    rows = []
    try:
        for i, observation in enumerate(keypoints['frames']):
            ok, image = cap.read()
            if not ok:
                raise ValueError('清晰帧视频不完整')
            if observation.get('real') and observation.get('mirror') and np.linalg.norm(np.asarray(observation['real']['points']['head_center'])-observation['mirror']['points']['head_center'])<20*size[0]/1280:
                continue  # Same contour assigned to both roles is not a clear stereo frame.
            pose = poses['frames'][i]
            if pose['status'] != 'fitted':
                continue
            grip = np.asarray(pose['grip_target_camera_m']); views = {}
            for view in ('real', 'mirror'):
                obs = observation.get(view)
                if not obs or obs['confidence'] < .2:
                    continue
                if tracking:
                    head=np.asarray(obs['points']['head_center']);own=tracking[i].get('real_box' if view=='real' else 'mirror_box');other=tracking[i].get('mirror_box' if view=='real' else 'real_box')
                    inside=lambda box:box is not None and np.all(head>=box[:2]) and np.all(head<=box[2:])
                    if inside(other) and not inside(own):continue
                camera = grip if view == 'real' else grip-2*(grip@n-mirror['distance_camera_m'])*n
                if camera[2] <= .1:
                    continue
                uv = camera[:2]/camera[2]*meta['focal'][i]+size/2
                box = crop_box(obs['points'], uv, size)
                x0,y0,x1,y1 = box
                crop = image[y0:y1,x0:x1]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(cv2.resize(crop, (320,240)), cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                views[view] = {'crop_box':box, 'sharpness':blur, 'confidence':obs['confidence'],
                               'visible_parts':list(obs['points']), 'grip_uv_hint':uv.tolist()}
            if 'real' not in views:
                continue
            real = views['real']
            score = real['confidence']*2+np.log1p(real['sharpness'])*.25
            score += .8*('handle_end' in real['visible_parts'])+.4*('throat' in real['visible_parts'])
            score += .25*('mirror' in views)+.2*bool(observation.get('stereo_shaft'))
            rows.append({'frame':i,'time_s':i/meta['fps'],'score':float(score),'views':views,
                         'manually_reviewed':False, 'physical_face_identity_verified':False})
    finally:
        cap.release()
    selected = choose(rows, meta['fps'])
    cap = cv2.VideoCapture(str(folder/'source.mp4')); tiles = []
    try:
        for row in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame']);ok,image=cap.read()
            if not ok:
                raise ValueError('清晰关键帧解码失败')
            pair = np.zeros((240,640,3),np.uint8)
            for j,view in enumerate(('real','mirror')):
                if view not in row['views']:
                    continue
                x0,y0,x1,y1 = row['views'][view]['crop_box'];crop=image[y0:y1,x0:x1]
                ratio=min(320/crop.shape[1],210/crop.shape[0]);w,h=round(crop.shape[1]*ratio),round(crop.shape[0]*ratio)
                thumb=cv2.resize(crop,(w,h));pair[30:30+h,j*320:j*320+w]=thumb
                label=f"{row['frame']+1} | {row['time_s']:.2f}s | {view}"
                cv2.putText(pair,label,(j*320+6,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(105,235,215),1,cv2.LINE_AA)
            name=f"racket_review_{row['frame']:04d}.png";cv2.imwrite(str(out/name),pair);row['thumbnail']=name;tiles.append(pair)
    finally:
        cap.release()
    if tiles:
        if len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
        sheet=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
        cv2.imwrite(str(out/'racket_review_sheet.png'),sheet)
    report={'schema_version':1, **{k:meta[k] for k in ('video_sha256','image_size','fps')},
            'keypoints_sha256':sha256(out/'racket_keypoints.json'),'method':'visibility / crop sharpness / temporal diversity',
            'selection_is_ground_truth':False,'frames':selected}
    write(out/'racket_review_frames.json',report)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--result',type=Path)
    a=p.parse_args();print(json.dumps({'selected':len(build(a.dataset,a.result)['frames'])}))
