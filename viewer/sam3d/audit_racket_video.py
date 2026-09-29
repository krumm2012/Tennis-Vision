"""Export all-frame visual review sheets and numerical diagnostics (not accuracy)."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    root=Path('output/sam3d_cloud')
    data=json.loads((root/'racket_poses.json').read_text())
    rows=data['frames']; folder=root/'racket_review';folder.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(root/'video.mp4'))
    sheets=[];tiles=[]
    for row in rows:
        ok,frame=cap.read()
        if not ok:raise RuntimeError('missing frame')
        observed=row['quality']=='silhouette_fitted'
        color=(80,220,80) if observed else (0,170,255)
        pts=np.round(row['projected_head_outline']).astype(np.int32)
        cv2.polylines(frame,[pts],True,color,2)
        tile=frame[40:390,520:1020].copy()
        tile=cv2.resize(tile,(300,210))
        label=f"{row['frame']+1}: {'fit' if observed else 'INTERP'}"
        cv2.putText(tile,label,(6,20),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        tiles.append(tile)
        if len(tiles)==25 or row is rows[-1]:
            while len(tiles)<25:tiles.append(np.zeros_like(tile))
            sheet=np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,25,5)])
            name=f'frames_{row["frame"]//25*25+1:03d}_{row["frame"]+1:03d}.jpg'
            cv2.imwrite(str(folder/name),sheet);sheets.append(name);tiles=[]
    rotations=Rotation.from_matrix([r['rotation_camera_columns'] for r in rows])
    jumps=np.degrees((rotations[:-1].inv()*rotations[1:]).magnitude())
    report={**data['summary'],'wrist_gap_max_m':max(r['wrist_gap_m'] for r in rows),
            'rotation_jump_median_deg':float(np.median(jumps)),
            'rotation_jump_max_deg':float(max(jumps)),
            'jump_over_45deg_frames_1based':(np.flatnonzero(jumps>45)+2).tolist(),
            'interpolated_frames_1based':[r['frame']+1 for r in rows if r['quality']=='interpolated'],
            'max_distance_to_observation_frames':max(r.get('nearest_observation_frames',0) for r in rows),
            'review_sheets':sheets,'note':'Internal consistency only; no held-out ground truth.'}
    (folder/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
