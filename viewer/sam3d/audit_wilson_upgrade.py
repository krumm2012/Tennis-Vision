"""Compare internal diagnostics and export all-frame Wilson reprojection sheets."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    root=Path('output/sam3d_cloud');folder=root/'wilson_review';folder.mkdir(exist_ok=True);report={}
    for name,file in [('v1','racket_poses_v1.json'),('v2','racket_poses_v2.json')]:
        data=json.loads((root/file).read_text());rows=data['frames'];rs=np.array([r['rotation_camera_columns'] for r in rows]);steps=np.degrees(Rotation.from_matrix(rs[:-1].transpose(0,2,1)@rs[1:]).magnitude());errors=[r['mask_fit_rms_px'] for r in rows if r.get('mask_fit_rms_px') is not None]
        report[name]={'summary':data['summary'],'contour_median_px':float(np.median(errors)),'contour_p90_px':float(np.percentile(errors,90)),'rotation_max_deg':float(max(steps)),'jump45_frames_1based':(np.flatnonzero(steps>45)+2).tolist()}
    report['caveat']='Different outlines/observations; residuals are not accuracy. Palm anchor is imposed, not measured. Calibration remains provisional.'
    (root/'wilson_upgrade_report.json').write_text(json.dumps(report,indent=2))
    capture=cv2.VideoCapture(str(root/'video.mp4'));tiles=[]
    for row in rows:
        ok,frame=capture.read();assert ok
        color=(70,225,95) if row['quality']=='silhouette_fitted' else (50,165,245)
        uv=np.round(row['projected_head_outline']).astype(np.int32);cv2.polylines(frame,[uv],True,color,2)
        end=np.round(row['projected_points']['handle_end']).astype(int);tip=np.round(row['projected_points']['tip']).astype(int);cv2.line(frame,tuple(end),tuple(tip),color,1)
        tile=cv2.resize(frame[40:390,520:1020],(300,210));cv2.putText(tile,f"{row['frame']+1} {'fit' if row['quality']=='silhouette_fitted' else 'ESTIMATE'}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1);tiles.append(tile)
        if len(tiles)==25:
            sheet=np.vstack([np.hstack(tiles[j:j+5]) for j in range(0,25,5)]);cv2.imwrite(str(folder/f"frames_{row['frame']-23:03d}_{row['frame']+1:03d}.jpg"),sheet);tiles=[]
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
