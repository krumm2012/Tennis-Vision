"""Upgrade playback/texture pixels while transferring existing SAM geometry honestly.

Only focal length and 2D evidence coordinates change. This is not SAM inference.
Same duration, frame count, aspect ratio and timeline are required.
"""
import argparse,json,shutil,subprocess
from pathlib import Path
import cv2,numpy as np
from package_result import package
from run_records import write,sha256,now
from mirror_calibration import apply_geometry

def prepare(folder):
    folder=Path(folder);old=folder/'result';meta=json.loads((old/'mesh_meta.json').read_text());record=json.loads((folder/'record.json').read_text());attempt=record['attempt']+1
    archive=folder/'attempts'/f"{record['attempt']:04d}"/'work/reconstruction.npz';run=folder/'attempts'/f'{attempt:04d}';work=run/'work';work.mkdir(parents=True,exist_ok=True)
    original=folder/'original.video';video=work/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(original),'-map','0:v:0','-an','-vf',"scale='trunc(min(2560,iw)/2)*2':-2,fps=25",'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    cap=cv2.VideoCapture(str(video));size=np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release();factor=size/np.array(meta['image_size'])
    if count!=meta['frames'] or fps!=meta['fps'] or not np.allclose(factor,factor[0]):raise ValueError('无法转移：时间轴或宽高比例不同')
    with np.load(archive,allow_pickle=False) as data:
        values={key:data[key].copy() for key in data.files};values['focal']*=factor[0]
        if 'joints2d' in values:values['joints2d']*=factor
    np.savez_compressed(work/'reconstruction.npz',**values);out=work/'result'
    if out.exists():shutil.rmtree(out)
    package(video,work/'reconstruction.npz',out)
    metadata=json.loads((out/'mesh_meta.json').read_text());metadata['geometry_provenance']='resolution_transfer_no_sam_inference';metadata['geometry_parent_video_sha256']=meta['video_sha256'];write(out/'mesh_meta.json',metadata)
    newhash=sha256(video)
    for name in ['person_candidates.json','racket_candidates.json','racket_roi_candidates.json','racket_poses_wrist.json','mirror_pose.json','calibration_annotations.json']:
        if not (old/name).exists():continue
        data=json.loads((old/name).read_text());data['video_sha256']=newhash;data['image_size']=size.astype(int).tolist()
        if name=='calibration_annotations.json':
            for key in ['ground','mirror']:
                if data.get(key,{}).get('points'):data[key]['points']=(np.asarray(data[key]['points'])*factor).tolist()
        for row in data.get('frames',[]):
            if name in ['person_candidates.json','racket_candidates.json','racket_roi_candidates.json']:
                for candidate in row.get('persons',row.get('candidates',[])):
                    candidate['box']=(np.asarray(candidate['box'])*np.tile(factor,2)).tolist();candidate['polygon']=(np.asarray(candidate['polygon'])*factor).tolist()
            elif name=='mirror_pose.json' and row.get('image') is not None:
                points=np.asarray(row['image']);points[:,:2]*=factor;row['image']=points.tolist();row['box']=(np.asarray(row['box'])*np.tile(factor,2)).tolist()
            elif name=='racket_poses_wrist.json':
                for key in ['observed_polygon','projected_head_outline']:
                    if row.get(key):row[key]=(np.asarray(row[key])*factor).tolist()
                if row.get('projected_points'):row['projected_points']={k:(np.asarray(v)*factor).tolist() for k,v in row['projected_points'].items()}
        write(out/name,data)
    for name in ['wilson_mesh.bin','wilson_model.json']:shutil.copy2(old/name,out/name)
    shutil.copy2(out/'racket_poses_wrist.json',out/'racket_poses.json')
    if meta.get('mirror_available'):
        geometry=json.loads((old/'mirror_geometry.json').read_text());geometry['video_sha256']=newhash
        for k,v in list(geometry.items()):
            if k.endswith('_px') and isinstance(v,(float,int)):geometry[k]=v*factor[0]
        geometry['transfer']='existing fixed plane; focal and pixel evidence scaled; no new fit'
        apply_geometry(folder,geometry,result=out)
    write(run/'run_manifest.json',{'status':'prepared','method':'resolution_transfer_no_sam_inference','parent_attempt':record['attempt'],'parent_video_sha256':meta['video_sha256'],'parent_npz_sha256':sha256(archive),'original_sha256':sha256(original),'source_sha256':newhash,'pixel_scale':factor.tolist(),'frames':count,'fps':fps,'finished_at':now()})
    write(out/'run_manifest.json',json.loads((run/'run_manifest.json').read_text()))
    # Fitters resolve archive by this attempt; live result remains intact until publish.
    write(folder/'record.json',{**record,'attempt':attempt,'pending_resolution_upgrade':True})
    print(work)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();prepare(a.dataset)
