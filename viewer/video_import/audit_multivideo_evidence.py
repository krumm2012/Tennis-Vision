"""Read-only, hash-bound input audit for multi-video appearance candidates.

Silhouette agreement and detector scores are consistency evidence, not GT.
The script never changes reconstructions, masks, calibrations or accepted data.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from multivideo_texture import depth_map, project
from run_records import now, sha256, write


def quantiles(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {"count": int(len(a)), "p05": float(np.quantile(a, .05)),
            "median": float(np.median(a)), "p95": float(np.quantile(a, .95))} if len(a) else {"count": 0}


def video_timeline(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
               "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)]
    data = json.loads(subprocess.check_output(command, timeout=120))
    return np.array([float(f["best_effort_timestamp_time"]) for f in data["frames"]])


def overlap(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def render(camera, faces, focal, image_size, mask_size):
    uv = project(camera, focal, image_size) * np.array(mask_size) / np.array(image_size)
    return depth_map(uv.astype(np.float32), camera[:, 2].astype(np.float32), faces,
                     mask_size[0], mask_size[1]) < 1e5


def crop_tile(image, mask, silhouette, box, label):
    overlay = image.copy()
    a = cv2.resize(mask, image.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 140
    b = cv2.resize(silhouette.astype(np.uint8), image.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    overlay[a] = overlay[a] * .72 + np.array([40, 230, 100]) * .28
    contour, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contour, -1, (230, 70, 230), 2)
    x0, y0, x1, y1 = np.array(box).astype(int)
    pad = 18
    crop = overlay[max(0, y0-pad):min(image.shape[0], y1+pad), max(0, x0-pad):min(image.shape[1], x1+pad)]
    tile = np.full((340, 240, 3), 22, np.uint8)
    if crop.size:
        scale = min(230/crop.shape[1], 300/crop.shape[0])
        crop = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)))
        y = 32; x = (240-crop.shape[1])//2
        tile[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
    cv2.putText(tile, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, .42, (230, 230, 230), 1)
    return tile


def audit(batch, output, step=5):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((batch/'batch_manifest.json').read_text())
    layout = batch/'mhr_uv.npz'
    with np.load(layout, allow_pickle=False) as data:
        layout_faces = data['faces'].copy()
    clips = []; tiles = []; all_views = []
    for entry in manifest['clips']:
        folder = Path(entry['folder']); work = folder/'attempts/0001/work'; result = folder/'result'
        meta = json.loads((result/'mesh_meta.json').read_text())
        remote = json.loads((work/'remote_manifest.json').read_text())
        multi = json.loads((result/'multiview_manifest.json').read_text())
        mirror = json.loads((result/'mirror_geometry.json').read_text())
        record = json.loads((folder/'record.json').read_text())
        files = {'original':folder/'original.video', 'video':folder/'source.mp4',
                 'archive':work/'reconstruction.npz', 'meta':result/'mesh_meta.json',
                 'multiview':result/'multiview_manifest.json', 'mirror':result/'mirror_geometry.json',
                 'display_raw':result/'mesh_local.bin', 'display_smooth':result/'mesh_smooth.bin'}
        hashes = {k:sha256(p) for k,p in files.items()}
        source_valid = (hashes['original'] == record['original_sha256'] == entry['original_sha256']
                        and hashes['video'] == record['source_sha256'] == meta['video_sha256'] == remote['source_sha256']
                        and hashes['archive'] == remote['artifact_sha256']['reconstruction.npz'])
        original_times = video_timeline(files['original']); video_times = video_timeline(files['video'])
        nearest = np.abs(video_times[:,None] - original_times[None,:]).argmin(1)
        timeline = {'original_frames':len(original_times), 'normalized_frames':len(video_times),
                    'normalized_interval_s':quantiles(np.diff(video_times)),
                    'nearest_original_pts_error_s':quantiles(abs(video_times-original_times[nearest])),
                    'nearest_original_index_monotonic':bool(np.all(np.diff(nearest)>=0)),
                    'unique_original_frames_mapped':int(len(np.unique(nearest))),
                    'original_pts_last_s':float(original_times[-1]), 'normalized_pts_last_s':float(video_times[-1]),
                    'note':'fps conversion drops/duplicates by timestamp; inference uses normalized indexing, not original frame indexing'}
        with np.load(files['archive'], allow_pickle=False) as archive:
            keys=['vertices','faces','source_roots','focal','masks','mirror_valid','mirror_vertices','mirror_roots','mirror_focal','masks_mirror_sam2']
            data = {k:archive[k] for k in keys}
        frames = len(data['vertices']); faces = data['faces']; size = meta['image_size']; masks = data['masks']
        smooth = np.memmap(files['display_smooth'], dtype='<f4', mode='r', shape=data['vertices'].shape)
        raw = np.memmap(files['display_raw'], dtype='<f4', mode='r', shape=data['vertices'].shape)
        sample_indices = np.arange(0, frames, step)
        delta = np.linalg.norm(smooth[sample_indices]-data['vertices'][sample_indices], axis=-1)
        correction_px = []
        mask_stats = {}
        for role,key in [('real','masks'),('mirror','masks_mirror_sam2')]:
            arr = data[key]; selected = arr>140; high = arr>234
            denominator = np.maximum(selected.sum((1,2)),1)
            mask_stats[role]={'nonempty_frames':int((selected.sum((1,2))>0).sum()),
                              'selected_pixel_area':quantiles(selected.sum((1,2))),
                              'high_logit_pixel_fraction_in_mask':quantiles(high.sum((1,2))/denominator),
                              'mask_value_unique_count':int(len(np.unique(arr))),
                              'nonbinary_pixel_fraction':float(((arr>0)&(arr<255)).mean())}
        valid = data['mirror_valid']; cap = cv2.VideoCapture(str(files['video'])); rows = []
        sheet_indices = {sample_indices[len(sample_indices)//5],sample_indices[len(sample_indices)//2],sample_indices[4*len(sample_indices)//5]}
        for index in sample_indices:
            index = int(index); cap.set(cv2.CAP_PROP_POS_FRAMES,index); ok,image=cap.read()
            if not ok:raise ValueError(f'Video decode failed: {folder.name}:{index}')
            root = data['source_roots'][index]; camera = data['vertices'][index]+root
            correction_px.extend(np.linalg.norm(project(smooth[index]+root,data['focal'][index],size)-project(camera,data['focal'][index],size),axis=-1).tolist())
            tracking = multi['tracking'][index]
            for view, role, key in [(0,'real','masks'),(1,'mirror','masks_mirror_sam2')]:
                mask=data[key][index]; mask_size=mask.shape[::-1]; selected=mask>140
                row={'clip':folder.name,'frame_index':index,'view':view,'role':role,'video_sha256':hashes['video'],
                     'archive_sha256':hashes['archive'],'heldout_for_existing_texture':index%25==0,
                     'native_valid':bool(view==0 or valid[index]),
                     'detector_confidence':float(tracking[role+'_confidence']),
                     'sam2_detector_polygon_iou':float(multi['mask_quality'][index][role+'_iou']),
                     'sam2_nonempty':bool(selected.any()),'calibrated_sam2_confidence_available':False}
                if not row['native_valid']:
                    if index in sheet_indices:
                        placeholder=np.full((340,240,3),22,np.uint8)
                        cv2.putText(placeholder,f'{folder.name} {index+1:03d} {role}',(7,21),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1)
                        cv2.putText(placeholder,'No native observation',(10,160),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,180,230),1)
                        tiles.append(placeholder)
                    row['recommended_texture_action']='skip_independent_mirror_no_native_observation';rows.append(row);continue
                cam=camera if view==0 else data['mirror_vertices'][index]+data['mirror_roots'][index]
                focal=data['focal'][index] if view==0 else data['mirror_focal'][index]
                silhouette=render(cam,faces,focal,size,mask_size)
                other=data['masks_mirror_sam2' if view==0 else 'masks'][index]>140
                supported=selected&silhouette
                box=tracking[role+'_box']; box=np.asarray(box) if box is not None else np.r_[0,0,size]
                x0,y0,x1,y1=box.astype(int)
                patch=image[max(y0,0):min(y1,size[1]),max(x0,0):min(x1,size[0])]
                sharpness=float(cv2.Laplacian(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var()) if patch.size else 0
                row.update(mesh_mask_iou=overlap(silhouette,selected),mesh_pixel_inside_sam2_fraction=float(supported.sum()/max(silhouette.sum(),1)),
                           sam2_pixel_explained_by_mesh_fraction=float(supported.sum()/max(selected.sum(),1)),
                           wrong_role_mask_overlap_fraction=float((silhouette&other).sum()/max(silhouette.sum(),1)),
                           sharpness_laplacian_variance=sharpness,detector_height_px=float(box[3]-box[1]),
                           projected_vertices_in_image_fraction=float(((project(cam,focal,size)>0)&(project(cam,focal,size)<size)).all(1).mean()))
                # Soft review suggestion only: preserve the input and disclose thresholds.
                reasons=[]
                if row['mesh_mask_iou']<.5:reasons.append('mesh_mask_iou_below_0.5')
                if row['wrong_role_mask_overlap_fraction']>.05:reasons.append('other_role_overlap_above_0.05')
                if row['detector_confidence']<.5:reasons.append('detector_confidence_below_0.5')
                row.update(review_reasons=reasons,recommended_texture_action='downweight_and_review' if reasons else 'normal_visibility_checks')
                if index in sheet_indices:tiles.append(crop_tile(image,mask,silhouette,box,f'{folder.name} {index+1:03d} {role} IoU {row["mesh_mask_iou"]:.2f}'))
                rows.append(row)
        cap.release()
        native_summary={role:{metric:quantiles([r[metric] for r in rows if r['role']==role and metric in r])
                             for metric in ['mesh_mask_iou','mesh_pixel_inside_sam2_fraction','wrong_role_mask_overlap_fraction','sharpness_laplacian_variance','detector_height_px']}
                        for role in ['real','mirror']}
        item={'clip':folder.name,'files':{k:str(p.resolve()) for k,p in files.items()},'sha256':hashes,
              'source_hash_chain_valid':source_valid,'timeline':timeline,'frames':frames,'mirror_native_valid_frames':int(valid.sum()),
              'mirror_invalid_frame_indices':np.flatnonzero(~valid).tolist(),'uv_topology_exact_match':bool(np.array_equal(faces,layout_faces)),
              'raw_mesh_file_exact_match':bool(np.array_equal(raw,data['vertices'])),
              'sampled_display_correction_m':quantiles(delta.ravel()),'sampled_display_projection_correction_px':quantiles(correction_px),
              'native_finite':bool(all(np.isfinite(data[k]).all() for k in ['vertices','source_roots','focal','mirror_vertices','mirror_roots'])),
              'mask':mask_stats,'sampled_native_projection':native_summary,
              'mirror_calibration':mirror,'view_rows':rows}
        clips.append(item);all_views.extend(rows); print('Evidence audit',folder.name,'real IoU',native_summary['real']['mesh_mask_iou']['median'],'mirror IoU',native_summary['mirror']['mesh_mask_iou']['median'],flush=True)
    normals=np.array([c['mirror_calibration']['normal_camera'] for c in clips]);normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    angles=np.rad2deg(np.arccos(np.clip(normals@normals.T,-1,1)))
    report={'schema_version':1,'created_at':now(),'status':'independent_input_audit_not_ground_truth',
            'code_sha256':sha256(Path(__file__)),'projection_code_sha256':sha256(Path(__file__).with_name('multivideo_texture.py')),
            'batch_manifest_sha256':sha256(batch/'batch_manifest.json'),'layout_sha256':sha256(layout),
            'sampling':{'step':step,'frame_index_base':0,'includes_existing_heldout':True},
            'mask_confidence_limits':'uint8 sigmoid SAM2 logits resized by area; not calibrated posterior certainty; independent detector polygon IoU is also not GT',
            'projection_limits':'projected mesh silhouette ignores clothing thickness; IoU flags consistency risk, not identity/pose correctness',
            'all_source_hash_chains_valid':all(c['source_hash_chain_valid'] for c in clips),
            'total_frames':sum(c['frames'] for c in clips),'mirror_native_valid_frames':sum(c['mirror_native_valid_frames'] for c in clips),
            'cross_clip_mirror_plane_normal_angle_max_deg':float(angles.max()),
            'cross_clip_mirror_plane_distance_range_m':[min(c['mirror_calibration']['distance_camera_m'] for c in clips),max(c['mirror_calibration']['distance_camera_m'] for c in clips)],
            'cross_clip_plane_limits':'per-clip body scales/camera roots are uncalibrated; distance differences cannot alone prove physical mirror moved',
            'clips':clips}
    write(output/'quality_inputs.json',report)
    write(output/'view_quality.json',{'schema_version':1,'quality_inputs_sha256':sha256(output/'quality_inputs.json'),'views':all_views})
    if tiles:
        columns=6;blank=np.zeros_like(tiles[0]); tiles+= [blank]*((-len(tiles))%columns)
        sheet=np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)])
        cv2.imwrite(str(output/'projection_contact_sheet.jpg'),sheet)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--step',type=int,default=5)
    args=parser.parse_args()
    if not 1<=args.step<=25:raise ValueError('Sampling step must be 1–25')
    audit(args.batch,args.output,args.step)
