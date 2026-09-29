"""Track the real racket with SAM2, seeded by wrist-associated YOLO masks."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def select_candidate(row):
    candidates = [c for c in row['candidates'] if c['wrist_distance_px'] < 65
                  and max(c['box'][2]-c['box'][0], c['box'][3]-c['box'][1]) < 180
                  and len(c['polygon']) >= 5]
    return max(candidates, key=lambda c: c['confidence'] * np.exp(-c['wrist_distance_px']/30), default=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('output/sam3d_cloud'))
    p.add_argument('--sam2', type=Path, default=Path('output/cloud_backup_20260926/sam2'))
    p.add_argument('--checkpoint', default='output/cloud_backup_20260926/sam2_work/sam2.1_hiera_small.pt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=250)
    args = p.parse_args()
    sys.path.insert(0, str(args.sam2.resolve()))
    from sam2.build_sam import build_sam2_video_predictor
    detections = json.loads((args.data/'racket_yolo_candidates.json').read_text())
    detections['frames'] = detections['frames'][args.start:args.end]
    chosen = [select_candidate(row) for row in detections['frames']]
    # Fixed crop retains the whole observed swing while giving the small racket
    # more image pixels in SAM2's encoder. Coordinates are restored on export.
    crop = [520, 40, 1020, 390]
    x0, y0, x1, y1 = crop
    folder = args.data/'racket_work'/f'frames_{args.start}_{args.end}'
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.data/'video.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,args.start)
    for n in range(len(chosen)):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'missing video frame {n}')
        cv2.imwrite(str(folder/f'{n:05d}.jpg'), frame[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    seeds = []
    for start in range(0, len(chosen), 12):
        eligible = [n for n in range(start, min(start+12, len(chosen)))
                    if chosen[n] and chosen[n]['confidence'] >= .10]
        if eligible:
            seeds.append(max(eligible, key=lambda n: chosen[n]['confidence'] * np.exp(-chosen[n]['wrist_distance_px']/30)))
    if chosen[0] and 0 not in seeds:
        seeds.insert(0, 0)
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', args.checkpoint,
                                          device=args.device, apply_postprocessing=False)
    if args.device in ('mps', 'cpu'):
        # SAM2 stores memory in bfloat16 for CUDA autocast. This run uses float32
        # CPU/MPS; mixed memory/attention dtypes can fail matrix multiplication.
        original_frame = predictor._run_single_frame_inference
        original_memory = predictor._run_memory_encoder
        def float_frame(*a, **kw):
            compact, pred = original_frame(*a, **kw)
            if compact['maskmem_features'] is not None:
                compact['maskmem_features'] = compact['maskmem_features'].float()
            return compact, pred
        def float_memory(*a, **kw):
            features, position = original_memory(*a, **kw)
            return features.float(), position
        predictor._run_single_frame_inference = float_frame
        predictor._run_memory_encoder = float_memory
    rows = []
    with torch.inference_mode():
        state = predictor.init_state(str(folder), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for n in seeds:
            mask = np.zeros((y1-y0, x1-x0), np.uint8)
            poly = np.round(np.array(chosen[n]['polygon'])-[x0,y0]).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
            predictor.add_new_mask(state, frame_idx=n, obj_id=1, mask=mask)
        print(f'tracking {len(chosen)} frames, {len(seeds)} conditioning masks', flush=True)
        for n, ids, logits in predictor.propagate_in_video(state, start_frame_idx=0):
            values = logits[0,0].float().cpu().numpy()
            mask = (values > 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv2.contourArea(c) > 8]
            wrist = np.array(detections['frames'][n]['wrist'])-[x0,y0]
            if contours:
                # Detached speckles are not useful racket shape observations.
                contour = max(contours, key=cv2.contourArea)
                points = contour[:,0,:].astype(float)+[x0,y0]
                area = float(cv2.contourArea(contour))
                distance = max(0., -cv2.pointPolygonTest(contour, tuple(wrist), True))
            else:
                points = np.empty((0,2));area=0.;distance=999.
            row = {'frame': n+args.start, 'polygon': points.tolist(), 'area_px': area,
                   'wrist_distance_px': round(distance,2), 'conditioned': n in seeds,
                   'yolo_confidence': chosen[n]['confidence'] if chosen[n] else 0.,
                   'mask_valid': bool(100 < area < 18000 and distance < 100)}
            rows.append(row)
            if n%25 == 24:
                print(f'{n+1}/{len(chosen)} masks; valid {sum(r["mask_valid"] for r in rows)}', flush=True)
    out = args.data/'racket_work'/f'masks_{args.start}_{args.end}.json'
    out.write_text(json.dumps({'video_id':'30.56','image_size':[1280,720], 'crop':crop,
                              'device':args.device,'seeds':seeds,'method':'SAM2.1_hiera_small_YOLO_wrist_seeds',
                              'frames':sorted(rows,key=lambda r:r['frame'])}, separators=(',',':')))
    print(out, flush=True)


if __name__ == '__main__':
    main()
