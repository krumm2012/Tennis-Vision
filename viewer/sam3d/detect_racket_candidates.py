"""Collect low-threshold YOLO racket boxes/masks for all SAM video frames.

Candidates are evidence, not final racket tracks. The anatomical right wrist is
recorded alongside each result to disambiguate the player from the mirror.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO


def distance_to_box(point, box):
    x, y = point
    left, top, right, bottom = box
    return math.hypot(max(left - x, 0, x - right),
                      max(top - y, 0, y - bottom))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/Users/krum5539/Downloads/yolo26s-seg.pt'))
    parser.add_argument('--data', type=Path, default=Path('output/sam3d_cloud'))
    parser.add_argument('--out', type=Path, default=Path('output/sam3d_cloud/racket_yolo_candidates.json'))
    parser.add_argument('--conf', type=float, default=.005)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    temporal = json.loads((args.data / 'temporal_pose.json').read_text())
    count = len(temporal['frames'])
    cap = cv2.VideoCapture(str(args.data / 'video.mp4'))
    if not cap.isOpened():
        raise RuntimeError('无法读取视频')
    model = YOLO(str(args.model))
    rows = []
    for frame in range(count):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f'视频只读到第 {frame} 帧')
        wrist = temporal['frames'][frame]['image'][41]
        result = model.predict(image, classes=[38], conf=args.conf, iou=.5,
                               imgsz=args.imgsz, device=args.device,
                               verbose=False)[0]
        masks = result.masks.xy if result.masks is not None else []
        candidates = []
        for index, box in enumerate(result.boxes):
            xyxy = [round(float(v), 2) for v in box.xyxy[0]]
            distance = distance_to_box(wrist, xyxy)
            # Keep low-confidence candidates near the hand and high-confidence
            # detections elsewhere for later mirror/false-positive auditing.
            if distance > 230 and float(box.conf) < .08:
                continue
            polygon = masks[index].tolist() if index < len(masks) else []
            candidates.append({'confidence': round(float(box.conf), 5),
                               'box': xyxy, 'wrist_distance_px': round(distance, 2),
                               'polygon': [[round(x, 1), round(y, 1)] for x, y in polygon]})
        candidates.sort(key=lambda row: row['confidence'], reverse=True)
        rows.append({'frame': frame, 'wrist': [round(x, 2) for x in wrist],
                     'candidates': candidates[:20]})
        if frame % 25 == 24 or frame + 1 == count:
            nearby = sum(any(c['wrist_distance_px'] < 55 and c['confidence'] >= .03
                             for c in row['candidates']) for row in rows)
            print(f'{frame + 1}/{count} 帧：近手腕 ≥0.03 候选 {nearby}/{len(rows)}', flush=True)
    payload = {'video_id': '30.56', 'fps': temporal['fps'], 'image_size': [1280, 720],
               'model': args.model.name, 'confidence_floor': args.conf,
               'imgsz': args.imgsz, 'frames': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    print(f'保存 {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
