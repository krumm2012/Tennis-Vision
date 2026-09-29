"""Small reproducible check for YOLO racket candidates near a reviewed frame."""
import argparse

import cv2
from ultralytics import YOLO


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    common = max(0, x2 - x1) * max(0, y2 - y1)
    return common / max(1, (a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/Users/krum5539/Downloads/yolo26s-seg.pt')
    parser.add_argument('--video', default='output/sam3d_cloud/video.mp4')
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 62)
    ok, image = cap.read()
    if not ok:
        raise RuntimeError('无法读取第 63 帧')
    result = YOLO(args.model).predict(image, classes=[38], conf=args.conf,
                                      imgsz=960, device=args.device, verbose=False)[0]
    reviewed = [760, 220, 870, 265]
    matches = [(round(float(box.conf), 3), round(overlap(box.xyxy[0].tolist(), reviewed), 3))
               for box in result.boxes]
    best = max((iou for _, iou in matches), default=0)
    print(f'frame=62 confidence={args.conf} candidates={len(matches)} '
          f'best_real_IoU={best:.3f} scores_and_IoU={matches}')
    if best < .3:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
