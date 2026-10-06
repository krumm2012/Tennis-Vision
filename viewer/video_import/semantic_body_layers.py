"""Local, hash-bound hair/skin/clothing parsing on native video person crops.

ONNX model predictions are automatic evidence, never manual segmentation truth.
Coordinates always refer to the unflipped source video, including the mirror.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from multivideo_texture import load_data, views, project, validate_fusion_group
from run_records import write, sha256

GROUPS = {'unknown': 0, 'hair': 1, 'skin': 2, 'upper_clothing': 3, 'lower_clothing': 4}
LABEL_GROUP = np.zeros(18, np.uint8)
LABEL_GROUP[2] = 1
LABEL_GROUP[[11, 12, 13, 14, 15]] = 2
LABEL_GROUP[[4, 7, 17]] = 3
LABEL_GROUP[[5, 6, 8]] = 4
PALETTE_BGR = np.array([[60, 60, 60], [180, 40, 180], [70, 175, 245],
                        [80, 200, 40], [220, 100, 60]], np.uint8)


def sample_layers(labels, confidence, boundary_distance, xy, box):
    """Nearest-neighbor categorical samples; never interpolate numeric class IDs."""
    ij = np.rint(np.asarray(xy) - np.asarray(box[:2])).astype(int)
    inside = ((ij[:, 0] >= 0) & (ij[:, 0] < labels.shape[1]) &
              (ij[:, 1] >= 0) & (ij[:, 1] < labels.shape[0]))
    group = np.zeros(len(xy), np.uint8); certainty = np.zeros(len(xy), np.float32)
    distance = np.zeros(len(xy), np.float32)
    ids = np.flatnonzero(inside); x, y = ij[ids].T
    group[ids] = labels[y, x]; certainty[ids] = confidence[y, x].astype(float) / 255
    distance[ids] = boundary_distance[y, x]
    return group, certainty, distance


def distances(labels):
    result = np.zeros(labels.shape, np.float32)
    for label in range(1, 5):
        mask = labels == label
        result[mask] = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)[mask]
    return result


class Parser:
    def __init__(self, model):
        self.model = Path(model)
        config = json.loads((self.model / 'config.json').read_text())
        if config['id2label']['2'] != 'Hair' or config['id2label']['4'] != 'Upper-clothes':
            raise ValueError('Unexpected model class mapping')
        self.processor = json.loads((self.model / 'preprocessor_config.json').read_text())
        settings = ort.SessionOptions(); settings.intra_op_num_threads = 4; settings.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model / 'onnx/model.onnx'), settings,
                                            providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, image):
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((512, 512), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb).astype(np.float32) / 255
        arr = (arr - np.asarray(self.processor['image_mean'], np.float32)) / np.asarray(self.processor['image_std'], np.float32)
        logits = self.session.run(None, {self.input_name: arr.transpose(2, 0, 1)[None]})[0][0]
        logits = cv2.resize(logits.transpose(1, 2, 0), image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        probability = np.exp(logits - logits.max(axis=2, keepdims=True))
        probability /= probability.sum(axis=2, keepdims=True)
        raw_label = logits.argmax(axis=2).astype(np.uint8)
        labels = LABEL_GROUP[raw_label]
        group_probability = np.stack([probability[:, :, LABEL_GROUP == k].sum(axis=2) for k in range(5)], axis=2)
        confidence = np.take_along_axis(group_probability, labels[:, :, None], axis=2)[:, :, 0]
        return labels, np.rint(confidence * 255).astype(np.uint8), raw_label


def generate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = args.model / 'provenance.json'
    manifest = args.output / 'model_binding.json'
    identity = {'model_sha256': sha256(args.model / 'onnx/model.onnx'),
                'provenance': json.loads(provenance.read_text()), 'code_sha256': sha256(Path(__file__)),
                'input_size': [512, 512], 'onnxruntime': ort.__version__,
                'groups': GROUPS, 'calibrated_confidence': False,
                'coordinate_space': 'native source video xy; mirror image is not flipped'}
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Existing semantic cache binding differs; use a new output')
    write(manifest, identity)
    batch = json.loads(args.batch.read_text())
    _, rows = validate_fusion_group(args.group, batch['clips'], args.layout)
    parser = Parser(args.model)
    history = []
    for clip, (entry, row) in enumerate(zip(batch['clips'], rows)):
        folder = Path(entry['folder']); name = folder.name
        if args.clips and name not in args.clips: continue
        frames = args.frames if args.frames else row['frames']['train_frames']
        if args.split == 'train' and not set(frames) <= set(row['frames']['train_frames']):
            raise ValueError('Nontraining frame in semantic TRAIN cache')
        if args.split == 'heldout' and not set(frames) <= set(row['frames']['heldout_frames']):
            raise ValueError('Unexpected frame in semantic heldout cache')
        if args.split == 'fresh' and set(frames) & (set(row['frames']['train_frames']) | set(row['frames']['heldout_frames'])):
            raise ValueError('Fresh frame already in texture splits')
        data = load_data(folder / 'attempts/0001/work/reconstruction.npz')
        meta = json.loads((folder / 'result/mesh_meta.json').read_text())
        mirror = json.loads((folder / 'result/mirror_geometry.json').read_text())
        cap = cv2.VideoCapture(str(folder / 'source.mp4'))
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame + 1: raise ValueError('Frame mismatch')
            for view, camera, mask, other in views(data, meta, mirror, frame, 'independent'):
                path = args.output / f'{name}_f{frame:03}_v{view}.npz'
                if path.exists(): continue
                yy, xx = np.where(mask > 234)
                if not len(xx): continue
                scale = np.asarray(meta['image_size']) / np.asarray(mask.shape[::-1])
                lo = np.maximum(0, np.floor(np.array([xx.min(), yy.min()]) * scale).astype(int) - 20)
                hi = np.minimum(meta['image_size'], np.ceil(np.array([xx.max() + 1, yy.max() + 1]) * scale).astype(int) + 20)
                crop = image[lo[1]:hi[1], lo[0]:hi[0]]
                started = time.monotonic(); label, confidence, raw_label = parser.infer(crop)
                box = np.r_[lo, hi]; distance = distances(label)
                face_xy = project(camera[data['faces']].mean(1), meta['focal'][frame], meta['image_size'])
                face_label, face_confidence, face_distance = sample_layers(label, confidence, distance, face_xy, box)
                np.savez_compressed(path, labels=label, confidence=confidence, raw_labels=raw_label,
                    box=box, face_label=face_label, face_confidence=face_confidence, face_distance=face_distance)
                record = dict(clip=name, frame=frame, view=view, split=args.split, native_crop_xyxy=box.tolist(),
                    video_sha256=row['inputs']['video']['sha256'], archive_sha256=row['inputs']['native_archive']['sha256'],
                    model_sha256=identity['model_sha256'], cache_sha256=sha256(path), seconds=time.monotonic()-started)
                write(path.with_suffix('.json'), record); history.append(record)
                if frame in [80, 140, 190, 137, 213]:
                    overlay = cv2.addWeighted(crop, .55, PALETTE_BGR[label], .45, 0)
                    # Source-pixel panels; no AI sharpening or image generation.
                    cv2.imwrite(str(path.with_suffix('.png')), np.concatenate([crop, overlay, PALETTE_BGR[label]], axis=1))
            print('SEMANTIC', name, frame, flush=True)
        cap.release()
    write(args.output / f'execution_{args.split}_{int(time.time())}.json', history)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['batch', 'group', 'layout', 'model', 'output']: p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--frames', type=int, nargs='+'); p.add_argument('--clips', nargs='+')
    p.add_argument('--split', choices=['train', 'heldout', 'fresh'], default='train')
    generate(p.parse_args())
