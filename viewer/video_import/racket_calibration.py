"""Video-scoped measured dimensions; assumed asset dimensions remain separate."""
import copy
import numpy as np

GRIPS = ('unknown', 'continental', 'eastern', 'semi_western', 'western', 'other')


def validate(value, meta):
    for key in ('video_sha256', 'image_size', 'fps'):
        if value.get(key) != meta[key]:
            raise ValueError('尺寸标定与视频不一致')
    dims = value.get('dimensions_cm', {})
    if not isinstance(dims, dict):
        raise ValueError('尺寸格式无效')
    limits = {'length': (45, 85), 'head_width': (15, 40), 'head_height': (20, 45),
              'grip_contact_from_butt': (1, 20), 'handle_diameter': (2, 5)}
    if set(dims)-set(limits):
        raise ValueError('未知尺寸字段')
    clean = {}
    for name, val in dims.items():
        lo, hi = limits[name]
        if type(val) not in (int, float) or not np.isfinite(val) or not lo <= val <= hi:
            raise ValueError('尺寸超出合理范围：'+name)
        clean[name] = float(val)
    if {'length', 'head_height'} <= set(clean) and clean['head_height'] >= clean['length']*.7:
        raise ValueError('拍框外高与总长不一致')
    grip = value.get('grip_style', 'unknown')
    if grip not in GRIPS:
        raise ValueError('握拍方式无效')
    measured = value.get('measured', False)
    if type(measured) is not bool:
        raise ValueError('实测标记无效')
    complete = {'length', 'head_width', 'head_height'} <= set(clean)
    if measured and not complete:
        raise ValueError('实测需填写总长、拍框外宽和外高')
    return {'schema_version': 1, **{k: meta[k] for k in ('video_sha256', 'image_size', 'fps')},
            'dimensions_cm': clean, 'measured': measured, 'grip_style': grip,
            'size_ready': measured and complete, 'physical_grip_bevel_verified': False,
            'camera_calibrated': False,'dimension_source':'user_measurement' if measured else 'assumed_asset_dimensions'}


def measured_model(base, calibration):
    """Change model geometry only for complete measured dimensions."""
    model = copy.deepcopy(base)
    if not calibration['size_ready']:
        return model
    d = calibration['dimensions_cm']
    old_ring = np.asarray(base['head_outline'])
    old_bottom = float(old_ring[:, 1].min())
    old_top = float(old_ring[:, 1].max())
    length = d['length']/100
    height = d['head_height']/100
    width = d['head_width']/100
    ring = old_ring.copy()
    ring[:, 0] *= width/(2*base['head_half_width_m'])
    ring[:, 1] = length-height+(old_ring[:, 1]-old_bottom)*height/(old_top-old_bottom)
    throat = length-height+(base['throat_y_m']-old_bottom)*height/(old_top-old_bottom)
    model.update(length_m=length, head_half_width_m=width/2,
                 head_center_y_m=length-height/2, throat_y_m=throat,
                 head_outline=ring.tolist(), dimensions_measured=True,
                 grip_y_m=d.get('grip_contact_from_butt', 4.5)/100,
                 grip_position_measured='grip_contact_from_butt' in d,
                 asset_file='wilson_mesh_directional.bin')
    return model


def resize_mesh(values, base, model):
    """Preserve baked colors; deform the isolated racket asset, never body meshes."""
    packed = np.asarray(values, dtype='<f4').reshape(-1, 6).copy()
    old_bottom = np.asarray(base['head_outline'])[:, 1].min()
    new_bottom = np.asarray(model['head_outline'])[:, 1].min()
    y = packed[:, 1].copy()
    packed[:, 0] *= model['head_half_width_m']/base['head_half_width_m']
    packed[:, 1] = np.where(y < old_bottom, y*new_bottom/old_bottom,
                            new_bottom+(y-old_bottom)*(model['length_m']-new_bottom)/(base['length_m']-old_bottom))
    return packed
