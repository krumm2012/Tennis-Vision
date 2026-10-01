"""Retain native SAM MHR parameters for real mesh replay, never invert mesh vertices."""
import numpy as np

FIELDS = {'mhr_model_params':'mhr_model_params', 'mhr_shape_params':'shape_params',
          'mhr_expr_params':'expr_params'}


def capture(person):
    result = {}
    for target, source in FIELDS.items():
        if source not in person:
            raise ValueError('SAM 未输出重拟合所需原生参数：'+source)
        a = np.asarray(person[source], dtype=np.float32).reshape(-1)
        if not a.size or not np.isfinite(a).all():
            raise ValueError('原生 MHR 参数无效：'+source)
        result[target] = a.copy()
    # Official MHR head: 136 native pose parameters + 68 skeletal scales.
    if result['mhr_model_params'].shape != (204,) or result['mhr_shape_params'].shape != (45,) or result['mhr_expr_params'].shape != (72,):
        raise ValueError('MHR 参数布局与当前官方 head 不一致；需检查模型版本')
    return result


def stack(rows, prefix=''):
    valid = [r for r in rows if r is not None]
    if not valid:
        return {}
    return {prefix+k:np.stack([r[k] if r is not None else np.zeros_like(valid[0][k]) for r in rows]) for k in FIELDS}


def available(archive, count):
    return all(k in archive and archive[k].shape == (count, width) and np.isfinite(archive[k]).all()
               for k,width in [('mhr_model_params',204),('mhr_shape_params',45),('mhr_expr_params',72)])
