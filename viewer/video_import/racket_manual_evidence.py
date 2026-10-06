"""Keep reviewed pixels authoritative and symmetric rim labels unordered."""
import numpy as np


def view_observations(automatic, reviewed, view, pixel_scale):
    if reviewed is not None:
        key = 'points' if view == 'real' else 'mirror_points'
        points = {k: np.asarray(p, float)/pixel_scale for k, p in reviewed.get(key, {}).items()}
        if not reviewed.get('grip_confirmed',{}).get(key,False):points.pop('grip_center',None)
        return points, {k: 3. for k in points}
    points = {k: np.asarray(p, float)/pixel_scale for k, p in automatic.get('points', {}).items()}
    weights = {k: min(1., automatic.get('confidence', 0)*3)*2.5/(automatic.get('uncertainty_px', {}).get(k, 5)/pixel_scale) for k in points}
    return points, weights


def pixel_terms(predicted, observed, weights, names, view_weight=1., rim_confirmed=False):
    terms = np.zeros((len(names), 2))
    for k, name in enumerate(names):
        if name in observed:
            terms[k] = (predicted[k]-observed[name])*weights[name]*view_weight*(6. if name == 'head_center' else 1.)
    if not rim_confirmed and {'rim_side', 'rim_opposite'} <= set(observed):
        a, b = names.index('rim_side'), names.index('rim_opposite')
        swapped = np.stack([(predicted[a]-observed['rim_opposite'])*weights['rim_opposite']*view_weight,
                            (predicted[b]-observed['rim_side'])*weights['rim_side']*view_weight])
        # Match the downstream pseudo-Huber objective, not raw L2 assignment.
        cost = lambda x: np.sum(np.sqrt(1+(x/3)**2)-1)
        if cost(swapped) < cost(terms[[a, b]]):
            terms[[a, b]] = swapped
    return terms
