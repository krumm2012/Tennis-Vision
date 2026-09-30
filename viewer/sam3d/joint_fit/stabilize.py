"""Small, time-aware display correction; never changes source-camera geometry."""
import numpy as np


def smooth_display(points, times, strength=.6, max_offset=.015):
    p=np.asarray(points,dtype=np.float64)
    t=np.asarray(times,dtype=float)
    if len(p)!=len(t) or not np.isfinite(p).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid stabilization sequence')
    out=p.copy()
    for i in range(1,len(p)-1):
        before,after=t[i]-t[i-1],t[i+1]-t[i]
        if max(before,after)>.12:continue  # Never blend across missing segments.
        speed=np.maximum(np.linalg.norm(p[i]-p[i-1],axis=-1)/before,np.linalg.norm(p[i+1]-p[i],axis=-1)/after)
        neighbor=(p[i-1]*after+p[i+1]*before)/(before+after)
        weight=strength/(1+(speed/.8)**2)
        delta=(neighbor-p[i])*np.asarray(weight)[...,None]
        length=np.linalg.norm(delta,axis=-1,keepdims=True)
        out[i]+=delta*np.minimum(1,max_offset/np.maximum(length,1e-12))
    return out.astype('<f4')


def stabilize_body(vertices,joints,times):
    stable=smooth_display(vertices,times)
    shown=smooth_display(joints,times)
    # Keep the native wrist and fingers intact, with a smooth transition at the
    # forearm. The already-fitted racket and hand therefore retain their grip.
    for i in range(len(vertices)):
        distance=np.minimum.reduce([np.linalg.norm(vertices[i]-joints[i,w],axis=1) for w in (41,62)])
        blend=np.clip((distance-.16)/.14,0,1)
        blend=blend*blend*(3-2*blend)
        stable[i]=vertices[i]+(stable[i]-vertices[i])*blend[:,None]
    shown[:,21:63]=joints[:,21:63]
    # Retain the same pelvis origin as the raw mesh.
    shown[:,[9,10]]=joints[:,[9,10]]
    return stable,shown
