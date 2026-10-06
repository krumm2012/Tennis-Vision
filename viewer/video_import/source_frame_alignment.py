"""Match the server's zero-start, round=near fps filter using source timestamps."""
import json
import subprocess
from fractions import Fraction

import numpy as np


def probe_timeline(path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
               '-show_entries', 'stream=time_base,width,height,duration:frame=best_effort_timestamp,duration,pkt_duration',
               '-of', 'json', str(path)]
    data = json.loads(subprocess.check_output(command, timeout=60))
    stream = data['streams'][0]
    unit = Fraction(stream['time_base'])
    times = [int(row['best_effort_timestamp']) * unit for row in data['frames']]
    final_duration = data['frames'][-1].get('duration', data['frames'][-1].get('pkt_duration'))
    end = times[-1] + int(final_duration)*unit if final_duration is not None else Fraction(stream['duration']) if 'duration' in stream else None
    return times, [stream['width'], stream['height']], end


def fps_filter_indices(original_times, normalized_times, fps, source_end=None):
    """Last input whose rounded output PTS is <= output PTS, as fps=...:round=near.

    This deliberately rejects other normalization pipelines rather than silently
    using average frame rate for VFR inputs. Inputs may be exact Fractions.
    """
    rate = Fraction(str(fps))
    source = [Fraction(str(t)) for t in original_times]
    target = [Fraction(str(t)) for t in normalized_times]
    if rate <= 0 or not source or not target or source[0] != 0 or target[0] != 0:
        raise ValueError('Expected zero-start fps normalization')
    if any(b <= a for a, b in zip(source, source[1:])):
        raise ValueError('Original timestamps must increase')
    if any(abs(t - Fraction(i, 1) / rate) > Fraction(1, 1000000) for i, t in enumerate(target)):
        raise ValueError('Normalized timestamps do not match fixed fps')
    ticks = []
    for time in source:
        q = time * rate + Fraction(1, 2)
        ticks.append(q.numerator // q.denominator)
    indices = np.searchsorted(ticks, np.arange(len(target)), side='right') - 1
    end_tick = None
    if source_end is not None:
        end_q = Fraction(str(source_end))*rate + Fraction(1, 2)
        end_tick = end_q.numerator // end_q.denominator
        if len(target) != end_tick:
            raise ValueError('Normalized frame count does not match source EOF')
    if np.any(indices < 0) or (end_tick is None and len(target) - 1 > ticks[-1]):
        raise ValueError('Normalized timeline extends beyond source timestamps')
    return indices


def source_frame_indices(original, normalized, fps):
    source_times, source_size, source_end = probe_timeline(original)
    target_times, target_size, _ = probe_timeline(normalized)
    return fps_filter_indices(source_times, target_times, fps, source_end), source_size, target_size
