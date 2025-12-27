# unsupervised/classification.py

import numpy as np

def classify_events(event_frames, y, vy, speed, accel):
    bounce = []
    hit = []

    accel_thr = np.percentile(accel, 80)
    y_ground = np.percentile(y, 70)

    for t in event_frames:
        if t < 2 or t > len(y) - 3:
            continue

        is_bounce = (
            vy[t-1] > 0 and vy[t+1] < 0 and
            y[t] >= y_ground and
            accel[t] > accel_thr
        )

        if is_bounce:
            bounce.append(t)
        else:
            hit.append(t)

    return bounce, hit