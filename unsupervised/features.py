# unsupervised/features.py

import numpy as np
from scipy.signal import savgol_filter

def compute_derivatives(x, y):
    vx = np.gradient(x)
    vy = np.gradient(y)
    ax = np.gradient(vx)
    ay = np.gradient(vy)
    speed = np.sqrt(vx**2 + vy**2)
    accel = np.sqrt(ax**2 + ay**2)
    return vx, vy, ax, ay, speed, accel


def sliding_window_stats(arr, t, w):
    left = max(0, t - w)
    right = min(len(arr), t + w + 1)
    window = arr[left:right]
    return window.mean(), window.std()


def build_features(ball_data, windows=(3, 5, 9)):
    frame_ids = sorted(ball_data.keys(), key=lambda x: int(x))

    x = np.array([ball_data[f]["x"] for f in frame_ids], dtype=float)
    y = np.array([ball_data[f]["y"] for f in frame_ids], dtype=float)
    visible = np.array([ball_data[f]["visible"] for f in frame_ids])

    # interpolation simple
    idx = np.arange(len(x))
    mask = visible & ~np.isnan(x)
    x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    y[~mask] = np.interp(idx[~mask], idx[mask], y[mask])

    # smoothing
    if len(x) >= 7:
        x = savgol_filter(x, 7, 3)
        y = savgol_filter(y, 7, 3)

    vx, vy, ax, ay, speed, accel = compute_derivatives(x, y)

    X = []
    valid_frames = []

    for t in range(3, len(x) - 3):
        feats = [
            y[t],
            vx[t], vy[t],
            ax[t], ay[t],
            speed[t],
            accel[t],
            vy[t] - vy[t-1],
            speed[t+1] - speed[t],
        ]

        for w in windows:
            feats.extend(sliding_window_stats(speed, t, w))
            feats.extend(sliding_window_stats(accel, t, w))
            feats.extend(sliding_window_stats(vy, t, w))

        X.append(feats)
        valid_frames.append(t)

    return np.array(X), valid_frames, frame_ids