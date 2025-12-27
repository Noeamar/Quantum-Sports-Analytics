import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def compute_raw_smooth_features(x, y, sg_window=15, poly=3):
    """
    Compute raw + smooth physics-based features from (x,y).
    """

    # RAW
    vx_raw = np.gradient(x)
    vy_raw = np.gradient(y)
    ax_raw = np.gradient(vx_raw)
    ay_raw = np.gradient(vy_raw)
    speed_raw = np.sqrt(vx_raw**2 + vy_raw**2)
    accel_raw = np.sqrt(ax_raw**2 + ay_raw**2)

    # SMOOTH
    x_s = savgol_filter(x, sg_window, poly)
    y_s = savgol_filter(y, sg_window, poly)

    vx = np.gradient(x_s)
    vy = np.gradient(y_s)
    ax = np.gradient(vx)
    ay = np.gradient(vy)

    speed = np.sqrt(vx**2 + vy**2)
    accel = np.sqrt(ax**2 + ay**2)
    jerk = np.gradient(accel)

    angle = np.degrees(np.arctan2(vy, vx))
    angle_change = np.gradient(angle)

    return pd.DataFrame({
        # RAW
        "x_raw": x, "y_raw": y,
        "vx_raw": vx_raw, "vy_raw": vy_raw,
        "ax_raw": ax_raw, "ay_raw": ay_raw,
        "speed_raw": speed_raw, "accel_raw": accel_raw,

        # SMOOTH
        "x_s": x_s, "y_s": y_s,
        "vx_s": vx, "vy_s": vy,
        "ax_s": ax, "ay_s": ay,
        "speed_s": speed,
        "accel_s": accel,
        "jerk_s": jerk,
        "angle_s": angle,
        "angle_change_s": angle_change,
    })


def build_multiwindow_features(df, t, windows):
    """
    Concatenate temporal windows around frame t.
    """
    vectors = []
    T = len(df)

    for W in windows:
        if t - W < 0 or t + W >= T:
            return None
        vectors.append(df.iloc[t-W:t+W+1].values.flatten())

    return np.concatenate(vectors)