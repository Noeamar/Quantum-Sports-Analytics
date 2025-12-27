import numpy as np
from feature_engineering import compute_raw_smooth_features, build_multiwindow_features


def supervised_hit_bounce_detection(ball_data, model, windows=(5,10,20,30)):
    frame_ids = sorted(ball_data.keys(), key=lambda x: int(x))

    x = np.array([ball_data[f]["x"] for f in frame_ids], float)
    y = np.array([ball_data[f]["y"] for f in frame_ids], float)
    visible = np.array([ball_data[f]["visible"] for f in frame_ids])

    feats = compute_raw_smooth_features(x, y)
    T = len(frame_ids)

    preds = ["air"] * T
    label_map = {0: "air", 1: "bounce", 2: "hit"}

    for t in range(max(windows), T - max(windows)):
        if not visible[t]:
            continue

        vec = build_multiwindow_features(feats, t, windows)
        if vec is None:
            continue

        p = model.predict(vec.reshape(1, -1))
        preds[t] = label_map[int(np.argmax(p))]

    for i, fid in enumerate(frame_ids):
        ball_data[fid]["pred_action"] = preds[i]

    return ball_data