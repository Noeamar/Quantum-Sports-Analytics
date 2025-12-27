# unsupervised/unsupervised_pipeline.py

import numpy as np
from .features import build_features
from .detection import detect_events_unsupervised, cluster_events
from .classification import classify_events

def unsupervised_hit_bounce_detection(ball_data):
    X, valid_frames, frame_ids = build_features(ball_data)

    anomaly_idx = detect_events_unsupervised(X)
    anomaly_frames = np.array(valid_frames)[anomaly_idx]

    event_centers = cluster_events(anomaly_frames)

    # recompute raw signals for physics
    x = np.array([ball_data[f]["x"] for f in frame_ids], dtype=float)
    y = np.array([ball_data[f]["y"] for f in frame_ids], dtype=float)

    vy = np.gradient(y)
    speed = np.sqrt(np.gradient(x)**2 + np.gradient(y)**2)
    accel = np.gradient(speed)

    bounce, hit = classify_events(event_centers, y, vy, speed, accel)

    bounce_set = set(bounce)
    hit_set = set(hit)

    for i, fid in enumerate(frame_ids):
        if i in bounce_set:
            ball_data[fid]["pred_action"] = "bounce"
        elif i in hit_set:
            ball_data[fid]["pred_action"] = "hit"
        else:
            ball_data[fid]["pred_action"] = "air"

    return ball_data