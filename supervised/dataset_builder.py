import json
import numpy as np
from pathlib import Path
from feature_engineering import compute_raw_smooth_features, build_multiwindow_features


def load_dataset_multiwindow(folder: Path, windows=(5, 10, 20, 30)):
    rows = []

    for file in folder.glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        frames = sorted(data.keys(), key=lambda x: int(x))
        x = np.array([data[f]["x"] for f in frames], float)
        y = np.array([data[f]["y"] for f in frames], float)
        labels = np.array([data[f]["action"] for f in frames])
        visible = np.array([data[f]["visible"] for f in frames])

        feats = compute_raw_smooth_features(x, y)
        T = len(frames)
        max_w = max(windows)

        for t in range(max_w, T - max_w):
            if not visible[t]:
                continue

            vec = build_multiwindow_features(feats, t, windows)
            if vec is None:
                continue

            rows.append((vec, labels[t]))

    return rows