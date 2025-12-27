# unsupervised/detection.py

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

def detect_events_unsupervised(X, contamination=0.03):
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    preds = model.predict(X)   # -1 = anomaly
    return np.where(preds == -1)[0]


def cluster_events(frame_indices, eps=5):
    if len(frame_indices) == 0:
        return []

    X = frame_indices.reshape(-1, 1)
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(X)

    centers = []
    for lab in np.unique(labels):
        cluster = frame_indices[labels == lab]
        centers.append(int(np.median(cluster)))

    return sorted(centers)