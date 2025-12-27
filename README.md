# 🎾 Hit & Bounce Detection — Roland-Garros 2025 Final

This repository contains two complete pipelines — **unsupervised** and **supervised** — to detect **tennis ball hits and bounces** from ball-tracking data extracted during the Roland-Garros 2025 Final.

The task consists in labeling each video frame as:
- `air`
- `hit`
- `bounce`

using only the ball trajectory `(x, y)` and visibility information.

---

All paths are **relative** and reusable after cloning the repository.

---

## 🎯 Objective

Given a ball trajectory time series:


## Method 1 — Unsupervised Detection

Principle

No labels are used.

Events are detected by identifying physical discontinuities in the trajectory:
	•	Sudden acceleration spikes
	•	Velocity sign inversions
	•	Direction changes
	•	Height extrema (bounces)

Pipeline
	1.	Interpolate missing positions
	2.	Smooth trajectories (Savitzky–Golay)
	3.	Compute derivatives (velocity, acceleration)
	4.	Detect anomalous frames using Isolation Forest
	5.	Cluster anomalies temporally (DBSCAN)
	6.	Classify events as hit or bounce using physics rules

Output

A per-frame prediction without any supervised learning.

⸻

## Method 2 — Supervised Detection

Principle

Use provided action labels to learn temporal dynamics.

Key Ideas
	•	Sliding temporal windows around each frame
	•	Multi-scale temporal context
	•	Raw + smoothed physical features
	•	Strong gradient-boosted classifier

Features
	•	Position, velocity, acceleration
	•	Speed, jerk, angle, angle changes
	•	Raw and smoothed signals
	•	Multi-window temporal embeddings

Model
	•	LightGBM (multiclass)
	•	Class imbalance handled via weighting
	•	Window sizes: [5, 10, 20, 30] frames

This approach achieves high recall and precision on hits and bounces.

⸻

## Running the Pipelines

Run both methods on the full dataset:

python main.py

This will:
	•	Apply the unsupervised detector to all points
	•	Apply the supervised model to all points
	•	Save enriched JSON files for both methods

## Evaluation

Evaluation is performed globally over the full match:
	•	Frame-level precision / recall / F1
	•	Confusion matrix
	•	Strong class imbalance handled explicitly

Metrics are printed once per run (not per point).

