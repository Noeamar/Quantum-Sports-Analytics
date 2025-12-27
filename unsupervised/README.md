# Unsupervised Hit & Bounce Detection

This folder contains the **unsupervised method** developed to detect **tennis hits and bounces**
from ball-tracking data of the Roland-Garros 2025 Final.

The method strictly follows the constraints of the exercise:
- No labels are used during detection
- Detection relies on physics-based analysis
- Only the (x, y) ball trajectory is exploited

---

## Input / Output

### Input
A JSON file per point containing ball positions:

```json
"56100": {
  "x": 894,
  "y": 395,
  "visible": true,
  "action": "air"
}

Output

Same structure enriched with a prediction:

"56100": {
  "x": 894,
  "y": 395,
  "visible": true,
  "action": "air",
  "pred_action": "bounce"
}


Method Overview (Unsupervised)

The pipeline is composed of three main steps.

1. Feature Extraction (No Supervision)

For each frame, physics-inspired features are extracted from the ball trajectory:
	•	vertical position (y)
	•	velocities and accelerations (x and y)
	•	speed and acceleration magnitude
	•	direction changes
	•	multi-scale sliding window statistics (mean and standard deviation)

Missing positions are interpolated on visible frames and smoothed using a
Savitzky–Golay filter to reduce tracking noise.

2. Event Detection (Anomaly Detection)

Hits and bounces are treated as rare events in the trajectory dynamics.
	•	Model: Isolation Forest
	•	Input: per-frame feature vectors
	•	Output: anomalous frames

Detected anomalies are temporally clustered using DBSCAN to obtain a single
representative frame per interaction event.


3. Physics-Based Classification (Hit vs Bounce)

Each detected event is classified using simple physical rules:
	•	Bounce:
	•	inversion of vertical velocity
	•	high vertical position (near the ground)
	•	strong acceleration peak
	•	Hit:
	•	strong acceleration and directional change
	•	does not satisfy bounce conditions

All other frames are labeled as “air”.

## Main Function

unsupervised_hit_bounce_detection(ball_data: dict) -> dict

## Limitations
	•	Frame-level labeling is difficult for unsupervised physics-based methods
	•	Numerical derivatives amplify tracking noise
	•	Bounce detection is sensitive to camera geometry and missing data

Despite these limitations, this pipeline provides a solid unsupervised baseline
and motivates the supervised approach implemented in the second part of the project.