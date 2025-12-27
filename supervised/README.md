# Supervised Hit & Bounce Detection

This folder contains the supervised pipeline for detecting tennis hits and bounces
from ball-tracking data.

## Method

A temporal classification model is trained using:
- Physics-based features (velocity, acceleration, direction)
- Raw + smoothed trajectories
- Multi-scale sliding windows
- Gradient boosting (LightGBM)

This approach explicitly learns the temporal dynamics of ball interactions.

## Final Model

- Model: LightGBM
- Input: multi-window temporal embeddings
- Output: air / hit / bounce
- Handles extreme class imbalance
- Strong frame-level performance

This supervised method significantly outperforms unsupervised baselines.