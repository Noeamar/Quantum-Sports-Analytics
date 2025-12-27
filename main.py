"""
main.py

Run both UNSUPERVISED and SUPERVISED hit & bounce detection
on the Roland-Garros 2025 ball-tracking dataset.

Outputs enriched JSON files with:
    "pred_action": "air" | "hit" | "bounce"
"""

import json
from pathlib import Path
import lightgbm as lgb

# ============================
# Import detection pipelines
# ============================

from unsupervised.unsupervised_detection import unsupervised_hit_bounce_detection
from supervised.supervised_detection import supervised_hit_bounce_detection


# ============================
# Paths (portable / GitHub-safe)
# ============================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "Data hit & bounce" / "per_point_v2"
OUT_UNSUP = ROOT / "outputs_unsupervised"
OUT_SUP = ROOT / "outputs_supervised"

MODEL_PATH = ROOT / "supervised" / "model_lightgbm.txt"


# ============================
# Main processing
# ============================

def run_unsupervised():
    print("\n=== Running UNSUPERVISED detection ===")
    OUT_UNSUP.mkdir(exist_ok=True, parents=True)

    for fp in sorted(DATA_DIR.glob("*.json")):
        with open(fp) as f:
            ball_data = json.load(f)

        ball_data = unsupervised_hit_bounce_detection(ball_data)

        out_path = OUT_UNSUP / fp.name
        with open(out_path, "w") as f:
            json.dump(ball_data, f, indent=2)

    print(f"Unsupervised results saved to: {OUT_UNSUP}")


def run_supervised():
    print("\n=== Running SUPERVISED detection (LightGBM) ===")
    OUT_SUP.mkdir(exist_ok=True, parents=True)

    # Load trained model
    model = lgb.Booster(model_file=str(MODEL_PATH))

    for fp in sorted(DATA_DIR.glob("*.json")):
        with open(fp) as f:
            ball_data = json.load(f)

        ball_data = supervised_hit_bounce_detection(ball_data, model)

        out_path = OUT_SUP / fp.name
        with open(out_path, "w") as f:
            json.dump(ball_data, f, indent=2)

    print(f"Supervised results saved to: {OUT_SUP}")


# ============================
# Entry point
# ============================

if __name__ == "__main__":

    print("===================================")
    print("Hit & Bounce Detection Pipeline")
    print("===================================")
    print(f"Data folder: {DATA_DIR}")
    print(f"Number of points: {len(list(DATA_DIR.glob('*.json')))}")

    run_unsupervised()
    run_supervised()

    print("\n=== DONE ===")