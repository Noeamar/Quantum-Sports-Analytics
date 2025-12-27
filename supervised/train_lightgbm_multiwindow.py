import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
from dataset_builder import load_dataset_multiwindow


LABEL_MAP = {"air": 0, "bounce": 1, "hit": 2}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}


def train_lightgbm(X_train, y_train):
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    return lgb.train(params, dtrain, num_boost_round=600)


if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[1]
    folder = ROOT / "Data hit & bounce" / "per_point_v2"

    print("Loading dataset...")
    rows = load_dataset_multiwindow(folder)

    X = np.stack([r[0] for r in rows])
    y = np.array([LABEL_MAP[r[1]] for r in rows])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    print("Training LightGBM...")
    model = train_lightgbm(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.argmax(preds, axis=1)

    print("\n=== LightGBM Multi-Window ===")
    print(classification_report(
        [INV_MAP[i] for i in y_test],
        [INV_MAP[i] for i in preds]
    ))
    print(confusion_matrix(y_test, preds))

    model.save_model("model_lightgbm.txt")