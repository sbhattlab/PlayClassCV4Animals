"""XGBoost baseline for behaviour classification with LOCO cross-validation.

Usage::

    pixi run -e classifier python -m script.train_xgboost

    pixi run -e classifier python -m script.train_xgboost --exclude social
"""

import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix, f1_score
from xgboost import XGBClassifier

from src._config import DEFAULT_CHECKPOINT_DIR, DEFAULT_DATASET_DIR, LABEL_ORDER
from src.classification.datamodule import LabelEncoder
from src.classification.model_selection import LOCO, LOVO
from src.classification.stats import aggregate_metrics


def parse_args():
    parser = ArgumentParser(description="XGBoost behaviour classifier (LOCO).")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing dataset files (default: %(default)s)",
    )
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument(
        "--cv",
        type=str,
        default="loco",
        choices=["lovo", "loco"],
        help="Cross-validation: loco (leave-one-cage-out, default) or lovo (leave-one-video-out)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load features and labels
    _skip_cols = {"video_id", "bird_id", "window", "n_frames"}
    features_df = pd.read_parquet(args.dataset_dir / "features_windowed.parquet")
    labels_df = pd.read_parquet(args.dataset_dir / "labels.parquet")

    # Median impute + z-score normalize (match datamodule)
    feat_cols = [c for c in features_df.columns if c not in _skip_cols]
    features_df[feat_cols] = features_df[feat_cols].fillna(
        features_df[feat_cols].median()
    )

    X = features_df[feat_cols].values
    video_ids = features_df["video_id"].values

    # Labels
    label_order = LABEL_ORDER.copy()
    if args.exclude:
        exclude_list = [args.exclude] if isinstance(args.exclude, str) else args.exclude
        label_order = [l for l in label_order if l not in exclude_list]
        mask = ~labels_df["behav_label"].isin(exclude_list)
        labels_df = labels_df[mask].reset_index(drop=True)
        X = X[mask.values]
        video_ids = video_ids[mask.values]

    le = LabelEncoder(label_order)
    y = np.array(le.encode(labels_df["behav_label"].tolist()))
    all_videos = sorted(set(video_ids))

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(DEFAULT_CHECKPOINT_DIR) / f"{timestamp}_xgboost"
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (run_dir / "cfg.json").open("w"), default=str, indent=2)

    splitter = LOCO() if args.cv == "loco" else LOVO()
    folds = list(splitter.split(all_videos))
    logger.info(f"{args.cv.upper()}: {len(folds)} folds")

    fold_results = []
    for fold_idx, (test_video, val_video) in enumerate(folds):
        logger.info(f"Fold {fold_idx}: test={test_video}, val={val_video}")

        if args.cv == "loco":
            test_mask = np.array([v.startswith(test_video) for v in video_ids])
            val_mask = np.array([v.startswith(val_video) for v in video_ids])
        else:
            test_mask = video_ids == test_video
            val_mask = video_ids == val_video
        train_mask = ~(test_mask | val_mask)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Inverse-sqrt class weights (match MLP baseline)
        counts = np.bincount(y_train, minlength=len(label_order)).astype(float)
        counts = np.clip(counts, 1, None)
        class_weights = 1 / np.sqrt(counts)
        class_weights = class_weights / class_weights.sum() * len(label_order)
        sample_weights = class_weights[y_train]

        clf = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_child_weight=5,
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            device="cuda",
            verbosity=1,
        )
        clf.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        result = {"test_id": test_video, "val_id": val_video}
        for split, X_s, y_s, prefix in [
            ("test", X_test, y_test, "test_"),
            ("train", X_train, y_train, "train_"),
            ("val", X_val, y_val, "val_"),
        ]:
            preds = clf.predict(X_s)
            result[f"{prefix}macro_f1"] = f1_score(y_s, preds, average="macro")
            result[f"{prefix}confusion_matrix"] = confusion_matrix(
                y_s, preds, labels=list(range(len(label_order)))
            )

        # Save model
        model_path = run_dir / f"fold_{fold_idx}_{test_video}.json"
        clf.save_model(model_path)

        fold_results.append(result)
        logger.info(
            f"  test_macro_f1={result['test_macro_f1']:.4f}  "
            f"val_macro_f1={result['val_macro_f1']:.4f}"
        )

    # Aggregate
    aggregate_metrics(fold_results, run_dir, label_order, prefix=args.cv)


if __name__ == "__main__":
    main()
