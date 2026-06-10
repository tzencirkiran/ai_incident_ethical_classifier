"""Reusable TF-IDF logistic regression artifact for serving."""
import logging
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from model.baseline import (
    C,
    DEFAULT_RESULTS_ROOT,
    MAX_FEATURES,
    MAX_ITER,
    MIN_DF,
    NGRAM_RANGE,
    load_split,
    split_name_from_processed_dir,
    tune_per_label_thresholds,
)


PROCESSED_ROOT = os.path.join(os.path.dirname(__file__), "processed")
DEFAULT_PROCESSED_DIR = os.path.join(PROCESSED_ROOT, "random")
DEFAULT_ARTIFACT_ROOT = os.path.join(os.path.dirname(__file__), "baseline_checkpoint")
ARTIFACT_FILENAME = "tfidf_logistic.pkl"

logger = logging.getLogger(__name__)


def artifact_path(processed_dir=DEFAULT_PROCESSED_DIR, artifact_root=DEFAULT_ARTIFACT_ROOT):
    split_name = split_name_from_processed_dir(processed_dir)
    return os.path.join(artifact_root, split_name, ARTIFACT_FILENAME)


def required_processed_files(processed_dir=DEFAULT_PROCESSED_DIR):
    return [
        os.path.join(processed_dir, "train.pt"),
        os.path.join(processed_dir, "val.pt"),
        os.path.join(processed_dir, "label_classes.npy"),
    ]


def baseline_status(processed_dir=DEFAULT_PROCESSED_DIR, artifact_root=DEFAULT_ARTIFACT_ROOT):
    required = required_processed_files(processed_dir)
    missing = [path for path in required if not os.path.exists(path)]
    path = artifact_path(processed_dir, artifact_root)
    artifact_exists = os.path.exists(path)
    return {
        "processed_dir": processed_dir,
        "artifact_path": path,
        "artifact_exists": artifact_exists,
        "missing": missing,
        "ok": artifact_exists or not missing,
    }


def train_baseline_artifact(processed_dir=DEFAULT_PROCESSED_DIR):
    label_classes = np.load(os.path.join(processed_dir, "label_classes.npy"), allow_pickle=True)
    train_texts, train_labels = load_split("train", processed_dir)
    val_texts, val_labels = load_split("val", processed_dir)

    logger.info(
        "Training TF-IDF logistic artifact | processed_dir=%s train=%d val=%d labels=%d",
        processed_dir,
        len(train_texts),
        len(val_texts),
        len(label_classes),
    )
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_texts)
    val_features = vectorizer.transform(val_texts)

    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=MAX_ITER,
            solver="liblinear",
        )
    )
    classifier.fit(train_features, train_labels)

    val_probs = classifier.predict_proba(val_features)
    thresholds = tune_per_label_thresholds(val_probs, val_labels, label_classes)

    return {
        "model_id": "tfidf",
        "display_name": "TF-IDF Logistic Regression",
        "processed_dir": processed_dir,
        "split": split_name_from_processed_dir(processed_dir),
        "vectorizer": vectorizer,
        "classifier": classifier,
        "label_classes": label_classes,
        "thresholds": thresholds,
        "config": {
            "max_features": MAX_FEATURES,
            "ngram_range": list(NGRAM_RANGE),
            "min_df": MIN_DF,
            "C": C,
            "class_weight": "balanced",
            "max_iter": MAX_ITER,
            "solver": "liblinear",
            "results_root": DEFAULT_RESULTS_ROOT,
        },
    }


def save_artifact(artifact, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Saved TF-IDF logistic artifact to %s", path)


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_or_train_baseline(processed_dir=DEFAULT_PROCESSED_DIR, artifact_root=DEFAULT_ARTIFACT_ROOT):
    path = artifact_path(processed_dir, artifact_root)
    if os.path.exists(path):
        logger.info("Loading TF-IDF logistic artifact from %s", path)
        return load_artifact(path)

    status = baseline_status(processed_dir, artifact_root)
    if status["missing"]:
        missing = ", ".join(status["missing"])
        raise FileNotFoundError(f"Cannot train TF-IDF artifact; missing processed files: {missing}")

    artifact = train_baseline_artifact(processed_dir)
    save_artifact(artifact, path)
    return artifact


def predict_baseline_scores(text, artifact):
    features = artifact["vectorizer"].transform([text])
    probabilities = artifact["classifier"].predict_proba(features)[0]
    label_classes = artifact["label_classes"]
    thresholds = artifact["thresholds"]

    scores = []
    for label, probability, threshold in zip(label_classes, probabilities, thresholds):
        prob = float(probability)
        cutoff = float(threshold)
        scores.append(
            {
                "label": str(label),
                "probability": prob,
                "threshold": cutoff,
                "selected": prob >= cutoff,
            }
        )
    return sorted(scores, key=lambda item: item["probability"], reverse=True)
