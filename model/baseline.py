"""Train and evaluate a TF-IDF + logistic regression multi-label baseline.

Uses the same processed train/val/test splits as the BERT pipeline. Thresholds
are tuned on validation data and reported on the held-out test split.
"""
import logging
import os
import argparse
import csv
import json

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier

PROCESSED_ROOT = os.path.join(os.path.dirname(__file__), 'processed')
DEFAULT_PROCESSED_DIR = os.path.join(PROCESSED_ROOT, 'random')
DEFAULT_RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results')

MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
C = 1.0
MAX_ITER = 2000
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.05)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train/evaluate the TF-IDF logistic baseline.')
    parser.add_argument('--processed-dir', default=DEFAULT_PROCESSED_DIR, help='Directory containing train/val/test .pt files')
    parser.add_argument('--run-name', default=None, help='Name for result folder under model/results')
    parser.add_argument('--results-root', default=DEFAULT_RESULTS_ROOT, help='Root directory for baseline result files')
    parser.add_argument('--no-save', action='store_true', help='Print metrics without writing result artifacts')
    return parser.parse_args()


def split_name_from_processed_dir(processed_dir):
    return os.path.basename(os.path.normpath(processed_dir)) or 'default'


def resolve_result_paths(args):
    run_name = args.run_name or split_name_from_processed_dir(args.processed_dir)
    results_dir = os.path.join(args.results_root, run_name)
    return (
        os.path.join(results_dir, 'baseline_metrics.json'),
        os.path.join(results_dir, 'baseline_per_label.csv'),
    )


def load_split(name, processed_dir):
    data = torch.load(os.path.join(processed_dir, f'{name}.pt'))
    return data['texts'], data['labels'].numpy()


def tune_per_label_thresholds(probs, labels, label_classes):
    thresholds = np.full(len(label_classes), 0.5)
    for j, name in enumerate(label_classes):
        best_f1, best_t = -1.0, 0.5
        for t in THRESHOLD_GRID:
            f1 = f1_score(labels[:, j], probs[:, j] >= t, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[j] = best_t
        logger.info('  %s: threshold=%.2f (val F1=%.4f)', name, best_t, best_f1)
    return thresholds


def save_results(args, metrics, per_label):
    metrics_path, per_label_path = resolve_result_paths(args)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    with open(per_label_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['label', 'threshold', 'precision', 'recall', 'f1', 'support'],
        )
        writer.writeheader()
        writer.writerows(per_label)

    logger.info('Baseline metrics saved to %s', metrics_path)
    logger.info('Baseline per-label metrics saved to %s', per_label_path)


def main():
    args = parse_args()
    logger.info('Loading processed split artifacts from %s', args.processed_dir)
    label_classes = np.load(os.path.join(args.processed_dir, 'label_classes.npy'), allow_pickle=True)

    train_texts, train_labels = load_split('train', args.processed_dir)
    val_texts, val_labels = load_split('val', args.processed_dir)
    test_texts, test_labels = load_split('test', args.processed_dir)

    logger.info(
        'Training TF-IDF logistic baseline | %d train | %d val | %d test | %d labels',
        len(train_texts), len(val_texts), len(test_texts), len(label_classes)
    )

    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_texts)
    val_features = vectorizer.transform(val_texts)
    test_features = vectorizer.transform(test_texts)

    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=MAX_ITER,
            solver='liblinear',
        )
    )
    classifier.fit(train_features, train_labels)

    logger.info('Tuning per-label thresholds on validation set...')
    val_probs = classifier.predict_proba(val_features)
    thresholds = tune_per_label_thresholds(val_probs, val_labels, label_classes)

    test_probs = classifier.predict_proba(test_features)
    preds = test_probs >= thresholds[np.newaxis, :]

    micro_f1 = f1_score(test_labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(test_labels, preds, average='macro', zero_division=0)
    logger.info('Test micro_f1=%.4f macro_f1=%.4f', micro_f1, macro_f1)

    logger.info('Per-label breakdown (threshold | precision | recall | f1 | support):')
    per_label = []
    for j, name in enumerate(label_classes):
        precision = precision_score(test_labels[:, j], preds[:, j], zero_division=0)
        recall = recall_score(test_labels[:, j], preds[:, j], zero_division=0)
        f1 = f1_score(test_labels[:, j], preds[:, j], zero_division=0)
        support = int(test_labels[:, j].sum())
        per_label.append({
            'label': str(name),
            'threshold': float(thresholds[j]),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': support,
        })
        logger.info(
            '  %-30s thr=%.2f  P=%.3f  R=%.3f  F1=%.3f  support=%d',
            name, thresholds[j], precision, recall, f1, support
        )

    metrics = {
        'model': 'tfidf_logistic',
        'processed_dir': args.processed_dir,
        'split': args.run_name or split_name_from_processed_dir(args.processed_dir),
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
        'thresholds': {str(name): float(thresholds[j]) for j, name in enumerate(label_classes)},
        'per_label': per_label,
        'config': {
            'max_features': MAX_FEATURES,
            'ngram_range': list(NGRAM_RANGE),
            'min_df': MIN_DF,
            'C': C,
            'class_weight': 'balanced',
            'max_iter': MAX_ITER,
            'solver': 'liblinear',
        },
        'counts': {
            'train': len(train_texts),
            'val': len(val_texts),
            'test': len(test_texts),
            'labels': len(label_classes),
        },
    }
    if not args.no_save:
        save_results(args, metrics, per_label)


if __name__ == '__main__':
    main()
