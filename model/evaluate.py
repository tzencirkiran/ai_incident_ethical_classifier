"""Evaluate the fine-tuned bert-tiny checkpoint on the held-out test split.

Reports overall micro/macro F1 (using the tuned per-label thresholds saved
alongside the checkpoint) and a per-label F1 / precision / recall breakdown.
"""
import logging
import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification

PROCESSED_ROOT = os.path.join(os.path.dirname(__file__), 'processed')
DEFAULT_PROCESSED_DIR = os.path.join(PROCESSED_ROOT, 'random')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoint')
BATCH_SIZE = 32
DEFAULT_THRESHOLD = 0.5  # fallback if a per-label thresholds.npy isn't found

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the fine-tuned checkpoint.')
    parser.add_argument('--processed-dir', default=DEFAULT_PROCESSED_DIR, help='Directory containing train/val/test .pt files')
    return parser.parse_args()


def load_split(name, processed_dir):
    data = torch.load(os.path.join(processed_dir, f'{name}.pt'))
    dataset = TensorDataset(data['input_ids'], data['attention_mask'], data['labels'])
    return dataset


def predict_probs(model, loader):
    """Run the model over a loader and return (probs, labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            probs = torch.sigmoid(model(input_ids=input_ids, attention_mask=attention_mask).logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    args = parse_args()
    logger.info('Loading processed split artifacts from %s', args.processed_dir)
    label_classes = np.load(os.path.join(args.processed_dir, 'label_classes.npy'), allow_pickle=True)

    thresholds_path = os.path.join(CHECKPOINT_DIR, 'thresholds.npy')
    if os.path.exists(thresholds_path):
        thresholds = np.load(thresholds_path)
        logger.info('Loaded per-label thresholds from %s', thresholds_path)
    else:
        thresholds = np.full(len(label_classes), DEFAULT_THRESHOLD)
        logger.warning('No thresholds.npy found; falling back to global threshold=%.2f', DEFAULT_THRESHOLD)

    logger.info('Loading checkpoint from %s', CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(DEVICE)

    test_loader = DataLoader(load_split('test', args.processed_dir), batch_size=BATCH_SIZE)
    logger.info('Evaluating on %d test examples', len(test_loader.dataset))

    probs, labels = predict_probs(model, test_loader)
    preds = probs >= thresholds[np.newaxis, :]

    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    logger.info('Test micro_f1=%.4f macro_f1=%.4f', micro_f1, macro_f1)

    logger.info('Per-label breakdown (threshold | precision | recall | f1 | support):')
    for j, name in enumerate(label_classes):
        precision = precision_score(labels[:, j], preds[:, j], zero_division=0)
        recall = recall_score(labels[:, j], preds[:, j], zero_division=0)
        f1 = f1_score(labels[:, j], preds[:, j], zero_division=0)
        support = int(labels[:, j].sum())
        logger.info(
            '  %-30s thr=%.2f  P=%.3f  R=%.3f  F1=%.3f  support=%d',
            name, thresholds[j], precision, recall, f1, support
        )


if __name__ == '__main__':
    main()
