"""Fine-tune prajjwal1/bert-tiny for multi-label ethical-issue classification.

Reads the tokenized splits produced by preprocessing.py and trains a
BertForSequenceClassification head with BCEWithLogitsLoss.
"""
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = 'prajjwal1/bert-tiny'
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'processed')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoint')

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
# Fixed threshold used only for model-selection during training, so "best epoch"
# doesn't depend on a threshold that gets tuned afterwards.
SELECTION_THRESHOLD = 0.5
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.05)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def load_split(name):
    data = torch.load(os.path.join(PROCESSED_DIR, f'{name}.pt'))
    dataset = TensorDataset(data['input_ids'], data['attention_mask'], data['labels'])
    return dataset


def predict_probs(model, loader):
    """Run the model over a loader and return (probs, labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            all_probs.append(torch.sigmoid(outputs.logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels), total_loss / len(loader)


def evaluate(model, loader, threshold=SELECTION_THRESHOLD):
    probs, labels, loss = predict_probs(model, loader)
    preds = probs >= threshold
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return loss, micro_f1, macro_f1


def evaluate_with_thresholds(model, loader, thresholds):
    """Evaluate using a per-label threshold vector instead of a single global cutoff."""
    probs, labels, loss = predict_probs(model, loader)
    preds = probs >= thresholds[np.newaxis, :]
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return loss, micro_f1, macro_f1


def tune_per_label_thresholds(model, loader, label_classes):
    """Find, for each label independently, the threshold in THRESHOLD_GRID that
    maximizes that label's F1 score on the given (validation) set."""
    probs, labels, _ = predict_probs(model, loader)

    thresholds = np.full(len(label_classes), SELECTION_THRESHOLD)
    for j, name in enumerate(label_classes):
        best_f1, best_t = -1.0, SELECTION_THRESHOLD
        for t in THRESHOLD_GRID:
            f1 = f1_score(labels[:, j], probs[:, j] >= t, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[j] = best_t
        logger.info('  %s: threshold=%.2f (val F1=%.4f)', name, best_t, best_f1)

    return thresholds


def build_label_maps(label_classes):
    labels = [str(label) for label in label_classes]
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in id2label.items()}
    return id2label, label2id


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    label_classes = np.load(os.path.join(PROCESSED_DIR, 'label_classes.npy'), allow_pickle=True)
    num_labels = len(label_classes)
    id2label, label2id = build_label_maps(label_classes)

    train_loader = DataLoader(load_split('train'), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(load_split('val'), batch_size=BATCH_SIZE)

    logger.info('Loading tokenizer for %s', MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info('Loading model %s', MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type='multi_label_classification',
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )

    logger.info('Training on %s | %d labels | %d train examples', DEVICE, num_labels, len(train_loader.dataset))

    best_val_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, micro_f1, macro_f1 = evaluate(model, val_loader)
        logger.info(
            'Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_micro_f1=%.4f | val_macro_f1=%.4f',
            epoch, EPOCHS, train_loss, val_loss, micro_f1, macro_f1
        )

        if micro_f1 > best_val_f1:
            best_val_f1 = micro_f1
            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)
            np.save(os.path.join(CHECKPOINT_DIR, 'label_classes.npy'), label_classes)
            logger.info('  -> new best (val_micro_f1=%.4f), checkpoint saved to %s', micro_f1, CHECKPOINT_DIR)

    logger.info('Done. Best val_micro_f1=%.4f. Checkpoint at %s', best_val_f1, CHECKPOINT_DIR)

    logger.info('Tuning per-label decision thresholds on the validation set...')
    best_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(DEVICE)
    thresholds = tune_per_label_thresholds(best_model, val_loader, label_classes)
    thresholds_path = os.path.join(CHECKPOINT_DIR, 'thresholds.npy')
    np.save(thresholds_path, thresholds)
    logger.info('Per-label thresholds saved to %s', thresholds_path)

    _, tuned_micro_f1, tuned_macro_f1 = evaluate_with_thresholds(best_model, val_loader, thresholds)
    logger.info('Validation with tuned thresholds: micro_f1=%.4f macro_f1=%.4f', tuned_micro_f1, tuned_macro_f1)


if __name__ == '__main__':
    main()
