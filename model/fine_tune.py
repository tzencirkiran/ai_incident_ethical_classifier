"""Fine-tune prajjwal1/bert-tiny for multi-label ethical-issue classification."""
import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = 'prajjwal1/bert-tiny'
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'processed')
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoint')

DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SEED = 42
# Fixed threshold used only for model-selection during training, so "best epoch"
# doesn't depend on a threshold that gets tuned afterwards.
SELECTION_THRESHOLD = 0.5
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.05)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune bert-tiny for multi-label incident tagging.')
    parser.add_argument('--output-dir', default=DEFAULT_CHECKPOINT_DIR, help='Directory for the best checkpoint')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument('--warmup-ratio', type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--weighted-bce', action='store_true', help='Use per-label positive class weights')
    parser.add_argument('--no-save', action='store_true', help='Train/evaluate without writing a checkpoint')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(name):
    data = torch.load(os.path.join(PROCESSED_DIR, f'{name}.pt'))
    dataset = TensorDataset(data['input_ids'], data['attention_mask'], data['labels'])
    return dataset


def build_loss(dataset, weighted_bce=False):
    if not weighted_bce:
        return torch.nn.BCEWithLogitsLoss()

    labels = dataset.tensors[2]
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    pos_weight = negatives / positives.clamp_min(1.0)
    logger.info('Using weighted BCE with pos_weight=%s', [round(float(x), 3) for x in pos_weight])
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))


def predict_probs(model, loader, criterion):
    """Run the model over a loader and return (probs, labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            total_loss += criterion(logits, labels).item()

            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels), total_loss / len(loader)


def evaluate(model, loader, criterion, threshold=SELECTION_THRESHOLD):
    probs, labels, loss = predict_probs(model, loader, criterion)
    preds = probs >= threshold
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return loss, micro_f1, macro_f1


def evaluate_with_thresholds(model, loader, criterion, thresholds):
    """Evaluate using a per-label threshold vector instead of a single global cutoff."""
    probs, labels, loss = predict_probs(model, loader, criterion)
    preds = probs >= thresholds[np.newaxis, :]
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return loss, micro_f1, macro_f1


def tune_per_label_thresholds(model, loader, criterion, label_classes):
    """Find, for each label independently, the threshold in THRESHOLD_GRID that
    maximizes that label's F1 score on the given (validation) set."""
    probs, labels, _ = predict_probs(model, loader, criterion)

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


def load_tokenizer(output_dir=DEFAULT_CHECKPOINT_DIR):
    if os.path.exists(os.path.join(output_dir, 'tokenizer_config.json')):
        logger.info('Loading tokenizer from local checkpoint %s', output_dir)
        return AutoTokenizer.from_pretrained(output_dir, local_files_only=True)

    logger.info('Loading tokenizer for %s', MODEL_NAME)
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def train_model(args):
    set_seed(args.seed)
    save_checkpoint = not args.no_save
    if save_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)

    label_classes = np.load(os.path.join(PROCESSED_DIR, 'label_classes.npy'), allow_pickle=True)
    num_labels = len(label_classes)
    id2label, label2id = build_label_maps(label_classes)

    train_dataset = load_split('train')
    val_dataset = load_split('val')
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    criterion = build_loss(train_dataset, args.weighted_bce)

    tokenizer = load_tokenizer(args.output_dir)

    logger.info('Loading model %s', MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type='multi_label_classification',
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    logger.info(
        'Training on %s | labels=%d train=%d batch=%d epochs=%d lr=%g wd=%g warmup=%.2f weighted_bce=%s seed=%d',
        DEVICE, num_labels, len(train_loader.dataset), args.batch_size, args.epochs, args.learning_rate,
        args.weight_decay, args.warmup_ratio, args.weighted_bce, args.seed
    )

    best_val_f1 = -1.0
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, micro_f1, macro_f1 = evaluate(model, val_loader, criterion)
        logger.info(
            'Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_micro_f1=%.4f | val_macro_f1=%.4f',
            epoch, args.epochs, train_loss, val_loss, micro_f1, macro_f1
        )

        if micro_f1 > best_val_f1:
            best_val_f1 = micro_f1
            best_val_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            if save_checkpoint:
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                np.save(os.path.join(args.output_dir, 'label_classes.npy'), label_classes)
                logger.info('  -> new best (val_micro_f1=%.4f), checkpoint saved to %s', micro_f1, args.output_dir)
            else:
                logger.info('  -> new best (val_micro_f1=%.4f)', micro_f1)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    logger.info('Done. Best val_micro_f1=%.4f at epoch %d', best_val_f1, best_epoch)

    logger.info('Tuning per-label decision thresholds on the validation set...')
    thresholds = tune_per_label_thresholds(model, val_loader, criterion, label_classes)
    if save_checkpoint:
        thresholds_path = os.path.join(args.output_dir, 'thresholds.npy')
        np.save(thresholds_path, thresholds)
        logger.info('Per-label thresholds saved to %s', thresholds_path)

    _, tuned_micro_f1, tuned_macro_f1 = evaluate_with_thresholds(model, val_loader, criterion, thresholds)
    logger.info('Validation with tuned thresholds: micro_f1=%.4f macro_f1=%.4f', tuned_micro_f1, tuned_macro_f1)

    metrics = {
        'best_epoch': best_epoch,
        'best_val_micro_f1': float(best_val_f1),
        'best_val_macro_f1': float(best_val_macro_f1),
        'tuned_val_micro_f1': float(tuned_micro_f1),
        'tuned_val_macro_f1': float(tuned_macro_f1),
        'thresholds': thresholds.tolist(),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'warmup_ratio': args.warmup_ratio,
            'seed': args.seed,
            'weighted_bce': args.weighted_bce,
        },
    }
    if save_checkpoint:
        metrics_path = os.path.join(args.output_dir, 'train_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info('Training metrics saved to %s', metrics_path)
    return metrics


def main():
    train_model(parse_args())


if __name__ == '__main__':
    main()
