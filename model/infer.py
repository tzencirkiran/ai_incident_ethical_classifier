"""Run multi-label ethical-issue inference with the fine-tuned bert-tiny checkpoint.

Usage:
    python infer.py "Headline text" --purpose "..." --technology "..."
"""
import argparse
import logging
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = 'prajjwal1/bert-tiny'
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoint')
MAX_LENGTH = 128
DEFAULT_THRESHOLD = 0.5  # fallback if a per-label thresholds.npy isn't found

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


TEXT_FIELDS = [
    ('purpose', 'Purpose'),
    ('technology', 'Technology'),
    ('deployer', 'Deployer'),
    ('developer', 'Developer'),
    ('system_name', 'System'),
    ('news_trigger', 'News trigger'),
    ('jurisdiction', 'Jurisdiction'),
    ('sector', 'Sector'),
]


def build_text(headline, **metadata):
    """Build the model input the same way preprocessing.py does."""
    parts = [headline.strip()]
    for key, label in TEXT_FIELDS:
        value = metadata.get(key)
        if value:
            parts.append(f'{label}: {value.strip()}')
    return '. '.join(parts)


def load_model():
    logger.info('Loading fine-tuned checkpoint from %s', CHECKPOINT_DIR)
    try:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
    except OSError as exc:
        raise RuntimeError(
            f'No tokenizer files found in {CHECKPOINT_DIR}. '
            'Run make train with the updated training script, or save the tokenizer into the checkpoint.'
        ) from exc

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(DEVICE)
    model.eval()
    label_classes = np.load(os.path.join(CHECKPOINT_DIR, 'label_classes.npy'), allow_pickle=True)

    thresholds_path = os.path.join(CHECKPOINT_DIR, 'thresholds.npy')
    if os.path.exists(thresholds_path):
        thresholds = np.load(thresholds_path)
        logger.info('Loaded per-label thresholds from %s', thresholds_path)
    else:
        thresholds = np.full(len(label_classes), DEFAULT_THRESHOLD)
        logger.warning('No thresholds.npy found; falling back to global threshold=%.2f', DEFAULT_THRESHOLD)

    return tokenizer, model, label_classes, thresholds


def predict(text, tokenizer, model, label_classes, thresholds):
    """Return a list of (label, probability) for tags whose probability exceeds
    that label's tuned threshold, sorted by probability descending."""
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt',
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = sorted(zip(label_classes, probs, thresholds), key=lambda x: x[1], reverse=True)
    return [(label, float(prob)) for label, prob, threshold in results if prob >= threshold]


def main():
    parser = argparse.ArgumentParser(description='Predict ethical-issue tags for an AI incident.')
    parser.add_argument('headline', help='Incident headline text')
    parser.add_argument('--purpose', default=None, help='Stated purpose of the AI system')
    parser.add_argument('--technology', default=None, help='Technology category')
    parser.add_argument('--deployer', default=None, help='Organization or person deploying the AI system')
    parser.add_argument('--developer', default=None, help='Organization or person developing the AI system')
    parser.add_argument('--system-name', default=None, help='AI system name')
    parser.add_argument('--news-trigger', default=None, help='News trigger taxonomy value')
    parser.add_argument('--jurisdiction', default=None, help='Impacted jurisdiction')
    parser.add_argument('--sector', default=None, help='Impacted sector')
    args = parser.parse_args()

    tokenizer, model, label_classes, thresholds = load_model()
    text = build_text(
        args.headline,
        purpose=args.purpose,
        technology=args.technology,
        deployer=args.deployer,
        developer=args.developer,
        system_name=args.system_name,
        news_trigger=args.news_trigger,
        jurisdiction=args.jurisdiction,
        sector=args.sector,
    )
    logger.info('Running inference on: %s', text)
    predictions = predict(text, tokenizer, model, label_classes, thresholds)

    print(f'Input: {text}')
    if not predictions:
        print('No tags exceeded the probability threshold.')
    else:
        print('Predicted ethical issues:')
        for label, prob in predictions:
            print(f'  {label}: {prob:.3f}')


if __name__ == '__main__':
    main()
