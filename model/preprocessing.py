"""Build train/val/test splits for fine-tuning bert-tiny on the AI incidents dataset.

Target: multi-label classification over a normalized set of canonical
"Ethical issue (taxonomy)" tags.
Input text: Headline plus safe incident metadata, concatenated into one string.
"""
import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

MODEL_NAME = 'prajjwal1/bert-tiny'
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'incidents_data.xlsx')
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), 'processed')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoint')

NUM_LABELS = 14
MAX_LENGTH = 128
RANDOM_STATE = 42
DEFAULT_SPLIT_NAME = 'random'
DEFAULT_VAL_YEAR = 2024
DEFAULT_TEST_YEAR = 2025

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Maps raw tag spellings/variants to a canonical category. Anything not listed
# here is left as-is and will be dropped later if it falls outside the top-N.
TAG_NORMALIZATION = {
    'transaprency': 'Transparency',
    'transpareny': 'Transparency',
    'accountabiity': 'Accountability',
    'accountabiilty': 'Accountability',
    'accuracy/relibaility': 'Accuracy/reliability',
    'accuracy/reliablity': 'Accuracy/reliability',
    'accuracy/reliabiity': 'Accuracy/reliability',
    'privacy/surveillamce': 'Privacy/surveillance',
    'privacy/surveillance/surveillance': 'Privacy/surveillance',
    'surveillanc': 'Privacy/surveillance',
    'surveillance': 'Privacy/surveillance',
    'privacy': 'Privacy/surveillance',
    'compeititon/monopolisation': 'Competition/monopolisation',
    'appropropriation': 'Appropriation',
    'accessiblity': 'Accessibility',
    'dual/multi-use': 'Dual use',
    'human/civil rights': 'Human rights/civil liberties',
    'employment - jobs': 'Employment/labour',
    'employment - jobs, pay': 'Employment/labour',
    'employment/labour - jobs': 'Employment/labour',
    'employment': 'Employment/labour',
    'job loss/losses loss': 'Employment/labour',
    'oversight/review': 'Oversight',
    'scope creep/normalisation': 'Normalisation',
}

TEXT_FIELDS = [
    ('Purpose', 'Purpose'),
    ('Technology', 'Technology'),
    ('Deployer', 'Deployer'),
    ('Developer', 'Developer'),
    ('System name', 'System'),
    ('News trigger (taxonomy)', 'News trigger'),
    ('Impacted area - Jurisdiction', 'Jurisdiction'),
    ('Impacted area - Sector', 'Sector'),
]


def normalize_tag(tag):
    """Collapse a raw tag string to a canonical category name.

    Compound 'Fairness - <attributes>' variants collapse to plain 'Fairness';
    known typos/synonyms are mapped via TAG_NORMALIZATION; everything else is
    passed through unchanged.
    """
    tag = tag.strip()
    if not tag:
        return None
    if re.match(r'(?i)^fairness\b', tag):
        return 'Fairness'
    key = tag.lower()
    if key in TAG_NORMALIZATION:
        return TAG_NORMALIZATION[key]
    return tag


def load_incidents(path=DATA_PATH):
    """Load the incidents spreadsheet and reconstruct its flat header.

    The sheet has a 3-row hierarchical header: row 1 holds main column names,
    row 2 holds sub-category names that only apply to the 'Impacted area' and
    'External harm' groups.
    """
    raw = pd.read_excel(path, header=None)
    main_row, sub_row = raw.iloc[1].tolist(), raw.iloc[2].tolist()

    columns = []
    current_main = None
    for main, sub in zip(main_row, sub_row):
        main = main.strip() if isinstance(main, str) else None
        sub = sub.strip() if isinstance(sub, str) else None
        if main is not None:
            current_main = main
        columns.append(f'{current_main} - {sub}' if sub is not None else current_main)

    df = raw.iloc[3:].reset_index(drop=True)
    df.columns = columns
    df = df.iloc[:, :-1]  # drop near-empty trailing duplicate 'Summary/links' column
    return df


def build_text(df):
    """Build the model input from headline plus safe incident metadata."""
    def join_row(row):
        parts = []
        if isinstance(row['Headline'], str):
            parts.append(row['Headline'].strip())
        for column, label in TEXT_FIELDS:
            value = row.get(column)
            if isinstance(value, str) and value.strip():
                parts.append(f'{label}: {value.strip()}')
        return '. '.join(parts)

    return df.apply(join_row, axis=1)


def build_labels(df):
    """Parse, normalize and binarize the 'Ethical issue (taxonomy)' column.

    Keeps only the NUM_LABELS most frequent canonical tags; rows whose tags are
    entirely outside this set are dropped (their label vector would be all-zero).
    Returns (kept_tags, mlb) where kept_tags[i] is the list of canonical tags
    retained for row i.
    """
    raw_tags = df['Ethical issue (taxonomy)'].apply(
        lambda s: [normalize_tag(t) for t in s.split(';')] if isinstance(s, str) else []
    )
    normalized = raw_tags.apply(lambda tags: [t for t in tags if t])

    tag_counts = normalized.explode().value_counts()
    top_tags = set(tag_counts.head(NUM_LABELS).index)

    kept = normalized.apply(lambda tags: [t for t in tags if t in top_tags])
    mlb = MultiLabelBinarizer(classes=sorted(top_tags))
    return kept, mlb


def tokenize(texts, tokenizer):
    return tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt',
    )


def load_tokenizer():
    if os.path.exists(os.path.join(CHECKPOINT_DIR, 'tokenizer_config.json')):
        logger.info('Loading tokenizer from local checkpoint %s', CHECKPOINT_DIR)
        return AutoTokenizer.from_pretrained(CHECKPOINT_DIR, local_files_only=True)

    logger.info('Loading tokenizer for %s', MODEL_NAME)
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def parse_args():
    parser = argparse.ArgumentParser(description='Build processed train/val/test split artifacts.')
    parser.add_argument('--split', choices=['random', 'temporal'], default='random')
    parser.add_argument(
        '--split-name',
        default=None,
        help='Output subdirectory under model/processed. Defaults to the split mode name.',
    )
    parser.add_argument('--output-root', default=OUTPUT_ROOT)
    parser.add_argument('--val-year', type=int, default=DEFAULT_VAL_YEAR)
    parser.add_argument('--test-year', type=int, default=DEFAULT_TEST_YEAR)
    return parser.parse_args()


def build_random_split(texts, labels):
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=RANDOM_STATE
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=RANDOM_STATE
    )
    return {
        'train': (train_texts, train_labels),
        'val': (val_texts, val_labels),
        'test': (test_texts, test_labels),
    }


def build_temporal_split(df, labels, val_year, test_year):
    years = pd.to_numeric(df['Occurred'], errors='coerce')
    valid_years = years.notna()
    dropped = int((~valid_years).sum())
    if dropped:
        logger.info('Dropping %d incidents with missing/non-numeric Occurred year for temporal split', dropped)

    df = df[valid_years].reset_index(drop=True)
    labels = labels[valid_years.to_numpy()]
    years = years[valid_years].astype(int).reset_index(drop=True)

    split_masks = {
        'train': years < val_year,
        'val': (years >= val_year) & (years < test_year),
        'test': years >= test_year,
    }
    splits = {}
    for split_name, mask in split_masks.items():
        split_texts = df.loc[mask, 'text'].tolist()
        split_labels = labels[mask.to_numpy()]
        if len(split_texts) == 0:
            raise ValueError(f'Temporal split produced no {split_name} examples')
        splits[split_name] = (split_texts, split_labels)
    return splits


def save_splits(splits, tokenizer, output_dir, label_classes, metadata):
    os.makedirs(output_dir, exist_ok=True)
    for split_name, (split_texts, split_labels) in splits.items():
        encodings = tokenize(split_texts, tokenizer)
        torch.save(
            {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': torch.tensor(split_labels, dtype=torch.float32),
                'texts': split_texts,
            },
            os.path.join(output_dir, f'{split_name}.pt'),
        )
        logger.info('%s: %d examples -> %s.pt', split_name, len(split_texts), split_name)

    np.save(os.path.join(output_dir, 'label_classes.npy'), np.array(label_classes))
    with open(os.path.join(output_dir, 'split_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info('Label classes saved to %s: %s', os.path.join(output_dir, 'label_classes.npy'), list(label_classes))
    logger.info('Split metadata saved to %s', os.path.join(output_dir, 'split_metadata.json'))


def main():
    args = parse_args()
    split_name = args.split_name or args.split
    output_dir = os.path.join(args.output_root, split_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info('Loading incidents from %s', DATA_PATH)
    df = load_incidents()
    logger.info('Loaded %d incidents', len(df))
    df['text'] = build_text(df)
    kept_tags, mlb = build_labels(df)

    mask = kept_tags.apply(len) > 0
    df = df[mask].reset_index(drop=True)
    kept_tags = kept_tags[mask].reset_index(drop=True)

    labels = mlb.fit_transform(kept_tags)
    logger.info('Kept %d incidents with %d canonical labels:', len(df), len(mlb.classes_))
    for cls, count in zip(mlb.classes_, labels.sum(axis=0)):
        logger.info('  %s: %d', cls, count)

    texts = df['text'].tolist()
    if args.split == 'random':
        splits = build_random_split(texts, labels)
        metadata = {
            'split': args.split,
            'split_name': split_name,
            'random_state': RANDOM_STATE,
            'counts': {name: len(split_texts) for name, (split_texts, _) in splits.items()},
        }
    else:
        splits = build_temporal_split(df, labels, args.val_year, args.test_year)
        years = pd.to_numeric(df['Occurred'], errors='coerce')
        metadata = {
            'split': args.split,
            'split_name': split_name,
            'val_year': args.val_year,
            'test_year': args.test_year,
            'year_ranges': {
                'train': f'< {args.val_year}',
                'val': f'{args.val_year} <= year < {args.test_year}',
                'test': f'>= {args.test_year}',
            },
            'dropped_missing_years': int(years.isna().sum()),
            'counts': {name: len(split_texts) for name, (split_texts, _) in splits.items()},
        }

    tokenizer = load_tokenizer()
    logger.info('Writing %s split artifacts to %s', args.split, output_dir)
    save_splits(splits, tokenizer, output_dir, mlb.classes_, metadata)


if __name__ == '__main__':
    main()
