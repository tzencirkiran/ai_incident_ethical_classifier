"""Grid-search TinyBERT fine-tuning hyperparameters on the validation split.

This script intentionally does not evaluate on the test split. Use the best
validation configuration for a final `make train ...` run, then run `make test`
once for held-out reporting.
"""
import argparse
import csv
import itertools
import json
import logging
import os
from types import SimpleNamespace

from fine_tune import DEFAULT_CHECKPOINT_DIR, DEFAULT_SEED, train_model

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'tuning_config.json')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'tuning')
RESULTS_JSON = os.path.join(RESULTS_DIR, 'tinybert_grid_results.json')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'tinybert_grid_results.csv')

DEFAULT_CONFIG = {
    'epochs': [10, 15],
    'batch_sizes': [32, 64],
    'learning_rates': [5e-5, 1e-4, 2e-4, 5e-4],
    'weight_decays': [0.0, 0.01],
    'warmup_ratios': [0.1],
    'weighted_bce': [False, True],
    'seeds': [DEFAULT_SEED],
    'max_runs': None,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_list(value, cast):
    if isinstance(value, list):
        return [cast(item) for item in value]
    return [cast(item.strip()) for item in value.split(',') if item.strip()]


def parse_bool_list(value):
    if isinstance(value, list):
        return [bool(item) for item in value]

    items = []
    for item in value.split(','):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized in {'1', 'true', 'yes', 'weighted'}:
            items.append(True)
        elif normalized in {'0', 'false', 'no', 'plain'}:
            items.append(False)
        else:
            raise argparse.ArgumentTypeError(f'Invalid boolean value: {item}')
    return items


def parse_args():
    parser = argparse.ArgumentParser(description='Run TinyBERT hyperparameter grid search.')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='JSON config file for the sweep')
    parser.add_argument('--epochs', default=None, help='Comma-separated epoch counts')
    parser.add_argument('--batch-sizes', default=None, help='Comma-separated batch sizes')
    parser.add_argument('--learning-rates', default=None)
    parser.add_argument('--weight-decays', default=None)
    parser.add_argument('--warmup-ratios', default=None)
    parser.add_argument('--weighted-bce', default=None, help='Comma-separated booleans')
    parser.add_argument('--seeds', default=None, help='Comma-separated random seeds')
    parser.add_argument('--max-runs', type=int, default=None, help='Limit the number of configs to run')
    parser.add_argument('--dry-run', action='store_true', help='Print planned configs without training')
    parser.add_argument('--save-best', action='store_true', help='Retrain the best config into model/checkpoint')
    return parser.parse_args()


def load_config(path):
    config = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            loaded = json.load(f)
        config.update(loaded)
        logger.info('Loaded tuning config from %s', path)
    elif path:
        logger.warning('Tuning config %s not found; using built-in defaults', path)
    return config


def resolve_config(args):
    config = load_config(args.config)
    overrides = {
        'epochs': args.epochs,
        'batch_sizes': args.batch_sizes,
        'learning_rates': args.learning_rates,
        'weight_decays': args.weight_decays,
        'warmup_ratios': args.warmup_ratios,
        'weighted_bce': args.weighted_bce,
        'seeds': args.seeds,
        'max_runs': args.max_runs,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def build_grid(config):
    grid = itertools.product(
        parse_list(config['epochs'], int),
        parse_list(config['batch_sizes'], int),
        parse_list(config['learning_rates'], float),
        parse_list(config['weight_decays'], float),
        parse_list(config['warmup_ratios'], float),
        parse_bool_list(config['weighted_bce']),
        parse_list(config['seeds'], int),
    )
    configs = []
    for epochs, batch_size, lr, weight_decay, warmup_ratio, weighted_bce, seed in grid:
        configs.append({
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'weighted_bce': weighted_bce,
            'seed': seed,
        })
    if config.get('max_runs') is not None:
        configs = configs[:int(config['max_runs'])]
    return configs


def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        'run',
        'epochs',
        'batch_size',
        'learning_rate',
        'weight_decay',
        'warmup_ratio',
        'weighted_bce',
        'seed',
        'best_epoch',
        'best_val_micro_f1',
        'best_val_macro_f1',
        'tuned_val_micro_f1',
        'tuned_val_macro_f1',
    ]
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in fieldnames})


def flatten_result(run_index, config, metrics):
    return {
        'run': run_index,
        **config,
        'best_epoch': metrics['best_epoch'],
        'best_val_micro_f1': metrics['best_val_micro_f1'],
        'best_val_macro_f1': metrics['best_val_macro_f1'],
        'tuned_val_micro_f1': metrics['tuned_val_micro_f1'],
        'tuned_val_macro_f1': metrics['tuned_val_macro_f1'],
    }


def as_train_args(config, no_save=True):
    return SimpleNamespace(
        output_dir=DEFAULT_CHECKPOINT_DIR,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        seed=config['seed'],
        weighted_bce=config['weighted_bce'],
        no_save=no_save,
    )


def main():
    args = parse_args()
    config = resolve_config(args)
    configs = build_grid(config)
    logger.info('Sweep config: %s', config)
    logger.info('Planned %d TinyBERT runs', len(configs))
    for i, config in enumerate(configs, start=1):
        logger.info('Run %d config: %s', i, config)

    if args.dry_run:
        return

    results = []
    for i, config in enumerate(configs, start=1):
        logger.info('Starting run %d/%d', i, len(configs))
        metrics = train_model(as_train_args(config, no_save=True))
        result = flatten_result(i, config, metrics)
        results.append(result)
        save_results(results)
        logger.info(
            'Run %d done | tuned_val_micro_f1=%.4f tuned_val_macro_f1=%.4f',
            i, result['tuned_val_micro_f1'], result['tuned_val_macro_f1']
        )

    best = max(results, key=lambda row: (row['tuned_val_micro_f1'], row['tuned_val_macro_f1']))
    logger.info('Best validation config: %s', best)
    logger.info('Results saved to %s and %s', RESULTS_JSON, RESULTS_CSV)

    if args.save_best:
        logger.info('Retraining best config into %s', DEFAULT_CHECKPOINT_DIR)
        best_config = {key: best[key] for key in [
            'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'warmup_ratio', 'weighted_bce', 'seed'
        ]}
        train_model(as_train_args(best_config, no_save=False))


if __name__ == '__main__':
    main()
