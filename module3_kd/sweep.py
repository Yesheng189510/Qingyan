"""Hyperparameter grid-search / sweep runner.

Usage:
    # Define a sweep JSON file (see example at bottom) and run:
    python sweep.py --sweep my_sweep.json

    # Run only specific fold for quick testing:
    python sweep.py --sweep my_sweep.json --folds "1"

The sweep file defines:
  - base: baseline config (same format as --config for train_kd.py)
  - search: dict of param_name → [value1, value2, ...]
  - n_runs: (optional) repeat each combo N times with different seeds

Each combination runs as a separate experiment with its own output directory.
Results are collected into sweep_results.jsonl for easy comparison.
"""

import sys
import json
import copy
import itertools
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT))


def expand_sweep(search_space: dict) -> list[dict]:
    """Cartesian product of all parameter lists → list of config overrides."""
    if not search_space:
        return [{}]
    keys = list(search_space.keys())
    values = [search_space[k] if isinstance(search_space[k], list) else [search_space[k]]
              for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_experiment(config: dict, folds: str, sweep_dir: Path,
                   run_name: str) -> dict:
    """Run a single experiment via train_kd.py, return parsed results."""
    # Write temp config
    config_path = sweep_dir / f'_tmp_{run_name}.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Run training
    out_dir = sweep_dir / run_name
    cmd = [
        sys.executable, '-u',
        str(_PROJECT / 'module3_kd' / 'train_kd.py'),
        '--config', str(config_path),
        '--folds', folds,
        '--out_dir', str(out_dir),
    ]
    print(f'\n{"="*70}')
    print(f'RUN: {run_name}')
    print(f'CMD: {" ".join(cmd)}')
    print(f'{"="*70}\n')

    result = subprocess.run(cmd, capture_output=False)
    success = result.returncode == 0

    # Collect results from the run's JSONL files
    fold_results = {}
    if success and out_dir.exists():
        for jsonl_file in sorted(out_dir.glob('fold_*.jsonl')):
            fold_name = jsonl_file.stem
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = [json.loads(l) for l in f if l.strip()]
            summary = next((l for l in lines if l.get('type') == 'summary'), None)
            if summary:
                fold_results[fold_name] = {
                    'best_acc': summary['best_acc'],
                    'best_epoch': summary['best_epoch'],
                    'total_min': summary['total_min'],
                }

    # Clean up temp config
    config_path.unlink(missing_ok=True)

    return {
        'run_name': run_name,
        'success': success,
        'fold_results': fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep runner')
    parser.add_argument('--sweep', type=str, required=True,
                        help='Path to sweep JSON file')
    parser.add_argument('--folds', type=str, default='0',
                        help='Folds to run (default: "0" for quick testing)')
    args = parser.parse_args()

    # Load sweep definition
    with open(args.sweep, 'r', encoding='utf-8') as f:
        sweep_def = json.load(f)

    base_cfg = sweep_def.get('base', {})
    search_space = sweep_def.get('search', {})
    n_runs = sweep_def.get('n_runs', 1)
    description = sweep_def.get('description', 'sweep')

    combos = expand_sweep(search_space)
    print(f'Sweep: {description}')
    print(f'Search space: {json.dumps(search_space, indent=2)}')
    print(f'Combinations: {len(combos)} x {n_runs} runs = {len(combos) * n_runs} total')
    print(f'Folds per run: {args.folds}')

    # Output directory
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = _PROJECT / 'module3_output_kd' / f'sweep_{ts}'
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep definition for reproducibility
    with open(sweep_dir / 'sweep_definition.json', 'w', encoding='utf-8') as f:
        json.dump(sweep_def, f, indent=2, ensure_ascii=False)

    results_log = sweep_dir / 'sweep_results.jsonl'
    all_results = []

    for run_i in range(n_runs):
        for combo_i, override in enumerate(combos):
            # Build config
            cfg = copy.deepcopy(base_cfg)
            cfg.update(override)

            # Unique run name
            param_str = '_'.join(f'{k}={v}' for k, v in override.items())
            if n_runs > 1:
                param_str += f'_seed{run_i}'
            if not param_str:
                param_str = 'baseline'
            run_name = f'{combo_i:03d}_{param_str}'

            # Run
            result = run_experiment(cfg, args.folds, sweep_dir, run_name)
            result['override'] = override
            result['run_index'] = run_i
            all_results.append(result)

            # Append to results log
            with open(results_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Summary
    print(f'\n{"="*70}')
    print(f'SWEEP COMPLETE: {len(all_results)} experiments')
    print(f'Results: {results_log}')
    print(f'Outputs: {sweep_dir}')
    print(f'\nTop results:')
    for r in sorted(all_results, key=lambda x: max(
        (v['best_acc'] for v in x['fold_results'].values()), default=0
    ), reverse=True)[:10]:
        best = max((v['best_acc'] for v in r['fold_results'].values()), default=0)
        status = 'OK' if r['success'] else 'FAIL'
        print(f'  {r["run_name"]:<50} best={best:.4f}  [{status}]')


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE SWEEP FILE
# ═══════════════════════════════════════════════════════════════════

EXAMPLE_SWEEP = {
    "description": "LR and temperature sweep for DKD",
    "base": {
        "student_arch": "resnet34",
        "num_classes": 4,
        "epochs": 60,
        "batch_size": 64,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "lr_scheduler": "cosine",
        "kd_method": "dkd",
        "temperature": 3.0,
        "alpha_kd": 0.7,
        "alpha_dkd": 0.5,
        "use_weighted_sampler": True,
        "use_mixup": False,
        "seed": 42
    },
    "search": {
        "lr": [0.0001, 0.0005, 0.001, 0.005],
        "temperature": [1.0, 2.0, 3.0, 5.0]
    },
    "n_runs": 1
}


if __name__ == '__main__':
    main()
