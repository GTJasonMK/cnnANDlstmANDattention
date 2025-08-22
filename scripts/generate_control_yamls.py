from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List
import yaml  # type: ignore

TCN_DEFAULT: Dict[str, Any] = {
    'enabled': True,
    'dropout': 0.05,
    'use_batchnorm': True,
    'layers': [
        {'out_channels': 64, 'kernel_size': 3, 'dilation': 1, 'activation': 'gated', 'use_weightnorm': True},
        {'out_channels': 128, 'kernel_size': 3, 'dilation': 2, 'activation': 'gated', 'use_weightnorm': True},
        {'out_channels': 128, 'kernel_size': 3, 'dilation': 4, 'activation': 'gated', 'use_weightnorm': True},
    ],
}

TEMPLATE: Dict[str, Any] = {
    'device': None,
    'model': {
        'fc_hidden': 128,
        'forecast_horizon': 3,
        'cnn': {
            'variant': 'standard',
            'dropout': 0.1,
            'use_batchnorm': True,
            'layers': [
                {'out_channels': 32, 'kernel_size': 5, 'activation': 'relu', 'pool': 'max', 'pool_kernel_size': 2},
                {'out_channels': 64, 'kernel_size': 3, 'activation': 'gelu', 'pool': 'max', 'pool_kernel_size': 2},
            ],
        },
        'lstm': {
            'rnn_type': 'lstm',
            'hidden_size': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.1,
        },
        'attention': {
            'enabled': True,
            'variant': 'standard',
            'num_heads': 4,
            'dropout': 0.1,
            'add_positional_encoding': False,
            'positional_mode': 'none',
        },
    },
    'data': {
        'data_path': 'CHANGE_ME',
        'sequence_length': 64,
        'horizon': 3,
        'feature_indices': None,
        'target_indices': None,
        'train_split': 0.7,
        'val_split': 0.15,
        'normalize': 'standard',
        'batch_size': 64,
        'num_workers': 0,
        'shuffle_train': True,
        'drop_last': False,
    },
    'train': {
        'epochs': 12,
        'loss': 'mse',
        'optimizer': {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0001},
        'scheduler': {'name': 'cosine', 'T_max': 12},
        'early_stopping': {'enabled': True, 'patience': 3, 'min_delta': 0.0},
        'checkpoints': {'dir': 'checkpoints_o', 'save_best_only': True, 'export_best_dir': ''},
        'gradient_clip': 1.0,
        'mixed_precision': False,
        'log_dir': 'runs',
        'seed': 42,
        'print_every': 50,
    },
}


def write_yaml(base: Dict[str, Any], out_path: Path):
    out_path.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding='utf-8')


def build_base(baseline: Dict[str, Any], data_path: str, export_root: Path, epochs: int | None) -> Dict[str, Any]:
    cfg = yaml.safe_load(yaml.dump(TEMPLATE))
    # baseline arch
    cfg['model']['cnn']['variant'] = baseline['cnn']
    # TCN special handling
    if baseline['cnn'] == 'tcn':
        cfg['model']['tcn'] = yaml.safe_load(yaml.dump(TCN_DEFAULT))
    else:
        # ensure TCN disabled if present
        if 'tcn' in cfg['model']:
            cfg['model']['tcn']['enabled'] = False
    cfg['model']['lstm']['rnn_type'] = baseline['rnn']
    cfg['model']['attention']['variant'] = baseline['attn']
    cfg['model']['attention']['positional_mode'] = baseline['pos']
    cfg['model']['attention']['add_positional_encoding'] = False
    if baseline['ca']:
        cfg['model']['cnn']['use_channel_attention'] = True
        cfg['model']['cnn']['channel_attention_type'] = 'eca'
    # data
    cfg['data']['data_path'] = data_path
    # train
    if epochs is not None:
        cfg['train']['epochs'] = int(epochs)
        if cfg['train'].get('scheduler', {}).get('name') == 'cosine':
            cfg['train']['scheduler']['T_max'] = int(epochs)
    # export dir to be filled by caller per-file
    return cfg


def main():
    ap = argparse.ArgumentParser(description='Generate control-variable YAMLs for all factors')
    ap.add_argument('--out-dir', default='configs/yaml/ctrl', help='Output directory for YAMLs')
    ap.add_argument('--data-path', required=True, help='Dataset path (CSV/NPZ/NPY) to set in each YAML')
    ap.add_argument('--export-best-root', required=True, help='Root directory for train.checkpoints.export_best_dir; each YAML uses a unique subdir by stem')
    ap.add_argument('--epochs', type=int, default=12, help='Train epochs to write into each YAML (also set scheduler.T_max if cosine)')
    # baseline
    ap.add_argument('--baseline-cnn', default='standard', choices=['standard','depthwise','dilated','inception','tcn'])
    ap.add_argument('--baseline-attn', default='standard', choices=['standard','multiscale','local','conformer'])
    ap.add_argument('--baseline-rnn', default='lstm', choices=['lstm','gru','ssm'])
    ap.add_argument('--baseline-ca', action='store_true', help='Use channel attention (ECA) in baseline')
    # positional sweep control
    ap.add_argument('--include-positional', action='store_true', help='Include positional sweep (alibi/rope) for factor=pos')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    export_root = Path(args.export_best_root); export_root.mkdir(parents=True, exist_ok=True)

    baseline = {
        'cnn': args.baseline_cnn,
        'attn': args.baseline_attn,
        'rnn': args.baseline_rnn,
        'pos': 'none',
        'ca': bool(args.baseline_ca),
    }
    rnn_types = ['lstm', 'gru', 'ssm']
    attn_variants = ['standard', 'multiscale', 'local', 'conformer']
    cnn_variants = ['standard', 'depthwise', 'dilated', 'inception', 'tcn']
    pos_modes = ['none', 'alibi', 'rope'] if args.include_positional else ['none']

    def save_cfg(cfg: Dict[str, Any], name: str):
        stem = Path(name).stem
        # unique export dir per yaml
        cfg['train'].setdefault('checkpoints', {})
        cfg['train']['checkpoints']['save_best_only'] = True
        cfg['train']['checkpoints']['export_best_dir'] = str(export_root)
        write_yaml(cfg, out_dir / name)

    # 1) baseline
    cfg_base = build_base(baseline, args.data_path, export_root, args.epochs)
    save_cfg(cfg_base, f"ctrl_baseline__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-ca{'on' if baseline['ca'] else 'off'}.yaml")

    # 2) RNN sweep
    for r in rnn_types:
        if r == baseline['rnn']:
            continue
        cfg = build_base({**baseline, 'rnn': r}, args.data_path, export_root, args.epochs)
        save_cfg(cfg, f"ctrl_rnn-{r}__{baseline['cnn']}-{baseline['attn']}-{r}-pos{baseline['pos']}-ca{'on' if baseline['ca'] else 'off'}.yaml")

    # 3) Attention sweep
    for a in attn_variants:
        if a == baseline['attn']:
            continue
        cfg = build_base({**baseline, 'attn': a}, args.data_path, export_root, args.epochs)
        save_cfg(cfg, f"ctrl_attn-{a}__{baseline['cnn']}-{a}-{baseline['rnn']}-pos{baseline['pos']}-ca{'on' if baseline['ca'] else 'off'}.yaml")

    # 4) CNN sweep
    for c in cnn_variants:
        if c == baseline['cnn']:
            continue
        cfg = build_base({**baseline, 'cnn': c}, args.data_path, export_root, args.epochs)
        save_cfg(cfg, f"ctrl_cnn-{c}__{c}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-ca{'on' if baseline['ca'] else 'off'}.yaml")

    # 5) Positional sweep (only if include_positional)
    if args.include_positional:
        for p in pos_modes:
            if p == baseline['pos']:
                continue
            cfg = build_base({**baseline, 'pos': p}, args.data_path, export_root, args.epochs)
            save_cfg(cfg, f"ctrl_pos-{p}__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{p}-ca{'on' if baseline['ca'] else 'off'}.yaml")

    # 6) Channel attention sweep (flip on/off)
    flipped_ca = not baseline['ca']
    cfg = build_base({**baseline, 'ca': flipped_ca}, args.data_path, export_root, args.epochs)
    save_cfg(cfg, f"ctrl_ca-{'on' if flipped_ca else 'off'}__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-ca{'on' if flipped_ca else 'off'}.yaml")

    print(f"Generated control-variable YAMLs into {out_dir}")


if __name__ == '__main__':
    main()

