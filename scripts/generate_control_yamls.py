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
        'wavelet': {
            'enabled': False,
            'wavelet': 'db4',
            'level': 3,
            'mode': 'symmetric',
            'take': 'all',
        },
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
    # 若选择了非 none 的位置编码，默认开启 add_positional_encoding
    cfg['model']['attention']['add_positional_encoding'] = bool(baseline.get('pos') not in (None, 'none'))
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
    ap.add_argument('--baseline-attn', default='standard', choices=['standard','multiscale','local','conformer','spatiotemporal'])
    ap.add_argument('--baseline-rnn', default='lstm', choices=['lstm','gru','ssm'])
    ap.add_argument('--baseline-ca', action='store_true', help='Use channel attention (ECA) in baseline')
    # sweep controls
    ap.add_argument('--include-positional', action='store_true', help='Include positional sweep (absolute/alibi/rope) for factor=pos')
    ap.add_argument('--include-wavelet', action='store_true', help='Include wavelet sweeps (on with different bases)')
    ap.add_argument('--wavelet-bases', default='db4,sym5,coif3', help='Comma-separated wavelet bases to test when include-wavelet is set')
    ap.add_argument('--ca-backbone', default='depthwise', choices=['depthwise','dilated','inception'], help='Which CNN variant to use when sweeping Channel Attention (since CA applies only to these CNNs)')
    ap.add_argument('--ca-per-cnn', action='store_true', help='Sweep CA (off/eca/se) for each CA-supported CNN (depthwise/dilated/inception)')
    ap.add_argument('--wavelet-levels', default='3', help='Comma-separated wavelet decomposition levels, e.g. "2,3,4"')
    ap.add_argument('--wavelet-modes', default='symmetric', help='Comma-separated wavelet padding modes, e.g. "symmetric,periodization"')
    ap.add_argument('--wavelet-takes', default='all', help='Comma-separated coefficient selections, e.g. "all,approx,details"')

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
    # 统一 CA 标签，避免出现 caon/caoff 的模糊命名
    ca_tag = 'caeca' if baseline['ca'] else 'caoff'
    rnn_types = ['lstm', 'gru', 'ssm']
    attn_variants = ['standard', 'multiscale', 'local', 'conformer', 'spatiotemporal']
    cnn_variants = ['standard', 'depthwise', 'dilated', 'inception', 'tcn']
    pos_modes = ['none', 'absolute', 'alibi', 'rope'] if args.include_positional else ['none']

    def save_cfg(cfg: Dict[str, Any], name: str, note: str = ""):
        # 统一最优模型导出目录（所有 YAML 的最佳模型导出到同一个目录）
        cfg['train'].setdefault('checkpoints', {})
        cfg['train']['checkpoints']['save_best_only'] = True
        cfg['train']['checkpoints']['export_best_dir'] = str(export_root)
        if note:
            cfg['notes'] = note
        write_yaml(cfg, out_dir / name)

    # 1) baseline
    cfg_base = build_base(baseline, args.data_path, export_root, args.epochs)
    save_cfg(cfg_base, f"ctrl_baseline__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-{ca_tag}.yaml", note="Baseline configuration for all sweeps")

    # 2) RNN sweep
    for r in rnn_types:
        if r == baseline['rnn']:
            continue
        cfg = build_base({**baseline, 'rnn': r}, args.data_path, export_root, args.epochs)
        save_cfg(cfg, f"ctrl_rnn-{r}__{baseline['cnn']}-{baseline['attn']}-{r}-pos{baseline['pos']}-{ca_tag}.yaml", note=f"Control variable: RNN={r}")

    # 3) Attention sweep
    for a in attn_variants:
        if a == baseline['attn']:
            continue
        cfg = build_base({**baseline, 'attn': a}, args.data_path, export_root, args.epochs)
        if a == 'spatiotemporal':
            cfg['model']['attention']['st_mode'] = 'serial'
            cfg['model']['attention']['st_fuse'] = 'sum'
        save_cfg(cfg, f"ctrl_attn-{a}__{baseline['cnn']}-{a}-{baseline['rnn']}-pos{baseline['pos']}-{ca_tag}.yaml", note=f"Control variable: Attention={a}")

    # 4) CNN/TCN backbone sweep
    for c in cnn_variants:
        if c == baseline['cnn']:
            continue
        cfg = build_base({**baseline, 'cnn': c}, args.data_path, export_root, args.epochs)
        save_cfg(cfg, f"ctrl_cnn-{c}__{c}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-{ca_tag}.yaml", note=f"Control variable: CNN/TCN={c}")

    # 5) Positional sweep (only if include_positional)
    if args.include_positional:
        for p in pos_modes:
            if p == baseline['pos']:
                continue
            cfg = build_base({**baseline, 'pos': p}, args.data_path, export_root, args.epochs)
            save_cfg(cfg, f"ctrl_pos-{p}__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{p}-{ca_tag}.yaml", note=f"Control variable: PositionEncoding={p}")

    # 6) Channel attention sweep：off + eca + se
    # 注意：CA 只在高级 CNN（depthwise/dilated/inception）中生效。
    # 若 baseline['cnn'] 不支持 CA，则在 CA 因子实验中固定使用一个可用的 cnn（默认为 depthwise），以保证“仅改变 CA 因子”。
    _ca_supported = ['depthwise', 'dilated', 'inception']
    ca_cnn_default = baseline['cnn'] if baseline['cnn'] in _ca_supported else (args.ca_backbone if args.ca_backbone in _ca_supported else 'depthwise')
    ca_targets: List[str] = _ca_supported if args.ca_per_cnn else [ca_cnn_default]
    for ca_cnn in ca_targets:
        # off
        cfg_off_ca = build_base({**baseline, 'cnn': ca_cnn, 'ca': False}, args.data_path, export_root, args.epochs)
        cfg_off_ca['model']['cnn']['use_channel_attention'] = False
        save_cfg(cfg_off_ca, f"ctrl_ca-off__{ca_cnn}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-caoff.yaml", note=f"Control variable: ChannelAttention=off (cnn={ca_cnn})")
        # on with ECA
        cfg_eca = build_base({**baseline, 'cnn': ca_cnn, 'ca': True}, args.data_path, export_root, args.epochs)
        cfg_eca['model']['cnn']['use_channel_attention'] = True
        cfg_eca['model']['cnn']['channel_attention_type'] = 'eca'
        save_cfg(cfg_eca, f"ctrl_ca-eca__{ca_cnn}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-caeca.yaml", note=f"Control variable: ChannelAttention type=eca (cnn={ca_cnn})")
        # on with SE
        cfg_se = build_base({**baseline, 'cnn': ca_cnn, 'ca': True}, args.data_path, export_root, args.epochs)
        cfg_se['model']['cnn']['use_channel_attention'] = True
        cfg_se['model']['cnn']['channel_attention_type'] = 'se'
        save_cfg(cfg_se, f"ctrl_ca-se__{ca_cnn}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-case.yaml", note=f"Control variable: ChannelAttention type=se (cnn={ca_cnn})")

    # 7) Normalization sweeps（data.normalize）
    for nm in ['standard', 'minmax', 'none']:
        cfg_n = build_base(baseline, args.data_path, export_root, args.epochs)
        cfg_n.setdefault('data', {})['normalize'] = nm
        save_cfg(cfg_n, f"ctrl_norm-{nm}__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-{ca_tag}.yaml", note=f"Control variable: data.normalize={nm}")

    # 8) RevIN sweeps（model.normalization.revin.enabled）
    for flag in [True, False]:
        cfg_r = build_base(baseline, args.data_path, export_root, args.epochs)
        cfg_r.setdefault('model', {}).setdefault('normalization', {}).setdefault('revin', {})
        cfg_r['model']['normalization']['revin']['enabled'] = bool(flag)
        save_cfg(cfg_r, f"ctrl_revin-{'on' if flag else 'off'}__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-{ca_tag}.yaml", note=f"Control variable: RevIN={'on' if flag else 'off'}")

    # 9) Decomposition sweeps（model.decomposition.enabled）
    for flag in [True, False]:
        cfg_d = build_base(baseline, args.data_path, export_root, args.epochs)
        cfg_d.setdefault('model', {}).setdefault('decomposition', {})
        cfg_d['model']['decomposition']['enabled'] = bool(flag)
        save_cfg(cfg_d, f"ctrl_decomp-{'on' if flag else 'off'}__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-{ca_tag}.yaml", note=f"Control variable: Decomposition={'on' if flag else 'off'}")

    # 10) Wavelet sweeps（可选）：
    if args.include_wavelet:
        # 解析 sweep 值集合
        bases = [s.strip() for s in args.wavelet_bases.split(',') if s.strip()]
        levels = [int(s.strip()) for s in str(args.wavelet_levels).split(',') if str(s).strip()]
        modes = [s.strip() for s in str(args.wavelet_modes).split(',') if s.strip()]
        takes = [s.strip() for s in str(args.wavelet_takes).split(',') if s.strip()]

        # on/off 开关（on 选择 bases 的第一个作为默认）
        cfg_on = build_base(baseline, args.data_path, export_root, args.epochs)
        cfg_on['data']['wavelet']['enabled'] = True
        if bases:
            cfg_on['data']['wavelet']['wavelet'] = bases[0]
        save_cfg(cfg_on, f"ctrl_wavelet-on__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-{ca_tag}.yaml", note="Control variable: Wavelet enabled=true")

        cfg_off = build_base(baseline, args.data_path, export_root, args.epochs)
        cfg_off['data']['wavelet']['enabled'] = False
        save_cfg(cfg_off, f"ctrl_wavelet-off__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-{ca_tag}.yaml", note="Control variable: Wavelet enabled=false")

        # 组合 sweep：base × level × mode × take
        for wb in bases:
            for lv in levels:
                for md in modes:
                    for tk in takes:
                        cfg_b = build_base(baseline, args.data_path, export_root, args.epochs)
                        w = cfg_b['data']['wavelet']
                        w['enabled'] = True
                        w['wavelet'] = wb
                        w['level'] = int(lv)
                        w['mode'] = md
                        w['take'] = tk
                        save_cfg(cfg_b, f"ctrl_wavelet-{wb}-L{int(lv)}-{md}-{tk}__{baseline['cnn']}-{baseline['attn']}-{baseline['rnn']}-pos{baseline['pos']}-{ca_tag}.yaml", note=f"Control variable: Wavelet base={wb}, level={lv}, mode={md}, take={tk}")

    print(f"Generated control-variable YAMLs into {out_dir}")


if __name__ == '__main__':
    main()

