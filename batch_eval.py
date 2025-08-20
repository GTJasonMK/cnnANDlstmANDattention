from __future__ import annotations
"""
批量评估脚本：在同一测试数据集上评估目录中的多个模型检查点，并汇总指标为CSV。

特性：
- 扫描 --checkpoints_dir 下所有 .pt 文件
- 复用 standalone_eval.evaluate() 完成单模型评估，避免重复代码
- 汇总每个模型的 MSE、MAE、RMSE、MAPE、R2、参数数量、状态 等
- 结果按 MSE 升序排序保存为 CSV，并在控制台输出简要对比
- 失败样本优雅跳过并记录错误信息
- 可选：生成对比柱状图（MSE）

用法示例：
  python batch_eval.py \
    --checkpoints_dir checkpoints/ \
    --data data/test.csv \
    --output_dir results/batch_eval \
    --device cuda --batch_size 256 --sequence_length 64 --horizon 3 --normalize standard
"""

import argparse
import os
import json
from typing import Optional, List, Dict, Any

import numpy as np

# 复用单模型评估逻辑
from standalone_eval import evaluate, parse_indices


def _ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _scan_checkpoints(dir_path: str) -> List[str]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Checkpoints dir not found: {dir_path}")
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.pt')]
    files.sort()
    return files


def _param_count_from_ckpt(ckpt_path: str) -> Optional[int]:
    """无需构建模型，直接统计 state_dict 中张量元素总数作为参数量。"""
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state', None)
        if not isinstance(state, dict):
            return None
        total = 0
        for v in state.values():
            try:
                # torch.Tensor
                total += int(v.numel())
            except Exception:
                try:
                    # 退化：numpy 数组
                    total += int(np.prod(getattr(v, 'shape', [])))
                except Exception:
                    pass
        return int(total)
    except Exception:
        return None


def _save_csv(rows: List[Dict[str, Any]], out_csv: str):
    _ensure_dir(os.path.dirname(out_csv))
    # 优先用 pandas 保存；若不可用则写入简易CSV
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        # 指标列按常见顺序重排
        cols = [
            'model', 'status', 'mse', 'mae', 'rmse', 'mape', 'r2', 'params', 'checkpoint'
        ]
        present = [c for c in cols if c in df.columns]
        df = df[present + [c for c in df.columns if c not in present]]
        df.to_csv(out_csv, index=False)
    except Exception:
        # 纯文本CSV
        if not rows:
            with open(out_csv, 'w', encoding='utf-8') as f:
                f.write('')
            return
        keys = list({k for r in rows for k in r.keys()})
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')


def _plot_summary(rows: List[Dict[str, Any]], out_path: str, metric: str = 'mse', top_k: int = 20):
    """简单柱状图对比前K个模型在指定指标上的表现。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('[WARN] matplotlib 不可用，跳过汇总图绘制')
        return
    # 过滤出成功的样本
    rows_ok = [r for r in rows if isinstance(r.get(metric), (int, float))]
    if not rows_ok:
        return
    rows_ok.sort(key=lambda r: (r.get(metric, float('inf'))))
    rows_ok = rows_ok[:top_k]
    names = [r['model'] for r in rows_ok]
    values = [r[metric] for r in rows_ok]
    plt.figure(figsize=(max(6, len(names) * 0.6), 4))
    plt.bar(names, values, color='steelblue', edgecolor='k', alpha=0.85)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric.upper()); plt.title(f'Model comparison ({metric.upper()})')
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser(description='Batch evaluator for multiple checkpoints on the same dataset')
    p.add_argument('--checkpoints_dir', required=True, type=str, help='Directory containing .pt checkpoints')
    p.add_argument('--data', required=True, type=str, help='Path to test data (CSV/NPZ/NPY)')
    p.add_argument('--output_dir', required=True, type=str, help='Directory to save batch results')

    # 与 standalone_eval 对齐的可选参数
    p.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--sequence_length', type=int, default=None)
    p.add_argument('--horizon', type=int, default=None)
    p.add_argument('--normalize', type=str, default=None, choices=['standard', 'minmax', 'none'])
    p.add_argument('--feature_indices', type=str, default=None, help='Comma-separated feature indices')
    p.add_argument('--target_indices', type=str, default=None, help='Comma-separated target indices')

    # 批量控制
    p.add_argument('--per_model_plots', action='store_true', help='是否为每个模型生成其单独的图像输出')
    p.add_argument('--save_summary_plot', action='store_true', help='是否生成汇总对比图（MSE）')

    args = p.parse_args()

    # 解析 indices
    feat_idx = parse_indices(args.feature_indices)
    targ_idx = parse_indices(args.target_indices)

    ckpts = _scan_checkpoints(args.checkpoints_dir)
    if not ckpts:
        print(f"[ERROR] 在目录中未找到 .pt 文件: {args.checkpoints_dir}")
        return

    _ensure_dir(args.output_dir)
    rows: List[Dict[str, Any]] = []

    print(f"[INFO] 共发现 {len(ckpts)} 个检查点，开始评估...\n")
    for i, ckpt in enumerate(ckpts, 1):
        name = os.path.splitext(os.path.basename(ckpt))[0]
        print(f"[{i}/{len(ckpts)}] 评估: {name}")
        out_dir_i = os.path.join(args.output_dir, name)
        try:
            metrics = evaluate(
                checkpoint=ckpt,
                data_path=args.data,
                output_dir=out_dir_i,
                device=args.device,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                horizon=args.horizon,
                normalize=args.normalize,
                feature_indices=feat_idx,
                target_indices=targ_idx,
                save_csv=False,            # 批量模式默认不逐模型保存 CSV，可视化也默认关闭
                plot_sequences=False,
                feature_names=None,
            )
            status = 'ok'
        except Exception as e:
            print(f"[ERROR] 评估失败: {name} -> {e}")
            metrics = None
            status = f"error: {e}"

        params = _param_count_from_ckpt(ckpt)
        row: Dict[str, Any] = {
            'model': name,
            'checkpoint': ckpt,
            'status': status,
            'params': params,
        }
        if isinstance(metrics, dict):
            row.update(metrics)
            # 控制台简报
            brief = {k: round(v, 6) for k, v in metrics.items()}
            print(json.dumps(brief, ensure_ascii=False))
        rows.append(row)

    # 成功样本按 MSE 升序排序
    rows_ok = [r for r in rows if isinstance(r.get('mse'), (int, float))]
    rows_err = [r for r in rows if r not in rows_ok]
    rows_ok.sort(key=lambda r: r.get('mse', float('inf')))
    rows_sorted = rows_ok + rows_err

    # 保存 CSV
    out_csv = os.path.join(args.output_dir, 'batch_metrics.csv')
    _save_csv(rows_sorted, out_csv)
    print(f"\n[INFO] 已保存汇总CSV: {out_csv}")

    # 可选：汇总图（MSE）
    if args.save_summary_plot:
        _plot_summary(rows_ok, os.path.join(args.output_dir, 'metrics_summary_mse.png'), metric='mse')
        print(f"[INFO] 已保存汇总图: {os.path.join(args.output_dir, 'metrics_summary_mse.png')}")

    # 控制台打印前若干条结果
    print("\n[RESULTS] 排名前列（按 MSE 升序）:")
    for r in rows_ok[:min(10, len(rows_ok))]:
        print(f"  - {r['model']}: mse={r.get('mse'):.6f}, mae={r.get('mae'):.6f}, rmse={r.get('rmse'):.6f}, r2={r.get('r2'):.6f}, params={r.get('params')}")


if __name__ == '__main__':
    main()

