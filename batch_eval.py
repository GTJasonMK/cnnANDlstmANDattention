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
import traceback
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# 并行执行
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from concurrent.futures.process import BrokenProcessPool  # type: ignore
# Use safer start method with CUDA to avoid fork-related crashes
try:
    mp.set_start_method('spawn')  # safe no-op if already set elsewhere
except RuntimeError:
    # start method was already set in this interpreter
    pass


# 复用单模型评估逻辑
from eval.standalone_eval import evaluate, parse_indices, _extract_arch_meta

# 强制严格模式：要求 checkpoint 中 cfg 完整准确
os.environ.setdefault("EVAL_STRICT", "1")
# 宽松自动对齐/推断开关：默认开启以减少评估失败（可被外部环境变量覆盖）
os.environ.setdefault("EVAL_AUTO_ALIGN_INPUT", "1")
os.environ.setdefault("EVAL_AUTO_ALIGN_TARGETS", "1")
os.environ.setdefault("EVAL_AUTO_INFER_HZ", "1")

# 进度条（可选）
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# Some environments (SLURM/queue) rename running files; detect and report this case clearly
def _looks_like_incomplete_checkpoint(name: str) -> bool:
    # Heuristic: names containing phrases suggesting in-progress
    lowers = name.lower()
    return any(s in lowers for s in [
        'pending', 'running', 'tmp', 'partial', '.inprogress', '.lock'
    ])


def _dbg_enabled() -> bool:
    return os.environ.get("EVAL_DEBUG", "0") != "0"


def _dlog(msg: str):
    if _dbg_enabled():
        print(f"[DEBUG] {msg}")


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


def _save_csv(rows: List[Dict[str, Any]], out_csv: str, columns: Optional[List[str]] = None, minimal: bool = False):
    _ensure_dir(os.path.dirname(out_csv))
    # 优先用 pandas 保存；若不可用则写入简易CSV
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        # 构造缺失但有用的派生列（如 arch）
        if 'arch' not in df.columns:
            def _mk_arch(r):
                c = str(r.get('cnn_variant', '') or '')
                rnn = str(r.get('rnn_type', '') or '')
                a = str(r.get('attn_variant', '') or '')
                parts = [p for p in [c, rnn, a] if p]
                return '+'.join(parts) if parts else None
            try:
                df['arch'] = df.apply(_mk_arch, axis=1)
            except Exception:
                pass
        # 列选择策略
        if columns and isinstance(columns, list):
            keep = [c for c in columns if c in df.columns]
            # 必保列
            for c in ['model','status','checkpoint']:
                if c not in keep and c in df.columns:
                    keep.insert(0, c)
            df = df[keep]
        else:
            if minimal:
                prefer_cols = ['model','status','error','mse','mae','rmse','r2','params','checkpoint']
            else:
                prefer_cols = [
                    'model', 'status', 'error',
                    'dataset', 'seq_len', 'horizon', 'normalize',
                    'cnn_variant', 'rnn_type', 'attn_variant', 'pos', 'ca', 'direction', 'heads',
                    'wav_on', 'wavelet_on', 'wavelet_base', 'wavelet_level', 'wavelet_mode', 'wavelet_take', 'arch',
                    'params', 'checkpoint',
                    'test_loss', 'test_mse', 'test_mae', 'test_rmse', 'test_r2',
                    'val_loss', 'val_mse', 'val_mae', 'val_rmse', 'val_r2',
                    'mse', 'mae', 'rmse', 'mape', 'r2'
                ]
            present = [c for c in prefer_cols if c in df.columns]
            others = [c for c in df.columns if c not in present]
            df = df[present + others]
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



class _EvalRunner:
    """Picklable evaluator wrapper with CUDA OOM-aware batch backoff.
    Defined at module scope to be picklable by ProcessPoolExecutor (spawn).
    """
    def __init__(self, args):
        self.args = args
    def __call__(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Optional GPU affinity
        gpu_id = task.get('gpu', None)
        device = task.get('device', 'cuda')
        if device == 'cuda':
            if gpu_id is not None and str(gpu_id) != '':
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # OOM-aware backoff loop
        bs = int(task.get('batch_size', 128))
        min_bs = max(1, int(getattr(self.args, 'min_batch_size', 8)))
        factor = max(1.1, float(getattr(self.args, 'oom_backoff', 2.0)))
        while True:
            try:
                t2 = dict(task); t2['batch_size'] = bs
                return _eval_worker(t2)
            except RuntimeError as e:
                msg = str(e)
                if ('CUDA out of memory' in msg or 'CUBLAS_STATUS_ALLOC_FAILED' in msg or 'out of memory' in msg.lower()) and device == 'cuda' and bs > min_bs:
                    new_bs = max(min_bs, int(bs // factor))
                    if new_bs < bs:
                        print(f"[WARN] CUDA OOM at batch_size={bs}. Retrying with batch_size={new_bs}")
                        bs = new_bs
                        continue
                # Not an OOM we can mitigate; fall back to a single attempt with original task
                return _eval_worker(task)

# -----------------------------

# 执行一批任务（批次内并行），带 per-GPU 限流与 OOM 回退
def _run_task_batch(tasks_batch: List[Dict[str, Any]], args, gpu_list: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    runner = _EvalRunner(args)
    if args.num_workers <= 1:
        iterator = tasks_batch
        if _HAS_TQDM:
            iterator = tqdm(tasks_batch, total=len(tasks_batch), desc='Evaluating', ncols=80)
        for t in iterator:
            rows.append(runner(t))
        return rows

    max_workers = int(args.num_workers)
    # 可选：按每卡最大并发限流
    if args.device == 'cuda' and gpu_list and getattr(args, 'max_per_gpu', 0) and int(args.max_per_gpu) > 0:
        from collections import defaultdict, deque
        buckets = defaultdict(deque)
        for t in tasks_batch:
            buckets[str(t.get('gpu') or 'none')].append(t)
        ex = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn'))
        try:
            active_per_gpu = {g: 0 for g in buckets.keys()}
            futures_map = {}
            total = len(tasks_batch)
            submitted = 0
            def can_submit_gpu(g):
                return active_per_gpu[g] < int(args.max_per_gpu)
            def submit_one(g):
                nonlocal submitted
                if buckets[g] and can_submit_gpu(g):
                    t = buckets[g].popleft()
                    fut = ex.submit(runner, t)
                    futures_map[fut] = (g, t)
                    active_per_gpu[g] += 1
                    submitted += 1
            # 初始填充
            for g in list(buckets.keys()):
                for _ in range(min(int(args.max_per_gpu), len(buckets[g]))):
                    submit_one(g)
            if _HAS_TQDM:
                from tqdm import tqdm as _tq
                pbar = _tq(total=total, desc='Evaluating', ncols=80)
            else:
                pbar = None
            while futures_map:
                done = list(as_completed(list(futures_map.keys()), timeout=None))
                for fut in done:
                    g, t = futures_map.pop(fut)
                    try:
                        rows.append(fut.result())
                    except Exception as e:
                        msg = str(e)
                        if isinstance(e, BrokenProcessPool) or 'terminated abruptly' in msg:
                            msg = 'worker crashed (BrokenProcessPool): ' + msg
                        rows.append({'model': os.path.splitext(os.path.basename(t['ckpt']))[0], 'checkpoint': t['ckpt'], 'status': f'error: {msg}'})
                    finally:
                        active_per_gpu[g] -= 1
                        if pbar: pbar.update(1)
                        submit_one(g)
                # 若还有任务未提交，继续尝试补充
                for gg in list(buckets.keys()):
                    while buckets[gg] and can_submit_gpu(gg):
                        submit_one(gg)
            if pbar: pbar.close()
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
        return rows
    # 无限流：直接提交
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as ex:
        futures = {ex.submit(runner, t): t for t in tasks_batch}
        def _append_error(t, e: Exception):
            msg = str(e)
            if isinstance(e, BrokenProcessPool) or 'terminated abruptly' in msg:
                msg = 'worker crashed (BrokenProcessPool): ' + msg
            rows.append({'model': os.path.splitext(os.path.basename(t['ckpt']))[0], 'checkpoint': t['ckpt'], 'status': f'error: {msg}'})
        if _HAS_TQDM:
            for fut in tqdm(as_completed(futures), total=len(tasks_batch), desc='Evaluating', ncols=80):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    _append_error(futures[fut], e)
        else:
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    _append_error(futures[fut], e)
    return rows

# Worker for parallel evaluation
# -----------------------------

def _eval_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single checkpoint in a separate process.
    Returns a row dict consistent with serial mode.
    """
    ckpt = task['ckpt']
    data_path = task['data']
    out_dir_i = task['out_dir']
    device = task['device']
    batch_size = int(task['batch_size'])
    seq_len_override = task['sequence_length']
    horizon_override = task['horizon']
    normalize_override = task['normalize']
    feat_idx = task['feature_indices']
    targ_idx = task['target_indices']
    per_model_plots = bool(task['per_model_plots'])
    debug = bool(task['debug'])
    retries = int(task.get('retries', 1))
    gpu_id = task.get('gpu', None)

    # Set device/GPU environment for this process
    if device == 'cuda':
        if gpu_id is not None and str(gpu_id) != '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if debug:
        os.environ['EVAL_DEBUG'] = '1'

    import re as _re

    import torch

    def build_row(status: str, metrics: Optional[Dict[str, Any]] = None, err: Optional[Exception] = None) -> Dict[str, Any]:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        row: Dict[str, Any] = {
            'model': name,
            'checkpoint': ckpt,
            'status': status,
            'params': None,
            'cnn_variant': None,
            'rnn_type': None,
            'attn_variant': None,
            'dataset': None,
            'seq_len': None,
            'horizon': None,
            'normalize': None,
            'wav_on': None,
            'wavelet_on': None,
            'wavelet_base': None,
            'wavelet_level': None,
            'wavelet_mode': None,
            'wavelet_take': None,
            'pos': None,
            'ca': None,
        }
        # param count (best effort)
        try:
            c = torch.load(ckpt, map_location='cpu')
            state = c.get('model_state', {}) if isinstance(c, dict) else {}
            total = 0
            for v in (state or {}).values():
                try:
                    total += int(v.numel())
                except Exception:
                    pass
            row['params'] = int(total) if total else None
        except Exception:
            pass
        # parse arch/meta from cfg (robust to different key styles)
        try:
            c = torch.load(ckpt, map_location='cpu')
            cfg_i = c.get('cfg', {}) if isinstance(c, dict) else {}
            data_i = (cfg_i.get('data') or {}) if isinstance(cfg_i, dict) else {}
            model_i = (cfg_i.get('model') or {}) if isinstance(cfg_i, dict) else {}
            def _first_nonempty(*vals):
                for v in vals:
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s != '' and s.lower() != 'none':
                        return v
                return None
            # cnn variant
            row['cnn_variant'] = _first_nonempty(
                (model_i.get('cnn', {}) or {}).get('variant'),
                model_i.get('cnn_variant'),
                (model_i.get('cnn', {}) or {}).get('type'),
            )
            # rnn type
            row['rnn_type'] = _first_nonempty(
                (model_i.get('lstm', {}) or {}).get('rnn_type'),
                (model_i.get('rnn', {}) or {}).get('type'),
                model_i.get('rnn_type'),
                (model_i.get('lstm', {}) or {}).get('type'),
            )
            # attention variant
            row['attn_variant'] = _first_nonempty(
                (model_i.get('attention', {}) or {}).get('variant'),
                (model_i.get('attn', {}) or {}).get('variant'),
                (model_i.get('attention', {}) or {}).get('type'),
                model_i.get('attention_variant'),
            )
            # dataset/seq_len/horizon/normalize (fallback to cfg if not parsed from name)
            if not row.get('dataset'):
                dp = data_i.get('data_path') or data_i.get('path')
                if isinstance(dp, str) and dp:
                    try:
                        base = os.path.splitext(os.path.basename(dp))[0]
                        row['dataset'] = base
                    except Exception:
                        pass
            if row.get('seq_len') is None:
                try:
                    row['seq_len'] = int(data_i.get('sequence_length')) if data_i.get('sequence_length') is not None else row.get('seq_len')
                except Exception:
                    pass
            if row.get('horizon') is None:
                try:
                    row['horizon'] = int((model_i.get('forecast_horizon') if isinstance(model_i, dict) else None) or data_i.get('horizon'))
                except Exception:
                    pass
            if row.get('normalize') is None:
                norm = data_i.get('normalize')
                if isinstance(norm, str) and norm:
                    row['normalize'] = norm
            # Wavelet backfill from cfg
            try:
                wave = (data_i.get('wavelet') or {}) if isinstance(data_i, dict) else {}
                if isinstance(wave, dict) and wave is not None:
                    w_on = bool(wave.get('enabled', False))
                    if row.get('wavelet_on') is None:
                        row['wavelet_on'] = w_on
                    if row.get('wav_on') is None:
                        row['wav_on'] = w_on
                    if row.get('wavelet_base') is None and wave.get('wavelet') is not None:
                        row['wavelet_base'] = wave.get('wavelet')
                    if row.get('wavelet_level') is None and wave.get('level') is not None:
                        row['wavelet_level'] = wave.get('level')
                    if row.get('wavelet_mode') is None and wave.get('mode') is not None:
                        row['wavelet_mode'] = wave.get('mode')
                    if row.get('wavelet_take') is None and wave.get('take') is not None:
                        row['wavelet_take'] = str(wave.get('take')).lower()
            except Exception:
                pass
            # Backfill core arch fields from cfg via helper (if missing)
            try:
                meta = _extract_arch_meta(cfg_i)
                for k in ['cnn_variant','rnn_type','attn_variant']:
                    if not row.get(k) and meta.get(k) is not None:
                        row[k] = meta.get(k)
            except Exception:
                pass
            # Backfill positional/CA/heads/direction from cfg (if missing)
            try:
                attn_cfg = (model_i.get('attention') or {}) if isinstance(model_i, dict) else {}
                lstm_cfg = (model_i.get('lstm') or {}) if isinstance(model_i, dict) else {}
                cnn_cfg = (model_i.get('cnn') or {}) if isinstance(model_i, dict) else {}
                if row.get('pos') is None:
                    pos_mode = attn_cfg.get('positional_mode', None)
                    if isinstance(pos_mode, str):
                        row['pos'] = pos_mode
                if row.get('heads') is None:
                    try:
                        row['heads'] = int(attn_cfg.get('num_heads')) if attn_cfg.get('num_heads') is not None else row.get('heads')
                    except Exception:
                        pass
                if row.get('direction') is None:
                    try:
                        bi = bool(lstm_cfg.get('bidirectional', True))
                        row['direction'] = 'bi' if bi else 'uni'
                    except Exception:
                        pass
                if row.get('ca') is None:
                    try:
                        ca_on = bool(cnn_cfg.get('use_channel_attention', False))
                        ca_type = str(cnn_cfg.get('channel_attention_type', 'eca')).lower() if ca_on else 'off'
                        row['ca'] = ca_type if ca_on else 'off'
                    except Exception:
                        pass
                # ReVIN/Decomposition flags from cfg (if missing)
                try:
                    norm_cfg = (model_i.get('normalization') or {}) if isinstance(model_i, dict) else {}
                    revin_cfg = (norm_cfg.get('revin') or {}) if isinstance(norm_cfg, dict) else {}
                    if row.get('rev_on') is None:
                        row['rev_on'] = bool(revin_cfg.get('enabled', False))
                except Exception:
                    pass
                try:
                    decomp_cfg = (model_i.get('decomposition') or {}) if isinstance(model_i, dict) else {}
                    if row.get('dec_on') is None:
                        row['dec_on'] = bool(decomp_cfg.get('enabled', False))
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass
        # parse filename: data_tag + slug
        try:
            base = os.path.basename(ckpt).lower()
            m = _re.search(r"__(?P<data>[a-z0-9_\-]+-l\d+-h\d+-(std|mm|none)-wav(on|off)(:[a-z0-9_\-]+)?-rev(on|off)-dec(on|off))__", base)
            if m:
                data_tag = m.group('data')
                parts = data_tag.split('-')
                dataset = parts[0]
                seq = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith('l') else None
                hor = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith('h') else None
                norm = parts[3]
                wav_on_tag = parts[4]
                wav_on2 = 'on' in wav_on_tag
                wav_base2 = 'none'
                if wav_on2 and ':' in wav_on_tag:
                    wav_base2 = wav_on_tag.split(':',1)[1]
                rev_flag = parts[5][3:] if len(parts) > 5 and parts[5].startswith('rev') else 'off'
                dec_flag = parts[6][3:] if len(parts) > 6 and parts[6].startswith('dec') else 'off'
                row.update({
                    'dataset': dataset,
                    'seq_len': seq,
                    'horizon': hor,
                    'normalize': norm,
                })
                # Do NOT override cfg-derived wavelet flags; only backfill if missing
                if row.get('wav_on') is None:
                    row['wav_on'] = wav_on2
                if row.get('wavelet_on') is None:
                    row['wavelet_on'] = wav_on2
                if row.get('wavelet_base') is None and wav_on2:
                    row['wavelet_base'] = wav_base2
                row['rev_on'] = (rev_flag == 'on')
                row['dec_on'] = (dec_flag == 'on')
            s = _re.search(r"__([a-z0-9_\-]+)__([a-z0-9]+)-([a-z0-9]+)-([a-z0-9]+)-pos(none|absolute|alibi|rope)-ca(off|eca|se)-(bi|uni)-[Hh](\d+)_", base)
            if s:
                # Groups: 2=cnn, 3=rnn, 4=attn, 5=pos, 6=ca, 7=bi/uni, 8=heads
                cnn2 = s.group(2)
                rnn2 = s.group(3)
                attn2 = s.group(4)
                pos_mode = s.group(5)
                ca_mode = s.group(6)
                # backfill if cfg didn’t provide
                if not row.get('cnn_variant'):
                    row['cnn_variant'] = cnn2
                if not row.get('rnn_type'):
                    row['rnn_type'] = rnn2
                if not row.get('attn_variant'):
                    row['attn_variant'] = attn2
                row.update({'pos': pos_mode, 'ca': ca_mode, 'direction': s.group(7), 'heads': int(s.group(8))})
        except Exception:
            pass
        if isinstance(metrics, dict):
            row.update(metrics)
        if err is not None:
            row['error'] = str(err)
        return row

    last_exc: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            # load ckpt to infer defaults when overrides are None
            import torch
            c = torch.load(ckpt, map_location='cpu')
            cfg_i = c.get('cfg', {}) if isinstance(c, dict) else {}
            data_i = (cfg_i.get('data') or {}) if isinstance(cfg_i, dict) else {}
            model_i = (cfg_i.get('model') or {}) if isinstance(cfg_i, dict) else {}
            feat_idx_i = feat_idx if feat_idx is not None else data_i.get('feature_indices')
            targ_idx_i = targ_idx if targ_idx is not None else data_i.get('target_indices')
            seq_len_i = seq_len_override if seq_len_override is not None else data_i.get('sequence_length')
            horizon_i = horizon_override if horizon_override is not None else (model_i.get('forecast_horizon', data_i.get('horizon')))
            normalize_i = normalize_override if normalize_override is not None else data_i.get('normalize')
            metrics = evaluate(
                checkpoint=ckpt,
                data_path=data_path,
                output_dir=out_dir_i,
                device=device,
                batch_size=batch_size,
                sequence_length=seq_len_i,
                horizon=horizon_i,
                normalize=normalize_i,
                feature_indices=feat_idx_i,
                target_indices=targ_idx_i,
                save_csv=False,
                plot_sequences=per_model_plots,
                feature_names=None,
            )
            # success
            r = build_row('ok', metrics=metrics, err=None)
            # 统一补齐评估时实际使用的元信息，避免不同来源缺省
            try:
                r['seq_len'] = int(seq_len_i) if seq_len_i is not None else r.get('seq_len')
            except Exception:
                pass
            try:
                r['horizon'] = int(horizon_i) if horizon_i is not None else r.get('horizon')
            except Exception:
                pass
            if normalize_i is not None and not r.get('normalize'):
                r['normalize'] = str(normalize_i)
            # 最终一致化：wav_on 与 wavelet_on 对齐
            try:
                if r.get('wavelet_on') is not None:
                    r['wav_on'] = bool(r['wavelet_on'])
            except Exception:
                pass
            # 数据集名始终取评估时的数据文件名（去扩展名），避免被其它解析覆盖
            try:
                r['dataset'] = os.path.splitext(os.path.basename(data_path))[0]
            except Exception:
                pass
            return r
        except Exception as e:
            last_exc = e
    # failed after retries
    return build_row(f"error: {last_exc}", metrics=None, err=last_exc)


def main():
    p = argparse.ArgumentParser(description='Batch evaluator for multiple checkpoints on the same dataset')
    p.add_argument('--checkpoints_dir', required=True, type=str, help='Directory containing .pt checkpoints')
    p.add_argument('--data', required=True, type=str, help='Path to test data (CSV/NPZ/NPY)')
    p.add_argument('--output_dir', required=True, type=str, help='Directory to save batch results')

    # 与 standalone_eval 对齐的可选参数
    p.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--min_batch_size', type=int, default=8, help='Minimum batch size to try when backing off on OOM')
    p.add_argument('--oom_backoff', type=float, default=2.0, help='Factor to reduce batch_size on CUDA OOM (>=1.5 recommended)')
    p.add_argument('--sequence_length', type=int, default=None)
    p.add_argument('--horizon', type=int, default=None)
    p.add_argument('--normalize', type=str, default=None, choices=['standard', 'minmax', 'none'])
    p.add_argument('--feature_indices', type=str, default=None, help='Comma-separated feature indices')
    p.add_argument('--target_indices', type=str, default=None, help='Comma-separated target indices')

    # 并行与批量控制
    p.add_argument('--num-workers', type=int, default=max(1, mp.cpu_count()), help='Number of parallel worker processes (1=serial)')
    p.add_argument('--gpus', type=str, default='', help='Comma-separated GPU IDs to distribute across workers (empty=CPU)')
    p.add_argument('--max-per-gpu', type=int, default=4, help='Max concurrent workers per single GPU (only used when device=cuda)')
    p.add_argument('--retries', type=int, default=1, help='Retry count for a failed evaluation')
    p.add_argument('--per_model_plots', action='store_true', help='是否为每个模型生成其单独的图像输出')
    p.add_argument('--save_summary_plot', action='store_true', help='是否生成汇总对比图（MSE）')
    # 输出列控制（可替代环境变量 EVAL_COLUMNS/EVAL_MINIMAL）
    p.add_argument('--columns', type=str, default=None, help='导出列（英文逗号分隔）。留空使用默认列集')
    p.add_argument('--minimal', action='store_true', help='仅导出简化列（model/status/error/mse/mae/rmse/r2/params/checkpoint）')
    p.add_argument('--debug', action='store_true', help='开启详细调试日志（等价于设置环境变量 EVAL_DEBUG=1）')

    args = p.parse_args()

    # 打开调试日志
    if args.debug:
        os.environ["EVAL_DEBUG"] = "1"
        _dlog("Debug mode enabled by --debug")

    # 解析 indices
    feat_idx = parse_indices(args.feature_indices)
    targ_idx = parse_indices(args.target_indices)

    ckpts = _scan_checkpoints(args.checkpoints_dir)
    if not ckpts:
        print(f"[ERROR] 在目录中未找到 .pt 文件: {args.checkpoints_dir}")
        return

    _ensure_dir(args.output_dir)

    # GPU/并发安全：若 device=cuda
    # - 未显式提供 --gpus：自动探测可见 GPU 列表
    auto_gpu_ids: List[str] = []
    if args.device == 'cuda':
        try:
            import torch
            n = torch.cuda.device_count()
            auto_gpu_ids = [g.strip() for g in (args.gpus.split(',') if args.gpus else [str(i) for i in range(n)]) if g.strip() != '']
        except Exception:
            auto_gpu_ids = []

    # GPU 列表（可为空）：允许并发数超过 GPU 数量；同一 GPU 上可跑多个任务
    gpu_list = []
    if args.device == 'cuda':
        gpu_list = [g.strip() for g in (args.gpus.split(',') if args.gpus else auto_gpu_ids) if g.strip() != '']

    tasks: List[Dict[str, Any]] = []
    for i, ckpt in enumerate(ckpts):
        # 循环分配 GPU（可为空表示走 CPU 或由 CUDA 自行选择默认设备）
        gpu = (gpu_list[i % len(gpu_list)]) if gpu_list else None
        tasks.append({
            'ckpt': ckpt,
            'data': args.data,
            'out_dir': os.path.join(args.output_dir, os.path.splitext(os.path.basename(ckpt))[0]),
            'device': args.device,
            'batch_size': args.batch_size,
            'sequence_length': args.sequence_length,
            'horizon': args.horizon,
            'normalize': args.normalize,
            'feature_indices': feat_idx,
            'target_indices': targ_idx,
            'per_model_plots': False,
            'debug': args.debug,
            'retries': max(1, int(args.retries)),
            'gpu': gpu,
        })

    rows: List[Dict[str, Any]] = []
    # 分批评估：一批跑完再跑下一批（批内并行）
    batch_size_eval = max(1, int(os.environ.get('BATCH_EVAL_GROUP', '0') or 0))
    if batch_size_eval <= 0:
        if args.device == 'cuda' and gpu_list and int(args.max_per_gpu) > 0:
            batch_size_eval = max(int(args.num_workers), int(args.max_per_gpu) * max(1, len(gpu_list)))
        else:
            batch_size_eval = int(args.num_workers)
    batches: List[List[Dict[str, Any]]] = [tasks[i:i+batch_size_eval] for i in range(0, len(tasks), batch_size_eval)]
    for bi, tasks_batch in enumerate(batches, 1):
        if _HAS_TQDM:
            print(f"[INFO] Evaluating batch {bi}/{len(batches)} (size={len(tasks_batch)})")
        rows.extend(_run_task_batch(tasks_batch, args, gpu_list))


    # 成功样本按 MSE 升序排序（兼容 test_loss==mse）
    rows_ok = [r for r in rows if isinstance(r.get('mse'), (int, float))]
    rows_err = [r for r in rows if r not in rows_ok]
    rows_ok.sort(key=lambda r: r.get('mse', float('inf')))
    rows_sorted = rows_ok + rows_err

    # 保存 CSV
    out_csv = os.path.join(args.output_dir, 'batch_metrics.csv')
    # 允许通过环境变量控制导出列（可选）。例如：
    #   EVAL_COLUMNS="model,status,mse,mae,rmse,r2,arch,params,checkpoint"
    # CLI 优先级高于环境变量
    cols = None
    if getattr(args, 'columns', None):
        cols = [c.strip() for c in args.columns.split(',') if c.strip()]
    else:
        cols_env = os.environ.get('EVAL_COLUMNS', '').strip()
        cols = [c.strip() for c in cols_env.split(',') if c.strip()] if cols_env else None
    minimal = bool(getattr(args, 'minimal', False)) or (os.environ.get('EVAL_MINIMAL','0') in ('1','true','yes','on'))
    _save_csv(rows_sorted, out_csv, columns=cols, minimal=minimal)
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

