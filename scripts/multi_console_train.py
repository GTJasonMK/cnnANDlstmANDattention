from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml  # type: ignore


def patch_yaml_for_output(src_yaml: Path, dst_yaml: Path, output_root: Path, data_path: Optional[str] = None, epochs: Optional[int] = None, preserve_export_best: bool = True) -> str:
    """为 YAML 打补丁，设置输出目录、数据路径等，并返回 export_best_dir 绝对路径（字符串）。
    注意：默认保留 YAML 中的 export_best_dir（不覆盖），以便将所有最优模型统一导出到你在生成 YAML 时设置的同一目录。
    """
    cfg = yaml.safe_load(src_yaml.read_text(encoding='utf-8'))

    # 设置输出目录为 output_root/yaml_stem
    stem = src_yaml.stem
    output_dir = output_root / stem

    # 确保各种输出目录都指向这个子目录（仅训练过程数据）
    cfg.setdefault('train', {})
    cfg['train'].setdefault('checkpoints', {})
    cfg['train']['checkpoints']['dir'] = str((output_dir / 'checkpoints_o').as_posix())
    cfg['train']['checkpoints']['save_best_only'] = True

    # export_best_dir：默认不覆盖，保持生成 YAML 时设置的“统一目录”
    if not preserve_export_best:
        cfg['train']['checkpoints']['export_best_dir'] = str((output_dir / 'export_best').as_posix())

    # 设置 log_dir
    cfg['train']['log_dir'] = str((output_dir / 'runs').as_posix())

    # 可选：覆盖数据路径
    if data_path:
        cfg.setdefault('data', {})
        cfg['data']['data_path'] = data_path

    # 可选：覆盖训练轮数
    if epochs is not None:
        cfg['train']['epochs'] = int(epochs)
        # 如果使用 cosine scheduler，同步更新 T_max
        if cfg['train'].get('scheduler', {}).get('name') == 'cosine':
            cfg['train']['scheduler']['T_max'] = int(epochs)

    # 写入目标文件
    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    dst_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')

    # 解析并返回 export_best_dir（若未设置，返回 output_root/export_best/yaml_stem）
    exp_dir = cfg['train']['checkpoints'].get('export_best_dir')
    if not exp_dir:
        exp_dir = str((output_root / 'export_best' / stem).as_posix())
    return exp_dir


def _launch_tmux(yaml_path: Path, output_dir: Path, gpu_id: Optional[str], session_name: str, window_name: str, keep_open: bool = True) -> bool:
    """使用 tmux 启动一个窗口运行训练，返回是否成功。
    keep_open=True 时训练结束后停在窗口等待回车；False 时训练结束后自动关闭窗口（便于分批等待）。
    同时将 stdout/stderr 通过 tee 记录到 output_dir/training.log，并写入 exit_code.txt 便于上层统计成功/失败。
    """
    if shutil.which('tmux') is None:
        return False
    # 创建会话（若不存在）
    try:
        subprocess.run(['tmux', 'has-session', '-t', session_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        session_exists = True
    except subprocess.CalledProcessError:
        session_exists = False
    if not session_exists:
        subprocess.run(['tmux', 'new-session', '-d', '-s', session_name, 'bash', '-lc', 'echo tmux session started; sleep 1'], check=True)
    # 组装命令
    cmd_parts = [sys.executable, 'main.py', '--config', str(yaml_path), '--output_dir', str(output_dir)]
    core = ' '.join([f'"{c}"' if " " in c else c for c in cmd_parts])
    if gpu_id is not None:
        core = f'CUDA_VISIBLE_DEVICES={gpu_id} ' + core
    # 统一：创建输出目录、日志与退出码文件
    outp = output_dir.as_posix()
    run_and_log = (
        f'mkdir -p "{outp}" && '
        f'({core}) 2>&1 | tee -a "{outp}/training.log"; '
        f'ec=${{PIPESTATUS[0]}}; echo $ec > "{outp}/exit_code.txt"; '
        f'exit $ec'
    )
    if keep_open:
        cmd_str = f'cd "{Path.cwd()}" && {run_and_log}; echo "[Done] {yaml_path.stem}. Press Enter to close..."; read'
    else:
        # 自动关闭窗口：训练结束后直接退出 shell
        cmd_str = f'cd "{Path.cwd()}" && {run_and_log}'
    subprocess.run(['tmux', 'new-window', '-t', session_name, '-n', window_name, 'bash', '-lc', cmd_str], check=True)
    print(f"[INFO] Launched tmux window '{window_name}' in session '{session_name}' for {yaml_path.stem}")
    return True


def _make_window_name(index: int, stem: str) -> str:
    # 保证唯一性：前缀索引 + 截断 stem
    safe = stem.replace(' ', '_')
    return f"{index:03d}_{safe[:40]}"


def launch_console_train(yaml_path: Path, output_dir: Path, gpu_id: Optional[str] = None, wait_seconds: float = 1.0, tmux_session: str = 'train', index: int = 0, mode: str = 'tmux') -> Optional[int]:
    """启动训练。
    mode:
      - 'tmux': 优先使用 tmux 窗口（无图形环境推荐）
      - 'background': 直接后台进程（记录 PID 和日志）
    返回：后台模式下返回 PID；tmux/终端模式返回 None
    """
    # 构建命令（传递 --output_dir 到 main）
    cmd_parts = [sys.executable, 'main.py', '--config', str(yaml_path), '--output_dir', str(output_dir)]

    # 设置环境变量
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 直接后台模式：总能并发，不依赖终端
    if mode == 'background':
        log_file = output_dir / 'training.log'
        f = open(log_file, 'w')
        proc = subprocess.Popen(cmd_parts, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(Path.cwd()))
        # 挂载日志文件句柄，便于上层在等待后关闭
        setattr(proc, '_augment_log_fp', f)
        print(f"[INFO] Launched background PID={proc.pid} for {yaml_path.stem}" + (f" (GPU {gpu_id})" if gpu_id else ""))
        time.sleep(wait_seconds)
        return proc

    # tmux 优先（Linux/macOS）
    if os.name != 'nt':
        win_name = _make_window_name(index, yaml_path.stem)
        if _launch_tmux(yaml_path, output_dir, gpu_id, tmux_session, win_name):
            time.sleep(wait_seconds)
            return None

    # Windows 控制台
    if os.name == 'nt':  # Windows
        console_cmd = ['cmd', '/c', 'start', 'cmd', '/k'] + cmd_parts
        title = f"Training {yaml_path.stem}"
        if gpu_id is not None:
            title += f" (GPU {gpu_id})"
        console_cmd.insert(-1, f'title {title} &&')
    else:  # Linux/macOS 图形终端
        terminal_commands = [
            ['gnome-terminal', '--', 'bash', '-c'],
            ['xterm', '-e', 'bash', '-c'],
            ['konsole', '-e', 'bash', '-c'],
            ['mate-terminal', '-e', 'bash', '-c'],
        ]
        cmd_str = ' '.join([f'"{part}"' if ' ' in part else part for part in cmd_parts])
        bash_cmd = f'cd "{Path.cwd()}" && {cmd_str}; echo "Training finished. Press Enter to close..."; read'
        console_cmd = None
        for term_cmd in terminal_commands:
            try:
                subprocess.run(['which', term_cmd[0]], check=True, capture_output=True)
                console_cmd = term_cmd + [bash_cmd]
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        if console_cmd is None:
            # 回退背景
            log_file = output_dir / 'training.log'
            f = open(log_file, 'w')
            proc = subprocess.Popen(cmd_parts, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(Path.cwd()))
            print(f"[INFO] Launched background PID={proc.pid} for {yaml_path.stem}" + (f" (GPU {gpu_id})" if gpu_id else ""))
            time.sleep(wait_seconds)
            return proc.pid

    # 启动新控制台
    try:
        subprocess.Popen(console_cmd, env=env, cwd=str(Path.cwd()))
        print(f"[INFO] Launched console for {yaml_path.stem}" + (f" (GPU {gpu_id})" if gpu_id else ""))
        time.sleep(wait_seconds)
        return None
    except Exception as e:
        print(f"[ERROR] Failed to launch console for {yaml_path.stem}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description='Launch training for each YAML in separate consoles')
    ap.add_argument('--yaml-dir', required=True, help='Directory containing YAML config files')
    ap.add_argument('--output-root', required=True, help='Root directory for outputs (each YAML gets a subdir)')
    ap.add_argument('--data-path', default=None, help='Override data.data_path for all YAMLs')
    ap.add_argument('--epochs', type=int, default=None, help='Override train.epochs for all YAMLs')
    ap.add_argument('--gpus', default='', help='Comma-separated GPU IDs to cycle through (empty for CPU)')
    ap.add_argument('--wait-between', type=float, default=2.0, help='Seconds to wait between launching consoles')
    ap.add_argument('--pattern', default='*.yaml', help='File pattern to match YAML files')
    ap.add_argument('--dry-run', action='store_true', help='Only show what would be done, do not actually launch')
    ap.add_argument('--tmux-session', default='train', help='tmux session name when launching consoles')
    ap.add_argument('--mode', choices=['tmux','background','auto'], default='tmux', help='How to launch each training: tmux (default), background (always), or auto (tmux else background)')
    ap.add_argument('--batch-size', type=int, default=None, help='Run YAMLs in batches of this size (sequential batches). Only applies to background/auto modes when tmux unavailable.')
    ap.add_argument('--wait-exit-seconds', type=float, default=10.0, help='Seconds to wait after a process exits before starting next batch (cooldown)')
    # 新增稳健性与调度参数
    ap.add_argument('--retry', type=int, default=1, help='Retry times for a failed training (background/auto modes)')
    ap.add_argument('--gpu-max-proc', type=int, default=1, help='Max concurrent trainings per single GPU (background/auto modes)')
    ap.add_argument('--cooldown-on-fail', type=float, default=15.0, help='Cooldown seconds after a failed job before retrying')
    ap.add_argument('--report', type=str, default='train_coverage_report.txt', help='Write a coverage report after all jobs finish')
    ap.add_argument('--debug', action='store_true', help='Print more verbose errors and keep temporary files')


    args = ap.parse_args()

    yaml_dir = Path(args.yaml_dir)
    output_root = Path(args.output_root)

    if not yaml_dir.exists():
        print(f"[ERROR] YAML directory not found: {yaml_dir}")
        sys.exit(1)

    # 查找所有 YAML 文件
    yaml_files = sorted(yaml_dir.glob(args.pattern))
    if not yaml_files:
        print(f"[ERROR] No YAML files found in {yaml_dir} matching pattern {args.pattern}")
        sys.exit(1)

    # 解析 GPU 列表
    gpus = [g.strip() for g in args.gpus.split(',') if g.strip() != '']
    if not gpus:
        gpus = [None]  # CPU mode

    print(f"[INFO] Found {len(yaml_files)} YAML files")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] GPUs: {gpus if gpus != [None] else 'CPU'}")
    print(f"[INFO] Wait between launches: {args.wait_between}s")

    if args.dry_run:
        print("\n[DRY RUN] Would launch the following:")
        for i, yaml_file in enumerate(yaml_files):
            gpu = gpus[i % len(gpus)]
            output_dir = output_root / yaml_file.stem
            print(f"  {yaml_file.name} -> {output_dir}" + (f" (GPU {gpu})" if gpu else " (CPU)"))
        return
    # 覆盖率与导出监控
    coverage = {
        'all': [y.stem for y in yaml_files],
        'launched': [],
        'succeeded': [],
        'failed': [],
        'exported': [],
        'missing_export': [],
    }

    def _schedule_background(task_files: List[Path]):
        # Per-GPU concurrency scheduler with retry and export checks
        from collections import deque
        pending = deque(task_files)
        active_per_gpu = {g: 0 for g in gpus}
        running: List[dict] = []
        total = len(task_files)
        print(f"[INFO] Scheduling {total} trainings in background with per-GPU cap={args.gpu_max_proc}")
        while pending or running:
            # fill capacity
            made = 0
            for g in gpus:
                while pending and active_per_gpu[g] < int(args.gpu_max_proc):
                    yaml_file = pending.popleft()
                    output_dir = output_root / yaml_file.stem
                    patched_yaml = temp_dir / f"{yaml_file.stem}_patched.yaml"
                    exp_dir = patch_yaml_for_output(yaml_file, patched_yaml, output_root, args.data_path, args.epochs)
                    try:
                        proc = launch_console_train(
                            patched_yaml,
                            output_dir,
                            g,
                            args.wait_between,
                            tmux_session=args.tmux_session,
                            index=0,
                            mode='background',
                        )
                    except Exception as e:
                        print(f"[ERROR] Launch failed for {yaml_file.stem} on GPU {g}: {e}")
                        proc = None
                    if isinstance(proc, subprocess.Popen):
                        running.append({'proc': proc, 'stem': yaml_file.stem, 'exp_dir': exp_dir, 'yaml': yaml_file, 'gpu': g, 'retries': int(args.retry)})
                        active_per_gpu[g] += 1
                        coverage['launched'].append(yaml_file.stem)
                        print(f"[INFO] Launch stem={yaml_file.stem} PID={proc.pid} GPU={g}")
                        made += 1
                    else:
                        # failed to launch -> count as failed
                        coverage['failed'].append(yaml_file.stem)
            if made == 0 and not running and pending:
                # cannot launch any (cap=0?) safeguard
                time.sleep(max(0.5, args.wait_between))
            # poll running
            i = 0
            while i < len(running):
                item = running[i]
                p: subprocess.Popen = item['proc']
                try:
                    rc = p.poll()
                except Exception as e:
                    print(f"[ERROR] Poll failed for {item['stem']}: {e}")
                    rc = -999
                if rc is None:
                    i += 1
                    continue
                # finished
                g = item['gpu']; stem=item['stem']; exp_dir=item['exp_dir']
                # close log
                lf = getattr(p, '_augment_log_fp', None)
                if lf:
                    try: lf.close()
                    except Exception: pass
                active_per_gpu[g] = max(0, active_per_gpu[g]-1)
                ok = (rc == 0)
                if ok:
                    coverage['succeeded'].append(stem)
                    # check export
                    any_pt = False
                    # 批次结束后，汇总 tmux 批次状态（读取 exit_code.txt）
                    succ=0; fail=0
                    for i, yaml_file in enumerate(batch):
                        outd = output_root / yaml_file.stem
                        ec_file = outd / 'exit_code.txt'
                        if ec_file.exists():
                            try:
                                ec = int(ec_file.read_text().strip())
                            except Exception:
                                ec = -999
                            if ec == 0:
                                succ += 1
                                coverage['succeeded'].append(yaml_file.stem)
                            else:
                                fail += 1
                                coverage['failed'].append(yaml_file.stem)
                        else:
                            # 可能是窗口被强制关闭/异常退出
                            coverage['failed'].append(yaml_file.stem)
                            fail += 1
                    print(f"[INFO] tmux batch {bi}: succeeded={succ}, failed={fail}")

                    try:
                        if os.path.isdir(exp_dir):
                            any_pt = any(fn.endswith('.pt') for fn in os.listdir(exp_dir))
                    except Exception:
                        any_pt = False
                    if any_pt:
                        coverage['exported'].append(stem)
                    else:
                        coverage['missing_export'].append(stem)
                else:
                    # retry if allowed
                    item['retries'] -= 1
                    if item['retries'] >= 0:
                        print(f"[WARN] Job {stem} rc={rc}. Retrying in {args.cooldown_on_fail}s (left={item['retries']})")
                        time.sleep(max(0.0, args.cooldown_on_fail))
                        pending.append(item['yaml'])
                    else:
                        coverage['failed'].append(stem)
                # remove from running
                running.pop(i)
            # small sleep to avoid busy loop
            time.sleep(0.5)


    # 为每个 YAML 文件启动训练
    temp_dir = Path('temp_configs')
    temp_dir.mkdir(exist_ok=True)

    try:
        launch_mode = ('auto' if args.mode == 'auto' else ('background' if args.mode == 'background' else 'tmux'))
        use_tmux = (shutil.which('tmux') is not None and launch_mode in ('tmux','auto'))

        total = len(yaml_files)
        if args.batch_size and args.batch_size > 0:
            # 分批：tmux 和 background 都支持
            batches = [yaml_files[i:i+args.batch_size] for i in range(0, total, args.batch_size)]
            print(f"[INFO] Launching in batches: {len(batches)} batches, batch_size={args.batch_size}")
            for bi, batch in enumerate(batches, 1):
                print(f"\n[INFO] Starting batch {bi}/{len(batches)} (size={len(batch)})")
                if use_tmux:
                    # tmux 批次：开窗口不 keep_open，等待所有窗口任务完成后再开下一批
                    batch_window_names: List[str] = []
                    for i, yaml_file in enumerate(batch):
                        gpu = gpus[i % len(gpus)]
                        output_dir = output_root / yaml_file.stem
                        patched_yaml = temp_dir / f"{yaml_file.stem}_patched.yaml"
                        exp_dir = patch_yaml_for_output(yaml_file, patched_yaml, output_root, args.data_path, args.epochs)
                        win_name = f"b{bi:03d}_" + _make_window_name(i, yaml_file.stem)
                        batch_window_names.append(win_name)
                        _launch_tmux(patched_yaml, output_dir, gpu, args.tmux_session, win_name, keep_open=False)
                        coverage['launched'].append(yaml_file.stem)
                        time.sleep(args.wait_between)
                    # 轮询检测本批窗口是否全部关闭（进程已结束）
                    while True:
                        try:
                            out = subprocess.check_output(['tmux', 'list-windows', '-F', '#{window_name}', '-t', args.tmux_session], stderr=subprocess.STDOUT, text=True)
                            active_names = set(l.strip() for l in out.strip().splitlines() if l.strip())
                        except subprocess.CalledProcessError:
                            active_names = set()
                        remaining = [n for n in batch_window_names if n in active_names]
                        if not remaining:
                            break
                        time.sleep(2)
                    print(f"[INFO] Batch {bi} finished. Cooling down {args.wait_exit_seconds}s...")
                    time.sleep(max(0.0, args.wait_exit_seconds))
                else:
                    # background 批次：改为 per-GPU 限流调度、带失败重试与导出检查
                    _schedule_background(batch)
                    print(f"[INFO] Batch {bi} finished. Cooling down {args.wait_exit_seconds}s...")
                    time.sleep(max(0.0, args.wait_exit_seconds))
                    # 批次结束：写覆盖率报告（仅 background 路径能收集到 succeeded/failed）
                    try:
                        report_path = Path(args.report)
                        with report_path.open('w', encoding='utf-8') as rf:
                            rf.write(f"Total YAMLs: {len(coverage['all'])}\n")
                            rf.write(f"Launched: {len(coverage['launched'])}\n")
                            rf.write(f"Succeeded: {len(coverage['succeeded'])}\n")
                            rf.write(f"Failed: {len(coverage['failed'])}\n")
                            rf.write(f"Exported (best): {len(coverage['exported'])}\n")
                            rf.write(f"Missing export: {len(coverage['missing_export'])}\n\n")
                            def _dump(name, items):
                                rf.write(f"[{name}] {len(items)}\n")
                                for it in sorted(set(items)):
                                    rf.write(f"  - {it}\n")
                                rf.write("\n")
                            _dump('ALL', coverage['all'])
                            _dump('LAUNCHED', coverage['launched'])
                            _dump('SUCCEEDED', coverage['succeeded'])
                            _dump('FAILED', coverage['failed'])
                            _dump('EXPORTED', coverage['exported'])
                            _dump('MISSING_EXPORT', coverage['missing_export'])
                        print(f"[INFO] Wrote coverage report to {report_path}")
                    except Exception as e:
                        print(f"[WARN] Failed to write coverage report: {e}")
            print(f"\n[INFO] All {total} YAML trainings finished in batched mode.")
        else:
            # 非分批：原逻辑（仅启动，不等待）
            for i, yaml_file in enumerate(yaml_files):
                gpu = gpus[i % len(gpus)]
                output_dir = output_root / yaml_file.stem
                patched_yaml = temp_dir / f"{yaml_file.stem}_patched.yaml"
                exp_dir = patch_yaml_for_output(yaml_file, patched_yaml, output_root, args.data_path, args.epochs)
                launched = launch_console_train(
                    patched_yaml,
                    output_dir,
                    gpu,
                    args.wait_between,
                    tmux_session=args.tmux_session,
                    index=i,
                    mode=launch_mode,
                )
                coverage['launched'].append(yaml_file.stem)
                if isinstance(launched, subprocess.Popen):
                    print(f"[INFO] PID[{launched.pid}] -> {yaml_file.name}")

            print(f"\n[INFO] Launched {len(yaml_files)} training sessions")
            print(f"[INFO] Check individual console windows for training progress")
            print(f"[INFO] Outputs will be saved to subdirectories under: {output_root}")

            if use_tmux:
                try:
                    out = subprocess.check_output(['tmux', 'list-windows', '-t', args.tmux_session], stderr=subprocess.STDOUT, text=True)
                    wins = [l for l in out.strip().splitlines() if l.strip()]
                    print(f"[INFO] tmux session '{args.tmux_session}' currently has {len(wins)} windows")
                except subprocess.CalledProcessError:
                    pass

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")


if __name__ == '__main__':
    main()
