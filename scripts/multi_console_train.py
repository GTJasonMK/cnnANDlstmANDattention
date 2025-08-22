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


def patch_yaml_for_output(src_yaml: Path, dst_yaml: Path, output_root: Path, data_path: Optional[str] = None, epochs: Optional[int] = None, preserve_export_best: bool = True):
    """为 YAML 打补丁，设置输出目录、数据路径等。
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


def _launch_tmux(yaml_path: Path, output_dir: Path, gpu_id: Optional[str], session_name: str, window_name: str) -> bool:
    """使用 tmux 启动一个窗口运行训练，返回是否成功"""
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
    if gpu_id is not None:
        cmd_str = f'CUDA_VISIBLE_DEVICES={gpu_id} ' + ' '.join([f'"{c}"' if " " in c else c for c in cmd_parts])
    else:
        cmd_str = ' '.join([f'"{c}"' if " " in c else c for c in cmd_parts])
    cmd_str = f'cd "{Path.cwd()}" && {cmd_str}; echo "[Done] {yaml_path.stem}. Press Enter to close..."; read'
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
        print(f"[INFO] Launched background PID={proc.pid} for {yaml_path.stem}" + (f" (GPU {gpu_id})" if gpu_id else ""))
        time.sleep(wait_seconds)
        return proc.pid

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
    
    # 为每个 YAML 文件启动训练
    temp_dir = Path('temp_configs')
    temp_dir.mkdir(exist_ok=True)
    
    try:
        for i, yaml_file in enumerate(yaml_files):
            gpu = gpus[i % len(gpus)]
            output_dir = output_root / yaml_file.stem

            # 创建打补丁的 YAML
            patched_yaml = temp_dir / f"{yaml_file.stem}_patched.yaml"
            patch_yaml_for_output(yaml_file, patched_yaml, output_root, args.data_path, args.epochs)

            # 启动新控制台训练
            launched_pid = launch_console_train(
                patched_yaml,
                output_dir,
                gpu,
                args.wait_between,
                tmux_session=args.tmux_session,
                index=i,
                mode=('auto' if args.mode == 'auto' else ('background' if args.mode == 'background' else 'tmux')),
            )
            if launched_pid is not None:
                print(f"[INFO] PID[{launched_pid}] -> {yaml_file.name}")

        print(f"\n[INFO] Launched {len(yaml_files)} training sessions")
        print(f"[INFO] Check individual console windows for training progress")
        print(f"[INFO] Outputs will be saved to subdirectories under: {output_root}")

        # 如使用 tmux，打印当前窗口统计
        if shutil.which('tmux') is not None and args.mode in ('tmux','auto'):
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
