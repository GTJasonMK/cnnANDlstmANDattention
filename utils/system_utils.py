"""
统一的系统工具
整合重复的环境检测和设备管理逻辑
"""
from __future__ import annotations

import os
import platform
import torch
import multiprocessing as mp
from typing import Optional, Dict, Any


def get_device(device_str: Optional[str] = None) -> torch.device:
    """统一的设备获取函数"""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_system_info() -> Dict[str, Any]:
    """统一的系统信息获取"""
    info = {
        'python_version': platform.python_version(),
        'os': f"{platform.system()} {platform.release()}",
        'cpu_count': mp.cpu_count(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info.update({
            'gpu_count': torch.cuda.device_count(),
            'current_gpu': torch.cuda.current_device(),
            'gpu_name': torch.cuda.get_device_name(),
            'gpu_memory_mb': torch.cuda.get_device_properties(0).total_memory // 1024**2
        })
    
    return info


def detect_cloud_environment() -> bool:
    """统一的云环境检测"""
    cloud_indicators = [
        '/autodl-tmp' in os.getcwd(),
        '/kaggle/' in os.getcwd(),
        '/content/' in os.getcwd(),
        os.path.exists('/etc/hostname') and 'gpu' in open('/etc/hostname', 'r', errors='ignore').read().lower()
    ]
    return any(cloud_indicators)


def setup_multiprocessing():
    """统一的多进程设置"""
    try:
        if detect_cloud_environment():
            mp.set_start_method('spawn', force=True)
        else:
            mp.set_start_method('spawn')
    except RuntimeError:
        pass  # 已经设置过了


def get_optimal_workers(max_workers: Optional[int] = None) -> int:
    """获取最佳worker数量"""
    cpu_count = mp.cpu_count() or 1
    
    if max_workers is not None:
        return min(max_workers, cpu_count)
    
    # 云环境通常限制更严格
    if detect_cloud_environment():
        return max(1, cpu_count // 2)
    
    return max(1, cpu_count - 1)


def set_seeds(seed: int = 42):
    """统一设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_environment(seed: Optional[int] = None, 
                     cuda_deterministic: bool = True):
    """统一的环境设置"""
    # 设置随机种子
    if seed is not None:
        set_seeds(seed)
    
    # CUDA设置
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 多进程设置
    setup_multiprocessing()
    
    # 环境变量
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if detect_cloud_environment():
        os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')


def print_system_info():
    """打印系统信息"""
    info = get_system_info()
    is_cloud = detect_cloud_environment()
    
    print("=" * 60)
    print("System Environment Information:")
    print(f"  Python Version: {info['python_version']}")
    print(f"  Operating System: {info['os']}")
    print(f"  PyTorch Version: {info['pytorch_version']}")
    print(f"  CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"  GPU Count: {info['gpu_count']}")
        print(f"  Current GPU: {info['current_gpu']}")
        print(f"  GPU Name: {info['gpu_name']}")
        print(f"  GPU Memory: {info['gpu_memory_mb']} MB")
    
    print(f"  CPU Cores: {info['cpu_count']}")
    print(f"  Working Directory: {os.getcwd()}")
    print(f"  Cloud Environment: {'Yes' if is_cloud else 'No'}")
    print("=" * 60)