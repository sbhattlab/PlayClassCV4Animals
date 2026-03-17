"""
GPU and system memory utilities
"""

import gc

import psutil
import torch
from loguru import logger


def free_gpu_memory(log_stats: bool = False):
    """Free GPU memory between chunks via garbage collection and cache clearing."""
    if log_stats and torch.cuda.is_available():
        before_alloc = torch.cuda.memory_allocated() / 1024**2
        before_res = torch.cuda.memory_reserved() / 1024**2
        logger.info(
            f"GPU memory before cleanup: {before_alloc:.1f} MB allocated, {before_res:.1f} MB reserved"
        )

    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()  # second pass clears anything freed by first-pass destructors
        torch.cuda.ipc_collect()

    if log_stats and torch.cuda.is_available():
        after_alloc = torch.cuda.memory_allocated() / 1024**2
        after_res = torch.cuda.memory_reserved() / 1024**2
        logger.info(
            f"GPU memory after cleanup: {after_alloc:.1f} MB allocated, {after_res:.1f} MB reserved"
        )


def free_system_memory(label: str = ""):
    """Run gc.collect() and log system RAM before/after."""
    prefix = f"{label}: " if label else ""

    vm = psutil.virtual_memory()
    before_gb = vm.used / 1024**3
    available_gb = vm.available / 1024**3
    logger.info(
        f"RAM before gc ({prefix}{before_gb:.1f} GB used, {available_gb:.1f} GB available)"
    )

    gc.collect()

    vm = psutil.virtual_memory()
    after_gb = vm.used / 1024**3
    available_gb = vm.available / 1024**3
    freed_gb = before_gb - after_gb
    logger.info(
        f"RAM after gc ({prefix}{after_gb:.1f} GB used, {available_gb:.1f} GB available, freed {freed_gb:.1f} GB)"
    )
