"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0. See the original Unsloth
repository at https://github.com/unslothai/unsloth.

The following line
https://github.com/linkedin/Liger-Kernel/blob/7382a8761f9af679482b968f9348013d933947c7/src/liger_kernel/ops/utils.py#L23
is based on code from Unsloth, located at:
https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

Modifications made by Yanning Chen, 2024.
"""

import functools
import importlib
from typing import Callable

import paddle
import triton
from packaging.version import Version


def custom_fwd(func):
    def wrapper(*args, **kwargs):
        ctx = args[0]
        if len(args) == 1:
            all_args = tuple(kwargs.values())
        else:
            all_args = args[1:] + tuple(kwargs.values())

        if not hasattr(ctx, "needs_input_grad"):
            ctx.needs_input_grad = [False] * len(all_args)
        for i, arg in enumerate(all_args):
            if isinstance(arg, paddle.Tensor):
                if not arg.stop_gradient:
                    ctx.needs_input_grad[i] = True
            else:
                ctx.needs_input_grad[i] = "not_tensor"
        return func(*args, **kwargs)

    return wrapper


def custom_bwd(func):
    def wrapper(*args, **kwargs):
        ctx = args[0]
        output = func(*args, **kwargs)
        result = []
        for each, need_input_grad in zip(output, ctx.needs_input_grad):
            if isinstance(need_input_grad, str) and need_input_grad == "not_tensor":
                continue
            if need_input_grad:
                result.append(each)
            else:
                result.append(None)
        while result and result[-1] is None:
            result.pop()
        return tuple(result)

    return wrapper


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, paddle.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def is_autocast_enabled():
    tracer = paddle.framework._dygraph_tracer()
    return False if tracer._amp_level == paddle.core.AmpLevel.O0 else True


def get_autocast_gpu_dtype():
    from paddle.amp.auto_cast import amp_global_state

    return amp_global_state().amp_dtype
