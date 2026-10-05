#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Adapted from YOLOX tools/benchmark_onnx.py for NanoDet.


"""

Benchmark NanoDet ONNX throughput (images per second).

Runs one warmup inference, then times --num-runs batches of dummy inputs
and reports how many images were processed per second.

Usage:
    python tools/benchmark_onnx.py \
        --model models/nanodet.onnx \
        --batch-size 1 \
        --num-runs 100

    python tools/benchmark_onnx.py \
        --model models/nanodet.onnx \
        --batch-size 1 \
        --device cpu \
        --num-runs 50 \
        --sweep-threads

    python tools/benchmark_onnx.py \
        --model models/nanodet.onnx \
        --batch-size 1 \
        --device cpu \
        --num-runs 50 \
        --sweep-threads \
        --intra-op-threads 8 \
        --workers 4

    python tools/benchmark_onnx.py \
        --model models/nanodet.onnx \
        --batch-size 1 \
        --device cpu \
        --intra-op-threads 2 \
        --workers 2 \
        --num-runs 50

    python tools/benchmark_onnx.py \
        --model models/nanodet.onnx \
        --batch-size 32 \
        --device cuda \
        --num-runs 50 \
        --sweep-batch-size

    python tools/benchmark_onnx.py \
        --model models/nanodet.onnx \
        --batch-size 32 \
        --device cuda \
        --num-runs 50 \
        --sweep

    python tools/benchmark_onnx.py \
        --batch-size 1 \
        --device coreml \
        --num-runs 50
"""

import argparse
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort


ONNX_DTYPE_TO_NUMPY = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}

CUDA_LOAD_HINT = """
CUDAExecutionProvider was requested but ONNX Runtime is using {providers}.
onnxruntime-gpu is installed, but CUDA/cuDNN libraries failed to load
(on Windows this is usually a missing cudnn64_9.dll or a cuDNN 9
sublibrary such as cudnn_engines_tensor_ir64_9.dll).

Current onnxruntime-gpu builds need CUDA 12.x, cuDNN 9.x (full set of
DLLs, not just cudnn64_9.dll), and the latest MSVC runtime. Try:

  pip install "onnxruntime-gpu[cuda,cudnn]"

or install the CUDA 12 toolkit + the full cuDNN 9 zip/installer and add
both bin directories to PATH, for example:
  C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin
  C:\\Program Files\\NVIDIA\\CUDNN\\v9.x\\bin\\12.x

If cuDNN was installed with NVIDIA's installer, use the CUDA 12 subfolder
(not bin\\11.8). Confirm this file exists next to cudnn64_9.dll:
  cudnn_engines_tensor_ir64_9.dll
See:
  https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
""".strip()

CUDNN_RUNTIME_HINT = """
CUDA convolution failed. cuDNN 9.21+ also needs cudnn_engines_tensor_ir64_9.dll
next to cudnn64_9.dll (ort.preload_dlls() often skips this file).

Confirm the DLL exists, then either:
  pip install -U "nvidia-cudnn-cu12>=9.21"
or add the directory that contains every cudnn*.dll to PATH, e.g.
  C:\\Program Files\\NVIDIA\\CUDNN\\v9.x\\bin\\12.x
  <venv>\\Lib\\site-packages\\nvidia\\cudnn\\bin
""".strip()


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL = os.path.join(REPO_ROOT, "models", "nanodet.onnx")


def image_model_input(session):
    """Pick the image tensor. Exported graphs may also list weights as inputs."""
    inputs = session.get_inputs()
    for item in inputs:
        if item.name == "data":
            return item
    for item in inputs:
        shape = item.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            return item
    return inputs[0]


def make_parser():
    parser = argparse.ArgumentParser("NanoDet ONNX throughput benchmark")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to the ONNX model. Defaults to models/nanodet.onnx.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=True,
        help="Batch size used for dummy inputs. "
        "With --sweep-batch-size or GPU --sweep this is the maximum batch to search.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        required=True,
        help="Number of timed inference runs after the warmup run.",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Height,width of the model input, e.g. 416,416. "
        "If omitted, spatial size is read from the ONNX graph.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "coreml"],
        help="ONNX Runtime execution device. "
        "'auto' uses CoreML when available, then CUDA, otherwise CPU. "
        "'coreml' falls back to CPU when CoreMLExecutionProvider is missing.",
    )
    parser.add_argument(
        "-t",
        "--intra-op-threads",
        type=int,
        default=None,
        help="CPU intra-op threads for one session.run(). "
        "Default is the ONNX Runtime default (typically all cores). "
        "With --sweep-threads this is the maximum thread count to search.",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=0,
        help="CPU inter-op threads. 0 uses the ONNX Runtime default. "
        "Rarely helpful for NanoDet's mostly sequential graph.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Concurrent inference threads sharing one session. "
        "Default is 1. Each worker performs --num-runs timed runs. "
        "With --sweep-threads this is the maximum worker count to search.",
    )
    parser.add_argument(
        "--sweep-threads",
        action="store_true",
        help="Search intra-op thread counts and worker counts (1,2,4,...) "
        "and report the combination with the highest images/s. "
        "Pairs that oversubscribe CPU cores (threads * workers > CPU count) "
        "are skipped. Caps: --intra-op-threads and --workers. "
        "This is the main CPU tuning knob; batch size usually does little on CPU.",
    )
    parser.add_argument(
        "--sweep-batch-size",
        action="store_true",
        help="Search batch sizes 1,2,4,... up to --batch-size and report "
        "the highest images/s. This is the main GPU tuning knob; "
        "CPU throughput usually stays flat as batch grows.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Device-aware search: CPU sweeps threads/workers, CUDA and CoreML "
        "sweep batch size. Equivalent to --sweep-threads or --sweep-batch-size.",
    )
    return parser


def is_static_dim(dim):
    return isinstance(dim, int) and dim > 0


def parse_input_shape(input_shape):
    try:
        height, width = tuple(map(int, input_shape.split(",")))
    except (AttributeError, TypeError, ValueError):
        raise ValueError(
            "Invalid --input-shape '{}'. Expected 'height,width', e.g. 416,416.".format(
                input_shape
            )
        )
    if height <= 0 or width <= 0:
        raise ValueError("--input-shape values must be positive integers.")
    return height, width


def _cudnn_bin_dirs():
    dirs = []
    try:
        import nvidia.cudnn as nvidia_cudnn

        roots = []
        cudnn_file = getattr(nvidia_cudnn, "__file__", None)
        if cudnn_file:
            roots.append(os.path.dirname(cudnn_file))
        roots.extend(getattr(nvidia_cudnn, "__path__", []) or [])
        for cudnn_root in roots:
            dirs.extend(
                (
                    os.path.join(cudnn_root, "bin"),
                    os.path.join(cudnn_root, "bin", "x86_64"),
                )
            )
    except ImportError:
        pass

    for site_dir in sys.path:
        dirs.extend(
            (
                os.path.join(site_dir, "nvidia", "cudnn", "bin"),
                os.path.join(site_dir, "nvidia", "cudnn", "bin", "x86_64"),
            )
        )

    for path in os.environ.get("PATH", "").split(os.pathsep):
        if path:
            dirs.append(path)

    unique = []
    seen = set()
    for path in dirs:
        real = os.path.abspath(path)
        if real in seen or not os.path.isdir(real):
            continue
        seen.add(real)
        unique.append(real)
        try:
            for name in os.listdir(real):
                child = os.path.join(real, name)
                if (
                    os.path.isdir(child)
                    and child not in seen
                    and (
                        name.startswith("12")
                        or name.startswith("13")
                        or name == "x86_64"
                    )
                ):
                    seen.add(child)
                    unique.append(child)
        except OSError:
            pass
    return [
        path
        for path in unique
        if glob.glob(os.path.join(path, "cudnn*.dll"))
    ]


def _prepend_dll_dir(path):
    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except OSError:
            pass


def _windows_load_extra_cudnn_dlls():
    """Load cuDNN 9 sublibraries that ort.preload_dlls() often skips.

    cuDNN 9.21+ added cudnn_engines_tensor_ir64_9.dll. If that file is not
    loaded, Conv nodes fail and ONNX Runtime silently retries on CPU.
    """
    if os.name != "nt":
        return

    import ctypes

    bin_dirs = _cudnn_bin_dirs()
    for path in bin_dirs:
        _prepend_dll_dir(path)

    get_module_handle_w = ctypes.windll.kernel32.GetModuleHandleW
    get_module_handle_w.argtypes = [ctypes.c_wchar_p]
    get_module_handle_w.restype = ctypes.c_void_p

    loaded = []
    failed = []
    seen = set()
    for bin_dir in bin_dirs:
        for dll_path in sorted(glob.glob(os.path.join(bin_dir, "cudnn*.dll"))):
            dll_name = os.path.basename(dll_path)
            key = dll_name.lower()
            if key in seen or get_module_handle_w(dll_name):
                seen.add(key)
                continue
            seen.add(key)
            try:
                ctypes.CDLL(dll_path)
                loaded.append(dll_path)
            except OSError as exc:
                failed.append("{} ({})".format(dll_path, exc))

    if loaded:
        print("loaded extra cuDNN DLLs:")
        for path in loaded:
            print("  {}".format(path))
    if failed:
        print("warning: failed to load some cuDNN DLLs:")
        for item in failed:
            print("  {}".format(item))


def preload_cuda_dependencies():
    """Load CUDA/cuDNN DLLs from PyTorch or NVIDIA pip packages when possible."""
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        except Exception as exc:
            print("warning: onnxruntime.preload_dlls() failed: {}".format(exc))
            try:
                import torch  # noqa: F401
            except ImportError:
                pass
    else:
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
    _windows_load_extra_cudnn_dlls()


def resolve_providers(device):
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu (not onnxruntime) or use --device cpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "coreml":
        if "CoreMLExecutionProvider" not in available:
            print(
                "CoreMLExecutionProvider is not available; using CPUExecutionProvider."
            )
            return ["CPUExecutionProvider"]
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_session(model_path, device, intra_op_threads=0, inter_op_threads=0):
    providers = resolve_providers(device)
    sess_options = ort.SessionOptions()
    if intra_op_threads > 0:
        sess_options.intra_op_num_threads = intra_op_threads
    if inter_op_threads > 0:
        sess_options.inter_op_num_threads = inter_op_threads
    try:
        session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
    except Exception as exc:
        if "CoreMLExecutionProvider" not in providers:
            raise
        print(
            "CoreMLExecutionProvider failed to load ({}); "
            "using CPUExecutionProvider.".format(exc)
        )
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
    active = session.get_providers()
    if device == "cuda" and "CUDAExecutionProvider" not in active:
        raise RuntimeError(CUDA_LOAD_HINT.format(providers=active))
    if (
        device == "auto"
        and "CUDAExecutionProvider" in providers
        and "CUDAExecutionProvider" not in active
    ):
        print(
            "warning: CUDAExecutionProvider is installed but failed to load; "
            "using CPU. Missing CUDA 12 / cuDNN 9 DLLs is the usual cause "
            "(cudnn64_9.dll / cudnn_engines_tensor_ir64_9.dll)."
        )
    if (
        "CoreMLExecutionProvider" in providers
        and "CoreMLExecutionProvider" not in active
    ):
        print(
            "warning: CoreMLExecutionProvider did not stay active; using {}.".format(
                active
            )
        )
    return session


def resolve_nchw_shape(model_input, batch_size, input_shape_arg):
    onnx_shape = list(model_input.shape)
    if len(onnx_shape) != 4:
        raise ValueError(
            "Expected a 4D NCHW input tensor, got {} with shape {}.".format(
                model_input.name, onnx_shape
            )
        )

    onnx_batch, onnx_channels, onnx_height, onnx_width = onnx_shape
    if is_static_dim(onnx_batch) and onnx_batch != batch_size:
        raise ValueError(
            "Model was exported with fixed batch size {}, but --batch-size is {}. "
            "Re-export with a dynamic batch axis, or pass a matching batch size.".format(
                onnx_batch, batch_size
            )
        )
    if is_static_dim(onnx_channels) and onnx_channels != 3:
        raise ValueError(
            "Expected 3 input channels, got {}.".format(onnx_channels)
        )

    if input_shape_arg is not None:
        height, width = parse_input_shape(input_shape_arg)
        if is_static_dim(onnx_height) and onnx_height != height:
            raise ValueError(
                "ONNX height is {}, but --input-shape height is {}.".format(
                    onnx_height, height
                )
            )
        if is_static_dim(onnx_width) and onnx_width != width:
            raise ValueError(
                "ONNX width is {}, but --input-shape width is {}.".format(
                    onnx_width, width
                )
            )
        return (batch_size, 3, height, width)

    if not is_static_dim(onnx_height) or not is_static_dim(onnx_width):
        raise ValueError(
            "ONNX spatial size is dynamic {}. Pass --input-shape height,width.".format(
                onnx_shape
            )
        )
    return (batch_size, 3, onnx_height, onnx_width)


def make_dummy_input(shape, onnx_type):
    dtype = ONNX_DTYPE_TO_NUMPY.get(onnx_type)
    if dtype is None:
        raise ValueError("Unsupported ONNX input type: {}".format(onnx_type))
    return np.random.randn(*shape).astype(dtype)


def power_of_two_counts(max_count):
    counts = []
    n = 1
    while n < max_count:
        counts.append(n)
        n *= 2
    counts.append(max_count)
    return counts


def sweep_combos(max_threads, max_workers, cpu_count):
    combos = []
    for threads in power_of_two_counts(max_threads):
        for workers in power_of_two_counts(max_workers):
            oversubscribed = threads * workers > cpu_count
            keep_axis = (workers == 1 and threads == max_threads) or (
                threads == 1 and workers == max_workers
            )
            if oversubscribed and not keep_axis:
                continue
            combos.append((threads, workers))
    return combos


def format_thread_count(value):
    return "default" if value <= 0 else str(value)


def timed_inference_runs(session, feeds, num_runs, workers):
    if workers == 1:
        start = time.perf_counter()
        for _ in range(num_runs):
            session.run(None, feeds)
        return time.perf_counter() - start

    barrier = threading.Barrier(workers + 1)

    def worker():
        local_feeds = {
            name: np.array(array, copy=True) for name, array in feeds.items()
        }
        barrier.wait()
        for _ in range(num_runs):
            session.run(None, local_feeds)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker) for _ in range(workers)]
        barrier.wait()
        start = time.perf_counter()
        for future in futures:
            future.result()
        return time.perf_counter() - start


def warmup_session(session, feeds, workers, device):
    try:
        session.run(None, feeds)
        if workers > 1:
            timed_inference_runs(session, feeds, num_runs=1, workers=workers)
    except Exception as exc:
        if device == "cuda":
            raise RuntimeError(CUDNN_RUNTIME_HINT) from exc
        raise


def measure_session(
    session,
    feeds,
    batch_size,
    num_runs,
    workers,
    intra_op_threads,
    inter_op_threads,
    device,
):
    warmup_session(session, feeds, workers, device)
    elapsed = timed_inference_runs(session, feeds, num_runs, workers)
    total_inferences = num_runs * workers
    total_images = total_inferences * batch_size
    model_input = image_model_input(session)
    dummy_input = next(iter(feeds.values()))
    return {
        "providers": session.get_providers(),
        "input_name": model_input.name,
        "input_shape": dummy_input.shape,
        "input_dtype": dummy_input.dtype,
        "intra_op_threads": intra_op_threads,
        "inter_op_threads": inter_op_threads,
        "workers": workers,
        "batch_size": batch_size,
        "elapsed": elapsed,
        "total_images": total_images,
        "images_per_second": total_images / elapsed,
        "ms_per_batch": (elapsed / num_runs) * 1000.0,
    }


def session_feeds(session, batch_size, input_shape_arg):
    model_input = image_model_input(session)
    input_shape = resolve_nchw_shape(model_input, batch_size, input_shape_arg)
    dummy_input = make_dummy_input(input_shape, model_input.type)
    return {model_input.name: dummy_input}


def run_one_config(
    model_path,
    device,
    input_shape_arg,
    batch_size,
    num_runs,
    workers,
    intra_op_threads,
    inter_op_threads,
):
    session = create_session(
        model_path, device, intra_op_threads, inter_op_threads
    )
    feeds = session_feeds(session, batch_size, input_shape_arg)
    return measure_session(
        session,
        feeds,
        batch_size,
        num_runs,
        workers,
        intra_op_threads,
        inter_op_threads,
        device,
    )


def print_gpu_thread_note(device):
    if device != "cuda":
        return
    print(
        "note: GPU throughput is driven by --batch-size. Intra-op threads "
        "do not speed up CUDA kernels."
    )
    print(
        "note: extra --workers can help a little at batch size 1 by overlapping "
        "copies and kernels, but increasing batch size is the main GPU lever."
    )


def print_device_notes(providers):
    if "CUDAExecutionProvider" in providers:
        print_gpu_thread_note("cuda")
    elif "CoreMLExecutionProvider" in providers:
        print(
            "note: CoreML throughput is driven by --batch-size. Intra-op threads "
            "do not speed up CoreML."
        )
        print(
            "note: extra --workers can help a little at batch size 1, "
            "but increasing batch size is the main CoreML lever."
        )


def print_single_result(result, num_runs):
    print("providers: {}".format(result["providers"]))
    print("input name: {}".format(result["input_name"]))
    print("input shape: {}".format(result["input_shape"]))
    print("input dtype: {}".format(result["input_dtype"]))
    print("intra-op-threads: {}".format(format_thread_count(result["intra_op_threads"])))
    print("inter-op-threads: {}".format(format_thread_count(result["inter_op_threads"])))
    print("workers: {}".format(result["workers"]))
    print("batch-size: {}".format(result["batch_size"]))
    print("warmup runs: 1")
    print("timed runs per worker: {}".format(num_runs))
    print("total images: {}".format(result["total_images"]))
    print("total time: {:.4f} s".format(result["elapsed"]))
    print("latency: {:.3f} ms / batch".format(result["ms_per_batch"]))
    print("throughput: {:.2f} images / s".format(result["images_per_second"]))


def print_sweep_table(results, num_runs):
    best = max(results, key=lambda item: item["images_per_second"])
    print("timed runs per worker: {}".format(num_runs))
    header = "{:>4} {:>8} {:>8} {:>8} {:>10} {:>12} {:>14}".format(
        "", "intra-op", "workers", "batch", "images", "ms/batch", "images/s"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        marker = "*" if result is best else ""
        print(
            "{:>4} {:>8} {:>8} {:>8} {:>10} {:>12.3f} {:>14.2f}".format(
                marker,
                format_thread_count(result["intra_op_threads"]),
                result["workers"],
                result["batch_size"],
                result["total_images"],
                result["ms_per_batch"],
                result["images_per_second"],
            )
        )
    print(
        "best: intra-op-threads={} workers={} batch-size={} "
        "throughput={:.2f} images / s".format(
            format_thread_count(best["intra_op_threads"]),
            best["workers"],
            best["batch_size"],
            best["images_per_second"],
        )
    )


def run_thread_sweep(args, cpu_count):
    max_threads = max(args.intra_op_threads or cpu_count, 1)
    max_workers = max(args.workers or cpu_count, 1)
    combos = sweep_combos(max_threads, max_workers, cpu_count)
    if not combos:
        raise RuntimeError("No thread/worker combinations to run.")
    print("cpu count: {}".format(cpu_count))
    print("sweep combos (intra-op-threads, workers): {}".format(combos))
    results = []
    sessions = {}
    feeds_by_threads = {}
    for thread_count, workers in combos:
        if thread_count not in sessions:
            sessions[thread_count] = create_session(
                args.model,
                args.device,
                thread_count,
                args.inter_op_threads,
            )
            feeds_by_threads[thread_count] = session_feeds(
                sessions[thread_count], args.batch_size, args.input_shape
            )
        result = measure_session(
            sessions[thread_count],
            feeds_by_threads[thread_count],
            args.batch_size,
            args.num_runs,
            workers,
            thread_count,
            args.inter_op_threads,
            args.device,
        )
        results.append(result)
        print(
            "intra-op-threads={} workers={} batch-size={} "
            "throughput={:.2f} images / s".format(
                thread_count,
                workers,
                args.batch_size,
                result["images_per_second"],
            )
        )
    return results


def run_batch_sweep(args):
    batch_sizes = power_of_two_counts(args.batch_size)
    print("sweep batch-sizes: {}".format(batch_sizes))
    session = create_session(
        args.model,
        args.device,
        args.intra_op_threads or 0,
        args.inter_op_threads,
    )
    workers = args.workers or 1
    results = []
    for batch_size in batch_sizes:
        try:
            feeds = session_feeds(session, batch_size, args.input_shape)
            result = measure_session(
                session,
                feeds,
                batch_size,
                args.num_runs,
                workers,
                args.intra_op_threads or 0,
                args.inter_op_threads,
                args.device,
            )
        except Exception as exc:
            print("batch-size={} failed: {}".format(batch_size, exc))
            print(
                "Stopping the batch sweep. For sizes above the exported "
                "batch, re-export with a dynamic batch axis. GPU out-of-memory "
                "also stops the sweep here."
            )
            break
        results.append(result)
        print(
            "batch-size={} throughput={:.2f} images / s latency={:.3f} ms / batch".format(
                batch_size,
                result["images_per_second"],
                result["ms_per_batch"],
            )
        )
    if not results:
        raise RuntimeError("No successful batch-size measurements.")
    return results


def session_uses_cuda(session):
    return "CUDAExecutionProvider" in session.get_providers()


def prefers_batch_sweep(providers):
    return (
        "CoreMLExecutionProvider" in providers
        or "CUDAExecutionProvider" in providers
    )


def should_preload_cuda(device):
    if device == "cuda":
        return True
    if device != "auto":
        return False
    available = ort.get_available_providers()
    return (
        "CoreMLExecutionProvider" not in available
        and "CUDAExecutionProvider" in available
    )


def main():
    args = make_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be a positive integer.")
    if args.intra_op_threads is not None and args.intra_op_threads < 0:
        raise ValueError("--intra-op-threads must be >= 0.")
    if args.inter_op_threads < 0:
        raise ValueError("--inter-op-threads must be >= 0.")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be a positive integer.")
    if not os.path.isfile(args.model):
        raise FileNotFoundError("ONNX model not found: {}".format(args.model))

    print("model: {}".format(os.path.abspath(args.model)))
    print("device: {}".format(args.device))
    ort.set_default_logger_severity(3)
    if should_preload_cuda(args.device):
        preload_cuda_dependencies()

    cpu_count = os.cpu_count() or 1
    sweep_threads = args.sweep_threads
    sweep_batch = args.sweep_batch_size
    if args.sweep and not sweep_threads and not sweep_batch:
        if args.device == "cpu":
            sweep_threads = True
        elif args.device == "cuda":
            sweep_batch = True
        else:
            probe = create_session(
                args.model,
                args.device,
                args.intra_op_threads or 0,
                args.inter_op_threads,
            )
            providers = probe.get_providers()
            sweep_label = "auto sweep" if args.device == "auto" else "sweep"
            if "CoreMLExecutionProvider" in providers:
                sweep_batch = True
                print("{}: CoreML detected, searching batch sizes".format(sweep_label))
            elif session_uses_cuda(probe):
                sweep_batch = True
                print("{}: CUDA detected, searching batch sizes".format(sweep_label))
            else:
                sweep_threads = True
                print(
                    "{}: CPU detected, searching threads/workers".format(sweep_label)
                )

    if sweep_threads:
        results = run_thread_sweep(args, cpu_count)
        print_device_notes(results[0]["providers"])
        print_sweep_table(results, args.num_runs)
        return

    if sweep_batch:
        if args.batch_size == 1:
            print(
                "note: --batch-size 1 only tests batch 1. Pass a larger "
                "--batch-size as the sweep maximum, e.g. --batch-size 32."
            )
        results = run_batch_sweep(args)
        if prefers_batch_sweep(results[0]["providers"]):
            print_device_notes(results[0]["providers"])
        else:
            print(
                "note: CPU images/s is usually almost flat across batch sizes; "
                "use --sweep-threads to tune CPU."
            )
        print_sweep_table(results, args.num_runs)
        return

    result = run_one_config(
        args.model,
        args.device,
        args.input_shape,
        args.batch_size,
        args.num_runs,
        args.workers or 1,
        args.intra_op_threads or 0,
        args.inter_op_threads,
    )
    print_device_notes(result["providers"])
    print_single_result(result, args.num_runs)


if __name__ == "__main__":
    main()
