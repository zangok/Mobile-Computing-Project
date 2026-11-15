"""Replay an image sequence through the eager FP32 FastDVDnet baseline."""

import argparse
import hashlib
import importlib.util
from itertools import cycle, islice
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

from pipeline import Frame, FrameHistory, process_window

EAGER_CONFIGURATIONS = ("b0", "inference", "cudnn", "tf32", "fp16", "fp16-channels-last", "fp16-reuse")
COMPILE_MODES = {
    "fp16-compile-default": "default",
    "fp16-compile-reduce-overhead": "reduce-overhead",
}
CONFIGURATIONS = (*EAGER_CONFIGURATIONS, *COMPILE_MODES)
CPU_CONFIGURATIONS = ("b0", "inference")


def load_module(path):
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    name = f"upstream_{path.stem}_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name)
        raise
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_frames(paths, width, height, sigma, seed):
    import numpy as np
    from PIL import Image
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        clean = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float().div_(255)
        noise = torch.randn(clean.shape, dtype=torch.float32, generator=generator)
        yield clean, (clean + sigma * noise).clamp_(0, 1)


def frame_paths(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def upstream_files(directory):
    return tuple(directory / name for name in ("models.py", "fastdvdnet.py", "model_clipped_noise.pth"))


def load_model(directory, device, dtype):
    import torch

    model_path, algorithm_path, checkpoint = upstream_files(directory)
    for path in (model_path, algorithm_path, checkpoint):
        if not path.is_file():
            raise ValueError(f"Missing upstream file: {path}")
    model = load_module(model_path).FastDVDnet(num_input_frames=5)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    model = model.to(device=device, dtype=dtype).eval()
    return model, load_module(algorithm_path).temp_denoise


def set_backend(configuration):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = configuration == "tf32"
    torch.backends.cudnn.allow_tf32 = configuration == "tf32"
    torch.backends.cudnn.benchmark = configuration not in CPU_CONFIGURATIONS


def load_configuration(directory, device, configuration):
    import torch

    dtype = torch.float16 if configuration.startswith("fp16") else torch.float32
    uses_channels_last = configuration in {"fp16-channels-last", "fp16-reuse", *COMPILE_MODES}
    layout = torch.channels_last if uses_channels_last else torch.contiguous_format
    model, denoise = load_model(directory, device, dtype)
    model = model.to(memory_format=layout)
    if configuration in COMPILE_MODES:
        model = torch.compile(model, mode=COMPILE_MODES[configuration], fullgraph=True, dynamic=False)
    execution = torch.no_grad if configuration == "b0" else torch.inference_mode
    return model, denoise, dtype, layout, execution


class WindowPacker:
    """Pack five CHW inputs, optionally reusing CPU and GPU staging tensors."""

    def __init__(self, device, dtype, layout, reuse):
        self.device = device
        self.dtype = dtype
        self.layout = layout
        self.reuse = reuse
        self.host = None
        self.packed = None

    def __call__(self, frames, scale=1):
        import torch

        if not self.reuse:
            host = torch.cat(frames).unsqueeze(0)
            if scale != 1:
                host = host.to(device=self.device, dtype=torch.float32).div_(scale)
            return host.to(device=self.device, dtype=self.dtype, memory_format=self.layout)

        shape = (1, 15, *frames[0].shape[-2:])
        if self.host is None or self.host.shape != shape:
            self.host = torch.empty(shape, dtype=torch.float32)
            self.packed = torch.empty(shape, device=self.device, dtype=self.dtype, memory_format=self.layout)
        torch.cat(frames, out=self.host[0])
        if scale != 1:
            self.host.div_(scale)
        return self.packed.copy_(self.host)


def windows(frames):
    history = FrameHistory(5)
    for frame_id, frame in enumerate(frames):
        window = history.push(Frame(frame_id, frame))
        if window is not None:
            yield window


def summarize(samples):
    return {
        "p50": statistics.median(samples),
        "p95": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        "samples": samples,
    }


def replay(args, paths):
    import torch

    device = torch.device(args.device)
    cuda = device.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch installation. Use --device cpu for a functional check.")

    model, temp_denoise, dtype, layout, execution = load_configuration(args.upstream, device, args.configuration)
    set_backend(args.configuration)
    pack = WindowPacker(device, dtype, layout, reuse=args.configuration == "fp16-reuse")

    sigma = args.sigma / 255
    frames = [noisy for _, noisy in paired_frames(paths, args.width, args.height, sigma, args.seed)]
    noise_map = torch.full((1, 1, args.height, args.width), sigma, device=device, dtype=dtype)
    if cuda:
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)

    def denoise(payloads):
        packed = pack(payloads)
        if cuda:
            gpu_start.record()
        result = temp_denoise(model, packed, noise_map)
        if cuda:
            gpu_end.record()
        return result.squeeze(0).float().cpu()

    replay_windows = windows(frames)
    if args.repeat_windows:
        replay_windows = islice(cycle(replay_windows), args.warmup + args.outputs)
    server_ms = []
    gpu_ms = []
    timed_ids = []
    expected_shape = (3, args.height, args.width)
    first_window_ms = None
    with execution():
        for output_index, window in enumerate(replay_windows):
            if cuda and output_index == args.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            start = perf_counter_ns()
            output = process_window(window, denoise)
            elapsed_ms = (perf_counter_ns() - start) / 1_000_000
            if output_index == 0:
                first_window_ms = elapsed_ms
            if output.frame_id != window[0].frame_id + 2 or output.payload.shape != expected_shape:
                raise RuntimeError(f"Invalid model output for window {output_index}")
            if not torch.isfinite(output.payload).all():
                raise RuntimeError(f"Nonfinite model output for window {output_index}")
            if output_index >= args.warmup:
                server_ms.append(elapsed_ms)
                timed_ids.append(output.frame_id)
                if cuda:
                    gpu_ms.append(gpu_start.elapsed_time(gpu_end))

    report = {
        "condition": "B0" if args.configuration == "b0" else "B1 candidate",
        "configuration": args.configuration,
        "device": args.device,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "width": args.width,
        "height": args.height,
        "sigma_8bit": args.sigma,
        "seed": args.seed,
        "warmup_outputs": args.warmup,
        "timed_outputs": len(timed_ids),
        "repeated_windows": args.repeat_windows,
        "first_window_ms": first_window_ms,
        "timed_frame_ids": timed_ids,
        "input_files": [path.name for path in paths],
        "fastdvdnet_files_sha256": {path.name: sha256(path) for path in upstream_files(args.upstream)},
        "server_window_ms": summarize(server_ms),
    }
    if cuda:
        report.update(
            gpu=torch.cuda.get_device_name(device),
            cudnn=torch.backends.cudnn.version(),
            gpu_region_ms=summarize(gpu_ms),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    if args.trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with execution(), torch.profiler.profile(activities=activities) as profile:
            process_window(window, denoise)
        profile.export_chrome_trace(str(args.trace))
    return report


def main():
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="Directory of lexically ordered PNG or JPEG frames")
    parser.add_argument("--upstream", type=Path, default=Path("checkpoints/fastdvdnet"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="b0")
    parser.add_argument("--width", type=int, default=protocol["workload"]["width"])
    parser.add_argument("--height", type=int, default=protocol["workload"]["height"])
    parser.add_argument("--sigma", type=int, choices=protocol["workload"]["noise_std_8bit"], default=25)
    parser.add_argument("--seed", type=int, default=protocol["workload"]["development_seeds"][0])
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--outputs", type=int, help="Limit the number of timed outputs")
    parser.add_argument("--repeat-windows", action="store_true", help="Cycle complete windows to reach the requested output counts")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, help="Export a separate single-window profiler trace after timing")
    args = parser.parse_args()
    if args.width < 4 or args.height < 4 or args.warmup < 0 or (args.outputs is not None and args.outputs < 1):
        parser.error("Dimensions must be at least four, warmup nonnegative and outputs positive")
    if not args.frames.is_dir():
        parser.error(f"Frame directory does not exist: {args.frames}")
    paths = frame_paths(args.frames)
    if args.configuration not in CPU_CONFIGURATIONS and args.device != "cuda":
        parser.error(f"{args.configuration} requires CUDA")
    if args.repeat_windows:
        if args.outputs is None or len(paths) < 5:
            parser.error("--repeat-windows requires --outputs and at least five frames")
    else:
        available = len(paths) - args.warmup - 4
        if available < 1 or (args.outputs is not None and args.outputs > available):
            parser.error(f"Insufficient frames for the requested replay ({max(0, available)} timed outputs available)")
        if args.outputs is not None:
            paths = paths[:args.warmup + args.outputs + 4]
    try:
        report = replay(args, paths)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {report['timed_outputs']} timed outputs to {args.report}")


if __name__ == "__main__":
    main()
