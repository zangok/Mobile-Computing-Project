"""Measure paired RGB reconstruction quality against the eager FP32 reference."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean

import torch
from torch.nn.functional import conv2d

from baseline import CONFIGURATIONS, CPU_CONFIGURATIONS, WindowPacker, frame_paths, load_configuration, paired_frames, set_backend, sha256, upstream_files
from pipeline import Frame, FrameHistory


class RGBMetrics:
    """PSNR and Gaussian-window SSIM on RGB tensors in [0, 1]."""

    def __init__(self, device):
        self.device = torch.device(device)
        offsets = torch.arange(-5, 6, device=device, dtype=torch.float32)
        weights = torch.exp(-0.5 * (offsets / 1.5).square())
        weights /= weights.sum()
        self.horizontal = weights.view(1, 1, 1, 11)
        self.vertical = weights.view(1, 1, 11, 1)

    def __call__(self, clean, estimates):
        if estimates.ndim != 4 or estimates.shape[1:] != clean.shape:
            raise ValueError("Quality estimates must match the clean RGB frame shape")
        if not torch.isfinite(estimates).all() or estimates.min() < 0 or estimates.max() > 1:
            raise ValueError("Quality estimates must be finite RGB values in [0, 1]")

        mse = (estimates - clean).square().mean(dim=(1, 2, 3))
        # A finite 120 dB ceiling keeps perfect-match reports and differences valid JSON.
        psnr = -10 * mse.clamp_min(1e-12).log10()
        clean = clean.expand_as(estimates)
        fields = torch.stack((clean, estimates, clean.square(), estimates.square(), clean * estimates))
        height, width = clean.shape[-2:]
        flattened = fields.reshape(-1, 1, height, width)
        smoothed = conv2d(conv2d(flattened, self.horizontal), self.vertical)
        mx, my, ex2, ey2, exy = smoothed.reshape(5, *clean.shape[:2], height - 10, width - 10)
        var_x = (ex2 - mx.square()).clamp_min(0)
        var_y = (ey2 - my.square()).clamp_min(0)
        covariance = exy - mx * my
        ssim = ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)) / (
            (mx.square() + my.square() + 0.01**2) * (var_x + var_y + 0.03**2)
        )
        return torch.stack((psnr, ssim.mean(dim=(1, 2, 3))), dim=1).tolist()


def evaluate_clip(paths, width, height, sigma, seed, reference, candidate, metrics):
    """Use a single seeded noisy stream for both model calls."""
    history = FrameHistory(5)
    frame_scores = []
    for frame_id, pair in enumerate(paired_frames(paths, width, height, sigma / 255, seed)):
        window = history.push(Frame(frame_id, pair))
        if window is None:
            continue
        clean, noisy = window[2].payload
        inputs = tuple(frame.payload[1] for frame in window)
        baseline = reference(inputs, sigma / 255)
        output = baseline if candidate is reference else candidate(inputs, sigma / 255)
        set_backend("b0")
        estimates = torch.stack(tuple(image.to(metrics.device) for image in (noisy, baseline, output)))
        frame_scores.append(metrics(clean.to(metrics.device), estimates))

    return {
        "seed": seed,
        "evaluated_frames": len(frame_scores),
        **{
            f"{label}_{metric}": mean(row[index][column] for row in frame_scores)
            for index, label in enumerate(("noisy", "b0", "candidate"))
            for column, metric in enumerate(("psnr_db", "ssim"))
        },
    }


def summarize(cases, limits):
    """Weight seeds within each clip/noise pair, then weight all pairs equally."""
    psnr_mean = mean(case["psnr_loss_db"] for case in cases)
    ssim_mean = mean(case["ssim_loss"] for case in cases)
    checks = {
        "mean_psnr": psnr_mean <= limits["mean_psnr_loss_db"],
        "mean_ssim": ssim_mean <= limits["mean_ssim_loss"],
        "per_clip_noise_psnr": all(
            case["psnr_loss_db"] <= limits["per_clip_noise_psnr_loss_db"] for case in cases
        ),
        "per_clip_noise_ssim": all(
            case["ssim_loss"] <= limits["per_clip_noise_ssim_loss"] for case in cases
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mean_psnr_loss_db": psnr_mean,
        "mean_ssim_loss": ssim_mean,
    }


def model_adapter(upstream, device, configuration):
    model, denoise, dtype, layout, execution = load_configuration(upstream, device, configuration)
    pack = WindowPacker(device, dtype, layout, reuse=configuration == "fp16-reuse")

    @execution()
    def predict(frames, sigma):
        set_backend(configuration)
        packed = pack(frames)
        noise_map = torch.full((1, 1, *packed.shape[-2:]), sigma, device=device, dtype=dtype)
        return denoise(model, packed, noise_map).squeeze(0).float()

    return predict


def score_cases(args, clips, reference, candidate, metrics):
    cases = []
    for directory, paths in clips:
        print(f"Scoring {directory.name} ({len(paths)} input frames)", flush=True)
        for sigma in args.sigma:
            scores = [
                evaluate_clip(paths, args.width, args.height, sigma, seed, reference, candidate, metrics)
                for seed in args.seed
            ]
            paired = {
                key: mean(score[key] for score in scores)
                for key in scores[0]
                if key not in {"seed", "evaluated_frames"}
            }
            cases.append({
                "clip": str(directory),
                "sigma_8bit": sigma,
                "input_frames": len(paths),
                "seeds": scores,
                **paired,
                "psnr_loss_db": paired["b0_psnr_db"] - paired["candidate_psnr_db"],
                "ssim_loss": paired["b0_ssim"] - paired["candidate_ssim"],
                "candidate_psnr_gain_over_noisy_db": paired["candidate_psnr_db"] - paired["noisy_psnr_db"],
                "candidate_ssim_gain_over_noisy": paired["candidate_ssim"] - paired["noisy_ssim"],
            })
    return cases


@torch.no_grad()
def run(args, protocol):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation")
    clips = []
    for directory in args.clips:
        paths = frame_paths(directory)[:args.max_frames]
        if len(paths) < 5:
            raise ValueError(f"At least five ordered frames are required in {directory}")
        clips.append((directory, paths))
    reference = model_adapter(args.upstream, device, "b0")
    candidate_upstream = args.candidate_upstream or args.upstream
    reference_hashes = {path.name: sha256(path) for path in upstream_files(args.upstream)}
    candidate_hashes = {path.name: sha256(path) for path in upstream_files(candidate_upstream)}
    if reference_hashes["model_clipped_noise.pth"] != candidate_hashes["model_clipped_noise.pth"]:
        raise ValueError("The candidate checkpoint differs from the reference checkpoint")
    candidate = reference
    if args.candidate == "noisy":
        candidate = lambda frames, sigma: frames[2]
    elif args.candidate != "b0" or args.candidate_upstream is not None:
        candidate = model_adapter(candidate_upstream, device, args.candidate)

    metrics = RGBMetrics(device)
    if args.candidate == "cudnn":
        # cuDNN v8 plans are thread-local, and their cache key excludes the
        # benchmark flag. Give autotuned FP32 its own cache.
        def predict_autotuned(frames, sigma):
            return candidate(frames, sigma).cpu()

        with ThreadPoolExecutor(max_workers=1) as worker:
            def isolated_candidate(frames, sigma):
                return worker.submit(predict_autotuned, frames, sigma).result()

            cases = score_cases(args, clips, reference, isolated_candidate, metrics)
    else:
        cases = score_cases(args, clips, reference, candidate, metrics)
    limits = protocol["quality_limits"]
    return {
        "candidate": args.candidate,
        "device": args.device,
        "torch": torch.__version__,
        "width": args.width,
        "height": args.height,
        "fastdvdnet_files_sha256": reference_hashes,
        "candidate_fastdvdnet_files_sha256": candidate_hashes,
        "quality_limits": limits,
        "cases": cases,
        **summarize(cases, limits),
    }


def main():
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
    workload = protocol["workload"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", type=Path, help="Directories of ordered PNG or JPEG frames")
    parser.add_argument("--candidate", choices=(*CONFIGURATIONS, "noisy"), required=True)
    parser.add_argument("--upstream", type=Path, default=Path("checkpoints/fastdvdnet"))
    parser.add_argument("--candidate-upstream", type=Path, help="Evaluate model code from a separate candidate directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--width", type=int, default=workload["width"])
    parser.add_argument("--height", type=int, default=workload["height"])
    parser.add_argument("--sigma", nargs="+", type=int, choices=workload["noise_std_8bit"], default=workload["noise_std_8bit"])
    parser.add_argument("--seed", nargs="+", type=int, default=workload["development_seeds"])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if min(args.width, args.height) < 11:
        parser.error("RGB SSIM requires both dimensions to be at least 11")
    if args.max_frames is not None and args.max_frames < 5:
        parser.error("--max-frames must be at least five")
    if args.candidate not in (*CPU_CONFIGURATIONS, "noisy") and args.device != "cuda":
        parser.error(f"{args.candidate} requires CUDA")
    if args.candidate == "noisy" and args.candidate_upstream is not None:
        parser.error("The noisy control does not load a candidate model")
    if len({path.resolve() for path in args.clips}) != len(args.clips) or any(
        len(set(values)) != len(values) for values in (args.sigma, args.seed)
    ):
        parser.error("Clip directories, noise levels and seeds must each be unique")
    try:
        report = run(args, protocol)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Quality gate {'passed' if report['passed'] else 'failed'}. Report: {args.report}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
