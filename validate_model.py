"""Check a candidate implementation against the same frozen model configuration."""

import argparse
import json
from pathlib import Path

import torch

from baseline import CONFIGURATIONS, CPU_CONFIGURATIONS, frame_paths, paired_frames, sha256, upstream_files
from quality import model_adapter

ROOT = Path(__file__).resolve().parent
SHAPES = ((320, 180), (640, 360), (960, 540), (321, 181))


def cases(development):
    generator = torch.Generator(device="cpu").manual_seed(11)
    for width, height in SHAPES:
        shape = 3, height, width
        for label, fill, sigma in (("black", 0.0, 10), ("white", 1.0, 50)):
            image = torch.full(shape, fill, dtype=torch.float32)
            yield f"{label}-{width}x{height}", width, height, sigma, (image,) * 5
        for sigma in (10, 50):
            frames = tuple(torch.rand(shape, generator=generator) for _ in range(5))
            yield f"random-{width}x{height}-sigma-{sigma}", width, height, sigma, frames
    for directory in development:
        paths = frame_paths(directory)[:5]
        if len(paths) != 5:
            raise ValueError(f"Expected at least five ordered frames in {directory}")
        frames = tuple(noisy for _, noisy in paired_frames(paths, 640, 360, 25 / 255, 11))
        yield f"development-{directory.name}", 640, 360, 25, frames


def agreement(reference, candidate, expected_shape, atol, rtol):
    if reference.shape != expected_shape or not torch.isfinite(reference).all():
        raise ValueError("The reference failed its shape or finite-output check")
    if candidate.shape != expected_shape or not torch.isfinite(candidate).all():
        return {"passed": False, "reason": "invalid candidate shape or nonfinite output"}
    difference = (reference - candidate).abs()
    allowed = atol + rtol * reference.abs()
    exceedances = int((difference > allowed).sum())
    return {
        "passed": exceedances == 0,
        "max_absolute_error": float(difference.max()),
        "values_outside_tolerance": exceedances,
    }


def run(args):
    limits = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))["quality_limits"]
    precision = "fp16" if args.configuration.startswith("fp16") else "fp32"
    atol, rtol = (limits[f"{precision}_{kind}"] for kind in ("atol", "rtol"))
    reference_files = {path.name: sha256(path) for path in upstream_files(args.upstream)}
    candidate_files = {path.name: sha256(path) for path in upstream_files(args.candidate_upstream)}
    if reference_files["model_clipped_noise.pth"] != candidate_files["model_clipped_noise.pth"]:
        raise ValueError("The candidate checkpoint differs from the reference checkpoint")

    device = torch.device(args.device)
    reference = model_adapter(args.upstream, device, args.configuration)
    candidate = model_adapter(args.candidate_upstream, device, args.configuration)
    results = []
    for name, width, height, sigma, frames in cases(args.development):
        expected = reference(frames, sigma / 255)
        actual = candidate(frames, sigma / 255)
        result = agreement(expected, actual, (3, height, width), atol, rtol)
        results.append(dict(case=name, **result))
    return {
        "configuration": args.configuration,
        "device": args.device,
        "torch": torch.__version__,
        "atol": atol,
        "rtol": rtol,
        "reference_files_sha256": reference_files,
        "candidate_files_sha256": candidate_files,
        "cases": results,
        "passed": all(case["passed"] for case in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("checkpoints/fastdvdnet"))
    parser.add_argument("--candidate-upstream", type=Path, required=True)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--development", type=Path, nargs="*", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cpu" and args.configuration not in CPU_CONFIGURATIONS:
        parser.error(f"{args.configuration} requires CUDA")
    try:
        report = run(args)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Numerical check {'passed' if report['passed'] else 'failed'}. Report: {args.report}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
