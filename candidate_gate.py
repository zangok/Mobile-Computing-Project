"""Apply the development acceptance rules to paired candidate reports."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def timing_medians(reference, candidate, expected_outputs, expected_warmup):
    if len(reference) != len(candidate) or len(reference) < 2:
        raise ValueError("Supply at least two paired server timing reports")
    fields = ("configuration", "device", "gpu", "torch", "width", "height",
              "sigma_8bit", "seed", "input_files", "timed_frame_ids", "warmup_outputs")
    for row in (*reference, *candidate):
        if any(key not in row or row[key] != reference[0][key] for key in fields):
            raise ValueError("All server reports must use the same configuration and workload")
        if row["timed_outputs"] != expected_outputs:
            raise ValueError(f"Each server report must contain {expected_outputs} timed outputs")
        if row["warmup_outputs"] != expected_warmup:
            raise ValueError(f"Each server report must follow {expected_warmup} warm-up outputs")
        value = row["server_window_ms"]["p50"]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Server latency must be positive and finite")
    return [median(row["server_window_ms"]["p50"] for row in reports)
            for reports in (reference, candidate)]


def compare(reference, candidate, numerical, quality, protocol):
    run = protocol["timing"]
    limits = protocol["candidate_limits"]
    b_ms, c_ms = timing_medians(reference, candidate, run["candidate_timed_outputs"], run["model_warmup_calls"])
    configuration = reference[0]["configuration"]
    if numerical.get("configuration") != configuration or quality.get("candidate") != configuration:
        raise ValueError("The numerical and quality reports must use the timed B1 configuration")

    if (any(row["fastdvdnet_files_sha256"] != numerical["reference_files_sha256"] for row in reference)
            or any(row["fastdvdnet_files_sha256"] != numerical["candidate_files_sha256"] for row in candidate)
            or quality["candidate_fastdvdnet_files_sha256"] != numerical["candidate_files_sha256"]):
        raise ValueError("Timing, numerical, and quality reports must refer to the same model files")
    server_reduction = 1 - c_ms / b_ms
    checks = {
        "numerical": numerical.get("passed") is True,
        "quality": quality.get("passed") is True,
        "server_latency": c_ms <= b_ms * (1 - limits["min_server_median_reduction_fraction"]),
    }
    return {"accepted": all(checks.values()), "checks": checks,
            "reference_server_ms": b_ms, "candidate_server_ms": c_ms,
            "minimum_server_reduction_fraction": limits["min_server_median_reduction_fraction"],
            "server_reduction_fraction": server_reduction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-timing", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate-timing", nargs="+", required=True, type=Path)
    parser.add_argument("--numerical", required=True, type=Path)
    parser.add_argument("--quality", required=True, type=Path)
    args = parser.parse_args()
    protocol = load(Path(__file__).with_name("protocol.json"))
    try:
        result = compare(
            [load(path) for path in args.reference_timing], [load(path) for path in args.candidate_timing],
            load(args.numerical), load(args.quality), protocol,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
