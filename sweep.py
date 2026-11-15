"""Quality-screen and time standard eager configurations on pinned DAVIS clips."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from baseline import CONFIGURATIONS, EAGER_CONFIGURATIONS, frame_paths
from prepare_davis import sequence_digest

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_cli(script, arguments, log):
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, arguments)],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode


def qualify(result, data, output):
    configuration = result["configuration"]
    report = output / f"{configuration}-quality.json"
    log = output / f"{configuration}-quality.log"
    code = run_cli("quality.py", [*data, "--candidate", configuration, "--report", report], log)
    if code in (0, 2):
        result["quality_report"] = report.name
    result["quality_status"] = {0: "passed", 2: "rejected"}.get(code, "error")
    if code:
        result.update(status="quality_rejected" if code == 2 else "quality_error", log=log.name)
    return code == 0


def run(args, data, protocol):
    timing = protocol["timing"]
    workload = protocol["workload"]
    seed = workload["development_seeds"][0]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for configuration in ("b0", *args.configurations):
        print(f"Evaluating {configuration}", flush=True)
        result = {"configuration": configuration, "quality_status": "reference" if configuration == "b0" else "pending"}
        results.append(result)
        if configuration != "b0" and not args.timing_first and not qualify(result, data, args.output):
            continue

        report = args.output / f"{configuration}-timing.json"
        log = args.output / f"{configuration}-timing.log"
        code = run_cli(
            "baseline.py",
            [
                args.clips / args.timing_clip, "--configuration", configuration, "--seed", seed,
                "--warmup", timing["model_warmup_calls"], "--outputs", timing["candidate_timed_outputs"],
                "--repeat-windows", "--report", report,
            ],
            log,
        )
        if code != 0:
            result.update(status="timing_error", log=log.name)
            if configuration == "b0":
                break
            continue
        measured = read_json(report)
        result.update(
            status="measured",
            timing_report=report.name,
            server_p50_ms=measured["server_window_ms"]["p50"],
            gpu_p50_ms=measured["gpu_region_ms"]["p50"],
        )
    ranked = sorted((result for result in results if "timing_report" in result), key=lambda row: row["server_p50_ms"])
    qualified = None
    for result in ranked:
        if result["quality_status"] == "pending":
            print(f"Quality-screening {result['configuration']}", flush=True)
            qualify(result, data, args.output)
        if result["quality_status"] in {"passed", "reference"}:
            qualified = result["configuration"]
            break
    return {
        "scope": "standard offline configuration screening",
        "screen_order": "timing first, then quality in latency order" if args.timing_first else "quality before timing",
        "development_clips": [path.name for path in data],
        "timing_clip": args.timing_clip,
        "timing_seed": seed,
        "timing_sigma_8bit": 25,
        "results": results,
        "fastest_measured_configuration": ranked[0]["configuration"] if ranked else None,
        "fastest_quality_validated_configuration": qualified,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("data/davis-development"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "development-clips.json")
    parser.add_argument("--timing-clip", default="drift-chicane")
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS[1:], default=EAGER_CONFIGURATIONS[1:])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-first", action="store_true", help="Rank timings before running the quality gate on the fastest options")
    args = parser.parse_args()
    if len(set(args.configurations)) != len(args.configurations):
        parser.error("Configurations must be unique")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory for each sweep")
    try:
        manifest = read_json(args.manifest)
        data = []
        for clip in manifest["clips"]:
            directory = args.clips / clip["name"]
            paths = frame_paths(directory)
            if len(paths) != clip["frames"] or sequence_digest(paths) != clip["ordered_frame_sha256"]:
                parser.error(f"Frames differ from the pinned manifest in {directory}")
            data.append(directory)
        if args.timing_clip not in {path.name for path in data}:
            parser.error("The timing clip must be one of the pinned development clips")
        report = run(args, data, read_json(ROOT / "protocol.json"))
        target = args.output / "summary.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Saved {target}")
    return int(any(result["status"].endswith("_error") for result in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
