"""Record the selected offline baseline and its reproducibility evidence."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from baseline import sha256, upstream_files

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("baseline.py", "pipeline.py", "server.py", "frame_tag.py", "quality.py",
                "validate_model.py", "replay_preview.py", "preview.html", "preview.js",
                "summarize_preview.py", "candidate_gate.py", "requirements-server.txt",
                "protocol.json", "development-clips.json")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def create(sweep_path, environment, upstream):
    sweep = read(sweep_path)
    name = sweep.get("fastest_quality_validated_configuration")
    if not name:
        raise ValueError("The sweep has no measured and quality-validated configuration")
    selected = next(row for row in sweep["results"] if row["configuration"] == name)
    timing = sweep_path.parent / selected["timing_report"]
    quality = sweep_path.parent / selected["quality_report"] if "quality_report" in selected else None
    if name != "b0" and (not quality or read(quality).get("passed") is not True):
        raise ValueError("The selected configuration is missing a passing quality report")
    model = {path.name: sha256(path) for path in upstream_files(upstream)}
    measured = read(timing)
    if measured["fastdvdnet_files_sha256"] != model:
        raise ValueError("Current upstream files differ from the benchmarked files")
    return {
        "scope": "offline B1 selection",
        "configuration": name,
        "gpu": measured["gpu"],
        "torch": measured["torch"],
        "cuda_build": measured["cuda_build"],
        "server_p50_ms": measured["server_window_ms"]["p50"],
        "gpu_p50_ms": measured["gpu_region_ms"]["p50"],
        "evidence_sha256": {str(path.relative_to(ROOT)): sha256(path)
                            for path in (sweep_path, environment, timing, quality) if path is not None},
        "application_files_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "development_replay_sha256": sha256(ROOT / "data/development-replay.mp4")
        if (ROOT / "data/development-replay.mp4").is_file() else None,
        "fastdvdnet_files_sha256": model,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=ROOT / "checkpoints/fastdvdnet")
    parser.add_argument("--output", type=Path, default=ROOT / "b1-manifest.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "runs/b1-snapshot.zip")
    args = parser.parse_args()
    try:
        report = create(args.sweep.resolve(), args.environment.resolve(), args.upstream)
        if args.output.exists() or args.snapshot.exists():
            raise ValueError("Choose unused manifest and snapshot paths")
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.snapshot, "x", compression=ZIP_DEFLATED) as archive:
            for name in SOURCE_FILES:
                archive.write(ROOT / name, name)
            for path in upstream_files(args.upstream):
                archive.write(path, f"checkpoints/fastdvdnet/{path.name}")
            archive.write(args.environment, "environment.txt")
        report["source_snapshot_sha256"] = sha256(args.snapshot)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Recorded offline B1 configuration {report['configuration']} in {args.output} and {args.snapshot}")


if __name__ == "__main__":
    main()
