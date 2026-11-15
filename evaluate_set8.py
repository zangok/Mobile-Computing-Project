"""Verify the staged Set8 bundle and run the fixed final quality evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def verify(root):
    vault = root.parent
    manifest = json.loads((vault / "set8-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned evaluator file changed: {name}")
    clips = []
    for clip in manifest["sequences"]:
        directory = vault / "frames" / clip["sequence"]
        paths = sorted(directory.iterdir())
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        if len(paths) != clip["frames"] or digest.hexdigest() != clip["sha256"]:
            raise ValueError(f"Pinned frame sequence changed: {clip['sequence']}")
        clips.append(directory)
    if len(clips) != 8:
        raise ValueError("The final quality evaluation requires all eight Set8 clips")
    return manifest, clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--configuration")
    parser.add_argument("--candidate-upstream", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.verify_only and not (args.configuration and args.candidate_upstream and args.report):
        parser.error("Final scoring requires a configuration, candidate model directory, and report path")
    try:
        manifest, clips = verify(ROOT)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    if args.verify_only:
        print("Verified the reference evaluator and all eight held-out Set8 sequences")
        return 0
    return subprocess.run([
        sys.executable, str(ROOT / "quality.py"), *map(str, clips),
        "--upstream", str(ROOT / "reference"), "--candidate", args.configuration,
        "--candidate-upstream", str(args.candidate_upstream),
        "--seed", *map(str, manifest["final_seeds"]), "--max-frames", "85",
        "--report", str(args.report),
    ]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
