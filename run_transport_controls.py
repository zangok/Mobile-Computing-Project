"""Compare transport-only and B1 loopback on identical pinned replay frames."""

import argparse
import asyncio
import json
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.request import ProxyHandler, build_opener

from baseline import sha256
from replay_preview import read_video, run

ROOT = Path(__file__).resolve().parent
CODECS = ("VP8", "H264")


def wait_for_server(process, url, codec, configuration):
    deadline = monotonic() + 60
    opener = build_opener(ProxyHandler({}))
    while process.poll() is None and monotonic() < deadline:
        try:
            with opener.open(f"{url}/health", timeout=2) as response:
                health = json.load(response)
        except OSError:
            sleep(0.5)
            continue
        if (health["codec"], health["configuration"], health["output_fps"]) != (codec, configuration, 30):
            raise RuntimeError("Unexpected loopback server settings")
        return
    raise RuntimeError("The loopback server did not become ready")


def stop_server(process):
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        # A venv's Windows Python launcher can leave its interpreter behind
        # if only the launcher PID is terminated.
        subprocess.run(["taskkill", "/pid", str(process.pid), "/t", "/f"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def measure(images, args, codec, configuration):
    # Bind only on loopback and let the OS choose an unused pilot port.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    label = f"{codec.lower()}-{configuration}"
    with (args.output / f"{label}.log").open("w", encoding="utf-8") as log:
        command = [
            sys.executable, "-u", str(ROOT / "server.py"), "--host", "127.0.0.1", "--port", str(port),
            "--codec", codec, "--configuration", configuration, "--output-fps", "30", "--warmup", "200",
            "--events", str(args.output / f"{label}-events.jsonl"),
        ]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for_server(process, url, codec, configuration)
            report = asyncio.run(run(url, images, args.frames, args.warmup_frames))
        finally:
            stop_server(process)
    target = args.output / f"{label}.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "codec": codec, "configuration": configuration, "report": target.name,
        "submitted": report["eligible_input_ids"], "returned": report["decoded_eligible_ids"],
        "delivery_percent": report["input_delivery_percent"],
        "server_input_gaps": report["server"]["frames"]["missing_input_ids"],
        "pending_replaced": report["server"]["frames"]["pending_replaced"],
        "decode_p95_ms": report["decode_delay_ms"]["p95"] if report["decode_delay_ms"] else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--warmup-frames", type=int, default=300)
    args = parser.parse_args()
    if args.frames < 5 or args.warmup_frames < 0:
        parser.error("Frames must be at least five and warmup cannot be negative")
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output = args.output.resolve()
    baseline = json.loads((ROOT / "b1-manifest.json").read_text(encoding="utf-8"))
    video = ROOT / "data/development-replay.mp4"
    digest = sha256(video)
    if baseline["development_replay_sha256"] != digest:
        parser.error("The replay does not match the pinned B1 manifest")
    images = read_video(video, args.frames + args.warmup_frames + 4)
    args.output.mkdir(parents=True)
    comparisons = []
    for codec in CODECS:
        for configuration in ("passthrough", baseline["configuration"]):
            print(f"Running {codec} {configuration}", flush=True)
            result = measure(images, args, codec, configuration)
            comparisons.append(result)
            print(json.dumps(result), flush=True)
    summary = {"scope": "laptop loopback control", "replay_sha256": digest,
               "warmup_frames": args.warmup_frames, "scored_frames": args.frames,
               "results": comparisons}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
