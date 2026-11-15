"""Build a fixed 640 by 360, 30 fps Safari development replay from pinned DAVIS frames."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from baseline import frame_paths, sha256
from prepare_davis import sequence_digest


def build(ffmpeg, clips, manifest, output):
    all_clips = []
    for item in manifest["clips"]:
        directory = clips / item["name"]
        paths = frame_paths(directory)
        if len(paths) != item["frames"] or sequence_digest(paths) != item["ordered_frame_sha256"]:
            raise ValueError(f"Development clip differs from the pinned manifest: {directory}")
        all_clips.append(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        segment = Path(temporary) / "segment.mp4"
        command = [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-filter_complex_threads", "1"]
        for directory in all_clips:
            command += ["-framerate", "30", "-start_number", "0", "-i", str(directory / "%05d.jpg")]
        filters = [f"[{i}:v]scale=640:360:flags=bilinear,setsar=1,format=yuv420p[v{i}]"
                   for i in range(len(all_clips))]
        filters.append("".join(f"[v{i}]" for i in range(len(all_clips))) + f"concat=n={len(all_clips)}:v=1:a=0[out]")
        subprocess.run([*command, "-filter_complex", ";".join(filters), "-map", "[out]", "-c:v", "libx264",
                        "-threads", "2", "-crf", "20", "-r", "30", str(segment)], check=True)
        subprocess.run([str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-stream_loop", "-1",
                        "-i", str(segment), "-t", "75", "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart",
                        str(output)], check=True)
    return {
        "source": "DAVIS 2017 480p development clips in manifest order, repeated",
        "clip_order": [path.name for path in all_clips],
        "width": 640, "height": 360, "fps": 30, "requested_seconds": 75,
        "file": output.name, "sha256": sha256(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--clips", type=Path, default=Path("data/davis-development"))
    parser.add_argument("--replay", type=Path, default=Path("data/development-replay.mp4"))
    parser.add_argument("--manifest", type=Path, default=Path("development-clips.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = build(args.ffmpeg.resolve(), args.clips, manifest, args.replay)
    args.replay.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.replay}")


if __name__ == "__main__":
    main()
