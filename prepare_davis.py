"""Extract and pin four image sequences from the official DAVIS 2017 archive."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from baseline import frame_paths

SOURCE = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
CLIPS = ("blackswan", "bmx-trees", "camel", "drift-chicane")


def sequence_digest(paths):
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare(archive_path, output):
    records = []
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        split_path = next(
            (name for name in sorted(names) if name.endswith("/ImageSets/2017/val.txt")),
            "ImageSets/2017/val.txt",
        )
        validation = set(archive.read(split_path).decode("utf-8").split())
        if not set(CLIPS) <= validation:
            raise ValueError("The archive validation list does not contain the selected clips")
        for clip in CLIPS:
            members = sorted(
                name for name in names
                if name.endswith(".jpg") and name.split("/")[-4:-1] == ["JPEGImages", "480p", clip]
            )
            filenames = [Path(name).name for name in members]
            if len(filenames) < 5 or filenames != [f"{index:05}.jpg" for index in range(len(filenames))]:
                raise ValueError(f"Missing, duplicate or nonsequential frames in {clip}")
            directory = output / clip
            directory.mkdir(parents=True, exist_ok=True)
            if any(path.name not in filenames for path in frame_paths(directory)):
                raise ValueError(f"Unexpected image files already present in {directory}")
            for member, filename in zip(members, filenames):
                image = archive.read(member)
                target = directory / filename
                if target.exists() and target.read_bytes() != image:
                    raise ValueError(f"Existing frame differs from the archive: {target}")
                if not target.exists():
                    target.write_bytes(image)
            records.append({
                "name": clip,
                "frames": len(filenames),
                "ordered_frame_sha256": sequence_digest(frame_paths(directory)),
            })
    return {"source": SOURCE, "split": "DAVIS 2017 480p validation", "clips": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/davis-development"))
    parser.add_argument("--manifest", type=Path, default=Path("development-clips.json"))
    args = parser.parse_args()
    manifest = prepare(args.archive, args.output)
    if args.manifest.exists():
        if json.loads(args.manifest.read_text(encoding="utf-8")) != manifest:
            parser.error(f"Extracted frames differ from the pinned manifest {args.manifest}")
    else:
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(manifest['clips'])} validation clips in {args.output}")


if __name__ == "__main__":
    main()
