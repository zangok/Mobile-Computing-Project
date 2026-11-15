"""Pin the official Set8 archives and stage the independent quality evaluator."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import BadZipFile, ZipFile

from baseline import upstream_files
from prepare_davis import sequence_digest

ROOT = Path(__file__).resolve().parent
SOURCE = "https://drive.google.com/drive/folders/11chLkbcX-oKGLOLONuDpXZM2-vujn_KD"
ARCHIVES = ("derf_540p_seqs.zip", "gopro_540p.zip")
SEQUENCES = {
    "tractor": {"tractor"},
    "touchdown": {"touchdown"},
    "park_joy": {"parkjoy"},
    "sunflower": {"sunflower"},
    "hypersmooth": {"hypersmooth", "hypesmooth"},
    "motorbike": {"motorbike"},
    "rafting": {"rafting"},
    "snowboard": {"snowboard"},
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing pinned data differ at {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def frame_number(member):
    names = re.findall(r"\d+", Path(member).stem)
    if not names:
        raise ValueError(f"Set8 member has no frame number: {member}")
    return int(names[-1])


def prepare(vault, upstream):
    if vault.resolve().is_relative_to(ROOT):
        raise ValueError("Place Set8 outside the application workspace")
    groups = {sequence: [] for sequence in SEQUENCES}
    archives = [vault / name for name in ARCHIVES]
    records = []
    with ExitStack() as stack:
        opened = {path: stack.enter_context(ZipFile(path)) for path in archives}
        for path, archive in opened.items():
            for member in archive.namelist():
                parts = Path(member).parts
                if Path(member).suffix.lower() not in {".png", ".jpg", ".jpeg"} or "__MACOSX" in parts:
                    continue
                parents = {re.sub(r"[^a-z0-9]", "", directory.lower()) for directory in parts[:-1]}
                for sequence, aliases in SEQUENCES.items():
                    if parents & aliases:
                        groups[sequence].append((frame_number(member), path, member))

        if any(len(group) < 5 or len({row[0] for row in group}) != len(group) for group in groups.values()):
            raise ValueError("Expected eight uniquely numbered Set8 image sequences")
        for sequence, group in groups.items():
            output = []
            for ordinal, (_, path, member) in enumerate(sorted(group)[:85]):
                target = vault / "frames" / sequence / f"{ordinal:05}{Path(member).suffix.lower()}"
                write_once(target, opened[path].read(member))
                output.append(target)
            if set(output) != set(output[0].parent.iterdir()):
                raise ValueError(f"Unexpected files in the pinned sequence {sequence}")
            records.append({"sequence": sequence, "frames": len(output), "sha256": sequence_digest(output)})

    evaluator = vault / "evaluator"
    sources = [ROOT / name for name in ("baseline.py", "pipeline.py", "quality.py", "protocol.json", "evaluate_set8.py")]
    copied = {evaluator / path.name: path for path in sources}
    copied.update({evaluator / "reference" / path.name: path for path in upstream_files(upstream)})
    for target, source in copied.items():
        write_once(target, source.read_bytes())
    manifest = {
        "source": SOURCE,
        "archives_sha256": {path.name: file_hash(path) for path in archives},
        "files_sha256": {path.relative_to(evaluator).as_posix(): file_hash(path)
                         for path in copied},
        "sequences": records,
        "final_seeds": json.loads((evaluator / "protocol.json").read_text(encoding="utf-8"))["workload"]["final_seeds"],
    }
    write_once(vault / "set8-manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vault", type=Path)
    parser.add_argument("--upstream", type=Path, default=ROOT / "checkpoints/fastdvdnet")
    args = parser.parse_args()
    try:
        manifest = prepare(args.vault, args.upstream)
    except (OSError, ValueError, BadZipFile) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Pinned {len(manifest['sequences'])} Set8 sequences and staged the evaluator outside the workspace")


if __name__ == "__main__":
    main()
