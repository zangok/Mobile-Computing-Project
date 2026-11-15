"""Collect counterbalanced B1 and pass-through replay measurements."""

import csv
import json
from pathlib import Path
import re
from statistics import median

from summarize_preview import score_replay_window

MODES = ("b1", "passthrough")


def trial_metadata(value):
    if (not isinstance(value, dict) or not isinstance(value.get("id"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", value["id"])
            or type(value.get("pairs")) is not int or not 1 <= value["pairs"] <= 5
            or type(value.get("index")) is not int or not 0 <= value["index"] < 2 * value["pairs"]):
        raise ValueError("Invalid replay batch trial")
    pair, position = divmod(value["index"], 2)
    return {"id": value["id"], "pairs": value["pairs"], "index": value["index"],
            "pair": pair + 1, "mode": MODES[(pair + position) % 2]}


def aggregate(trials):
    rows = []
    for duration in (30, 60):
        for mode in MODES:
            windows = [trial["windows"][str(duration)] for trial in trials
                       if trial["mode"] == mode and str(duration) in trial["windows"]]
            if not windows:
                continue
            point = {"mode": mode, "scored_seconds": duration, "runs": len(windows)}
            for field, values in (
                ("delivery_percent", [window["input_delivery_percent"] for window in windows]),
                ("preview_p95_ms", [window["preview_delay_ms"]["p95"] for window in windows
                                    if window["preview_delay_ms"] is not None]),
            ):
                point[field] = {"min": min(values), "median": median(values), "max": max(values),
                                "runs": len(values)} if values else None
            rows.append(point)
    return rows


def save_trial(output, trial, session, report, replay_digest):
    if (report.get("source") != "replay" or report.get("finish_reason") != "replay-ended"
            or not isinstance(report.get("replay"), dict)
            or report["replay"].get("sha256") != replay_digest):
        raise ValueError("Batch scoring requires the complete pinned replay")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"batch-{trial['id']}.json"
    batch = (json.loads(target.read_text(encoding="utf-8")) if target.exists() else
             {"batch_id": trial["id"], "pairs": trial["pairs"], "replay_sha256": replay_digest, "trials": []})
    if batch["pairs"] != trial["pairs"] or batch["replay_sha256"] != replay_digest:
        raise ValueError("Batch settings changed between trials")
    if trial["index"] < len(batch["trials"]):
        if batch["trials"][trial["index"]]["session"] == session:
            return batch
        raise ValueError("This batch trial is already saved")
    if trial["index"] != len(batch["trials"]):
        raise ValueError("Batch trials must be saved in order")
    saved = {**trial, "session": session, "windows": {}, "errors": {}}
    for length in (30, 60):
        try:
            score = score_replay_window(session, report, 10, length)
            if score["preview_delay_ms"]:
                score["preview_delay_ms"].pop("samples")
            saved["windows"][str(length)] = score
        except ValueError as error:
            saved["errors"][str(length)] = str(error)
    batch["trials"].append(saved)
    batch["aggregate"] = aggregate(batch["trials"])
    batch["valid_runs"] = sum(not item["errors"] for item in batch["trials"])
    batch["complete"] = len(batch["trials"]) == batch["pairs"] * 2
    pending = target.with_suffix(".tmp")
    pending.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    pending.replace(target)
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("trial", "pair", "mode", "session", "scored_seconds", "submitted",
                         "displayed", "delivery_percent", "preview_p95_ms", "error"))
        for item in batch["trials"]:
            for duration in ("30", "60"):
                score = item["windows"].get(duration, {})
                delay = score.get("preview_delay_ms") or {}
                writer.writerow((item["index"] + 1, item["pair"], item["mode"], item["session"], duration,
                                 score.get("submitted_frames"), score.get("displayed_frames"),
                                 score.get("input_delivery_percent"), delay.get("p95"), item["errors"].get(duration)))
    return batch
