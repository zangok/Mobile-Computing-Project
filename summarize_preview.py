"""Join saved phone display measurements to laptop events by session and frame ID."""

import argparse
import json
from pathlib import Path

from baseline import summarize


def startup_summary(events, report):
    first_rtp = next((row for row in events if row["event"] == "startup_rtp"
                      and row["inbound_rtp_packets"] > 0), None)
    first_decode = next((row for row in events if row["event"] == "first_decoded_input"), None)
    first_tagged = next((row for row in events if row["event"] == "received"), None)
    first_encoded = next((row for row in report["rtp"] if (row.get("out") or [None])[0]), None)
    first_sent = next((row for row in report["rtp"] if len(row.get("out") or []) >= 4 and row["out"][3]), None)
    origin = report.get("first_submitted_ms")

    def browser_offset(row):
        return row["ms"] - origin if row is not None and origin is not None else None

    return {
        "rtp_sample_interval_ms": 250,
        "first_encoded_sample_after_phone_submission_ms": browser_offset(first_encoded),
        "first_send_sample_after_phone_submission_ms": browser_offset(first_sent),
        "first_decoded_ms_after_first_positive_server_rtp_sample":
            (first_decode["monotonic_ns"] - first_rtp["monotonic_ns"]) / 1e6
            if first_rtp is not None and first_decode is not None else None,
        "first_tagged_frame_id": first_tagged["frame_id"] if first_tagged is not None else None,
    }


def read_summary(path, requested_session=None, replay_window=None):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    candidates = [item for item in events if item["event"] == "browser_diagnostics"
                  and (requested_session is None or item["session"] == requested_session)]
    if not candidates:
        raise ValueError("No saved Safari diagnostics found")
    saved = candidates[-1]
    report = saved["report"]
    if replay_window is not None:
        return score_replay_window(saved["session"], report, *replay_window)
    session_events = [item for item in events if item["session"] == saved["session"]]
    processed = {item["frame_id"]: item for item in session_events if item["event"] == "processed"}
    sent = {item["frame_id"]: item["server_residence_ms"] for item in session_events
            if item["event"] == "sent_to_encoder" and "server_residence_ms" in item}
    columns = {}
    for frame_id, preview_ms, late_ms, decode_ms, *_ in report["display"]:
        if frame_id not in processed or frame_id not in sent:
            continue
        model = processed[frame_id]
        values = {
            "submission_to_expected_display_ms": preview_ms,
            "display_callback_vs_expected_ms": late_ms,
            "server_lookahead_ms": model["lookahead_ms"],
            "server_queue_ms": model["queue_ms"],
            "server_processing_ms": model["server_ms"],
            "server_residence_ms": sent[frame_id],
            "outside_server_residence_ms": preview_ms - sent[frame_id],
            "browser_reported_decode_processing_ms": decode_ms,
        }
        for name, value in values.items():
            if value is not None:
                columns.setdefault(name, []).append(value)
    return {
        "session": saved["session"],
        "display_samples": len(report["display"]),
        "matched_samples": len(columns.get("server_residence_ms", [])),
        "rtp_snapshots": len(report["rtp"]),
        "startup": startup_summary(session_events, report),
        "timings": {
            name: {"samples": len(values), **{k: v for k, v in summarize(values).items() if k != "samples"}}
            for name, values in columns.items()
        },
    }


def score_replay_window(session, report, warmup, duration):
    if report.get("source") != "replay" or not report.get("replay"):
        raise ValueError("Fixed-window scoring requires a saved prerecorded replay")
    ingestion = report.get("ingestion", [])
    start = round(warmup * 1000)
    end = round((warmup + duration) * 1000)
    if not ingestion or ingestion[0][1] > start or ingestion[-1][1] < end + 100:
        raise ValueError("The saved ingestion trace must cover the complete scored window and 100 ms of drain")
    stride = 30 // report["output_fps"]
    submitted = {frame_id for frame_id, elapsed, *_ in ingestion if frame_id >= 2 and start <= elapsed < end}
    eligible = {frame_id for frame_id in submitted if frame_id % stride == 0}
    if not eligible:
        raise ValueError("No eligible submitted frames were logged in the scored window")
    displayed = {sample[0]: sample[1] for sample in report["display"] if sample[0] in eligible}
    return {
        "session": session,
        "measurement": "Safari submission to expected display",
        "replay": report["replay"],
        "output_fps": report["output_fps"],
        "codec": report.get("codec"),
        "warmup_seconds": warmup,
        "scored_seconds": duration,
        "submitted_frames": len(submitted),
        "submitted_fps": len(submitted) / duration,
        "eligible_frames": len(eligible),
        "displayed_frames": len(displayed),
        "displayed_fps": len(displayed) / duration,
        "finish_reason": report.get("finish_reason", "manual"),
        "eligible_delivery_percent": 100 * len(displayed) / len(eligible),
        "input_delivery_percent": 100 * len(displayed) / len(submitted),
        "preview_delay_ms": summarize(list(displayed.values())) if displayed else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--replay-seconds", type=float)
    parser.add_argument("--warmup-seconds", type=float, default=10)
    args = parser.parse_args()
    window = None
    if args.replay_seconds is not None:
        if args.replay_seconds <= 0 or args.warmup_seconds < 0:
            parser.error("Replay duration must be positive and warmup must be nonnegative")
        window = (args.warmup_seconds, args.replay_seconds)
    try:
        print(json.dumps(read_summary(args.log, args.session, window), indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
