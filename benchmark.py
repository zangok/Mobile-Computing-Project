"""Run a dependency-free, synthetic check of the offline evaluation harness."""

import argparse
import json
import math
import platform
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

from pipeline import Frame, FrameHistory, process_window


def passthrough(payloads):
    return payloads[len(payloads) // 2]


def run_smoke(context: int, warmup: int, outputs: int) -> dict:
    if warmup < 0 or outputs < 1:
        raise ValueError("Warmup must be nonnegative and outputs must be positive")

    history = FrameHistory[int](context)
    timings_ms = []
    first_center = context // 2
    input_count = warmup + outputs + context - 1
    for frame_id in range(input_count):
        window = history.push(Frame(frame_id, frame_id))
        if window is None:
            continue

        start = perf_counter_ns()
        result = process_window(window, passthrough)
        elapsed_ms = (perf_counter_ns() - start) / 1_000_000
        expected_center = frame_id - first_center
        if result.frame_id != expected_center or result.payload != expected_center:
            raise RuntimeError(f"Incorrect center frame for input {frame_id}")
        if expected_center >= first_center + warmup:
            timings_ms.append(elapsed_ms)

    if len(timings_ms) != outputs:
        raise RuntimeError(f"Expected {outputs} timed outputs, got {len(timings_ms)}")
    return {
        "workload": "synthetic_passthrough",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "context": context,
        "input_count": input_count,
        "warmup_outputs": warmup,
        "timed_outputs": outputs,
        "first_timed_frame_id": first_center + warmup,
        "last_timed_frame_id": first_center + warmup + outputs - 1,
        "cpu_pipeline_p50_ms": statistics.median(timings_ms),
        "cpu_pipeline_p95_ms": sorted(timings_ms)[math.ceil(0.95 * outputs) - 1],
        "cpu_pipeline_samples_ms": timings_ms,
    }


def main():
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=protocol["timing"]["model_warmup_calls"])
    parser.add_argument("--outputs", type=int, default=protocol["timing"]["candidate_timed_outputs"])
    parser.add_argument("--report", type=Path, help="Write the full JSON report here")
    args = parser.parse_args()
    try:
        report = run_smoke(protocol["workload"]["temporal_context"], args.warmup, args.outputs)
    except ValueError as error:
        parser.error(str(error))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "cpu_pipeline_samples_ms"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
