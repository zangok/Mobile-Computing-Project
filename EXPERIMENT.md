# Pre-experiment run sheet

Run the initial experiment on the native Windows CUDA environment used by the iPhone service. Keep the WSL measurements as feasibility results. Close unrelated GPU workloads and record the power mode, driver, package versions, iPhone and iOS versions, negotiated codec, and network conditions for each comparison.

## Baseline and reproducible replay

1. Use the offline sweep and its evidence to record B1 in `b1-manifest.json`. Keep the source hashes, checkpoint, model execution settings, and development inputs together.
2. Run B1 with 640 × 360 model input, 30 fps capture and output, the fixed five-frame history, and VP8 if the phone negotiates it. Verify the 2 Mbit/s ceiling on both legs and record measured rates. Keep the 15 fps mode labeled as a fallback.
3. In Safari, use **Start replay** with the pinned development MP4. The browser finishes downloading the file before opening the media connection. Leave the page open until it saves diagnostics automatically. Score a 30-second window after the 10-second warm-up for development, and a 60-second window after the same warm-up for final paired trials. Use the same video digest and scored interval on every comparison.
4. Use **Start paired replays** on the batch-enabled preview before candidate trials to characterize timing variation. Each set includes one B1 and one pass-through replay. Successive sets reverse the order. Keep Safari in the foreground until **Batch complete**. The server saves each run immediately and exports the 30- and 60-second measurements under its event log's `batches/` directory. The project can proceed with a measured shortfall against the 97% delivery and 200 ms p95 targets if the instrumentation works. Report the shortfall.

## Protocol amendment on November 15 2026

Use repeated offline server timings and correctness checks to accept or reject development candidates. Move the iPhone preview comparison to the final paired evaluation of selected candidates against B1. This change applies to both the one-shot and iterative Codex conditions and was recorded before formal trials.

The original development gate allowed a candidate to lose at most one percentage point of frame delivery or increase p95 preview delay by at most 5%. In the three matched 60-second pass-through pilot runs, delivery ranged from 20.4% to 62.6% and p95 delay ranged from 345 to 8,077 ms. That variation occurred with denoising bypassed and makes the original gate unreliable for individual submissions. [Paired pilot results](evidence/b1/iphone-pilot.json)

Keep the 640 × 360 workload at 30 fps, the pinned B1, the numerical and quality thresholds, and the one-shot versus iterative comparison. Report final iPhone delivery and p95 against B1 and the original deployment targets of 97% delivery and 200 ms p95. Keep any fallback measurements labeled separately.

## Evaluator and acceptance

- Use the numerical checker against B1 and the four DAVIS development clips against B0 for quality. Retain failures and the tested configuration.
- Accept a development candidate when it passes the numerical and quality checks and reduces median server processing time by at least 10% against B1 across repeated paired measurements. Keep the raw reports alongside the acceptance result.
- Use `candidate_gate.py` with the paired offline timing reports and both correctness reports. Keep final paired Safari trials as a separate evaluation of the selected candidates.
- Keep the staged Set8 bundle outside each candidate checkout. Use its verification command before final scoring. Candidate session tools must be tested for access isolation before using any held-out results. Keep the final evaluation runs and reports out of both conditions' working context.

## Codex condition pilots

- Use the `b1-baseline` Git tag as the starting point for every independent session. For example, create branches `codex/n-pilot` and `codex/a-pilot` from that tag. Keep each iterative submission as a distinct commit on its session branch. Start each one-shot proposal from the same tag and retain its patch and commit ID for evaluation.
- Prepare one fixed task packet with the frozen B1 source, configuration, baseline profile, allowed edit surface, objectives, and development validation commands. Start both conditions from that packet and give each session a fresh candidate copy.
- **N** collects separate one-shot proposals before giving the agent candidate feedback. Evaluate the collected proposals in the researcher context.
- **A** allows profiling, testing, and revision against development inputs during the session. The independent evaluator scores each versioned submission.
- Pilot both conditions before starting the three sessions per condition prescribed in the protocol. Check the five-candidate and 45-minute limits, including failed submissions, review command and decision logs, and confirm B1 can be restored. Keep pilot attempts out of the final comparison.
