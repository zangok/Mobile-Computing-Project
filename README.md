# Profiling Guided Video Denoising

The [B1 evidence bundle](evidence/b1/README.md) contains the five offline reference runs, development quality results, and the iPhone repeatability pilot. Start independent Codex branches from the `b1-baseline` Git tag. The [experiment run sheet](EXPERIMENT.md) defines the acceptance rules and branch workflow.

This course project studies whether Codex can use profiling and repeated validation to improve the performance of a fixed FastDVDnet video-denoising application on an RTX laptop. The planned application receives iPhone video over WebRTC and returns a denoised preview.

## Run the offline smoke check

From this directory with Python 3.10 or later:

```powershell
python benchmark.py --report runs/smoke.json
```

The harness uses synthetic integer payloads and a pass-through model adapter. It checks that each complete five-frame window returns the middle frame's ID and payload. By default it warms up for 200 outputs, times the next 200, and writes the raw CPU timings to the JSON report. These timings characterize the smoke check only.

- `pipeline.py` assembles ordered frame windows and hands their payloads to a model adapter. The caller supplies frames in stream order. The offline replay emits only complete windows, with no boundary padding.
- `benchmark.py` exercises that interface and records CPU timing and sample counts.
- `protocol.json` records the proposal's starting workload, timing, quality, and candidate-selection thresholds. Freeze it with the source and evaluator after the pilot.

## Run eager FP32 FastDVDnet

The baseline uses the [official FastDVDnet implementation and checkpoint](https://github.com/m-tassano/fastdvdnet/tree/c8fdf6182a0340e89dd18f5df25b47337cbede6f). Use an environment with PyTorch 2.7.1, Pillow, and NumPy. GPU measurements require a CUDA-enabled PyTorch build. The Windows Python environment used for the functional check has PyTorch 2.7.1+cpu, Pillow 11.0.0, and NumPy 2.2.2.

```powershell
git clone https://github.com/m-tassano/fastdvdnet.git checkpoints/fastdvdnet
git -C checkpoints/fastdvdnet checkout --detach c8fdf6182a0340e89dd18f5df25b47337cbede6f
python baseline.py data/pilot-clip --warmup 200 --outputs 200 --report runs/b0-cuda.json --trace runs/b0-trace.json
```

The checkpoint SHA-256 is `8118974ac7defaa5037f73caf87e0cb53efcfa49ae77d55c05ab187f59e55949`. The report records hashes for the checkpoint and both upstream source files. Supply numbered PNG or JPEG images in lexical order. The example requires at least 404 frames. For a functional check, add `--device cpu` and use fewer warm-up and timed outputs.

Replay decodes and resizes images before timing, then applies seeded, clipped Gaussian noise. Server window timing includes packing, inference, and transfers. CUDA events time the padded model region, and the report includes peak allocated and reserved device memory. The optional trace profiles one extra window after measurement.

## Reproduce the CUDA feasibility pilot

In Ubuntu WSL2, from the project directory, create a dedicated environment:

```bash
python3 -m venv ~/.venvs/6301-fastdvdnet
source ~/.venvs/6301-fastdvdnet/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install pillow==11.0.0 numpy==2.2.2
```

The pilot uses the 300-frame Foreman CIF clip from [Xiph test media](https://media.xiph.org/video/derf/), downloaded as `data/foreman_cif.y4m`. Its SHA-256 is `b0e3c35eff9fe1da91521a02eb9f0e3ef4bc5aec341ecef1567cd87f1711267b`. Decode the frames with FFmpeg 7.1. The replay runner resizes the decoded RGB frames to 640 × 360 with bilinear interpolation.

```bash
mkdir -p data/foreman
ffmpeg -i data/foreman_cif.y4m -start_number 0 data/foreman/%06d.png
python baseline.py data/foreman --warmup 200 --outputs 96 --report runs/b0-foreman-cuda.json --trace runs/b0-foreman-trace.json
```

This single run checks execution, memory fit, and profiler capture with the target input dimensions. The formal development and evaluation runs use the workloads and sample counts specified in `protocol.json`.

The pilot completed on the RTX 3050 using PyTorch 2.7.1+cu126. Across 96 timed outputs, GPU region latency was 86.8 ms p50 and 87.3 ms p95. Server window latency was 90.5 ms p50 and 93.3 ms p95. Peak PyTorch GPU memory was 217 MiB allocated and 278 MiB reserved. The eager FP32 reference exceeds the proposed 25 ms GPU service target. The report and trace are under `runs/`, alongside the package snapshot `b0-foreman-environment.txt`.

## Check paired reconstruction quality

From the Ubuntu environment above, run a short check on the feasibility clip:

```bash
python quality.py data/foreman --candidate fp16 --max-frames 12 --seed 11 --report runs/quality-foreman-fp16.json
```

Pass one directory per clip. By default the evaluator uses noise standard deviations 10, 25, and 50 and all three development seeds. Both models receive the same seeded, clipped noisy frames and are scored against the resized clean RGB frames before transport. The evaluator excludes the first and last two frames of each clip. It computes PSNR over RGB with a 120 dB ceiling for near-exact matches and averages SSIM over RGB channels and valid spatial locations using an 11 × 11 Gaussian window with sigma 1.5 and population covariance.

The report averages frame scores within each seed, then weights seeds equally within each clip and noise level. The overall gate weights clip and noise pairs equally, with separate limits for each pair. Exit status 0 indicates a pass, 2 indicates a quality rejection with a written report, and 1 indicates a runtime or input-file error. Use `--candidate b0` for an identity control and `--candidate noisy` to exercise rejection. This gate checks reconstruction quality against B0. The later B1 equivalence check will use the separate numerical tolerances in the protocol.

On Foreman, FP16 passed this short pilot at all three noise levels. Across eight scored frames per level and one seed, its mean losses versus FP32 were 0.00020 dB PSNR and 0.000020 SSIM.

## Sweep the eager configurations on DAVIS

Use the official [DAVIS 2017 TrainVal 480p archive](https://davischallenge.org/davis2017/code.html). Extract the four selected sequences and verify them against the pinned development manifest:

```bash
python prepare_davis.py data/DAVIS-2017-trainval-480p.zip
python sweep.py --output runs/davis-eager
```

The default sweep tests FP32 inference mode, FP32 with cuDNN autotuning, TF32 with autotuning, and FP16 with autotuning, including channels-last layout and reusable packing buffers. Each candidate must pass the paired gate on all four clips with all three noise levels and seeds. The timing run uses the same drift-chicane windows for each configuration at sigma 25 and seed 11. It cycles complete windows to reach 200 warm-up and 200 timed outputs without constructing windows across clip boundaries. The report ranks measured configurations by median server latency. Select a subset with `--configurations`. The optional `fp16-compile-default` and `fp16-compile-reduce-overhead` configurations compile the fixed-shape model. The benchmark logs first-window wall time separately from warmed samples.

The first development run compared B0 with FP16 plus cuDNN autotuning. FP16 passed all quality checks across 2,304 scored outputs. Mean losses were 0.00026 dB PSNR and 0.000039 SSIM. On the matched timing replay, median server latency fell from 90.9 ms to 57.6 ms, and median GPU latency fell from 86.6 ms to 54.5 ms. Reports are under `runs/davis-eager-fp16/`. The GPU service target is 25 ms, and B1 selection is still open.

Channels-last FP16 also passed the full DAVIS gate and reached 46.8 ms median GPU latency over 200 timed windows. Buffer reuse and both compiled modes ran successfully in 10-window smoke timings. The reduce-overhead mode recorded a CUDA graph and passed the short drift-chicane quality check. Its 39.6 ms GPU median is preliminary. Run the full timing and quality sweep before freezing B1.

The native Windows sweep selected `fp16-reuse` for offline B1. Median server time was 52.5 ms over 200 timed windows, compared with 99.7 ms for B0. It passed all four DAVIS clips at all three noise levels and seeds. The two compiled configurations had no working Triton installation in this environment. The frozen source and environment are recorded in `b1-manifest.json` and `runs/b1-snapshot.zip`. The Windows and WSL runs use separate environments.

## Run the laptop preview server

Run these from the project directory in the CUDA-enabled Ubuntu WSL2 environment:

```bash
source ~/.venvs/6301-fastdvdnet/bin/activate
python -m pip install -r requirements-server.txt
python server.py --configuration fp16-channels-last
```

The service binds to `127.0.0.1:8080` by default. It uses VP8 and one bidirectional video track with host ICE candidates. Channels-last FP16 passed the full DAVIS gate and is suitable for the integration pilot. Use `--configuration b0` for the FP32 reference, or `--configuration passthrough` to test only the media path. For the phone session, the server can take `--host`, `--port`, and the paths to a certificate and key via `--cert` and `--key`. Local network reachability and certificate trust are part of the phone setup.

In a second Ubuntu terminal, run a local replay:

```bash
source ~/.venvs/6301-fastdvdnet/bin/activate
python replay_preview.py --clip data/davis-development/drift-chicane --warmup-frames 300 --frames 900 --report runs/preview-replay.json
```

The replay loops the supplied clip as needed and timestamps submitted frames and decoded returns on the same laptop clock. The report contains eligible delivery, recovered IDs, invalid tags, duplicate IDs, and submission-to-decode delay. A few frames after the scored interval keep decoder jitter buffers advancing. The phone will measure expected display time on its own clock.

### Signaling and frame format

The client sends a WebRTC SDP offer as `{type, sdp}` to `POST /offer` and applies the returned SDP answer. Use one send/receive video transceiver and gather ICE before posting. `GET /health` shows server settings. `GET /stats` returns frame counters and cumulative RTP counters. Use `DELETE /session` when ending the preview. The server writes per-frame events to `runs/preview-events.jsonl` by default.

Each transmitted image is 640 × 384, with the 640 × 360 image at the top. The bottom 24 rows carry 48 black or white cells, each 13 pixels wide, starting at column 8. Encode bytes most significant bit first. The packet is the byte `0xD3`, a big-endian unsigned 32-bit frame ID, then CRC-8 (polynomial `0x07`, initial value zero, no reflection or final XOR) over the preceding five bytes. Start IDs at zero for each session and increment by one per submitted frame. The server crops the tag before inference and restores the middle frame's tag and media timestamp in the return.

The receiver drains input independently of the model worker. An input ID gap resets the five-frame history, and a newer ready window replaces the single pending window when the worker is busy. The service records pending and output replacements so replay delivery can be explained from the logs. The browser's upload limit and the negotiated codec and bitrate on the actual LAN still need verification.

The separate `--output-fps 15` fallback schedules even center IDs while consuming the full 30 fps input and keeping the same five-frame context. Replay reports delivery for both the requested outputs and all eligible inputs. Keep the default of 30 output fps for the primary experiment.

In laptop loopback, pass-through returned all 118 eligible IDs. Channels-last FP16 at 30 output fps returned 74 of 118, with 262 ms p95 submission-to-decode delay. At 15 output fps, it returned all 450 scheduled frames over a 30-second scored interval after a 10-second warm-up, with 243 ms p95 submission-to-decode delay. Those measurements are in `runs/loopback-*.json`. The 15 fps mode is useful for the phone integration pilot while the primary performance target remains open.

### Safari over Wi-Fi

The preview at `/` requests the rear camera and returns a tagged video stream. It measures ingestion to expected display using `requestVideoFrameCallback` on the phone's clock. For a private LAN pilot, run the server on the laptop's Wi-Fi IPv4 address with a certificate for that address. The native Windows environment is an option when Ubuntu WSL is using NAT. Its pilot measurements should be labeled with the Windows environment.

From PowerShell, set `$previewAddress` to the laptop's Wi-Fi IPv4 address and run:

```powershell
.\.venv\Scripts\python.exe make_preview_tls.py $previewAddress
.\.venv\Scripts\python.exe server.py --host $previewAddress --port 8443 --configuration fp16-channels-last --output-fps 15 --cert runs/preview-tls/server.pem --key runs/preview-tls/server-key.pem
```

The certificate generator writes the short-lived public CA certificate to `runs/preview-tls/public/preview-ca.cer` and discards the CA signing key. Transfer the public certificate to the phone and enable full trust for **6301 Preview Test CA** before opening the HTTPS preview. Remove the certificate profile from the phone after the pilot. [Apple's certificate trust instructions](https://support.apple.com/en-us/102390)

For a timing diagnostic, let the iPhone preview run for about 30 seconds, tap **Save diagnostics**, wait for confirmation, then tap **Stop**. This saves recent per-frame display timings and available WebRTC counters to the server's event log. The display samples contain the frame ID, submission-to-expected-display delay, callback offset from expected display, and browser-reported decode processing time where available. To join them to server timing, run `python summarize_preview.py <event-log>`. The resulting outside-server interval covers upload and return transport, codec work outside the server's measured window, and browser scheduling. The clocks need no alignment because the comparison subtracts elapsed durations for the same ID.

For a reproducible replay, provide the pinned MP4 to `server.py` with `--replay-file data/development-replay.mp4 --output-fps 30`. In Safari, **Start replay** downloads the file before transmission and logs its SHA-256. Leave the page open through **Replay complete** so it can save diagnostics and close the session. Score the 30-second development window after 10 seconds of warm-up with `python summarize_preview.py runs/iphone-primary-b1-events.jsonl --replay-seconds 30`. Use the [pre-experiment run sheet](EXPERIMENT.md) for the remaining validation and study pilots.

Run `python run_transport_controls.py --output runs/transport-controls-repeat` in the CUDA environment to compare pass-through and B1 under VP8 and H.264. The controls decode the same pinned MP4 before streaming, use 10 seconds of warm-up and 30 seconds of scored frames, and report submission to client decode on a single laptop clock. Safari tests separately report submission to expected display on the phone.

Start the preview with `--paired-replay` alongside `--replay-file`, `--configuration fp16-reuse`, and `--output-fps 30` for one-tap B1/pass-through collection. Select the number of sets in Safari and tap **Start paired replays**. The clip downloads once. Every trial starts a fresh connection and saves automatically, with B1 first in odd-numbered sets and pass-through first in even-numbered sets. JSON and CSV summaries appear in `batches/` next to the server event log as each trial finishes. An incomplete window is marked in the export.

## Check numerical equivalence

Once B1 is selected, run the checker with its exact configuration. This example uses the channels-last configuration and the same implementation on both sides as a control:

```bash
python validate_model.py --configuration fp16-channels-last \
  --candidate-upstream checkpoints/fastdvdnet \
  --development data/davis-development/{blackswan,bmx-trees,camel,drift-chicane} \
  --report runs/numerical-control.json
```

The checker verifies checkpoint identity, output shape, finiteness, and elementwise tolerance on constant and random windows, three supported resolutions, an extra spatial padding case, and the supplied development clips. Set `--candidate-upstream` to the candidate model-code directory to check an alternate implementation. The same option on `quality.py` runs that implementation against B0. The preview replay checks frame IDs through the media path.

## Pilot milestones

1. CUDA execution and profiler capture verified in Ubuntu WSL2. Both Windows and Ubuntu WSL2 see the NVIDIA GeForce RTX 3050 laptop GPU with 6144 MiB VRAM and driver 581.29.
2. Eager FP32 timing and memory fit verified on the Foreman feasibility clip. Repeat on the pinned development clips when they are ready.
3. Use the four pinned DAVIS development clips to screen standard configurations. Run the numerical checker against B1 during candidate validation. Keep held-out inputs and the final evaluator outside the application's writable checkout before candidate sessions.
4. Connect the iPhone preview to the laptop service. Laptop loopback covers VP8 framing, center-frame IDs, reconnection, and replacement of stale windows under load. Expected-display timing and LAN settings are verified with the phone.
5. Select and freeze B1 using the conventional tuning sweep. Record the hardware and software versions, workload hashes, reference measurements, and run order before starting the noniterative and iterative Codex sessions.
