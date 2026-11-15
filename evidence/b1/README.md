# B1 reference evidence

The native Windows baseline uses FP16, channels-last layout, and reusable buffers on the RTX 3050. Five separate processes each warmed up for 200 outputs, then timed 200 more on the pinned drift-chicane development clip at 640 × 360, sigma 25, and seed 11. The pooled server median was 57.6 ms and p95 was 64.0 ms. The pooled GPU median was 47.2 ms and p95 was 48.6 ms. [Summary](summary.json) · [Individual timing reports](timing/) · [Python packages](environment.txt)

The development quality check passed over 2,304 outputs. The numerical report is a 20-case identity control against B1. Future submissions need their own numerical and quality reports. [DAVIS quality](quality-davis.json) · [Numerical control](numerical-control.json)

The iPhone file contains per-trial metrics from the counterbalanced pre-experiment pilot. It records observed transport variation and the resulting protocol amendment. It does not contain held-out scores or results from the Codex conditions. [iPhone pilot](iphone-pilot.json) · [Protocol amendment](../../EXPERIMENT.md)

A separate warmed profile recorded 322 CUDA kernel events. The original tuning sweep and raw traces are kept with the local working artifacts. Use the setup instructions and the pinned source revision, checkpoint, and development data to reproduce the baseline. [Profile summary](profile-summary.json) · [Project setup](../../README.md)
