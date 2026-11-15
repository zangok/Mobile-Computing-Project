const WIDTH = 640;
const IMAGE_HEIGHT = 360;
const TAG_HEIGHT = 24;
const CELL_WIDTH = 13;
const LEFT = 8;
const MAGIC = 0xD3;
const FPS = 30;
const startButton = document.querySelector("#start");
const replayButton = document.querySelector("#start-replay");
const replayFile = document.querySelector("#replay-file");
const batchControls = document.querySelector("#batch-controls");
const batchButton = document.querySelector("#start-batch");
const batchPairs = document.querySelector("#batch-pairs");
const batchStatus = document.querySelector("#batch-status");
const stopButton = document.querySelector("#stop");
const saveButton = document.querySelector("#save");
const camera = document.querySelector("#camera");
const returned = document.querySelector("#returned");
const status = document.querySelector("#status");
const stats = document.querySelector("#stats");
const diagnostics = document.querySelector("#diagnostics");
const replayProgress = document.querySelector("#replay-progress");
const outgoing = document.createElement("canvas");
outgoing.width = WIDTH;
outgoing.height = IMAGE_HEIGHT + TAG_HEIGHT;
const picture = outgoing.getContext("2d");
const incomingTag = document.createElement("canvas");
incomingTag.width = WIDTH;
incomingTag.height = TAG_HEIGHT;
const pixels = incomingTag.getContext("2d", {willReadFrequently: true});
let session;
let standardReplayAvailable = false;
let batchRun = null;

function crc8(bytes) {
  let crc = 0;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) {
      crc = ((crc << 1) ^ (crc & 0x80 ? 0x07 : 0)) & 0xFF;
    }
  }
  return crc;
}

function drawTag(frameId) {
  const bytes = new Uint8Array(6);
  bytes[0] = MAGIC;
  new DataView(bytes.buffer).setUint32(1, frameId);
  bytes[5] = crc8(bytes.subarray(0, 5));
  picture.fillStyle = "#000";
  picture.fillRect(0, IMAGE_HEIGHT, WIDTH, TAG_HEIGHT);
  picture.fillStyle = "#fff";
  for (let cell = 0; cell < 48; cell++) {
    if (bytes[cell >> 3] & (0x80 >> (cell % 8))) {
      picture.fillRect(LEFT + cell * CELL_WIDTH, IMAGE_HEIGHT, CELL_WIDTH, TAG_HEIGHT);
    }
  }
}

function readTag() {
  pixels.drawImage(returned, 0, IMAGE_HEIGHT, WIDTH, TAG_HEIGHT, 0, 0, WIDTH, TAG_HEIGHT);
  const rgb = pixels.getImageData(0, 6, WIDTH, 12).data;
  const bytes = new Uint8Array(6);
  for (let cell = 0; cell < 48; cell++) {
    let sum = 0;
    for (let row = 0; row < 12; row++) {
      for (let column = 3; column < 10; column++) {
        const offset = 4 * (row * WIDTH + LEFT + cell * CELL_WIDTH + column);
        sum += rgb[offset] + rgb[offset + 1] + rgb[offset + 2];
      }
    }
    bytes[cell >> 3] |= Number(sum >= 128 * 12 * 7 * 3) << (7 - cell % 8);
  }
  if (bytes[0] !== MAGIC || bytes[5] !== crc8(bytes.subarray(0, 5))) {
    return null;
  }
  return new DataView(bytes.buffer).getUint32(1);
}

function submitFrame(active, sourceTime = null) {
  const cropWidth = Math.min(camera.videoWidth, camera.videoHeight * WIDTH / IMAGE_HEIGHT);
  const cropHeight = cropWidth * IMAGE_HEIGHT / WIDTH;
  const frameId = active.nextId++;
  const submitted = performance.now();
  if (frameId === 0) active.firstSubmittedMs = Math.round(submitted);
  active.submitted.set(frameId, submitted);
  if (active.source === "replay") {
    active.ingestion.push([frameId, Math.round(submitted - active.firstSubmittedMs),
      Math.round(sourceTime * 1000)]);
    if (active.ingestion.length > active.displayLimit) active.ingestion.shift();
  }
  picture.drawImage(camera, (camera.videoWidth - cropWidth) / 2, (camera.videoHeight - cropHeight) / 2,
    cropWidth, cropHeight, 0, 0, WIDTH, IMAGE_HEIGHT);
  drawTag(frameId);
  if (active.manualCapture) active.canvasTrack.requestFrame();
  active.submitted.delete(frameId - 900);
}

function produce(active, now) {
  if (session !== active) return;
  if (camera.readyState >= 2 && now >= active.nextFrame) {
    submitFrame(active);
    active.nextFrame = Math.max(now - 10, active.nextFrame) + 1000 / FPS;
  }
  active.animation = requestAnimationFrame((next) => produce(active, next));
}

function produceReplay(active) {
  const capture = (now, metadata) => {
    if (session !== active) return;
    if (active.lastSourceFrame !== undefined) {
      active.skippedSourceFrames += Math.max(0, metadata.presentedFrames - active.lastSourceFrame - 1);
    }
    active.lastSourceFrame = metadata.presentedFrames;
    submitFrame(active, metadata.mediaTime);
    active.sourceCallback = camera.requestVideoFrameCallback(capture);
  };
  active.sourceCallback = camera.requestVideoFrameCallback(capture);
}

function watchReturn(active) {
  const observe = (now, metadata) => {
    if (session !== active) return;
    const id = returned.videoWidth === WIDTH && returned.videoHeight === IMAGE_HEIGHT + TAG_HEIGHT ? readTag() : null;
    if (id === null) active.badTags++;
    else if (id > active.lastDisplayed) {
      active.lastDisplayed = id;
      const submitted = active.submitted.get(id);
      if (submitted === undefined || !Number.isFinite(metadata.expectedDisplayTime)) active.unmatched++;
      else {
        if (metadata.expectedDisplayTime < now) active.late++;
        active.delays.push(metadata.expectedDisplayTime - submitted);
        if (active.delays.length > 900) active.delays.shift();
        active.display.push([
          id, Math.round(metadata.expectedDisplayTime - submitted),
          Math.round(now - metadata.expectedDisplayTime),
          Number.isFinite(metadata.processingDuration) ? Math.round(metadata.processingDuration * 1000) : null,
          Math.round(submitted - active.firstSubmittedMs),
        ]);
        if (active.display.length > active.displayLimit) active.display.shift();
      }
      active.displayed++;
      status.textContent = `Preview connected · ${active.outputFps} fps output mode`;
    }
    active.videoCallback = returned.requestVideoFrameCallback(observe);
  };
  active.videoCallback = returned.requestVideoFrameCallback(observe);
}

function refresh(active) {
  const delays = [...active.delays].sort((a, b) => a - b);
  const p95 = delays.length ? Math.round(delays[Math.ceil(delays.length * 0.95) - 1]) + " ms" : "—";
  stats.textContent = `Displayed ${active.displayed} of ${active.nextId} submitted · p95 ${p95} · ${active.badTags} unreadable tags · ${active.unmatched} unmatched · ${active.late} late callbacks`;
  if (active.source === "replay" && active.firstSubmittedMs !== undefined) {
    const elapsed = Math.round((performance.now() - active.firstSubmittedMs) / 1000);
    if (!camera.ended) {
      replayProgress.textContent = `Replay ${elapsed} s · diagnostics save at the end`;
    }
  }
}

async function sampleRtp(active) {
  try {
    const snapshots = await active.peer.getStats();
    if (session !== active) return;
    const sample = {ms: Math.round(performance.now())};
    for (const row of snapshots.values()) {
      if (row.type === "outbound-rtp" && (row.kind ?? row.mediaType) === "video") {
        sample.out = [row.framesEncoded, row.totalEncodeTime, row.bytesSent, row.packetsSent,
          row.frameWidth, row.frameHeight, row.qualityLimitationReason];
      }
      if (row.type === "inbound-rtp" && (row.kind ?? row.mediaType) === "video") {
        sample.in = [row.framesDecoded, row.totalDecodeTime, row.jitterBufferDelay,
          row.jitterBufferEmittedCount, row.bytesReceived, row.packetsLost, row.framesDropped];
      }
      if (row.type === "candidate-pair" && row.nominated && row.state === "succeeded") {
        sample.rtt_s = row.currentRoundTripTime;
      }
    }
    active.rtp.push(sample);
    // Retain the startup sequence, even if the preview runs for several minutes.
    if (active.rtp.length > 60) active.rtp.splice(48, 1);
  } catch {
    // Display measurements remain available on Safari versions with limited RTP stats.
  }
}

function save(active, reason = "manual") {
  if (!active?.id || session !== active) return Promise.resolve(false);
  if (active.savePromise) return active.savePromise;
  active.savePromise = postDiagnostics(active, reason);
  return active.savePromise;
}

async function postDiagnostics(active, reason) {
  saveButton.disabled = true;
  try {
    const payload = {session: active.id, output_fps: active.outputFps, codec: active.codec,
      submitted: active.nextId, displayed: active.displayed, unreadable: active.badTags,
      unmatched: active.unmatched, late: active.late, display: active.display, rtp: active.rtp,
      first_submitted_ms: active.firstSubmittedMs, source: active.source, ingestion: active.ingestion,
      replay: active.replay, sender: active.senderSettings,
      capture_mode: active.manualCapture ? "manual" : "canvas-rate",
      skipped_source_frames: active.skippedSourceFrames, finish_reason: reason};
    // A laptop running the earlier diagnostics service still accepts the last 600 samples.
    if (active.displayLimit === 600) payload.display = active.display.slice(-600);
    const response = await fetch("/diagnostics", {method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify(payload)});
    if (!response.ok) throw new Error(await response.text());
    if (active.batch && reason === "replay-ended") active.batch.progress = await response.json();
    diagnostics.textContent = "Diagnostics saved";
    return true;
  } catch (error) {
    diagnostics.textContent = "Could not save diagnostics · " + error.message;
    return false;
  } finally {
    active.savePromise = null;
    if (session === active) saveButton.disabled = false;
  }
}

function finishReplay(active) {
  if (session !== active) return;
  replayProgress.textContent = "Replay finished. Waiting for returned frames.";
  // Keep the connection open briefly so the final display callbacks can arrive.
  active.finishTimer = setTimeout(async () => {
    if (session !== active) return;
    await active.savePromise;
    await sampleRtp(active);
    if (await save(active, "replay-ended")) {
      if (active.batch) {
        await advanceBatch(active);
      } else {
        await stop(active, "Replay complete");
        replayProgress.textContent = "Diagnostics saved";
      }
    } else if (session === active) {
      replayProgress.textContent = "Replay finished. Tap Save diagnostics to retry.";
      if (active.batch) active.finished = true;
    }
  }, 1500);
}

function iceComplete(peer) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const changed = () => {
      if (peer.iceGatheringState === "complete") {
        clearTimeout(timer);
        peer.removeEventListener("icegatheringstatechange", changed);
        resolve();
      }
    };
    const timer = setTimeout(() => {
      peer.removeEventListener("icegatheringstatechange", changed);
      reject(new Error("Local network discovery timed out"));
    }, 8000);
    peer.addEventListener("icegatheringstatechange", changed);
  });
}

function idleControls() {
  startButton.disabled = Boolean(batchRun);
  replayFile.disabled = Boolean(batchRun);
  replayButton.disabled = Boolean(batchRun) || (!replayFile.files.length && !standardReplayAvailable);
  batchButton.disabled = Boolean(batchRun);
  batchPairs.disabled = Boolean(batchRun);
  stopButton.disabled = !batchRun;
}

function endBatch(batch, message) {
  if (batchRun !== batch) return;
  clearTimeout(batch.nextTimer);
  batchRun = null;
  batchStatus.textContent = message;
  idleControls();
}

async function advanceBatch(active) {
  const batch = active.batch;
  const progress = batch.progress;
  const closed = await stop(active, "Replay saved", true);
  if (batchRun !== batch) return;
  if (!closed || !progress || progress.saved_runs !== batch.index + 1) {
    endBatch(batch, "Batch stopped. Saved trials are on the laptop.");
    return;
  }
  batch.index++;
  if (progress.complete) {
    endBatch(batch, `Batch complete · ${progress.valid_runs} of ${progress.saved_runs} runs scored · ${batch.id.slice(0, 8)}`);
  } else {
    batchStatus.textContent = `${progress.saved_runs} of ${batch.pairs * 2} runs saved. Next run starting shortly.`;
    batch.nextTimer = setTimeout(() => start("standard", batch), 1000);
  }
}

function startBatch() {
  if (session || batchRun) return;
  const batch = {id: crypto.randomUUID().replaceAll("-", ""), pairs: Number(batchPairs.value), index: 0};
  batchRun = batch;
  start("standard", batch);
}

async function stop(active, message = "Stopped", keepBatch = false) {
  if (!active || session !== active) return false;
  session = null;
  stopButton.disabled = true;
  saveButton.disabled = true;
  cancelAnimationFrame(active.animation);
  if (active.sourceCallback !== undefined) camera.cancelVideoFrameCallback(active.sourceCallback);
  if (active.videoCallback !== undefined) returned.cancelVideoFrameCallback(active.videoCallback);
  clearInterval(active.refresh);
  clearInterval(active.rtpTimer);
  clearTimeout(active.rtpStartupTimer);
  clearTimeout(active.finishTimer);
  if (active.onReplayEnded) camera.removeEventListener("ended", active.onReplayEnded);
  active.peer?.close();
  active.media?.getTracks().forEach((track) => track.stop());
  active.canvasTrack?.stop();
  camera.pause();
  camera.srcObject = null;
  if (active.replayUrl) {
    camera.removeAttribute("src");
    camera.load();
    URL.revokeObjectURL(active.replayUrl);
  }
  returned.srcObject = null;
  status.textContent = message;
  let closed = true;
  if (active.accepted) {
    try {
      const path = active.id ? `/session?session=${encodeURIComponent(active.id)}` : "/session";
      const response = await fetch(path, {method: "DELETE"});
      closed = response.ok;
    } catch {
      closed = false;
    }
  }
  if (!closed) status.textContent = message + " · Server did not acknowledge stop";
  if (active.batch && !keepBatch) endBatch(active.batch, "Batch stopped. Saved trials are on the laptop.");
  idleControls();
  return closed;
}

async function prepareReplay(active, file) {
  status.textContent = "Preparing the selected replay";
  if (file === "standard") {
    file = active.batch?.file;
    if (!file) {
      status.textContent = "Downloading the standard replay";
      const response = await fetch("/replay.mp4");
      if (!response.ok) throw new Error("The standard replay is unavailable");
      file = await response.blob();
      if (active.batch) active.batch.file = file;
    }
  }
  let digest = active.batch?.digest;
  if (!digest) {
    const bytes = new Uint8Array(await crypto.subtle.digest("SHA-256", await file.arrayBuffer()));
    digest = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (active.batch) active.batch.digest = digest;
  }
  active.replayUrl = URL.createObjectURL(file);
  await new Promise((resolve, reject) => {
    camera.addEventListener("loadedmetadata", resolve, {once: true});
    camera.addEventListener("error", () => reject(new Error("Safari could not decode the selected video")),
      {once: true});
    camera.src = active.replayUrl;
    camera.load();
  });
  active.replay = {sha256: digest, bytes: file.size, duration_ms: Math.round(camera.duration * 1000)};
}

async function configureSender(active, sender) {
  const parameters = sender.getParameters();
  if (!parameters.encodings?.length) return;
  parameters.encodings[0].maxBitrate = 2_000_000;
  parameters.encodings[0].scaleResolutionDownBy = 1;
  parameters.degradationPreference = "maintain-resolution";
  try {
    await sender.setParameters(parameters);
    const applied = sender.getParameters();
    active.senderSettings = {
      max_bitrate: applied.encodings?.[0]?.maxBitrate ?? null,
      scale_resolution_down_by: applied.encodings?.[0]?.scaleResolutionDownBy ?? null,
      degradation_preference: applied.degradationPreference ?? null,
    };
  } catch (error) {
    active.senderSettings = {error: error.name};
  }
}

async function start(file = null, batch = null) {
  if (session || batchRun !== batch) return;
  const active = {submitted: new Map(), nextId: 0, nextFrame: 0, displayed: 0,
    lastDisplayed: -1, badTags: 0, unmatched: 0, late: 0, delays: [], display: [], rtp: [], ingestion: [],
    source: file ? "replay" : "camera", skippedSourceFrames: 0, batch};
  session = active;
  startButton.disabled = true;
  replayButton.disabled = true;
  replayFile.disabled = true;
  batchButton.disabled = true;
  batchPairs.disabled = true;
  stopButton.disabled = true;
  if (batch) batchStatus.textContent = `Starting run ${batch.index + 1} of ${batch.pairs * 2}`;
  stats.textContent = "";
  diagnostics.textContent = "";
  replayProgress.textContent = "";
  try {
    if ((!file && !navigator.mediaDevices?.getUserMedia) || !outgoing.captureStream || !returned.requestVideoFrameCallback) {
      throw new Error("Safari needs a trusted HTTPS connection with camera and video frame support");
    }
    if (file) {
      await prepareReplay(active, file);
    } else {
      status.textContent = "Waiting for camera permission";
      active.media = await navigator.mediaDevices.getUserMedia({audio: false,
        video: {width: {ideal: WIDTH}, height: {ideal: IMAGE_HEIGHT}, frameRate: {ideal: FPS, max: FPS},
          facingMode: {ideal: "environment"}}});
      camera.srcObject = active.media;
      await camera.play();
    }
    picture.clearRect(0, 0, outgoing.width, outgoing.height);
    let media = outgoing.captureStream(file ? 0 : FPS);
    active.canvasTrack = media.getVideoTracks()[0];
    active.manualCapture = Boolean(file && typeof active.canvasTrack.requestFrame === "function");
    if (file && !active.manualCapture) {
      active.canvasTrack.stop();
      media = outgoing.captureStream(FPS);
      active.canvasTrack = media.getVideoTracks()[0];
    }
    if ("contentHint" in active.canvasTrack) active.canvasTrack.contentHint = "detail";
    const settings = await fetch("/health").then((response) => response.json());
    active.displayLimit = settings.diagnostics_display_limit ?? 600;
    active.outputFps = settings.output_fps;
    active.codec = settings.codec;
    document.querySelector("#return-label").textContent =
      settings.configuration === "passthrough" ? "Returned (transport test)" : "Denoised";
    const peer = new RTCPeerConnection({iceServers: []});
    active.peer = peer;
    const transceiver = peer.addTransceiver(active.canvasTrack, {direction: "sendrecv", streams: [media],
      sendEncodings: [{maxBitrate: 2_000_000, scaleResolutionDownBy: 1}]});
    const supported = RTCRtpSender.getCapabilities?.("video")?.codecs ?? [];
    const codecs = supported.filter((codec) => codec.mimeType.toLowerCase() === "video/" + settings.codec.toLowerCase());
    if (codecs.length && transceiver.setCodecPreferences) transceiver.setCodecPreferences(codecs);
    await configureSender(active, transceiver.sender);
    peer.addEventListener("track", async (event) => {
      returned.srcObject = new MediaStream([event.track]);
      try {
        await returned.play();
        if (session === active) watchReturn(active);
      } catch (error) {
        await stop(active, error.message);
      }
    });
    peer.addEventListener("connectionstatechange", () => {
      if (peer.connectionState === "failed") stop(active, "Media connection failed. Check the local network and try again.");
    });
    status.textContent = `Connecting · ${settings.output_fps} fps output mode`;
    await peer.setLocalDescription(await peer.createOffer());
    await iceComplete(peer);
    const offer = {sdp: peer.localDescription.sdp, type: peer.localDescription.type};
    if (batch) offer.batch = {id: batch.id, pairs: batch.pairs, index: batch.index};
    const response = await fetch("/offer", {method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify(offer)});
    if (!response.ok) throw new Error(await response.text());
    active.accepted = true;
    const answer = await response.json();
    active.id = answer.session;
    if (batch) {
      const mode = answer.configuration === "passthrough" ? "Pass-through" : "B1";
      document.querySelector("#return-label").textContent = mode;
      batchStatus.textContent = `Run ${batch.index + 1} of ${batch.pairs * 2} · ${mode}`;
    }
    await peer.setRemoteDescription({type: answer.type, sdp: answer.sdp});
    await configureSender(active, transceiver.sender);
    if (file) {
      if (peer.connectionState !== "connected") {
        await new Promise((resolve, reject) => {
          const changed = () => {
            if (peer.connectionState === "connected" || peer.connectionState === "failed") {
              clearTimeout(timer);
              peer.removeEventListener("connectionstatechange", changed);
              if (peer.connectionState === "connected") resolve();
              else reject(new Error("Replay media connection failed"));
            }
          };
          const timer = setTimeout(() => {
            peer.removeEventListener("connectionstatechange", changed);
            reject(new Error("Replay media connection timed out"));
          }, 12000);
          peer.addEventListener("connectionstatechange", changed);
        });
      }
      active.onReplayEnded = () => finishReplay(active);
      camera.addEventListener("ended", active.onReplayEnded, {once: true});
      produceReplay(active);
      await camera.play();
    }
    stopButton.disabled = false;
    saveButton.disabled = !active.id;
    active.refresh = setInterval(() => refresh(active), 1000);
    active.rtpTimer = setInterval(() => sampleRtp(active), 250);
    active.rtpStartupTimer = setTimeout(() => {
      clearInterval(active.rtpTimer);
      active.rtpTimer = setInterval(() => sampleRtp(active), 1000);
    }, 12000);
    if (!file) active.animation = requestAnimationFrame((now) => produce(active, now));
  } catch (error) {
    await stop(active, error.message);
  }
}

startButton.addEventListener("click", () => start());
replayButton.addEventListener("click", () => start(replayFile.files[0] ?? "standard"));
replayFile.addEventListener("change", () => {
  replayButton.disabled = !replayFile.files.length && !standardReplayAvailable;
});
fetch("/health").then((response) => response.json()).then((settings) => {
  standardReplayAvailable = settings.replay_available === true;
  batchControls.hidden = settings.paired_replay !== true;
  if (!session && standardReplayAvailable) replayButton.disabled = false;
}).catch(() => {});
batchButton.addEventListener("click", startBatch);
stopButton.addEventListener("click", () => {
  if (session) stop(session);
  else if (batchRun) endBatch(batchRun, "Batch stopped. Saved trials are on the laptop.");
});
saveButton.addEventListener("click", async () => {
  const active = session;
  if (await save(active, active?.finished ? "replay-ended" : "manual") && active.finished) {
    await advanceBatch(active);
  }
});
