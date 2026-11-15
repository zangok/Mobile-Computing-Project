"""Run the laptop-side WebRTC denoising service for a single local client."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
import json
import logging
from pathlib import Path
import ssl
from time import perf_counter_ns
from uuid import uuid4

from aiohttp import web
from aiortc import MediaStreamTrack, RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
import numpy as np

from baseline import CONFIGURATIONS, CPU_CONFIGURATIONS, WindowPacker, load_configuration, set_backend, sha256
from frame_tag import HEIGHT, IMAGE_HEIGHT, TAG_ROWS, WIDTH, attach_tag, read_tag
from pipeline import Frame, FrameHistory
from replay_batch import save_trial, trial_metadata

LOG = logging.getLogger(__name__)
INPUT_FPS = 30
DIAGNOSTIC_DISPLAY_LIMIT = 2400
BITRATE_CEILING = 2_000_000


@dataclass(frozen=True)
class Incoming:
    image: np.ndarray
    pts: int
    time_base: Fraction
    received_ns: int


class Denoiser:
    def __init__(self, upstream, configuration, device, sigma):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        set_backend(configuration)
        self.model, self.denoise, self.dtype, self.layout, self.execution = load_configuration(
            upstream, self.device, configuration
        )
        self.pack = WindowPacker(self.device, self.dtype, self.layout, reuse=configuration == "fp16-reuse")
        self.noise_map = torch.full((1, 1, IMAGE_HEIGHT, WIDTH), sigma / 255, device=self.device, dtype=self.dtype)

    def __call__(self, images):
        torch = self.torch
        with self.execution():
            frames = tuple(torch.from_numpy(image).permute(2, 0, 1) for image in images)
            result = self.denoise(self.model, self.pack(frames, scale=255), self.noise_map)
            if not torch.isfinite(result).all():
                raise ValueError("Denoiser returned nonfinite pixels")
            return result[0].permute(1, 2, 0).float().mul_(255).round_().to(torch.uint8).cpu().numpy()


class PreviewTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, source, processor, executor, record, output_fps):
        super().__init__()
        self.source = source
        self.processor = processor
        self.executor = executor
        self.record = record
        self.output_stride = INPUT_FPS // output_fps
        self.stats = dict(received=0, bad_geometry=0, bad_tag=0, bad_timestamp=0,
                          out_of_order=0, missing_input_ids=0,
                          pending_replaced=0, output_replaced=0, processed=0, sent_to_encoder=0,
                          processing_errors=0, policy_skipped=0)
        self.pending = None
        self.output = None
        self.source_closed = False
        self.worker_closed = False
        self.work_ready = asyncio.Event()
        self.output_ready = asyncio.Event()
        self.tasks = (asyncio.create_task(self._receive()), asyncio.create_task(self._process()))

    async def _receive(self):
        last_id = None
        history = FrameHistory(5)
        try:
            while True:
                frame = await self.source.recv()
                self.stats["received"] += 1
                if self.stats["received"] == 1:
                    self.record("first_decoded_input", width=frame.width, height=frame.height)
                if (frame.width, frame.height) != (WIDTH, HEIGHT):
                    self.stats["bad_geometry"] += 1
                    self.record("bad_geometry", width=frame.width, height=frame.height)
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                frame_id = read_tag(rgb)
                if frame_id is None:
                    self.stats["bad_tag"] += 1
                    self.record("bad_tag", pts=frame.pts)
                    continue
                if frame.pts is None or frame.time_base is None:
                    self.stats["bad_timestamp"] += 1
                    self.record("bad_timestamp", frame_id=frame_id)
                    continue
                if last_id is not None:
                    if frame_id <= last_id:
                        self.stats["out_of_order"] += 1
                        self.record("out_of_order", frame_id=frame_id)
                        continue
                    gap = frame_id - last_id - 1
                    if gap:
                        self.stats["missing_input_ids"] += gap
                        self.record("input_gap", after_id=last_id, count=gap)
                        history = FrameHistory(5)
                        if self.pending is not None:
                            self._discard_pending()
                            self.pending = None
                last_id = frame_id
                self.record("received", frame_id=frame_id)
                window = history.push(Frame(frame_id, Incoming(
                    rgb[:IMAGE_HEIGHT].copy(), frame.pts, frame.time_base, perf_counter_ns()
                )))
                if window is not None:
                    center_id = window[2].frame_id
                    if center_id % self.output_stride:
                        self.stats["policy_skipped"] += 1
                        self.record("policy_skipped", frame_id=center_id)
                        continue
                    if self.pending is not None:
                        self._discard_pending()
                    self.pending = window
                    self.work_ready.set()
        except MediaStreamError:
            pass
        except Exception:
            LOG.exception("Could not receive preview frame")
        finally:
            self.source_closed = True
            self.work_ready.set()

    def _discard_pending(self):
        self.stats["pending_replaced"] += 1
        self.record("pending_replaced", frame_id=self.pending[2].frame_id)

    async def _process(self):
        loop = asyncio.get_running_loop()
        try:
            while True:
                await self.work_ready.wait()
                self.work_ready.clear()
                window, self.pending = self.pending, None
                if window is None:
                    if self.source_closed:
                        return
                    continue
                center = window[2]
                started = perf_counter_ns()
                image = await loop.run_in_executor(
                    self.executor, self.processor, tuple(entry.payload.image for entry in window)
                )
                frame = VideoFrame.from_ndarray(attach_tag(image, center.frame_id), format="rgb24")
                frame.pts = center.payload.pts
                frame.time_base = center.payload.time_base
                if self.output is not None:
                    self.stats["output_replaced"] += 1
                    self.record("output_replaced", frame_id=self.output[0])
                self.output = center.frame_id, frame, center.payload.received_ns
                self.stats["processed"] += 1
                finished = perf_counter_ns()
                ready = window[-1].payload.received_ns
                self.record("processed", frame_id=center.frame_id,
                            lookahead_ms=(ready - center.payload.received_ns) / 1e6,
                            queue_ms=(started - ready) / 1e6, server_ms=(finished - started) / 1e6)
                self.output_ready.set()
                if self.source_closed and self.pending is None:
                    return
        except Exception:
            self.stats["processing_errors"] += 1
            self.record("processing_error")
            LOG.exception("Could not process preview frame")
            self.source.stop()
            self.tasks[0].cancel()
        finally:
            self.worker_closed = True
            self.output_ready.set()

    async def recv(self):
        while True:
            if self.output is not None:
                frame_id, frame, received_ns = self.output
                self.output = None
                self.stats["sent_to_encoder"] += 1
                self.record("sent_to_encoder", frame_id=frame_id,
                            server_residence_ms=(perf_counter_ns() - received_ns) / 1e6)
                return frame
            if self.worker_closed or self.readyState != "live":
                raise MediaStreamError
            self.output_ready.clear()
            await self.output_ready.wait()

    def stop(self):
        super().stop()
        self.source.stop()
        for task in self.tasks:
            task.cancel()
        self.output_ready.set()

    async def close(self):
        self.stop()
        await asyncio.gather(*self.tasks, return_exceptions=True)


class PreviewServer:
    def __init__(self, args, processor, executor):
        self.args = args
        self.processor = processor
        self.executor = executor
        self.peer = None
        self.track = None
        self.last_stats = None
        self.last_transport = None
        self.session_id = None
        self.events = None
        self.close_lock = asyncio.Lock()
        self.startup_task = None
        self.trial = None
        self.replay_digest = sha256(args.replay_file) if args.paired_replay else None

    def record(self, event, **values):
        if self.events:
            entry = dict(session=self.session_id, event=event, monotonic_ns=perf_counter_ns(), **values)
            self.events.write(json.dumps(entry) + "\n")

    async def offer(self, request):
        if self.peer is not None:
            raise web.HTTPConflict(text="A preview session already exists")
        try:
            offer = await request.json()
            if offer["type"] != "offer" or not isinstance(offer["sdp"], str):
                raise ValueError("Expected a WebRTC offer")
        except (KeyError, TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text="Expected a WebRTC offer") from error
        trial = None
        if offer.get("batch") is not None:
            if not self.args.paired_replay:
                raise web.HTTPBadRequest(text="Paired replay is not enabled")
            try:
                trial = trial_metadata(offer["batch"])
            except ValueError as error:
                raise web.HTTPBadRequest(text=str(error)) from error
        if self.peer is not None:
            raise web.HTTPConflict(text="A preview session already exists")
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.peer = peer
        self.session_id = uuid4().hex
        self.trial = trial
        configuration = "passthrough" if trial and trial["mode"] == "passthrough" else self.args.configuration
        processor = passthrough if configuration == "passthrough" else self.processor
        self.record("session_started", configuration=configuration, codec=self.args.codec,
                    input_fps=INPUT_FPS, output_fps=self.args.output_fps, batch=trial)
        transceiver = peer.addTransceiver("video", direction="sendrecv")
        transceiver.setCodecPreferences([
            codec for codec in RTCRtpSender.getCapabilities("video").codecs
            if codec.mimeType.lower() == f"video/{self.args.codec.lower()}"
        ])
        incoming = []

        @peer.on("track")
        def track_received(track):
            incoming.append(track)

        @peer.on("connectionstatechange")
        async def connection_changed():
            state = peer.connectionState
            if peer is self.peer:
                self.record("connection_state", state=state)
                if state == "failed":
                    await self.close(peer)

        self.startup_task = asyncio.create_task(self.sample_startup(peer))
        try:
            async with asyncio.timeout(15):
                await peer.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type="offer"))
                if len(incoming) != 1 or incoming[0].kind != "video" or len(peer.getTransceivers()) != 1:
                    raise ValueError("Offer must provide one bidirectional video transceiver")
                self.track = PreviewTrack(incoming[0], processor, self.executor, self.record, self.args.output_fps)
                transceiver.sender.replaceTrack(self.track)
                await peer.setLocalDescription(await peer.createAnswer())
                if transceiver.currentDirection != "sendrecv":
                    raise ValueError("The preview requires bidirectional video")
            return web.json_response(dict(sdp=peer.localDescription.sdp, type=peer.localDescription.type,
                                          session=self.session_id, configuration=configuration))
        except asyncio.CancelledError:
            await self.close(peer)
            raise
        except Exception as error:
            LOG.warning("Could not negotiate preview: %s", error)
            await self.close(peer)
            raise web.HTTPBadRequest(text="Could not negotiate the preview offer") from error

    async def close(self, peer):
        async with self.close_lock:
            if peer is None or peer is not self.peer:
                return
            if self.startup_task is not None:
                self.startup_task.cancel()
                await asyncio.gather(self.startup_task, return_exceptions=True)
                self.startup_task = None
            track, self.track = self.track, None
            if track is not None:
                await track.close()
                self.last_stats = dict(track.stats)
                self.last_transport = await self.transport(peer)
                self.record("session_summary", **self.last_stats, rtp=self.last_transport)
            await peer.close()
            if self.events:
                self.events.flush()
            self.peer = None

    async def sample_startup(self, peer):
        if self.events is None:
            return
        for _ in range(80):
            counters = await self.transport(peer)
            self.record("startup_rtp", **counters, state=peer.connectionState)
            await asyncio.sleep(0.25)

    async def transport(self, peer):
        stats = (await peer.getStats()).values()
        return {
            "outbound_rtp_bytes": sum(item.bytesSent for item in stats if item.type == "outbound-rtp"),
            "outbound_rtp_packets": sum(item.packetsSent for item in stats if item.type == "outbound-rtp"),
            "inbound_rtp_packets": sum(item.packetsReceived for item in stats if item.type == "inbound-rtp"),
            "inbound_rtp_packets_lost": sum(item.packetsLost for item in stats if item.type == "inbound-rtp"),
        }

    async def delete(self, request):
        if request.query.get("session", self.session_id) != self.session_id:
            raise web.HTTPConflict(text="The preview session has changed")
        await self.close(self.peer)
        return web.Response(status=204)

    async def stats(self, request):
        peer = self.peer
        return web.json_response(dict(
            state=peer.connectionState if peer else "idle",
            session=self.session_id,
            frames=dict(self.track.stats) if self.track else self.last_stats,
            rtp=await self.transport(peer) if peer else self.last_transport,
        ))

    async def health(self, request):
        return web.json_response(dict(configuration=self.args.configuration, codec=self.args.codec,
                                      width=WIDTH, height=IMAGE_HEIGHT, tag_rows=TAG_ROWS,
                                      input_fps=INPUT_FPS, output_fps=self.args.output_fps,
                                      replay_available=self.args.replay_file is not None,
                                      paired_replay=self.args.paired_replay,
                                      diagnostics_display_limit=DIAGNOSTIC_DISPLAY_LIMIT))

    async def diagnostics(self, request):
        try:
            report = await request.json()
            if not isinstance(report, dict) or not isinstance(report.get("display"), list) or not isinstance(report.get("rtp"), list):
                raise ValueError
            if len(report["display"]) > DIAGNOSTIC_DISPLAY_LIMIT or len(report["rtp"]) > 60:
                raise ValueError
            ingestion = report.get("ingestion", [])
            if not isinstance(ingestion, list) or len(ingestion) > DIAGNOSTIC_DISPLAY_LIMIT:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text="Invalid diagnostic report") from error
        if self.track is None or report.get("session") != self.session_id:
            raise web.HTTPConflict(text="The preview session has ended")
        if self.events is None:
            raise web.HTTPServiceUnavailable(text="Server diagnostics logging is unavailable")
        self.record("browser_diagnostics", report=report)
        if self.trial is not None and report.get("finish_reason") == "replay-ended":
            if report.get("codec") != self.args.codec or report.get("output_fps") != self.args.output_fps:
                raise web.HTTPBadRequest(text="Batch transport settings are inconsistent")
            try:
                batch = save_trial(self.args.events.parent / "batches", self.trial,
                                   self.session_id, report, self.replay_digest)
            except ValueError as error:
                raise web.HTTPBadRequest(text=str(error)) from error
            return web.json_response({"batch_id": batch["batch_id"], "saved_runs": len(batch["trials"]),
                                      "valid_runs": batch["valid_runs"], "complete": batch["complete"]})
        return web.Response(status=204)

    async def lifecycle(self, app):
        if self.args.events:
            self.args.events.parent.mkdir(parents=True, exist_ok=True)
            self.events = self.args.events.open("a", encoding="utf-8", buffering=1)
        try:
            yield
        finally:
            await self.close(self.peer)
            if self.events:
                self.events.close()


def create_app(args, processor, executor):
    server = PreviewServer(args, processor, executor)
    app = web.Application(client_max_size=256 * 1024)
    directory = Path(__file__).resolve().parent

    async def page(request):
        return web.FileResponse(directory / "preview.html")

    async def script(request):
        return web.FileResponse(directory / "preview.js", headers={"Content-Type": "text/javascript"})

    async def replay_file(request):
        if args.replay_file is None:
            raise web.HTTPNotFound()
        return web.FileResponse(args.replay_file, headers={"Content-Type": "video/mp4"})

    app.cleanup_ctx.append(server.lifecycle)
    app.add_routes([
        web.get("/", page), web.get("/preview.js", script), web.get("/replay.mp4", replay_file),
        web.post("/offer", server.offer), web.get("/stats", server.stats),
        web.get("/health", server.health), web.delete("/session", server.delete),
        web.post("/diagnostics", server.diagnostics),
    ])
    return app


def passthrough(images):
    return images[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--configuration", choices=("passthrough", *CONFIGURATIONS), default="fp16")
    parser.add_argument("--upstream", type=Path, default=Path("checkpoints/fastdvdnet"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--sigma", choices=(10, 25, 50), type=int, default=25)
    parser.add_argument("--output-fps", type=int, choices=(15, 30), default=30)
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--paired-replay", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--codec", choices=("VP8", "H264"), default="VP8")
    parser.add_argument("--cert", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--events", type=Path, default=Path("runs/preview-events.jsonl"))
    args = parser.parse_args()
    if bool(args.cert) != bool(args.key):
        parser.error("--cert and --key must be supplied together")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.replay_file is not None and not args.replay_file.is_file():
        parser.error("--replay-file must point to an existing MP4")
    if args.paired_replay and (args.replay_file is None or args.configuration == "passthrough" or args.output_fps != 30):
        parser.error("Paired replay requires a replay file, a denoising configuration, and 30 output fps")
    if args.device == "cpu" and args.configuration not in (*CPU_CONFIGURATIONS, "passthrough"):
        parser.error(f"{args.configuration} requires CUDA")
    if args.codec == "H264":
        from aiortc.codecs import h264

        # aiortc caps the encoder target in its bitrate setter.
        h264.MAX_BITRATE = min(h264.MAX_BITRATE, BITRATE_CEILING)
    tls = None
    if args.cert:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(args.cert, args.key)
    logging.basicConfig(level=logging.INFO)
    with ThreadPoolExecutor(max_workers=1) as executor:
        if args.configuration == "passthrough":
            processor = passthrough
        else:
            build = partial(Denoiser, args.upstream, args.configuration, args.device, args.sigma)
            processor = executor.submit(build).result()
            sample = np.zeros((IMAGE_HEIGHT, WIDTH, 3), dtype=np.uint8)
            for _ in range(args.warmup):
                executor.submit(processor, (sample,) * 5).result()
        with suppress(KeyboardInterrupt):
            web.run_app(create_app(args, processor, executor), host=args.host, port=args.port, ssl_context=tls)


if __name__ == "__main__":
    main()
