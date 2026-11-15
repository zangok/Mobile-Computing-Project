"""Send tagged frames to the preview server and measure decoded loopback delivery."""

import argparse
import asyncio
from fractions import Fraction
import json
from pathlib import Path
import ssl
from time import perf_counter

from aiohttp import ClientSession, ClientTimeout, TCPConnector
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame, open as open_video
import numpy as np
from PIL import Image

from baseline import frame_paths, summarize
from frame_tag import IMAGE_HEIGHT, WIDTH, attach_tag, read_tag


class ReplayTrack(VideoStreamTrack):
    def __init__(self, images, count):
        super().__init__()
        self.images = images
        self.count = count
        self.sent = {}
        self.finished = asyncio.Event()
        self.started = None

    async def recv(self):
        frame_id = len(self.sent)
        if frame_id == self.count:
            self.finished.set()
            await asyncio.Future()
        if self.started is None:
            self.started = perf_counter()
        deadline = self.started + frame_id / 30
        await asyncio.sleep(max(0, deadline - perf_counter()))
        image = self.images[frame_id % len(self.images)]
        frame = VideoFrame.from_ndarray(attach_tag(image, frame_id), format="rgb24")
        frame.pts = frame_id * 3000
        frame.time_base = Fraction(1, 90000)
        self.sent[frame_id] = perf_counter()
        return frame


def read_images(clip):
    if clip is None:
        image = np.zeros((IMAGE_HEIGHT, WIDTH, 3), dtype=np.uint8)
        image[:, :, 1] = np.arange(WIDTH, dtype=np.uint16) % 256
        return [image]
    images = []
    for path in frame_paths(clip):
        with Image.open(path) as image:
            images.append(np.array(image.convert("RGB").resize((WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)))
    if not images:
        raise ValueError(f"No replay frames in {clip}")
    return images


def read_video(path, count):
    images = []
    with open_video(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if (frame.width, frame.height) != (WIDTH, IMAGE_HEIGHT):
                raise ValueError("The replay file must decode at 640 by 360")
            images.append(frame.to_ndarray(format="rgb24"))
            if len(images) == count:
                return images
    raise ValueError(f"The replay contains fewer than {count} frames")


async def run(url, images, count, warmup, ca_certificate=None):
    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    # Extra frames keep both decoder jitter buffers advancing at the end of
    # the measured interval.
    source = ReplayTrack(images, warmup + count + 4)
    peer.addTrack(source)
    received = {}
    receiver = None
    invalid_tags = 0
    duplicates = 0

    async def receive(track):
        nonlocal invalid_tags, duplicates
        try:
            while True:
                frame = await track.recv()
                decoded = perf_counter()
                frame_id = read_tag(frame.to_ndarray(format="rgb24"))
                if frame_id is None or frame_id not in source.sent:
                    invalid_tags += 1
                elif frame_id in received:
                    duplicates += 1
                else:
                    received[frame_id] = decoded
        except MediaStreamError:
            return

    @peer.on("track")
    def on_track(track):
        nonlocal receiver
        receiver = asyncio.create_task(receive(track))

    connector = TCPConnector(ssl=ssl.create_default_context(cafile=ca_certificate)) if ca_certificate else None
    async with ClientSession(timeout=ClientTimeout(total=20), connector=connector) as client:
        accepted = False
        try:
            async with client.get(f"{url}/health") as response:
                response.raise_for_status()
                settings = await response.json()
            await peer.setLocalDescription(await peer.createOffer())
            local = peer.localDescription
            async with client.post(f"{url}/offer", json=dict(type=local.type, sdp=local.sdp)) as response:
                response.raise_for_status()
                accepted = True
                answer = await response.json()
            await peer.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
            await asyncio.wait_for(source.finished.wait(), timeout=source.count / 30 + 15)
            await asyncio.sleep(1)
            async with client.get(f"{url}/stats") as response:
                response.raise_for_status()
                stats = await response.json()
        finally:
            await peer.close()
            if receiver is not None:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            if accepted:
                async with client.delete(f"{url}/session") as response:
                    response.raise_for_status()
    input_ids = set(range(max(2, warmup), warmup + count))
    output_stride = settings["input_fps"] // settings["output_fps"]
    measurable = {frame_id for frame_id in input_ids if frame_id % output_stride == 0}
    intersection = measurable & received.keys()
    samples = [(received[frame_id] - source.sent[frame_id]) * 1000 for frame_id in sorted(intersection)]
    decoded_times = sorted(received[frame_id] for frame_id in intersection)
    intervals = [(end - start) * 1000 for start, end in zip(decoded_times, decoded_times[1:])]
    interval_seconds = source.sent[warmup + count] - source.sent[min(input_ids)]
    return {
        "measurement": "loopback submission to decoded frame",
        "sent": len(source.sent),
        "received_ids": list(received),
        "warmup_input_frames": warmup,
        "duplicate_ids": duplicates,
        "invalid_return_tags": invalid_tags,
        "eligible_input_ids": len(measurable),
        "decoded_eligible_ids": len(intersection),
        "submitted_fps": len(input_ids) / interval_seconds,
        "decoded_eligible_fps": len(intersection) / interval_seconds,
        "eligible_delivery_percent": 100 * len(intersection) / len(measurable),
        "input_delivery_percent": 100 * len(input_ids & received.keys()) / len(input_ids),
        "missing_eligible_ids": sorted(measurable - intersection),
        "decode_delay_ms": summarize(samples) if samples else None,
        "inter_decode_ms": summarize(intervals) if intervals else None,
        "server_settings": settings,
        "server": stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--ca-certificate", type=Path, help="Trust the preview CA for this replay")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 5 or args.warmup_frames < 0:
        parser.error("--frames must be at least five and warm-up must be nonnegative")
    report = asyncio.run(run(args.url.rstrip("/"), read_images(args.clip), args.frames,
                             args.warmup_frames, args.ca_certificate))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Recovered {report['decoded_eligible_ids']}/{report['eligible_input_ids']} eligible IDs")
    return 0 if report["decoded_eligible_ids"] and not report["invalid_return_tags"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
