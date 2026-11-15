"""Frame ordering shared by offline replay and the WebRTC stream worker."""

from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

Payload = TypeVar("Payload")


@dataclass(frozen=True)
class Frame(Generic[Payload]):
    frame_id: int
    payload: Payload


class FrameHistory(Generic[Payload]):
    """Emit one complete window per arrival after the initial four frames."""

    def __init__(self, context: int = 5):
        if context < 1 or context % 2 == 0:
            raise ValueError("Temporal context must be positive and odd")
        self._frames: deque[Frame[Payload]] = deque(maxlen=context)
        self._context = context

    def push(self, frame: Frame[Payload]) -> tuple[Frame[Payload], ...] | None:
        if self._frames and frame.frame_id <= self._frames[-1].frame_id:
            raise ValueError("Frame IDs must increase within a stream")
        self._frames.append(frame)
        if len(self._frames) == self._context:
            return tuple(self._frames)
        return None


def process_window(window, denoise):
    """Call the model adapter with ordered payloads and retain the middle ID."""
    middle = window[len(window) // 2]
    return Frame(middle.frame_id, denoise(tuple(frame.payload for frame in window)))
