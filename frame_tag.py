"""Encode the preview frame ID into the reserved bottom 24 RGB rows."""

import numpy as np

WIDTH = 640
IMAGE_HEIGHT = 360
TAG_ROWS = 24
HEIGHT = IMAGE_HEIGHT + TAG_ROWS
LEFT = 8
CELL_WIDTH = 13
MAGIC = 0xD3


def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ (0x07 if crc & 0x80 else 0)) & 0xFF
    return crc


def attach_tag(image, frame_id):
    if image.shape != (IMAGE_HEIGHT, WIDTH, 3) or image.dtype != np.uint8:
        raise ValueError("Expected a 640 by 360 RGB uint8 image")
    body = bytes([MAGIC]) + frame_id.to_bytes(4, "big")
    bits = np.unpackbits(np.frombuffer(body + bytes([crc8(body)]), dtype=np.uint8))
    tagged = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    tagged[:IMAGE_HEIGHT] = image
    tagged[IMAGE_HEIGHT:, LEFT:LEFT + len(bits) * CELL_WIDTH] = np.repeat(bits * 255, CELL_WIDTH)[None, :, None]
    return tagged


def read_tag(image):
    if image.shape != (HEIGHT, WIDTH, 3):
        return None
    cells = image[IMAGE_HEIGHT + 6:HEIGHT - 6, LEFT:LEFT + 48 * CELL_WIDTH]
    centers = cells.reshape(12, 48, CELL_WIDTH, 3)[:, :, 3:-3]
    bits = centers.mean(axis=(0, 2, 3)) >= 128
    packet = np.packbits(bits).tobytes()
    if packet[0] != MAGIC or crc8(packet[:-1]) != packet[-1]:
        return None
    return int.from_bytes(packet[1:5], "big")
