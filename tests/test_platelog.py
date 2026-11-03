
import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import patch

import platelog


# ============================================================
# Helpers and fixtures
# ============================================================

def observation(text, confidence=0.9, sharp=20, crop=None, t=0.0):
    """Create a simulated plate observation."""
    return {
        "text": text,
        "conf": confidence,
        "sharp": sharp,
        "crop": crop if crop is not None else np.zeros((10, 10, 3)),
        "t": t,
    }


@pytest.fixture
def truck_image():
    return np.ones((30, 30, 3), dtype=np.uint8)


# ============================================================
# Plate decision tests
# ============================================================

def test_majority_vote():
    """The most frequently recognized plate should win."""
    obs = [
        observation("PUY052"),
        observation("PUY052"),
        observation("PUY427"),
        observation("PUY052"),
    ]

    plate, confidence = platelog.decide_plate(
        obs, min_reads=3, min_conf=0.5
    )

    assert plate == "PUY052"
    assert confidence > 0.5


def test_insufficient_agreement():
    """A plate needs enough agreeing observations."""
    obs = [
        observation("PUY052"),
        observation("PUY052"),
        observation("PUY427"),
    ]

    plate, _ = platelog.decide_plate(
        obs, min_reads=3, min_conf=0.5
    )

    assert plate == "UNREADABLE"


def test_low_confidence():
    """Repeated readings with low OCR confidence should be rejected."""
    obs = [
        observation("PUY052", confidence=0.3),
        observation("PUY052", confidence=0.3),
        observation("PUY052", confidence=0.3),
    ]

    plate, confidence = platelog.decide_plate(
        obs, min_reads=3, min_conf=0.5
    )

    assert plate == "UNREADABLE"
    assert confidence < 0.5


def test_invalid_readings_are_ignored():
    """Readings outside the accepted 4-8 character range are ignored."""
    obs = [
        observation(""),
        observation("ABC"),
        observation("PUY052"),
        observation("PUY052"),
        observation("PUY052"),
    ]

    plate, _ = platelog.decide_plate(
        obs, min_reads=3, min_conf=0.5
    )

    assert plate == "PUY052"


def test_no_valid_readings():
    """No valid OCR readings should produce UNREADABLE."""
    obs = [
        observation(""),
        observation("ABC"),
        observation("ABCDEFGHI"),
    ]

    plate, confidence = platelog.decide_plate(
        obs, min_reads=3, min_conf=0.5
    )

    assert plate == "UNREADABLE"
    assert confidence == 0.0


def test_majority_confidence_uses_agreement_ratio():
    """Confidence should account for disagreement between readings."""
    obs = [
        observation("PUY052", confidence=0.9),
        observation("PUY052", confidence=0.9),
        observation("PUY052", confidence=0.9),
        observation("PUY427", confidence=0.9),
    ]

    plate, confidence = platelog.decide_plate(
        obs, min_reads=3, min_conf=0.5
    )

    assert plate == "PUY052"
    assert confidence == pytest.approx(0.675)


# ============================================================
# Event generation tests
# ============================================================

def test_sharpest_plate_crop_is_selected(tmp_path):
    """The sharpest crop among agreeing readings should be saved."""
    blurry = np.zeros((10, 10, 3), dtype=np.uint8)
    sharp = np.ones((10, 10, 3), dtype=np.uint8)

    truck = {
        "image": np.zeros((30, 30, 3), dtype=np.uint8),
        "obs": [
            observation("PUY052", sharp=12, crop=blurry),
            observation("PUY052", sharp=30, crop=sharp),
            observation("PUY052", sharp=20, crop=blurry),
        ],
    }

    saved = {}

    with patch.object(
        platelog.cv2,
        "imwrite",
        side_effect=lambda path, image: saved.update(image=image) or True,
    ):
        event = platelog.make_event(
            Path("clip.mp4"), 1, truck, None, tmp_path
        )

    assert event["plate"] == "PUY052"
    assert np.array_equal(saved["image"], sharp)


def test_unreadable_plate_uses_truck_image(tmp_path, truck_image):
    """If the plate is unreadable, save the best truck image instead."""
    truck = {
        "image": truck_image,
        "obs": [
            observation("PUY052"),
            observation("PUY427"),
        ],
    }

    saved = {}

    with patch.object(
        platelog.cv2,
        "imwrite",
        side_effect=lambda path, image: saved.update(image=image) or True,
    ):
        event = platelog.make_event(
            Path("clip.mp4"), 1, truck, None, tmp_path
        )

    assert event["plate"] == "UNREADABLE"
    assert np.array_equal(saved["image"], truck_image)
    assert event["first_seen"] == "MISSING"
    assert event["last_seen"] == "MISSING"


def test_event_timestamps_are_formatted(tmp_path):
    """Event timestamps should be calculated from clip start time."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    start = datetime(
        2026, 11, 3, 12, 0, 0,
        tzinfo=ZoneInfo(platelog.SITE_TIMEZONE),
    )

    truck = {
        "image": np.zeros((30, 30, 3), dtype=np.uint8),
        "obs": [
            observation("PUY052", t=1.0),
            observation("PUY052", t=3.0),
            observation("PUY052", t=5.0),
        ],
    }

    with patch.object(platelog.cv2, "imwrite", return_value=True):
        event = platelog.make_event(
            Path("clip.mp4"), 1, truck, start, tmp_path
        )

    assert event["first_seen"] == (start + platelog.timedelta(seconds=1)).strftime(
        platelog.TIME_FORMAT
    )
    assert event["last_seen"] == (start + platelog.timedelta(seconds=5)).strftime(
        platelog.TIME_FORMAT
    )


# ============================================================
# Fake video and model objects
# ============================================================

class FakeTensor:
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, boxes, ids):
        self.xyxy = FakeTensor(boxes)
        self.id = FakeTensor(ids)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)

    def get(self, prop):
        return 5.0

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        pass


class FakeYOLO:
    def __init__(self, results):
        self.results = iter(results)

    def track(self, *args, **kwargs):
        return [next(self.results)]


class FakeALPR:
    class Detector:
        def predict(self, image):
            return []

    def __init__(self):
        self.detector = self.Detector()


# ============================================================
# Truck tracking and deduplication tests
# ============================================================

def test_same_tracking_id_produces_one_truck():
    """Repeated detections with the same tracking ID are grouped."""
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(3)
    ]

    results = [
        FakeResult(FakeBoxes([[0, 0, 20, 20]], [7]))
        for _ in range(3)
    ]

    yolo = FakeYOLO(results)

    with patch.object(
        platelog.cv2,
        "VideoCapture",
        return_value=FakeCapture(frames),
    ):
        trucks = platelog.read_clip(
            Path("clip.mp4"), yolo, FakeALPR()
        )

    assert len(trucks) == 1
    assert len(trucks[0]["obs"]) == 3


def test_different_tracking_ids_produce_separate_trucks():
    """Different tracking IDs should create separate truck entries."""
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(3)
    ]

    results = [
        FakeResult(
            FakeBoxes(
                [[0, 0, 20, 20], [30, 30, 50, 50]],
                [7, 8],
            )
        )
        for _ in range(3)
    ]

    yolo = FakeYOLO(results)

    with patch.object(
        platelog.cv2,
        "VideoCapture",
        return_value=FakeCapture(frames),
    ):
        trucks = platelog.read_clip(
            Path("clip.mp4"), yolo, FakeALPR()
        )

    assert len(trucks) == 2


def test_trucks_below_minimum_frames_are_discarded():
    """Tracks with too few observations should be filtered out."""
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(3)
    ]

    results = [
        FakeResult(FakeBoxes([[0, 0, 20, 20]], [7])),
        FakeResult(FakeBoxes([[0, 0, 20, 20]], [7])),
        FakeResult(FakeBoxes([], [])),
    ]

    yolo = FakeYOLO(results)

    with patch.object(
        platelog.cv2,
        "VideoCapture",
        return_value=FakeCapture(frames),
    ):
        trucks = platelog.read_clip(
            Path("clip.mp4"), yolo, FakeALPR()
        )

    assert len(trucks) == 0


def test_largest_truck_crop_supersedes_previous_image():
    """The largest detected truck crop should be retained."""
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(3)
    ]

    results = [
        FakeResult(FakeBoxes([[0, 0, 10, 10]], [7])),
        FakeResult(FakeBoxes([[0, 0, 20, 20]], [7])),
        FakeResult(FakeBoxes([[0, 0, 15, 15]], [7])),
    ]

    yolo = FakeYOLO(results)

    with patch.object(
        platelog.cv2,
        "VideoCapture",
        return_value=FakeCapture(frames),
    ):
        trucks = platelog.read_clip(
            Path("clip.mp4"), yolo, FakeALPR()
        )

    assert len(trucks) == 1
    assert trucks[0]["image"].shape[:2] == (20, 20)
