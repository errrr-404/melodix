"""Unit tests for :mod:`melodix.vision.detector`.

Everything here runs without torch, without ultralytics, and without a
checkpoint. The seam that makes that possible is the ``model_factory``
argument to :class:`SymbolDetector`: the real one imports ultralytics, and
these tests inject one that returns a stub exposing the same three attributes
ultralytics puts on a result — ``xyxyn``, ``conf`` and ``cls``.

That is a deliberate limitation worth stating. These tests prove the wrapper
logic — lazy loading, threshold handling, clamping, ordering, schema
validation — and prove nothing whatsoever about detection accuracy. Only a
trained checkpoint against held-out pages can do that.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from melodix.vision.dataset import BoundingBox
from melodix.vision.detector import (
    Detection,
    DetectorConfig,
    DetectorNotAvailableError,
    PageDetections,
    SymbolDetector,
)
from melodix.vision.labels import NUM_CLASSES, SymbolCategory, SymbolClass

PAGE_W = 800
PAGE_H = 600


class FakeBoxes:
    """Stands in for an ultralytics ``Boxes``.

    Exposes exactly the three attributes the parser reads, as plain lists, so
    no tensor library is involved.
    """

    def __init__(self, rows):
        self.xyxyn = [row[:4] for row in rows]
        self.conf = [row[4] for row in rows]
        self.cls = [row[5] for row in rows]


class FakeResult:
    """One page of results."""

    def __init__(self, rows=None, boxes=None):
        self.boxes = boxes if rows is None else FakeBoxes(rows)


class FakeModel:
    """Records how it was called and replays canned results."""

    def __init__(self, results=None):
        self._results = [FakeResult([])] if results is None else results
        self.calls: list[dict] = []
        self.moved_to: str | None = None

    def predict(self, image, **kwargs):
        self.calls.append({"image": image, **kwargs})
        return self._results

    def to(self, device):
        self.moved_to = device
        return self


def detector_with(rows=None, results=None, weights=None, tmp_path=None, **config_kwargs):
    """Build a detector backed by a fake model, plus the model itself."""
    if weights is None:
        weights = tmp_path / "fake.pt"
        weights.write_bytes(b"not a real checkpoint")

    if results is None:
        results = [FakeResult(rows if rows is not None else [])]
    model = FakeModel(results)

    config = DetectorConfig(weights=weights, **config_kwargs)
    detector = SymbolDetector(config, model_factory=lambda path, cfg: model)
    return detector, model


def page(width=PAGE_W, height=PAGE_H):
    """A blank grayscale page."""
    return np.full((height, width), 255, dtype=np.uint8)


# A box occupying the middle tenth of the page, as normalised xyxy.
CENTRE_BOX = (0.45, 0.45, 0.55, 0.55)


# --------------------------------------------------------------------------- #
# The lazy import contract
# --------------------------------------------------------------------------- #


def test_importing_the_detector_does_not_pull_in_torch():
    """The whole point of the module: Stage 1 and Stage 3 never pay for it.

    Checked in a fresh interpreter rather than against this process's
    ``sys.modules``. Once any test in the suite touches ultralytics, a
    process-wide assertion would report whatever ran first rather than what
    importing the detector actually costs.
    """
    probe = (
        "import sys; import melodix.vision.detector; "
        "print(int('torch' in sys.modules), int('ultralytics' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    assert result.stdout.split() == ["0", "0"], f"detector imported: {result.stdout!r}"


def test_building_a_detector_does_not_load_the_model(tmp_path):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"x")
    called = []

    detector = SymbolDetector(
        DetectorConfig(weights=weights),
        model_factory=lambda path, cfg: called.append(path) or FakeModel(),
    )

    assert not detector.is_loaded
    assert called == []


def test_the_model_loads_on_first_detection(tmp_path):
    detector, model = detector_with(tmp_path=tmp_path)

    assert not detector.is_loaded
    detector.detect(page())

    assert detector.is_loaded
    assert len(model.calls) == 1


def test_loading_can_be_forced_up_front(tmp_path):
    """So a missing checkpoint fails at startup, not after a page ingest."""
    detector, _ = detector_with(tmp_path=tmp_path)

    detector.load()

    assert detector.is_loaded


def test_loading_twice_only_builds_one_model(tmp_path):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"x")
    built = []

    detector = SymbolDetector(
        DetectorConfig(weights=weights),
        model_factory=lambda path, cfg: built.append(1) or FakeModel(),
    )
    detector.load()
    detector.load()
    detector.detect(page())

    assert len(built) == 1


def test_a_missing_checkpoint_is_reported_before_the_model_is_built(tmp_path):
    built = []
    detector = SymbolDetector(
        DetectorConfig(weights=tmp_path / "absent.pt"),
        model_factory=lambda path, cfg: built.append(1) or FakeModel(),
    )

    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        detector.load()
    assert built == []


def test_the_missing_extra_error_names_the_install_command():
    """A bare ImportError sends the reader to the wrong problem."""
    error = DetectorNotAvailableError('install the vision extra: pip install -e ".[dev,vision]"')

    assert "vision" in str(error)


def test_the_unavailable_error_can_be_caught_as_an_import_error():
    assert issubclass(DetectorNotAvailableError, ImportError)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_threshold", -0.1),
        ("confidence_threshold", 1.1),
        ("iou_threshold", -0.1),
        ("iou_threshold", 1.1),
        ("image_size", 0),
        ("image_size", 16),
        ("max_detections", 0),
        ("max_detections", -5),
    ],
)
def test_invalid_config_values_are_rejected(field, value, tmp_path):
    with pytest.raises(ValueError, match=field):
        DetectorConfig(weights=tmp_path / "w.pt", **{field: value})


def test_an_image_size_off_the_stride_is_rejected():
    """YOLO downsamples by 32 and silently rounds, which would make the
    reported and actual inference size disagree.
    """
    with pytest.raises(ValueError, match="multiple of 32"):
        DetectorConfig(weights=Path("w.pt"), image_size=1000)


def test_the_default_image_size_suits_sheet_music():
    """640 leaves a notehead a few pixels across on a 300 DPI page."""
    assert DetectorConfig(weights=Path("w.pt")).image_size >= 1280


def test_the_default_detection_cap_clears_a_dense_page():
    """The ultralytics default of 300 truncates an ensemble page."""
    assert DetectorConfig(weights=Path("w.pt")).max_detections > 300


def test_the_config_is_reachable_from_the_detector(tmp_path):
    detector, _ = detector_with(tmp_path=tmp_path, image_size=1600)

    assert detector.config.image_size == 1600


# --------------------------------------------------------------------------- #
# What gets passed to the model
# --------------------------------------------------------------------------- #


def test_configured_settings_reach_the_model(tmp_path):
    detector, model = detector_with(
        tmp_path=tmp_path,
        confidence_threshold=0.4,
        iou_threshold=0.6,
        image_size=960,
        max_detections=1234,
    )

    detector.detect(page())

    call = model.calls[0]
    assert call["conf"] == 0.4
    assert call["iou"] == 0.6
    assert call["imgsz"] == 960
    assert call["max_det"] == 1234


def test_a_per_call_threshold_overrides_the_configured_one(tmp_path):
    detector, model = detector_with(tmp_path=tmp_path, confidence_threshold=0.25)

    detector.detect(page(), min_confidence=0.8)

    assert model.calls[0]["conf"] == 0.8


def test_the_model_is_asked_to_stay_quiet(tmp_path):
    """Per-page ultralytics logging buries everything else in a batch run."""
    detector, model = detector_with(tmp_path=tmp_path)

    detector.detect(page())

    assert model.calls[0]["verbose"] is False


def test_an_explicit_device_is_passed_through(tmp_path):
    detector, model = detector_with(tmp_path=tmp_path, device="cpu")

    detector.detect(page())

    assert model.calls[0]["device"] == "cpu"


def test_auto_device_is_left_to_ultralytics(tmp_path):
    """Passing device="auto" verbatim would be rejected by ultralytics."""
    detector, model = detector_with(tmp_path=tmp_path, device="auto")

    detector.detect(page())

    assert "device" not in model.calls[0]


# --------------------------------------------------------------------------- #
# Image preparation
# --------------------------------------------------------------------------- #


def test_a_grayscale_page_is_expanded_to_three_channels(tmp_path):
    detector, model = detector_with(tmp_path=tmp_path)

    detector.detect(page())

    assert model.calls[0]["image"].shape == (PAGE_H, PAGE_W, 3)


def test_a_colour_page_is_passed_through(tmp_path):
    detector, model = detector_with(tmp_path=tmp_path)

    detector.detect(np.full((PAGE_H, PAGE_W, 3), 255, dtype=np.uint8))

    assert model.calls[0]["image"].shape == (PAGE_H, PAGE_W, 3)


def test_an_alpha_channel_is_dropped(tmp_path):
    detector, model = detector_with(tmp_path=tmp_path)

    detector.detect(np.full((PAGE_H, PAGE_W, 4), 255, dtype=np.uint8))

    assert model.calls[0]["image"].shape == (PAGE_H, PAGE_W, 3)


def test_a_single_channel_page_is_expanded(tmp_path):
    detector, model = detector_with(tmp_path=tmp_path)

    detector.detect(np.full((PAGE_H, PAGE_W, 1), 255, dtype=np.uint8))

    assert model.calls[0]["image"].shape == (PAGE_H, PAGE_W, 3)


def test_the_grayscale_expansion_repeats_the_channel(tmp_path):
    detector, model = detector_with(tmp_path=tmp_path)
    original = page()
    original[10, 20] = 0

    detector.detect(original)

    prepared = model.calls[0]["image"]
    assert list(prepared[10, 20]) == [0, 0, 0]


def test_an_empty_image_is_rejected(tmp_path):
    detector, _ = detector_with(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="empty image"):
        detector.detect(np.zeros((0, 0), dtype=np.uint8))


def test_an_unsupported_shape_is_rejected(tmp_path):
    detector, _ = detector_with(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="unsupported image shape"):
        detector.detect(np.zeros((10, 10, 2), dtype=np.uint8))


def test_the_page_is_not_loaded_when_the_image_is_invalid(tmp_path):
    """Validation happens before the checkpoint is read."""
    detector, _ = detector_with(tmp_path=tmp_path)

    with pytest.raises(ValueError):
        detector.detect(np.zeros((0, 0), dtype=np.uint8))
    assert not detector.is_loaded


# --------------------------------------------------------------------------- #
# Parsing results
# --------------------------------------------------------------------------- #


def test_a_detection_carries_its_class_and_score(tmp_path):
    detector, _ = detector_with(
        rows=[(*CENTRE_BOX, 0.9, int(SymbolClass.CROSS_NOTEHEAD))], tmp_path=tmp_path
    )

    found = detector.detect(page())

    assert len(found) == 1
    assert found.detections[0].symbol is SymbolClass.CROSS_NOTEHEAD
    assert found.detections[0].confidence == pytest.approx(0.9)


def test_normalised_corners_become_a_centre_and_size(tmp_path):
    detector, _ = detector_with(rows=[(0.2, 0.4, 0.6, 0.8, 0.9, 0)], tmp_path=tmp_path)

    hit = detector.detect(page()).detections[0]

    assert hit.box.cx == pytest.approx(0.4)
    assert hit.box.cy == pytest.approx(0.6)
    assert hit.box.w == pytest.approx(0.4)
    assert hit.box.h == pytest.approx(0.4)


def test_the_page_size_comes_back_with_the_detections(tmp_path):
    detector, _ = detector_with(rows=[(*CENTRE_BOX, 0.9, 0)], tmp_path=tmp_path)

    found = detector.detect(page(width=1024, height=768))

    assert (found.image_width, found.image_height) == (1024, 768)


def test_an_empty_result_yields_no_detections(tmp_path):
    detector, _ = detector_with(rows=[], tmp_path=tmp_path)

    assert len(detector.detect(page())) == 0


def test_a_result_without_boxes_is_tolerated(tmp_path):
    """Ultralytics sets boxes to None when nothing was found."""
    detector, _ = detector_with(results=[FakeResult(boxes=None)], tmp_path=tmp_path)

    assert len(detector.detect(page())) == 0


def test_detections_from_several_results_are_combined(tmp_path):
    detector, _ = detector_with(
        results=[
            FakeResult([(*CENTRE_BOX, 0.9, 0)]),
            FakeResult([(0.1, 0.1, 0.2, 0.2, 0.8, 2)]),
        ],
        tmp_path=tmp_path,
    )

    assert len(detector.detect(page())) == 2


def test_inverted_corners_are_normalised(tmp_path):
    """A box given bottom-right first still describes the same region."""
    detector, _ = detector_with(rows=[(0.6, 0.8, 0.2, 0.4, 0.9, 0)], tmp_path=tmp_path)

    hit = detector.detect(page()).detections[0]

    assert hit.box.cx == pytest.approx(0.4)
    assert hit.box.w == pytest.approx(0.4)


def test_a_ragged_result_is_rejected(tmp_path):
    """Mismatched box, score and class counts mean a malformed result."""
    boxes = FakeBoxes([(*CENTRE_BOX, 0.9, 0)])
    boxes.conf = [0.9, 0.8]
    detector, _ = detector_with(results=[FakeResult(boxes=boxes)], tmp_path=tmp_path)

    with pytest.raises(ValueError, match="inconsistent"):
        detector.detect(page())


def test_a_box_without_four_coordinates_is_rejected(tmp_path):
    boxes = FakeBoxes([(*CENTRE_BOX, 0.9, 0)])
    boxes.xyxyn = [(0.1, 0.2, 0.3)]
    detector, _ = detector_with(results=[FakeResult(boxes=boxes)], tmp_path=tmp_path)

    with pytest.raises(ValueError, match="expected 4 box coordinates"):
        detector.detect(page())


# --------------------------------------------------------------------------- #
# Clamping and filtering
# --------------------------------------------------------------------------- #


def test_a_box_spilling_past_the_page_edge_is_clamped(tmp_path):
    """YOLO routinely emits corners a hair outside [0, 1] for an edge symbol,
    and BoundingBox validates strictly. Clamping keeps a real detection.
    """
    detector, _ = detector_with(rows=[(-0.02, -0.01, 0.1, 0.1, 0.9, 0)], tmp_path=tmp_path)

    hit = detector.detect(page()).detections[0]

    assert hit.box.x_min >= 0.0
    assert hit.box.y_min >= 0.0


def test_a_box_spilling_past_the_far_edge_is_clamped(tmp_path):
    detector, _ = detector_with(rows=[(0.9, 0.9, 1.05, 1.02, 0.9, 0)], tmp_path=tmp_path)

    hit = detector.detect(page()).detections[0]

    assert hit.box.x_max <= 1.0
    assert hit.box.y_max <= 1.0


def test_a_box_entirely_off_the_page_is_dropped(tmp_path):
    """Clamping collapses it to no area, which is not a symbol."""
    detector, _ = detector_with(rows=[(1.2, 1.2, 1.4, 1.4, 0.9, 0)], tmp_path=tmp_path)

    assert len(detector.detect(page())) == 0


def test_a_zero_area_box_is_dropped(tmp_path):
    detector, _ = detector_with(rows=[(0.5, 0.5, 0.5, 0.5, 0.9, 0)], tmp_path=tmp_path)

    assert len(detector.detect(page())) == 0


def test_low_scoring_detections_are_filtered(tmp_path):
    """Applied again after parsing, since an injected model may ignore conf."""
    detector, _ = detector_with(
        rows=[(*CENTRE_BOX, 0.9, 0), (0.1, 0.1, 0.2, 0.2, 0.1, 0)],
        tmp_path=tmp_path,
        confidence_threshold=0.5,
    )

    assert len(detector.detect(page())) == 1


def test_the_per_call_threshold_filters_too(tmp_path):
    detector, _ = detector_with(
        rows=[(*CENTRE_BOX, 0.9, 0), (0.1, 0.1, 0.2, 0.2, 0.6, 0)],
        tmp_path=tmp_path,
        confidence_threshold=0.1,
    )

    assert len(detector.detect(page(), min_confidence=0.8)) == 1


def test_a_confidence_above_one_is_clamped(tmp_path):
    detector, _ = detector_with(rows=[(*CENTRE_BOX, 1.000001, 0)], tmp_path=tmp_path)

    assert detector.detect(page()).detections[0].confidence == 1.0


def test_a_class_outside_the_schema_is_rejected(tmp_path):
    """The checkpoint was trained against a different label set; guessing
    would report the wrong drum for every symbol of that class.
    """
    detector, _ = detector_with(rows=[(*CENTRE_BOX, 0.9, NUM_CLASSES)], tmp_path=tmp_path)

    with pytest.raises(ValueError, match="outside the schema"):
        detector.detect(page())


def test_the_schema_error_names_the_label_module(tmp_path):
    detector, _ = detector_with(rows=[(*CENTRE_BOX, 0.9, 99)], tmp_path=tmp_path)

    with pytest.raises(ValueError, match="melodix.vision.labels"):
        detector.detect(page())


# --------------------------------------------------------------------------- #
# Ordering and access
# --------------------------------------------------------------------------- #


def test_detections_come_back_in_confidence_order(tmp_path):
    detector, _ = detector_with(
        rows=[
            (0.1, 0.1, 0.2, 0.2, 0.3, 0),
            (0.3, 0.3, 0.4, 0.4, 0.9, 0),
            (0.5, 0.5, 0.6, 0.6, 0.6, 0),
        ],
        tmp_path=tmp_path,
    )

    scores = [hit.confidence for hit in detector.detect(page())]

    assert scores == sorted(scores, reverse=True)


def test_reading_order_runs_down_the_page_then_across(tmp_path):
    detector, _ = detector_with(
        rows=[
            (0.7, 0.7, 0.8, 0.8, 0.9, 0),  # low on the page, highest score
            (0.1, 0.1, 0.2, 0.2, 0.5, 0),  # top left
            (0.5, 0.1, 0.6, 0.2, 0.4, 0),  # top right
        ],
        tmp_path=tmp_path,
    )

    ordered = detector.detect(page()).in_reading_order()

    assert [round(hit.box.cx, 2) for hit in ordered] == [0.15, 0.55, 0.75]


def test_a_page_of_detections_is_iterable(tmp_path):
    detector, _ = detector_with(
        rows=[(*CENTRE_BOX, 0.9, 0), (0.1, 0.1, 0.2, 0.2, 0.8, 0)], tmp_path=tmp_path
    )

    found = detector.detect(page())

    assert len(list(found)) == len(found) == 2


def test_detections_can_be_filtered_by_category(tmp_path):
    detector, _ = detector_with(
        rows=[
            (0.1, 0.1, 0.2, 0.2, 0.9, int(SymbolClass.CROSS_NOTEHEAD)),
            (0.3, 0.3, 0.4, 0.4, 0.8, int(SymbolClass.ACCENT)),
            (0.5, 0.5, 0.6, 0.6, 0.7, int(SymbolClass.REST_QUARTER)),
        ],
        tmp_path=tmp_path,
    )

    found = detector.detect(page())

    assert len(found.of_category(SymbolCategory.NOTEHEAD)) == 1
    assert len(found.of_category(SymbolCategory.MODIFIER)) == 1
    assert len(found.of_category(SymbolCategory.REST)) == 1


def test_noteheads_are_the_detections_carrying_a_position(tmp_path):
    detector, _ = detector_with(
        rows=[
            (0.1, 0.1, 0.2, 0.2, 0.9, int(SymbolClass.CROSS_NOTEHEAD)),
            (0.3, 0.3, 0.4, 0.4, 0.8, int(SymbolClass.ROUND_NOTEHEAD)),
            (0.5, 0.5, 0.6, 0.6, 0.7, int(SymbolClass.ACCENT)),
        ],
        tmp_path=tmp_path,
    )

    found = detector.detect(page())

    assert len(found.noteheads()) == 2
    assert all(hit.carries_position for hit in found.noteheads())


def test_detections_can_be_thresholded_after_the_fact(tmp_path):
    detector, _ = detector_with(
        rows=[(0.1, 0.1, 0.2, 0.2, 0.9, 0), (0.3, 0.3, 0.4, 0.4, 0.4, 0)],
        tmp_path=tmp_path,
        confidence_threshold=0.1,
    )

    found = detector.detect(page())

    assert len(found.above_confidence(0.5)) == 1


# --------------------------------------------------------------------------- #
# The handoff to Stage 1
# --------------------------------------------------------------------------- #


def test_a_detection_reports_its_centre_in_pixels(tmp_path):
    """The row is what StaffGrid.snap turns into a staff position."""
    detector, _ = detector_with(rows=[(0.25, 0.5, 0.35, 0.6, 0.9, 0)], tmp_path=tmp_path)

    found = detector.detect(page())
    x, y = found.detections[0].center_pixels(found.image_width, found.image_height)

    assert x == pytest.approx(0.3 * PAGE_W)
    assert y == pytest.approx(0.55 * PAGE_H)


def test_a_notehead_centre_snaps_to_a_staff_position(tmp_path):
    """End to end across the stage boundary, with a grid built by hand."""
    from melodix.geometry.staff import StaffGrid, StaffLine

    grid = StaffGrid(
        lines=tuple(
            StaffLine(y=200.0 + step * 20, x_start=100, x_end=700, thickness=2)
            for step in range(5)
        )
    )
    # A notehead centred on row 240 of a 600px page is the middle line.
    cy = 240 / PAGE_H
    detector, _ = detector_with(
        rows=[(0.4, cy - 0.01, 0.45, cy + 0.01, 0.9, int(SymbolClass.ROUND_NOTEHEAD))],
        tmp_path=tmp_path,
    )

    found = detector.detect(page())
    hit = found.noteheads()[0]
    _, y = hit.center_pixels(found.image_width, found.image_height)

    assert grid.snap(y) == 4


def test_a_detection_reaches_its_schema_row(tmp_path):
    detector, _ = detector_with(
        rows=[(*CENTRE_BOX, 0.9, int(SymbolClass.ACCENT))], tmp_path=tmp_path
    )

    hit = detector.detect(page()).detections[0]

    assert hit.label.name == "accent"
    assert hit.category is SymbolCategory.MODIFIER
    assert not hit.carries_position


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #


def test_a_batch_returns_one_result_per_image(tmp_path):
    detector, _ = detector_with(rows=[(*CENTRE_BOX, 0.9, 0)], tmp_path=tmp_path)

    results = detector.detect_batch([page(), page(), page()])

    assert len(results) == 3
    assert all(len(result) == 1 for result in results)


def test_batch_members_keep_their_own_dimensions(tmp_path):
    """Pages of a scanned book vary in size; letterboxing them to a common
    shape would shift every normalised box.
    """
    detector, _ = detector_with(rows=[(*CENTRE_BOX, 0.9, 0)], tmp_path=tmp_path)

    results = detector.detect_batch([page(800, 600), page(1024, 768)])

    assert (results[0].image_width, results[0].image_height) == (800, 600)
    assert (results[1].image_width, results[1].image_height) == (1024, 768)


def test_an_empty_batch_returns_nothing(tmp_path):
    detector, _ = detector_with(tmp_path=tmp_path)

    assert detector.detect_batch([]) == []


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_a_detection_rejects_an_impossible_confidence(confidence):
    with pytest.raises(ValueError, match="confidence must be in"):
        Detection(SymbolClass.ACCENT, BoundingBox(0.5, 0.5, 0.1, 0.1), confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_the_confidence_bounds_are_inclusive(confidence):
    assert Detection(SymbolClass.ACCENT, BoundingBox(0.5, 0.5, 0.1, 0.1), confidence)


@pytest.mark.parametrize(("width", "height"), [(0, 600), (800, 0), (-1, 600)])
def test_a_page_rejects_an_impossible_size(width, height):
    with pytest.raises(ValueError, match="image size must be positive"):
        PageDetections(detections=(), image_width=width, image_height=height)


def test_an_empty_page_of_detections_is_valid():
    assert len(PageDetections(detections=(), image_width=800, image_height=600)) == 0


def test_a_detection_is_immutable():
    hit = Detection(SymbolClass.ACCENT, BoundingBox(0.5, 0.5, 0.1, 0.1), 0.9)

    with pytest.raises(AttributeError):
        hit.confidence = 0.5  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Fidelity of the stubs above
# --------------------------------------------------------------------------- #

pytestmark_vision = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None,
    reason="needs the vision extra: pip install -e '.[dev,vision]'",
)


@pytestmark_vision
def test_the_parser_reads_a_real_ultralytics_boxes():
    """The one thing the stubs cannot check about themselves.

    Every other test in this file asserts against FakeBoxes, so they all pass
    just as happily if the real ultralytics API differs from what FakeBoxes
    imitates. This builds a genuine `Boxes` and runs the same parser over it.
    """
    import torch
    from ultralytics.engine.results import Boxes

    from melodix.vision.detector import _detections_from_results

    # Real Boxes take pixel xyxy plus the source shape, and derive .xyxyn.
    boxes = Boxes(
        torch.tensor([[100.0, 200.0, 140.0, 240.0, 0.91, float(SymbolClass.CROSS_NOTEHEAD)]]),
        orig_shape=(PAGE_H, PAGE_W),
    )

    class Result:
        pass

    result = Result()
    result.boxes = boxes

    found = _detections_from_results([result], threshold=0.25)

    assert len(found) == 1
    assert found[0].symbol is SymbolClass.CROSS_NOTEHEAD
    assert found[0].confidence == pytest.approx(0.91, abs=1e-6)
    # x 100..140 of 800 -> centre 0.15, width 0.05
    assert found[0].box.cx == pytest.approx(0.15, abs=1e-6)
    assert found[0].box.w == pytest.approx(0.05, abs=1e-6)
    assert found[0].box.cy == pytest.approx(220 / PAGE_H, abs=1e-6)


@pytestmark_vision
def test_the_real_boxes_expose_the_attributes_the_stubs_imitate():
    """Guards FakeBoxes against an ultralytics API change."""
    import torch
    from ultralytics.engine.results import Boxes

    boxes = Boxes(torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.5, 0.0]]), orig_shape=(600, 800))

    for attribute in ("xyxyn", "conf", "cls"):
        assert hasattr(boxes, attribute), f"FakeBoxes imitates .{attribute}"
