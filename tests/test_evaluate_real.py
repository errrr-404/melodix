"""Unit tests for ``scripts/evaluate_real.py``.

Metric code is where a plausible-looking bug does the most damage, because its
output is a number nobody can sanity-check by eye. So the matching and averaging
here are verified against cases worked out by hand — a perfect detector must
score exactly 1.0, a detector that finds nothing exactly 0.0, and a half-right
one exactly 0.5 — rather than against whatever the implementation happens to
produce.

A stub detector supplies fixed boxes, so none of this needs a checkpoint, torch,
or a GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from melodix.vision.dataset import Annotation, BoundingBox, write_label_file
from melodix.vision.detector import Detection, PageDetections
from melodix.vision.labels import SymbolClass

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_real.py"


def _load():
    """Import the evaluation script as a module."""
    spec = importlib.util.spec_from_file_location("evaluate_real", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load()

PAGE_W = 400
PAGE_H = 500


class StubDetector:
    """Returns canned detections, keyed by call order."""

    def __init__(self, per_page):
        self._per_page = list(per_page)
        self.calls = 0

    def detect(self, image):
        detections = self._per_page[min(self.calls, len(self._per_page) - 1)]
        self.calls += 1
        return PageDetections(
            detections=tuple(detections), image_width=PAGE_W, image_height=PAGE_H
        )


def detection(symbol, x0, y0, x1, y1, confidence=0.9) -> Detection:
    """A detection in pixel space."""
    return Detection(symbol, confidence, x0, y0, x1, y1)


def annotation(symbol, x0, y0, x1, y1) -> Annotation:
    """A stored annotation, normalised as on disk."""
    return Annotation(
        symbol=symbol, box=BoundingBox.from_pixels(x0, y0, x1, y1, PAGE_W, PAGE_H)
    )


def build(root: Path, pages, split: str = "val", ensemble: tuple[str, ...] = ()) -> Path:
    """Write a dataset: one blank image and label file per page of truth."""
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for index, truth in enumerate(pages):
        image = np.full((PAGE_H, PAGE_W), 255, np.uint8)
        cv2.imwrite(str(root / "images" / split / f"page_{index}.png"), image)
        write_label_file(root / "labels" / split / f"page_{index}.txt", tuple(truth))
    if ensemble:
        (root / ev.MANIFEST_NAME).write_text("\n".join(ensemble) + "\n", encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# Box overlap
# --------------------------------------------------------------------------- #


def test_a_box_fully_overlaps_itself():
    box = ev.Box(0, 10, 10, 30, 30)

    assert box.iou(box) == pytest.approx(1.0)


def test_disjoint_boxes_do_not_overlap():
    assert ev.Box(0, 0, 0, 10, 10).iou(ev.Box(0, 50, 50, 60, 60)) == 0.0


def test_half_overlapping_boxes_score_a_third():
    """Two equal boxes sharing half their area: 0.5 over 1.5."""
    left = ev.Box(0, 0, 0, 20, 10)
    right = ev.Box(0, 10, 0, 30, 10)

    assert left.iou(right) == pytest.approx(1 / 3)


def test_overlap_never_exceeds_one():
    box = ev.Box(0, 1.1, 2.2, 3.3, 4.4)

    assert box.iou(box) <= 1.0


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def test_an_exact_prediction_matches():
    truth = [ev.Box(0, 10, 10, 30, 30)]
    predictions = [ev.Box(0, 10, 10, 30, 30)]

    pairs, spare, missed = ev.match_boxes(truth, predictions)

    assert len(pairs) == 1
    assert spare == []
    assert missed == []


def test_a_prediction_of_the_wrong_class_does_not_match():
    """Right place, wrong symbol, is a miss and a false positive both."""
    truth = [ev.Box(0, 10, 10, 30, 30)]
    predictions = [ev.Box(5, 10, 10, 30, 30)]

    pairs, spare, missed = ev.match_boxes(truth, predictions)

    assert pairs == []
    assert len(spare) == 1
    assert len(missed) == 1


def test_a_prediction_below_the_iou_threshold_does_not_match():
    truth = [ev.Box(0, 0, 0, 20, 20)]
    predictions = [ev.Box(0, 18, 18, 38, 38)]

    pairs, spare, missed = ev.match_boxes(truth, predictions, iou_threshold=0.5)

    assert pairs == []
    assert len(spare) == len(missed) == 1


def test_only_one_prediction_can_claim_a_truth_box():
    """Two overlapping predictions on one symbol is one hit and one duplicate."""
    truth = [ev.Box(0, 10, 10, 30, 30)]
    predictions = [
        ev.Box(0, 10, 10, 30, 30, confidence=0.9),
        ev.Box(0, 11, 11, 31, 31, confidence=0.8),
    ]

    pairs, spare, missed = ev.match_boxes(truth, predictions)

    assert len(pairs) == 1
    assert len(spare) == 1
    assert missed == []


def test_the_most_confident_prediction_claims_the_box():
    """Greedy by confidence, as COCO and ultralytics do."""
    truth = [ev.Box(0, 10, 10, 30, 30)]
    predictions = [
        ev.Box(0, 12, 12, 32, 32, confidence=0.4),
        ev.Box(0, 10, 10, 30, 30, confidence=0.95),
    ]

    pairs, _, _ = ev.match_boxes(truth, predictions)

    assert pairs[0][1].confidence == pytest.approx(0.95)


def test_unfound_truth_is_reported_missing():
    pairs, spare, missed = ev.match_boxes([ev.Box(0, 0, 0, 10, 10)], [])

    assert pairs == [] and spare == []
    assert len(missed) == 1


def test_predictions_with_no_truth_are_all_spare():
    pairs, spare, missed = ev.match_boxes([], [ev.Box(0, 0, 0, 10, 10)])

    assert pairs == [] and missed == []
    assert len(spare) == 1


# --------------------------------------------------------------------------- #
# Metrics against hand-worked cases
# --------------------------------------------------------------------------- #


def test_a_perfect_detector_scores_one():
    truth = [ev.Box(0, 10, 10, 30, 30), ev.Box(1, 50, 50, 70, 70)]
    evaluation = ev.Evaluation()

    evaluation.add_page(truth, list(truth), 0.5)

    aggregate = evaluation.aggregate()
    assert aggregate["precision"] == pytest.approx(1.0)
    assert aggregate["recall"] == pytest.approx(1.0)
    assert aggregate["map50"] == pytest.approx(1.0)


def test_a_detector_that_finds_nothing_scores_zero():
    evaluation = ev.Evaluation()

    evaluation.add_page([ev.Box(0, 10, 10, 30, 30)], [], 0.5)

    aggregate = evaluation.aggregate()
    assert aggregate["recall"] == 0.0
    assert aggregate["map50"] == 0.0


def test_finding_half_the_symbols_halves_recall():
    truth = [ev.Box(0, 0, 0, 10, 10), ev.Box(0, 50, 50, 60, 60)]
    evaluation = ev.Evaluation()

    evaluation.add_page(truth, [truth[0]], 0.5)

    assert evaluation.tallies[0].recall == pytest.approx(0.5)
    assert evaluation.tallies[0].precision == pytest.approx(1.0)


def test_one_right_and_one_wrong_halves_precision():
    truth = [ev.Box(0, 0, 0, 10, 10)]
    predictions = [ev.Box(0, 0, 0, 10, 10), ev.Box(0, 200, 200, 210, 210)]
    evaluation = ev.Evaluation()

    evaluation.add_page(truth, predictions, 0.5)

    assert evaluation.tallies[0].precision == pytest.approx(0.5)
    assert evaluation.tallies[0].recall == pytest.approx(1.0)


def test_a_class_with_no_ground_truth_is_left_out_of_the_average():
    """Otherwise 27 absent classes drag every aggregate toward zero."""
    evaluation = ev.Evaluation()
    evaluation.add_page([ev.Box(0, 0, 0, 10, 10)], [ev.Box(0, 0, 0, 10, 10)], 0.5)

    assert evaluation.aggregate()["classes"] == 1
    assert evaluation.aggregate()["map50"] == pytest.approx(1.0)


def test_average_precision_is_one_when_every_prediction_is_right():
    assert ev.average_precision([(0.9, True), (0.8, True)], support=2) == pytest.approx(1.0)


def test_average_precision_is_zero_without_ground_truth():
    assert ev.average_precision([(0.9, False)], support=0) == 0.0


def test_average_precision_rewards_ranking_hits_first():
    """The property AP exists to measure: confident-and-right beats
    confident-and-wrong, at identical precision and recall.
    """
    good = ev.average_precision([(0.9, True), (0.5, False)], support=1)
    bad = ev.average_precision([(0.9, False), (0.5, True)], support=1)

    assert good > bad


def test_support_counts_ground_truth_not_predictions():
    evaluation = ev.Evaluation()
    evaluation.add_page(
        [ev.Box(0, 0, 0, 10, 10)],
        [ev.Box(0, 0, 0, 10, 10), ev.Box(0, 99, 99, 109, 109)],
        0.5,
    )

    assert evaluation.tallies[0].support == 1


# --------------------------------------------------------------------------- #
# Confusion
# --------------------------------------------------------------------------- #


def test_a_class_confusion_is_recorded():
    truth = [ev.Box(int(SymbolClass.AUGMENTATION_DOT), 10, 10, 20, 20)]
    predictions = [ev.Box(int(SymbolClass.STACCATO), 10, 10, 20, 20)]
    evaluation = ev.Evaluation()

    evaluation.add_page(truth, predictions, 0.5)

    key = (int(SymbolClass.AUGMENTATION_DOT), int(SymbolClass.STACCATO))
    assert evaluation.confusion[key] == 1


def test_a_false_positive_over_nothing_is_background():
    evaluation = ev.Evaluation()

    evaluation.add_page([], [ev.Box(0, 10, 10, 20, 20)], 0.5)

    assert evaluation.confusion[(-1, 0)] == 1


def test_a_missed_symbol_is_recorded_as_missed():
    evaluation = ev.Evaluation()

    evaluation.add_page([ev.Box(3, 10, 10, 20, 20)], [], 0.5)

    assert evaluation.confusion[(3, -1)] == 1


def test_the_confusable_pair_is_the_documented_one():
    assert ev.CONFUSABLE_PAIR == ("augmentation_dot", "staccato")


# --------------------------------------------------------------------------- #
# The ensemble slice
# --------------------------------------------------------------------------- #


def test_ensemble_pages_are_read_from_a_manifest(tmp_path):
    root = build(tmp_path, [[], []], ensemble=("page_1",))

    assert ev.load_manifest(root) == {"page_1"}


def test_manifest_comments_and_blank_lines_are_ignored(tmp_path):
    root = build(tmp_path, [[]])
    (root / ev.MANIFEST_NAME).write_text(
        "# ensemble pages\n\npage_0\n  # trailing\n", encoding="utf-8"
    )

    assert ev.load_manifest(root) == {"page_0"}


def test_a_directory_named_ensemble_marks_its_pages(tmp_path):
    """The convention alternative to a manifest."""
    folder = tmp_path / "images" / "val" / "ensemble"
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "duet.png"), np.full((10, 10), 255, np.uint8))

    assert "duet" in ev.load_manifest(tmp_path)


def test_no_marking_yields_no_ensemble_pages(tmp_path):
    root = build(tmp_path, [[]])

    assert ev.load_manifest(root) == set()


def test_the_ensemble_slice_is_reported_separately(tmp_path):
    """The whole point: a strong single-staff average must not be able to hide
    a weak ensemble result.
    """
    hit = annotation(SymbolClass.ROUND_NOTEHEAD, 10, 10, 30, 30)
    root = build(tmp_path, [[hit], [hit]], ensemble=("page_1",))

    # Page 0 (single) detected correctly, page 1 (ensemble) missed entirely.
    stub = StubDetector([[detection(SymbolClass.ROUND_NOTEHEAD, 10, 10, 30, 30)], []])
    report = ev.evaluate(Path("unused.pt"), root, detector=stub)

    assert report["ensemble"]["pages"] == 1
    assert report["single_staff"]["pages"] == 1
    assert report["single_staff"]["aggregate"]["recall"] == pytest.approx(1.0)
    assert report["ensemble"]["aggregate"]["recall"] == pytest.approx(0.0)


def test_the_aggregate_would_have_hidden_the_ensemble_failure(tmp_path):
    """Demonstrates why the slice exists, using the same data."""
    hit = annotation(SymbolClass.ROUND_NOTEHEAD, 10, 10, 30, 30)
    root = build(tmp_path, [[hit], [hit]], ensemble=("page_1",))
    stub = StubDetector([[detection(SymbolClass.ROUND_NOTEHEAD, 10, 10, 30, 30)], []])

    report = ev.evaluate(Path("unused.pt"), root, detector=stub)

    assert report["aggregate"]["recall"] == pytest.approx(0.5)
    assert report["ensemble"]["aggregate"]["recall"] == pytest.approx(0.0)


def test_the_report_warns_when_ensembles_are_worse(tmp_path, capsys):
    hit = annotation(SymbolClass.ROUND_NOTEHEAD, 10, 10, 30, 30)
    root = build(tmp_path, [[hit], [hit]], ensemble=("page_1",))
    stub = StubDetector([[detection(SymbolClass.ROUND_NOTEHEAD, 10, 10, 30, 30)], []])

    ev.print_report(ev.evaluate(Path("unused.pt"), root, detector=stub))

    assert "ENSEMBLE IS MEASURABLY WORSE" in capsys.readouterr().out


def test_the_report_says_when_nothing_is_marked(tmp_path, capsys):
    root = build(tmp_path, [[annotation(SymbolClass.ROUND_NOTEHEAD, 10, 10, 30, 30)]])
    stub = StubDetector([[]])

    ev.print_report(ev.evaluate(Path("unused.pt"), root, detector=stub))

    assert "none marked" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The synthetic warning
# --------------------------------------------------------------------------- #


def test_a_synthetic_dataset_is_detected(tmp_path):
    folder = tmp_path / "images" / "val"
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "synthetic_00007.png"), np.full((10, 10), 255, np.uint8))

    assert ev.looks_synthetic(tmp_path)


def test_a_real_dataset_is_not_flagged(tmp_path):
    root = build(tmp_path, [[]])

    assert not ev.looks_synthetic(root)


def test_the_synthetic_warning_is_printed(tmp_path, capsys):
    """Scoring against the training generator produces a beautiful number that
    means nothing, so it must be impossible to miss.
    """
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    cv2.imwrite(
        str(tmp_path / "images" / "val" / "synthetic_00000.png"),
        np.full((PAGE_H, PAGE_W), 255, np.uint8),
    )
    write_label_file(tmp_path / "labels" / "val" / "synthetic_00000.txt", ())

    report = ev.evaluate(Path("unused.pt"), tmp_path, detector=StubDetector([[]]))
    ev.print_report(report)

    out = capsys.readouterr().out
    assert report["synthetic_source"]
    assert "say nothing about real pages" in out


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_a_perfect_run_scores_one_end_to_end(tmp_path):
    truth = [
        annotation(SymbolClass.CROSS_NOTEHEAD, 10, 10, 30, 30),
        annotation(SymbolClass.ROUND_NOTEHEAD, 100, 100, 120, 120),
    ]
    root = build(tmp_path, [truth])
    stub = StubDetector(
        [
            [
                detection(SymbolClass.CROSS_NOTEHEAD, 10, 10, 30, 30),
                detection(SymbolClass.ROUND_NOTEHEAD, 100, 100, 120, 120),
            ]
        ]
    )

    report = ev.evaluate(Path("unused.pt"), root, detector=stub)

    assert report["pages"] == 1
    assert report["aggregate"]["map50"] == pytest.approx(1.0)
    assert report["confusion"] == []


def test_every_page_is_scored(tmp_path):
    root = build(tmp_path, [[], [], []])
    stub = StubDetector([[]])

    report = ev.evaluate(Path("unused.pt"), root, detector=stub)

    assert report["pages"] == 3
    assert stub.calls == 3


def test_a_missing_split_is_reported(tmp_path):
    root = build(tmp_path, [[]])

    with pytest.raises(FileNotFoundError, match="no train images"):
        ev.evaluate(Path("unused.pt"), root, split="train", detector=StubDetector([[]]))


def test_the_report_is_json_serialisable(tmp_path):
    import json

    root = build(tmp_path, [[annotation(SymbolClass.ROUND_NOTEHEAD, 10, 10, 30, 30)]])
    stub = StubDetector([[detection(SymbolClass.ROUND_NOTEHEAD, 10, 10, 30, 30)]])

    report = ev.evaluate(Path("unused.pt"), root, detector=stub)

    assert json.loads(json.dumps(report))["pages"] == 1


def test_the_iou_threshold_is_honoured(tmp_path):
    truth = [annotation(SymbolClass.ROUND_NOTEHEAD, 0, 0, 40, 40)]
    root = build(tmp_path, [truth])
    # Overlaps about 0.39 of the union: passes at 0.3, fails at 0.5.
    stub_boxes = [detection(SymbolClass.ROUND_NOTEHEAD, 10, 10, 50, 50)]

    lax = ev.evaluate(
        Path("u.pt"), root, iou_threshold=0.3, detector=StubDetector([stub_boxes])
    )
    strict = ev.evaluate(
        Path("u.pt"), root, iou_threshold=0.5, detector=StubDetector([stub_boxes])
    )

    assert lax["aggregate"]["recall"] == pytest.approx(1.0)
    assert strict["aggregate"]["recall"] == pytest.approx(0.0)


def test_all_twenty_eight_classes_appear_in_the_report(tmp_path):
    """Including those with no ground truth, so a silently absent class is
    visible rather than simply missing from the table.
    """
    root = build(tmp_path, [[]])

    report = ev.evaluate(Path("unused.pt"), root, detector=StubDetector([[]]))

    assert len(report["per_class"]) == 28


# --------------------------------------------------------------------------- #
# Metrics the application actually consumes
# --------------------------------------------------------------------------- #


def test_the_snap_tolerance_comes_from_the_geometry_module():
    """Derived, not hard-coded, so it tracks StaffGrid.snap.

    One staff position is half a line spacing, and snap's default tolerance is
    0.4 positions, so at 14 px spacing it absorbs +/-2.8 px.
    """
    import inspect

    from melodix.geometry.staff import StaffGrid

    tolerance = inspect.signature(StaffGrid.snap).parameters["tolerance"].default

    assert ev.snap_tolerance_px(14.0) == pytest.approx(tolerance * 7.0)
    assert ev.snap_tolerance_px(14.0) == pytest.approx(2.8)


def test_the_tolerance_scales_with_staff_spacing():
    assert ev.snap_tolerance_px(28.0) == pytest.approx(2 * ev.snap_tolerance_px(14.0))


def test_a_small_class_is_identified_by_its_measured_box(tmp_path):
    """Measured per corpus rather than listed, so the set adapts when the
    engraving size does.
    """
    truth = [
        annotation(SymbolClass.AUGMENTATION_DOT, 200, 200, 206, 206),
        annotation(SymbolClass.CROSS_NOTEHEAD, 100, 100, 119, 119),
    ]
    root = build(tmp_path, [truth])
    stub = StubDetector([[detection(SymbolClass.AUGMENTATION_DOT, 200, 200, 206, 206)]])

    report = ev.evaluate(Path("u.pt"), root, detector=stub)

    assert report["small_classes"]["classes"] == ["augmentation_dot"]


def test_a_large_class_is_not_called_small(tmp_path):
    truth = [annotation(SymbolClass.CROSS_NOTEHEAD, 100, 100, 119, 119)]
    root = build(tmp_path, [truth])

    report = ev.evaluate(Path("u.pt"), root, detector=StubDetector([[]]))

    assert "cross_notehead" not in report["small_classes"]["classes"]


def test_centroid_error_is_measured_in_pixels(tmp_path):
    """A prediction offset two pixels down must report two pixels, not an IoU."""
    truth = [annotation(SymbolClass.CROSS_NOTEHEAD, 100, 100, 120, 120)]
    root = build(tmp_path, [truth])
    stub = StubDetector([[detection(SymbolClass.CROSS_NOTEHEAD, 100, 102, 120, 122)]])

    report = ev.evaluate(Path("u.pt"), root, detector=stub)

    row = next(r for r in report["per_class"] if r["name"] == "cross_notehead")
    assert row["centroid"]["dy_median"] == pytest.approx(2.0)
    assert row["centroid"]["dx_median"] == pytest.approx(0.0)


def test_vertical_and_horizontal_error_are_reported_apart(tmp_path):
    """They feed different things: the row goes to snap, the column decides
    which notehead a modifier attaches to, and the tolerances differ.
    """
    truth = [annotation(SymbolClass.CROSS_NOTEHEAD, 100, 100, 120, 120)]
    root = build(tmp_path, [truth])
    stub = StubDetector([[detection(SymbolClass.CROSS_NOTEHEAD, 105, 101, 125, 121)]])

    report = ev.evaluate(Path("u.pt"), root, detector=stub, iou_threshold=0.3)

    row = next(r for r in report["per_class"] if r["name"] == "cross_notehead")
    assert row["centroid"]["dx_median"] == pytest.approx(5.0)
    assert row["centroid"]["dy_median"] == pytest.approx(1.0)


def test_a_class_inside_the_snap_budget_passes(tmp_path):
    """2 px of vertical error is inside the 2.8 px snap absorbs."""
    truth = [annotation(SymbolClass.CROSS_NOTEHEAD, 100, 100, 120, 120)]
    root = build(tmp_path, [truth])
    stub = StubDetector([[detection(SymbolClass.CROSS_NOTEHEAD, 100, 102, 120, 122)]])

    report = ev.evaluate(Path("u.pt"), root, detector=stub)

    assert report["centroid_pass"]["failing"] == []
    assert report["centroid_pass"]["passing"] == 1


def test_a_class_outside_the_snap_budget_is_named(tmp_path):
    """5 px of vertical error puts snap outside its tolerance, so the centroid
    resolves to the wrong staff position or to None.
    """
    truth = [annotation(SymbolClass.CROSS_NOTEHEAD, 100, 100, 130, 130)]
    root = build(tmp_path, [truth])
    stub = StubDetector([[detection(SymbolClass.CROSS_NOTEHEAD, 100, 105, 130, 135)]])

    report = ev.evaluate(Path("u.pt"), root, detector=stub, iou_threshold=0.3)

    failing = report["centroid_pass"]["failing"]
    assert [row["name"] for row in failing] == ["cross_notehead"]


def test_the_pass_threshold_follows_the_staff_spacing(tmp_path):
    """The same error passes at a large engraving and fails at a small one."""
    truth = [annotation(SymbolClass.CROSS_NOTEHEAD, 100, 100, 130, 130)]
    root = build(tmp_path, [truth])
    boxes = [detection(SymbolClass.CROSS_NOTEHEAD, 100, 104, 130, 134)]

    tight = ev.evaluate(
        Path("u.pt"), root, detector=StubDetector([boxes]), iou_threshold=0.3,
        staff_spacing=14.0,
    )
    loose = ev.evaluate(
        Path("u.pt"), root, detector=StubDetector([boxes]), iou_threshold=0.3,
        staff_spacing=28.0,
    )

    assert tight["centroid_pass"]["failing"]
    assert not loose["centroid_pass"]["failing"]


def test_an_unmatched_class_reports_no_centroid(tmp_path):
    truth = [annotation(SymbolClass.CROSS_NOTEHEAD, 100, 100, 120, 120)]
    root = build(tmp_path, [truth])

    report = ev.evaluate(Path("u.pt"), root, detector=StubDetector([[]]))

    row = next(r for r in report["per_class"] if r["name"] == "cross_notehead")
    assert row["centroid"]["matched"] == 0


def test_the_report_prints_the_application_sections(tmp_path, capsys):
    truth = [
        annotation(SymbolClass.AUGMENTATION_DOT, 200, 200, 206, 206),
        annotation(SymbolClass.CROSS_NOTEHEAD, 100, 100, 120, 120),
    ]
    root = build(tmp_path, [truth])
    stub = StubDetector([[detection(SymbolClass.AUGMENTATION_DOT, 200, 200, 206, 206),
                          detection(SymbolClass.CROSS_NOTEHEAD, 100, 100, 120, 120)]])

    ev.print_report(ev.evaluate(Path("u.pt"), root, detector=stub))

    out = capsys.readouterr().out
    assert "SMALL CLASSES" in out
    assert "CENTROID ERROR vs what the application needs" in out
