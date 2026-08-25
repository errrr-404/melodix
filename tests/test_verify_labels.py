"""Unit tests for ``scripts/verify_labels.py``.

This tool exists to answer one question — are the boxes on the glyphs — and it
got that question wrong the first time it was asked. It thresholded ink at a
fixed 128, so on a page that augmentation had brightened and blurred, thin
one-pixel strokes read as paper and accents looked catastrophically misaligned
while their boxes were pixel-exact. It very nearly triggered an unnecessary
regeneration of a 1.4 GB dataset.

So the tests below cover both directions: a correct label set must pass, and a
deliberately broken one must fail. A checker that cannot fail is worse than no
checker, because it launders a bad dataset as a good one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from melodix.vision.dataset import Annotation, BoundingBox, write_label_file
from melodix.vision.labels import SymbolClass

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_labels.py"


def _load():
    """Import the verification script as a module."""
    spec = importlib.util.spec_from_file_location("verify_labels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vl = _load()

PAGE_W = 400
PAGE_H = 500


def page_with_discs(centres=((100, 120), (250, 300), (330, 430)), radius=14, shift=0):
    """A page of black discs and boxes for them, optionally shifted off.

    ``shift`` displaces every box by that many pixels, which is what a stale
    label set looks like.
    """
    image = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    boxes = []
    for x, y in centres:
        cv2.circle(image, (x, y), radius, 0, -1)
        boxes.append(
            Annotation(
                symbol=SymbolClass.ROUND_NOTEHEAD,
                box=BoundingBox.from_pixels(
                    x - radius + shift,
                    y - radius + shift,
                    x + radius + shift,
                    y + radius + shift,
                    PAGE_W,
                    PAGE_H,
                ),
            )
        )
    return image, tuple(boxes)


def build(root: Path, shift=0, pages=3, split="train", faint=False) -> Path:
    """Write a small dataset, optionally with stale boxes or faint ink."""
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for index in range(pages):
        image, boxes = page_with_discs(shift=shift)
        if faint:
            # Thin, pale strokes: what augmentation does to a 1 px glyph.
            image = cv2.GaussianBlur(image, (0, 0), 3.0)
            image = np.clip(image.astype(np.float32) * 1.05 + 40, 0, 255).astype(np.uint8)
        cv2.imwrite(str(root / "images" / split / f"page_{index}.png"), image)
        write_label_file(root / "labels" / split / f"page_{index}.txt", boxes)
    return root


# --------------------------------------------------------------------------- #
# Thresholding
# --------------------------------------------------------------------------- #


def test_ink_is_found_on_a_normal_page():
    image, _ = page_with_discs()

    mask = vl.ink_mask(image)

    assert mask.any()
    assert not mask.all()


def test_ink_is_still_found_on_a_faded_page():
    """The bug this tool shipped with. A fixed cutoff of 128 misses ink that
    brightening and blurring has lifted above it; Otsu adapts.
    """
    image, _ = page_with_discs()
    faded = np.clip(image.astype(np.float32) * 0.4 + 150, 0, 255).astype(np.uint8)
    assert faded.min() > 128, "fixture is not actually faded past the old cutoff"

    assert (faded <= vl.FALLBACK_INK_THRESHOLD).sum() == 0  # what the old code saw
    assert vl.ink_mask(faded).any()  # what it sees now


def test_a_blank_page_yields_no_ink():
    blank = np.full((50, 50), 255, np.uint8)

    assert not vl.ink_mask(blank).any()


# --------------------------------------------------------------------------- #
# Measuring one box
# --------------------------------------------------------------------------- #


def test_a_box_on_its_glyph_measures_a_small_offset():
    image, boxes = page_with_discs()
    mask = vl.ink_mask(image)
    x0, y0, x1, y1 = boxes[0].box.to_pixels(PAGE_W, PAGE_H)

    report = vl.measure_box(mask, x0, y0, x1, y1, 0)

    assert report.centroid_offset < 0.05
    assert report.ink_fraction > 0.5
    assert not report.suspicious


def test_a_box_on_paper_is_suspicious():
    image, _ = page_with_discs()
    mask = vl.ink_mask(image)

    report = vl.measure_box(mask, 5, 5, 40, 40, 0)

    assert report.suspicious


def test_a_box_half_off_its_glyph_is_suspicious():
    image, _ = page_with_discs()
    mask = vl.ink_mask(image)
    # The first disc is centred at (100, 120) with radius 14.
    report = vl.measure_box(mask, 100, 120, 128, 148, 0)

    assert report.centroid_offset > vl.CENTROID_TOLERANCE
    assert report.suspicious


def test_a_degenerate_box_is_skipped():
    image, _ = page_with_discs()

    assert vl.measure_box(vl.ink_mask(image), 10, 10, 11, 11, 0) is None


def test_a_tight_box_reports_high_tightness():
    image, boxes = page_with_discs()
    mask = vl.ink_mask(image)
    x0, y0, x1, y1 = boxes[0].box.to_pixels(PAGE_W, PAGE_H)

    assert vl.measure_box(mask, x0, y0, x1, y1, 0).tightness > 0.8


# --------------------------------------------------------------------------- #
# Judging a dataset
# --------------------------------------------------------------------------- #


def test_a_correct_dataset_passes(tmp_path):
    report = vl.verify_split(build(tmp_path), sample=None)

    assert report["suspicious"] == 0
    assert not vl.verdict(report).startswith("MISALIGNED")


def test_a_faint_dataset_still_passes(tmp_path):
    """The false alarm this tool originally raised."""
    report = vl.verify_split(build(tmp_path, faint=True), sample=None)

    assert not vl.verdict(report).startswith("MISALIGNED")


def test_a_shifted_dataset_is_caught(tmp_path):
    """A checker that cannot fail launders a bad dataset as a good one."""
    report = vl.verify_split(build(tmp_path, shift=20), sample=None)

    assert report["suspicious"] > 0
    assert vl.verdict(report).startswith("MISALIGNED")


def test_the_report_counts_every_box(tmp_path):
    report = vl.verify_split(build(tmp_path, pages=4), sample=None)

    assert report["pages"] == 4
    assert report["boxes"] == 12


def test_sampling_limits_the_pages_read(tmp_path):
    report = vl.verify_split(build(tmp_path, pages=6), sample=2)

    assert report["pages"] == 2


def test_sampling_is_reproducible(tmp_path):
    root = build(tmp_path, pages=6)

    first = vl.verify_split(root, sample=3, seed=4)
    second = vl.verify_split(root, sample=3, seed=4)

    assert first["boxes"] == second["boxes"]
    assert first["mean_offset"] == pytest.approx(second["mean_offset"])


def test_a_missing_split_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no val images"):
        vl.verify_split(build(tmp_path), split="val")


def test_every_class_appears_in_the_report(tmp_path):
    report = vl.verify_split(build(tmp_path), sample=None)

    assert len(report["per_class"]) == 28


# --------------------------------------------------------------------------- #
# The unconfounded check
# --------------------------------------------------------------------------- #


def test_a_correct_dataset_shows_no_meaningful_radius_trend(tmp_path):
    """A rotation-stale label set displaces boxes in proportion to distance
    from the page centre. A correct one shows no such trend.

    Judged on effect size, not correlation. A clean fixture correlates at 0.79
    across offsets spanning 0.001 to 0.009 — a real trend statistically, and
    nothing at all in practice. Asserting on the correlation alone made this
    test fail against a dataset that was perfectly labelled.
    """
    result = vl.radius_correlation(build(tmp_path, pages=4), pages=4)

    assert result["boxes"] > 0
    assert result["offset_span"] < vl.MIN_RADIUS_EFFECT


def test_a_rotation_stale_dataset_shows_a_radius_trend(tmp_path):
    """Build the defect deliberately: rotate the pages, leave the boxes."""
    root = tmp_path / "stale"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    centres = [(x, y) for x in (60, 200, 340) for y in (70, 250, 430)]
    for index in range(4):
        image, boxes = page_with_discs(centres=tuple(centres), radius=12)
        matrix = cv2.getRotationMatrix2D((PAGE_W / 2, PAGE_H / 2), 6.0, 1.0)
        rotated = cv2.warpAffine(image, matrix, (PAGE_W, PAGE_H), borderValue=255)
        cv2.imwrite(str(root / "images" / "train" / f"p{index}.png"), rotated)
        write_label_file(root / "labels" / "train" / f"p{index}.txt", boxes)

    result = vl.radius_correlation(root, pages=4)

    assert result["correlation"] > 0.3
    assert result["offset_span"] >= vl.MIN_RADIUS_EFFECT


def test_the_verdict_prefers_the_radius_signal(tmp_path):
    """A raised flag rate with a flat radius trend is the neighbouring-ink
    confound, not misalignment, and must not demand a regeneration.
    """
    report = {
        "suspicious_rate": 0.05,
        "radius": {"boxes": 500, "correlation": -0.05, "bands": []},
    }

    assert not vl.verdict(report).startswith("MISALIGNED")
    assert "No regeneration needed" in vl.verdict(report)


def test_a_rising_radius_trend_overrides_a_low_flag_rate():
    report = {
        "suspicious_rate": 0.0,
        "radius": {
            "boxes": 500,
            "correlation": 0.8,
            "offset_span": 0.4,
            "bands": [],
        },
    }

    assert vl.verdict(report).startswith("MISALIGNED")


def test_a_strong_correlation_at_a_trivial_magnitude_is_not_misalignment():
    """Both signals are required. This combination — a clean correlation across
    offsets that are all near zero — is what a correct dataset looks like, and
    treating it as a defect would demand a needless regeneration.
    """
    report = {
        "suspicious_rate": 0.0,
        "radius": {
            "boxes": 500,
            "correlation": 0.79,
            "offset_span": 0.008,
            "bands": [],
        },
    }

    assert not vl.verdict(report).startswith("MISALIGNED")


def test_an_overwhelming_flag_rate_is_reported_even_without_a_radius_trend():
    report = {
        "suspicious_rate": 0.6,
        "radius": {"boxes": 500, "correlation": 0.0, "bands": []},
    }

    assert vl.verdict(report).startswith("MISALIGNED")


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_the_cli_passes_a_correct_dataset(tmp_path, capsys):
    code = vl.main(["--data", str(build(tmp_path)), "--sample", "0"])

    assert code == 0
    assert "VERDICT: aligned" in capsys.readouterr().out


def test_the_cli_fails_a_shifted_dataset(tmp_path, capsys):
    code = vl.main(["--data", str(build(tmp_path, shift=20)), "--sample", "0"])

    assert code == 1
    assert "MISALIGNED" in capsys.readouterr().out


def test_the_cli_writes_json(tmp_path):
    out = tmp_path / "report.json"

    vl.main(["--data", str(build(tmp_path)), "--sample", "0", "--json", str(out)])

    import json

    assert json.loads(out.read_text(encoding="utf-8"))["boxes"] == 9
