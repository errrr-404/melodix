"""Unit tests for :mod:`melodix.geometry.deskew`.

The sign convention is asserted directly rather than only through round-trips:
a round-trip passes just as happily when both the estimate and the correction
are inverted, which is precisely the bug that would corrupt every page.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from melodix.geometry.deskew import (
    DeskewConfig,
    SkewEstimate,
    deskew,
    estimate_skew,
    estimate_skew_hough,
    estimate_skew_projection,
    rotate_image,
)
from melodix.geometry.staff import detect_staff_grids

from tests.helpers import blank_page, draw_notehead, draw_staff, draw_stem

# Estimator accuracy is bounded by the fine search step (0.05 degrees); allow a
# little over one step for interpolation noise on rotated synthetic input.
ANGLE_TOL = 0.12
HOUGH_TOL = 0.25


def tilted_page(angle_deg: float, spacing: float = 14.0, top_row: int = 200) -> np.ndarray:
    """Draw a staff, then rotate the page counter-clockwise by ``angle_deg``."""
    page = blank_page(width=800, height=600)
    draw_staff(page, top_row=top_row, spacing=spacing)
    return rotate_image(page, angle_deg, border_value=255)


# --------------------------------------------------------------------------- #
# Sign convention
# --------------------------------------------------------------------------- #


def test_lines_rising_to_the_right_give_positive_skew():
    """The convention, asserted against the drawing rather than a round-trip."""
    page = blank_page()
    # y decreasing as x increases: the line rises towards the right.
    cv2.line(page, (60, 260), (740, 200), color=0, thickness=2)
    cv2.line(page, (60, 300), (740, 240), color=0, thickness=2)

    estimate = estimate_skew(page, DeskewConfig(method="projection"))

    assert estimate.skew_deg > 0.0


def test_lines_falling_to_the_right_give_negative_skew():
    page = blank_page()
    cv2.line(page, (60, 200), (740, 260), color=0, thickness=2)
    cv2.line(page, (60, 240), (740, 300), color=0, thickness=2)

    estimate = estimate_skew(page, DeskewConfig(method="projection"))

    assert estimate.skew_deg < 0.0


def test_correction_is_the_negation_of_skew():
    estimate = SkewEstimate(skew_deg=2.5, confidence=0.9, method="test")

    assert estimate.correction_deg == pytest.approx(-2.5)


def test_applying_the_correction_levels_the_page():
    """The property that actually matters: correcting drives skew to zero."""
    page = tilted_page(3.0)

    first = estimate_skew(page, DeskewConfig(method="projection"))
    levelled = rotate_image(page, first.correction_deg)
    second = estimate_skew(levelled, DeskewConfig(method="projection"))

    assert abs(second.skew_deg) <= ANGLE_TOL


# --------------------------------------------------------------------------- #
# Projection estimator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("angle", [-6.0, -3.5, -0.8, 0.0, 0.8, 2.0, 3.5, 6.0])
def test_projection_recovers_the_applied_angle(angle):
    estimate = estimate_skew_projection(tilted_page(angle))

    assert estimate.skew_deg == pytest.approx(angle, abs=ANGLE_TOL)


def test_projection_reports_zero_confidence_on_a_blank_page():
    estimate = estimate_skew_projection(blank_page())

    assert estimate.confidence == 0.0
    assert estimate.skew_deg == 0.0


def test_projection_is_confident_on_a_staff():
    assert estimate_skew_projection(tilted_page(2.0)).confidence > 0.5


def test_projection_search_can_be_narrowed():
    """A seeded search only resolves angles inside its own window."""
    page = tilted_page(5.0)

    wide = estimate_skew_projection(page)
    narrow = estimate_skew_projection(page, center_deg=0.0, radius_deg=1.0)

    assert wide.skew_deg == pytest.approx(5.0, abs=ANGLE_TOL)
    assert abs(narrow.skew_deg) <= 1.0


def test_angles_beyond_max_are_not_wrapped_to_a_wrong_answer():
    config = DeskewConfig(method="projection", max_angle_deg=2.0)

    estimate = estimate_skew(tilted_page(8.0), config)

    assert abs(estimate.skew_deg) <= 2.0


# --------------------------------------------------------------------------- #
# Hough estimator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("angle", [-6.0, -2.0, 0.0, 2.0, 6.0])
def test_hough_recovers_the_applied_angle(angle):
    estimate = estimate_skew_hough(tilted_page(angle))

    assert estimate.skew_deg == pytest.approx(angle, abs=HOUGH_TOL)


def test_hough_reports_zero_confidence_on_a_blank_page():
    estimate = estimate_skew_hough(blank_page())

    assert estimate.confidence == 0.0
    assert estimate.skew_deg == 0.0


def test_hough_ignores_near_vertical_strokes():
    """Stems are long and straight but must not be read as skew evidence."""
    page = blank_page()
    draw_staff(page, top_row=200, spacing=14)
    for x in range(120, 700, 40):
        draw_stem(page, x=x, top_row=120, bottom_row=300, width=3)

    estimate = estimate_skew_hough(page)

    assert estimate.skew_deg == pytest.approx(0.0, abs=HOUGH_TOL)


def test_hough_confidence_drops_when_long_strokes_disagree():
    """A staff competing with a tilted rule should report doubt, not an average."""
    agreeing = blank_page()
    draw_staff(agreeing, top_row=200, spacing=14)

    conflicted = blank_page()
    draw_staff(conflicted, top_row=200, spacing=14)
    for offset in range(0, 60, 12):
        cv2.line(conflicted, (60, 400 + offset), (740, 340 + offset), color=0, thickness=2)

    assert estimate_skew_hough(conflicted).confidence < estimate_skew_hough(agreeing).confidence


# --------------------------------------------------------------------------- #
# Auto strategy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("angle", [-4.0, -1.5, 0.0, 1.5, 4.0])
def test_auto_recovers_the_applied_angle(angle):
    estimate = estimate_skew(tilted_page(angle), DeskewConfig(method="auto"))

    assert estimate.skew_deg == pytest.approx(angle, abs=HOUGH_TOL)


def test_auto_reports_which_path_it_took():
    config = DeskewConfig(method="auto")

    assert estimate_skew(tilted_page(2.0), config).method == "auto:hough+projection"


def test_auto_falls_back_to_projection_without_hough_evidence():
    """No segment long enough for Hough, so the full sweep must still run."""
    page = blank_page()
    draw_staff(page, top_row=200, spacing=14)
    config = DeskewConfig(method="auto", hough_min_line_ratio=0.99, hough_threshold=100000)

    assert estimate_skew(page, config).method == "auto:projection"


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #


def test_rotation_preserves_shape_and_dtype_by_default():
    page = blank_page(width=800, height=600)

    rotated = rotate_image(page, 5.0)

    assert rotated.shape == (600, 800)
    assert rotated.dtype == np.uint8


def test_rotation_fills_the_border_with_the_given_value():
    page = blank_page(width=200, height=200)
    page[:, :] = 0

    rotated = rotate_image(page, 10.0, border_value=255)

    assert rotated[0, 0] == 255


def test_expand_grows_the_canvas_to_avoid_clipping():
    page = blank_page(width=400, height=200)

    rotated = rotate_image(page, 30.0, expand=True)

    assert rotated.shape[0] > 200
    assert rotated.shape[1] > 400


def test_zero_rotation_is_a_no_op():
    page = blank_page()
    draw_staff(page, top_row=100, spacing=12)

    assert np.array_equal(rotate_image(page, 0.0), page)


def test_nearest_interpolation_keeps_a_mask_binary():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:42, :] = 255

    rotated = rotate_image(mask, 3.0, border_value=0, interpolation=cv2.INTER_NEAREST)

    assert set(np.unique(rotated)).issubset({0, 255})


def test_rotating_an_empty_image_is_rejected():
    with pytest.raises(ValueError, match="empty image"):
        rotate_image(np.zeros((0, 0), dtype=np.uint8), 1.0)


def test_colour_pages_are_supported():
    gray = blank_page()
    draw_staff(gray, top_row=200, spacing=14)
    colour = np.repeat(gray[:, :, None], 3, axis=2)
    tilted = rotate_image(colour, 2.0, border_value=255)

    estimate = estimate_skew(tilted, DeskewConfig(method="projection"))

    assert estimate.skew_deg == pytest.approx(2.0, abs=ANGLE_TOL)


# --------------------------------------------------------------------------- #
# deskew()
# --------------------------------------------------------------------------- #


def test_deskew_levels_a_tilted_page():
    result = deskew(tilted_page(3.0))

    assert result.applied
    assert abs(estimate_skew(result.image).skew_deg) <= HOUGH_TOL


def test_deskew_leaves_a_level_page_untouched():
    """A clean scan must not pay a resampling penalty for nothing."""
    page = blank_page()
    draw_staff(page, top_row=200, spacing=14)

    result = deskew(page)

    assert not result.applied
    assert result.image is page


def test_deskew_skips_a_negligible_tilt():
    result = deskew(tilted_page(0.02))

    assert not result.applied


def test_deskew_skips_an_unconfident_estimate():
    result = deskew(blank_page())

    assert not result.applied
    assert result.estimate.confidence == 0.0


def test_deskew_reports_the_estimate_even_when_it_declines_to_rotate():
    result = deskew(blank_page())

    assert result.estimate.skew_deg == 0.0
    assert result.estimate.method == "projection"


def test_deskew_on_an_empty_image_is_rejected():
    with pytest.raises(ValueError, match="empty image"):
        deskew(np.zeros((0, 0), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Integration with staff detection
# --------------------------------------------------------------------------- #


def test_staff_detection_fails_on_a_tilted_page():
    """The failure this module exists to prevent, pinned as a test."""
    assert detect_staff_grids(tilted_page(2.5)) == []


def test_deskew_restores_staff_detection():
    page = tilted_page(2.5)

    result = deskew(page)
    grids = detect_staff_grids(result.image)

    assert result.applied
    assert len(grids) == 1
    assert grids[0].line_spacing == pytest.approx(14.0, abs=0.5)


@pytest.mark.parametrize("angle", [-4.0, -2.5, -1.2, 1.2, 2.5, 4.0])
def test_deskew_restores_detection_across_the_common_range(angle):
    grids = detect_staff_grids(deskew(tilted_page(angle)).image)

    assert len(grids) == 1


def test_deskew_survives_a_realistic_page_with_notes():
    page = blank_page(width=800, height=600)
    draw_staff(page, top_row=200, spacing=14)
    for x in range(120, 700, 50):
        draw_notehead(page, center_x=x, center_y=228, radius=5)
        draw_stem(page, x=x + 7, top_row=180, bottom_row=228)
    tilted = rotate_image(page, 2.0, border_value=255)

    result = deskew(tilted)

    assert result.estimate.skew_deg == pytest.approx(2.0, abs=HOUGH_TOL)
    assert len(detect_staff_grids(result.image)) == 1


# --------------------------------------------------------------------------- #
# Configuration and estimate validation
# --------------------------------------------------------------------------- #


def test_default_method_is_projection():
    """Measured faster and more accurate than Hough on realistic pages."""
    assert DeskewConfig().method == "projection"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "radon"),
        ("max_angle_deg", 0.0),
        ("max_angle_deg", 90.0),
        ("coarse_step_deg", 0.0),
        ("coarse_step_deg", 50.0),
        ("fine_step_deg", 0.0),
        ("fine_step_deg", 5.0),
        ("working_width", 50),
        ("min_confidence", -0.1),
        ("min_confidence", 1.5),
        ("min_correction_deg", -1.0),
        ("hough_min_line_ratio", 0.0),
        ("hough_min_line_ratio", 1.5),
        ("hough_threshold", 0),
        ("hough_max_line_gap", -1),
    ],
)
def test_invalid_config_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        DeskewConfig(**{field: value})


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_estimate_rejects_out_of_range_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        SkewEstimate(skew_deg=0.0, confidence=confidence, method="test")


def test_is_actionable_requires_both_confidence_and_magnitude():
    config = DeskewConfig()

    assert SkewEstimate(2.0, 0.9, "test").is_actionable(config)
    assert not SkewEstimate(2.0, 0.01, "test").is_actionable(config)
    assert not SkewEstimate(0.01, 0.9, "test").is_actionable(config)


def test_min_correction_threshold_sits_above_the_search_resolution():
    """Otherwise search quantisation alone rotates a level page."""
    config = DeskewConfig()

    assert config.min_correction_deg > config.fine_step_deg