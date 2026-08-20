"""Unit tests for :mod:`melodix.geometry.barlines`.

The module's defining property is what it refuses to do: it reports every tall
vertical stroke without deciding which are barlines. Several tests below assert
that a note stem *is* returned, which reads as backwards until you recall that
discriminating one from the other needs the staff grid and therefore belongs to
:mod:`melodix.geometry.systems`. Detecting the stem here is correct behaviour,
not a false positive.
"""

from __future__ import annotations

import numpy as np
import pytest

from melodix.geometry.barlines import (
    BarlineDetectionConfig,
    VerticalSegment,
    detect_vertical_segments,
    extract_vertical_segments,
    isolate_vertical_runs,
    merge_collinear,
)
from melodix.geometry.staff import binarize, to_grayscale
from tests.helpers import blank_page, draw_notehead, draw_staff, draw_stem

# A drawn stroke rasterises with rounded caps, so its detected extent runs a
# pixel or two past the requested endpoints.
CAP_TOL = 4


def vertical_mask(page):
    """Binarise a page and keep only its tall vertical structures."""
    return isolate_vertical_runs(binarize(to_grayscale(page)), kernel_height=31)


def segment(x=100.0, y_start=100, y_end=200, thickness=2) -> VerticalSegment:
    """Build a segment without going near an image."""
    return VerticalSegment(x=x, y_start=y_start, y_end=y_end, thickness=thickness)


def staff_page(top_row=200, spacing=20, x_start=100, x_end=700):
    """A page carrying one staff and nothing else."""
    page = blank_page(600, 800)
    draw_staff(page, top_row=top_row, spacing=spacing, x_start=x_start, x_end=x_end)
    return page


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vertical_kernel_ratio", 0.0),
        ("vertical_kernel_ratio", 1.5),
        ("min_height_px", 0),
        ("min_height_px", -5),
        ("min_height_staff_spans", 0.0),
        ("min_height_staff_spans", -1.0),
        ("min_height_ratio", 0.0),
        ("min_height_ratio", 1.5),
        ("max_thickness_px", 0),
        ("min_aspect_ratio", 0.0),
        ("merge_x_tolerance_px", -1.0),
        ("merge_max_gap_px", -1),
    ],
)
def test_invalid_config_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        BarlineDetectionConfig(**{field: value})


def test_min_height_px_may_be_left_unset():
    assert BarlineDetectionConfig().min_height_px is None


def test_the_kernel_is_shorter_than_the_height_floor():
    """So a stroke just under the floor survives the opening and is rejected
    explicitly, rather than being erased and silently vanishing.
    """
    assert BarlineDetectionConfig().vertical_kernel_ratio < 1.0


# --------------------------------------------------------------------------- #
# VerticalSegment
# --------------------------------------------------------------------------- #


def test_an_inverted_extent_is_rejected():
    with pytest.raises(ValueError, match="y_end must be at least y_start"):
        VerticalSegment(x=100.0, y_start=200, y_end=100, thickness=2)


@pytest.mark.parametrize("thickness", [0, -3])
def test_a_non_positive_thickness_is_rejected(thickness):
    with pytest.raises(ValueError, match="thickness must be positive"):
        VerticalSegment(x=100.0, y_start=100, y_end=200, thickness=thickness)


def test_a_single_row_stroke_is_permitted():
    assert segment(y_start=100, y_end=100).height == 1


def test_height_counts_both_endpoints():
    assert segment(y_start=100, y_end=200).height == 101


def test_centre_row_is_the_midpoint():
    assert segment(y_start=100, y_end=200).y_center == pytest.approx(150.0)


def test_aspect_ratio_is_height_over_thickness():
    assert segment(y_start=0, y_end=99, thickness=4).aspect_ratio == pytest.approx(25.0)


# --------------------------------------------------------------------------- #
# Spanning
# --------------------------------------------------------------------------- #


def test_a_stroke_covering_the_range_spans_it():
    assert segment(y_start=100, y_end=200).spans(100.0, 200.0)


def test_a_stroke_overshooting_the_range_spans_it():
    """A system-wide stroke spans every staff it passes through."""
    assert segment(y_start=50, y_end=400).spans(100.0, 200.0)


def test_a_stroke_stopping_short_does_not_span():
    assert not segment(y_start=130, y_end=200).spans(100.0, 200.0)


def test_tolerance_admits_a_stroke_a_few_pixels_short():
    stroke = segment(y_start=102, y_end=198)

    assert not stroke.spans(100.0, 200.0)
    assert stroke.spans(100.0, 200.0, tolerance=3.0)


def test_tolerance_does_not_admit_a_stroke_far_short():
    assert not segment(y_start=140, y_end=200).spans(100.0, 200.0, tolerance=3.0)


def test_spanning_an_inverted_range_is_rejected():
    with pytest.raises(ValueError, match="y_bottom must be at least y_top"):
        segment().spans(200.0, 100.0)


def test_overlapping_strokes_share_rows():
    assert segment(y_start=100, y_end=200).overlaps(segment(y_start=150, y_end=250))


def test_disjoint_strokes_do_not_overlap():
    assert not segment(y_start=100, y_end=200).overlaps(segment(y_start=250, y_end=300))


def test_touching_strokes_overlap():
    assert segment(y_start=100, y_end=200).overlaps(segment(y_start=200, y_end=300))


# --------------------------------------------------------------------------- #
# Vertical isolation
# --------------------------------------------------------------------------- #


def test_isolation_keeps_a_tall_stroke():
    page = blank_page(600, 800)
    draw_stem(page, x=400, top_row=200, bottom_row=300, width=3)

    assert vertical_mask(page).any()


def test_isolation_erases_staff_lines():
    """The mirror of the horizontal pass: level ink must not survive."""
    assert not vertical_mask(staff_page()).any()


def test_isolation_erases_a_notehead():
    page = blank_page(600, 800)
    draw_notehead(page, center_x=400, center_y=300, radius=10)

    assert not vertical_mask(page).any()


def test_isolation_erases_a_stroke_shorter_than_the_kernel():
    page = blank_page(600, 800)
    draw_stem(page, x=400, top_row=200, bottom_row=210, width=3)

    assert not vertical_mask(page).any()


@pytest.mark.parametrize("kernel_height", [0, -1])
def test_a_non_positive_kernel_is_rejected(kernel_height):
    binary = binarize(to_grayscale(blank_page(100, 100)))

    with pytest.raises(ValueError, match="kernel_height must be positive"):
        isolate_vertical_runs(binary, kernel_height)


def test_an_even_kernel_does_not_shift_the_stroke():
    """An off-centre anchor would drag every stroke a pixel down the page."""
    page = blank_page(600, 800)
    draw_stem(page, x=400, top_row=200, bottom_row=300, width=3)
    binary = binarize(to_grayscale(page))

    rows = np.flatnonzero(isolate_vertical_runs(binary, 32).any(axis=1))

    assert int(rows[0]) == pytest.approx(200, abs=CAP_TOL)
    assert int(rows[-1]) == pytest.approx(300, abs=CAP_TOL)


# --------------------------------------------------------------------------- #
# Segment extraction
# --------------------------------------------------------------------------- #


def test_a_stroke_taller_than_the_floor_is_kept():
    page = blank_page(600, 800)
    draw_stem(page, x=400, top_row=200, bottom_row=300, width=3)

    found = extract_vertical_segments(vertical_mask(page), min_height=50)

    assert len(found) == 1
    assert found[0].x == pytest.approx(400, abs=CAP_TOL)


def test_a_stroke_shorter_than_the_floor_is_dropped():
    page = blank_page(600, 800)
    draw_stem(page, x=400, top_row=200, bottom_row=300, width=3)

    assert extract_vertical_segments(vertical_mask(page), min_height=200) == []


def test_a_stroke_thicker_than_the_limit_is_dropped():
    """Wide ink is a blob or a bracket, not a rule."""
    page = blank_page(600, 800)
    page[200:400, 400:440] = 0

    assert extract_vertical_segments(vertical_mask(page), min_height=50) == []


def test_a_squat_component_fails_the_aspect_check():
    """Clearing the height bar by being large overall is not enough."""
    page = blank_page(600, 800)
    page[200:240, 400:415] = 0
    config = BarlineDetectionConfig(min_height_px=20)

    strict = extract_vertical_segments(vertical_mask(page), 20, config)
    lax = extract_vertical_segments(
        vertical_mask(page), 20, BarlineDetectionConfig(min_aspect_ratio=2.0)
    )

    assert strict == []
    assert len(lax) == 1


def test_segments_are_ordered_left_to_right():
    page = blank_page(600, 800)
    for x in (600, 200, 400):
        draw_stem(page, x=x, top_row=200, bottom_row=300, width=3)

    found = extract_vertical_segments(vertical_mask(page), min_height=50)

    assert [round(s.x) for s in found] == sorted(round(s.x) for s in found)


def test_two_strokes_in_one_column_stay_separate():
    """Connected components, not a column projection: a projection would fuse
    these into one impossible stroke spanning the gap between systems.
    """
    page = blank_page(600, 800)
    draw_stem(page, x=400, top_row=100, bottom_row=180, width=3)
    draw_stem(page, x=400, top_row=400, bottom_row=480, width=3)

    found = extract_vertical_segments(vertical_mask(page), min_height=50)

    assert len(found) == 2
    assert found[0].y_end < found[1].y_start


def test_no_segments_on_a_blank_mask():
    assert extract_vertical_segments(vertical_mask(blank_page(600, 800)), min_height=50) == []


# --------------------------------------------------------------------------- #
# Merging fragments
# --------------------------------------------------------------------------- #


def test_two_fragments_of_one_stroke_are_rejoined():
    merged = merge_collinear([segment(y_start=10, y_end=40), segment(y_start=43, y_end=70)])

    assert len(merged) == 1
    assert (merged[0].y_start, merged[0].y_end) == (10, 70)


def test_a_gap_wider_than_the_limit_is_not_bridged():
    """Keeps a barline from fusing with the barline on the staff below."""
    distant = [segment(y_start=10, y_end=40), segment(y_start=90, y_end=140)]

    assert len(merge_collinear(distant)) == 2


def test_fragments_in_different_columns_are_not_merged():
    fragments = [segment(x=100.0, y_start=10, y_end=40), segment(x=140.0, y_start=43, y_end=70)]

    assert len(merge_collinear(fragments)) == 2


def test_a_small_column_drift_is_tolerated():
    fragments = [segment(x=100.0, y_start=10, y_end=40), segment(x=101.0, y_start=43, y_end=70)]

    assert len(merge_collinear(fragments)) == 1


def test_the_merge_gap_can_be_widened():
    fragments = [segment(y_start=10, y_end=40), segment(y_start=60, y_end=90)]

    assert len(merge_collinear(fragments)) == 2
    assert len(merge_collinear(fragments, BarlineDetectionConfig(merge_max_gap_px=25))) == 1


def test_merging_weights_the_column_by_fragment_height():
    fragments = [
        segment(x=100.0, y_start=0, y_end=89, thickness=2),
        segment(x=110.0, y_start=90, y_end=99, thickness=2),
    ]

    merged = merge_collinear(fragments, BarlineDetectionConfig(merge_x_tolerance_px=20.0))

    assert merged[0].x == pytest.approx(101.0)


def test_merging_takes_the_thicker_stroke():
    fragments = [
        segment(y_start=10, y_end=40, thickness=2),
        segment(y_start=43, y_end=70, thickness=5),
    ]

    assert merge_collinear(fragments)[0].thickness == 5


def test_merging_nothing_yields_nothing():
    assert merge_collinear([]) == []


def test_a_lone_fragment_survives_unchanged():
    only = segment()

    assert merge_collinear([only]) == [only]


# --------------------------------------------------------------------------- #
# Height floors
# --------------------------------------------------------------------------- #


def test_staff_spacing_sets_the_floor_at_three_spans():
    """Barline 80px tall, stem 30px: at 3 spans of 20px the floor is 60."""
    page = staff_page()
    draw_stem(page, x=300, top_row=200, bottom_row=280, width=3)
    draw_stem(page, x=500, top_row=250, bottom_row=280, width=3)

    found = detect_vertical_segments(page, staff_spacing=20.0)

    assert len(found) == 1
    assert found[0].x == pytest.approx(300, abs=CAP_TOL)


def test_a_shorter_span_requirement_admits_the_stem():
    page = staff_page()
    draw_stem(page, x=300, top_row=200, bottom_row=280, width=3)
    draw_stem(page, x=500, top_row=250, bottom_row=280, width=3)
    config = BarlineDetectionConfig(min_height_staff_spans=1.0)

    assert len(detect_vertical_segments(page, config, staff_spacing=20.0)) == 2


def test_an_explicit_pixel_floor_overrides_the_spacing_hint():
    """Documented precedence: min_height_px wins over everything."""
    page = staff_page()
    draw_stem(page, x=300, top_row=200, bottom_row=280, width=3)
    config = BarlineDetectionConfig(min_height_px=200)

    assert detect_vertical_segments(page, config, staff_spacing=1.0) == []


def test_the_image_height_fallback_applies_without_a_spacing_hint():
    """2% of a 600px page is 12, low enough to admit a 30px stem."""
    page = staff_page()
    draw_stem(page, x=500, top_row=250, bottom_row=280, width=3)

    assert len(detect_vertical_segments(page)) == 1


@pytest.mark.parametrize("spacing", [0.0, -20.0])
def test_a_non_positive_spacing_hint_is_rejected(spacing):
    with pytest.raises(ValueError, match="staff_spacing must be positive"):
        detect_vertical_segments(staff_page(), staff_spacing=spacing)


# --------------------------------------------------------------------------- #
# End-to-end detection
# --------------------------------------------------------------------------- #


def test_barlines_are_found_at_the_drawn_columns():
    page = staff_page()
    for x in (100, 300, 500, 700):
        draw_stem(page, x=x, top_row=200, bottom_row=280, width=3)

    found = detect_vertical_segments(page, staff_spacing=20.0)

    assert [round(s.x) for s in found] == pytest.approx([100, 300, 500, 700], abs=CAP_TOL)


def test_a_barline_reaches_both_staff_lines():
    page = staff_page()
    draw_stem(page, x=300, top_row=200, bottom_row=280, width=3)

    found = detect_vertical_segments(page, staff_spacing=20.0)

    assert found[0].spans(200.0, 280.0, tolerance=3.0)


def test_a_stem_is_returned_but_does_not_span_the_staff():
    """The whole reason classification lives one module up."""
    page = staff_page()
    draw_notehead(page, center_x=400, center_y=270, radius=8)
    draw_stem(page, x=410, top_row=215, bottom_row=270, width=3)

    found = detect_vertical_segments(page, staff_spacing=10.0)

    assert len(found) == 1
    assert not found[0].spans(200.0, 280.0, tolerance=3.0)


def test_staff_lines_alone_produce_no_segments():
    assert detect_vertical_segments(staff_page(), staff_spacing=20.0) == []


def test_a_blank_page_produces_no_segments():
    assert detect_vertical_segments(blank_page(600, 800), staff_spacing=20.0) == []


def test_detection_on_an_empty_image_is_rejected():
    with pytest.raises(ValueError, match="empty image"):
        detect_vertical_segments(np.zeros((0, 0), dtype=np.uint8))


def test_a_colour_page_is_supported():
    page = np.full((600, 800, 3), 255, dtype=np.uint8)
    page[200:280, 300:303] = 0

    assert len(detect_vertical_segments(page, staff_spacing=20.0)) == 1


def test_a_stroke_crossing_two_staves_is_reported_once():
    """The continuous engraving style: one stroke, one segment."""
    page = blank_page(700, 800)
    draw_staff(page, top_row=100, spacing=20)
    draw_staff(page, top_row=260, spacing=20)
    draw_stem(page, x=300, top_row=100, bottom_row=340, width=3)

    found = detect_vertical_segments(page, staff_spacing=20.0)

    assert len(found) == 1
    assert found[0].spans(100.0, 340.0, tolerance=3.0)


def test_per_staff_strokes_are_reported_separately():
    """The other engraving style: two strokes at one column, two segments."""
    page = blank_page(700, 800)
    draw_staff(page, top_row=100, spacing=20)
    draw_staff(page, top_row=260, spacing=20)
    draw_stem(page, x=300, top_row=100, bottom_row=180, width=3)
    draw_stem(page, x=300, top_row=260, bottom_row=340, width=3)

    found = detect_vertical_segments(page, staff_spacing=20.0)

    assert len(found) == 2
