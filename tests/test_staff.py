"""Unit tests for :mod:`melodix.geometry.staff`.

Two kinds of test live here. The detection tests render a synthetic page and
assert on what comes back, which exercises the OpenCV pipeline end to end. The
coordinate tests build a :class:`StaffGrid` directly from known line rows,
because the 0-8 position mapping is the contract Stage 3 depends on and it
deserves assertions that cannot be perturbed by a thresholding change.
"""

from __future__ import annotations

import numpy as np
import pytest

from melodix.geometry.staff import (
    LINES_PER_STAFF,
    STAFF_TOP_POSITION,
    LineBand,
    StaffDetectionConfig,
    StaffGrid,
    StaffLine,
    binarize,
    detect_staff_grids,
    extract_line_bands,
    group_bands_into_grids,
    isolate_horizontal_runs,
    to_grayscale,
)

from tests.helpers import blank_page, draw_notehead, draw_staff, draw_stem

# Line centroids land on the drawn row exactly, but the horizontal opening can
# shift an edge column by a pixel or two.
X_TOL = 3


def rendered_staff(top_row: int = 200, spacing: int = 20, **kwargs) -> StaffGrid:
    """Draw one staff on a fresh page and return the detected grid."""
    page = blank_page(600, 800)
    draw_staff(page, top_row=top_row, spacing=spacing, **kwargs)
    grids = detect_staff_grids(page)
    assert len(grids) == 1
    return grids[0]


def synthetic_grid(top: float = 200.0, spacing: float = 20.0, index: int = 0) -> StaffGrid:
    """Build a grid from exact line rows, bypassing detection entirely."""
    return StaffGrid(
        lines=tuple(
            StaffLine(y=top + step * spacing, x_start=100, x_end=700, thickness=2)
            for step in range(LINES_PER_STAFF)
        ),
        index=index,
    )


def bands(*ys: float, thickness: int = 2) -> list[LineBand]:
    """Build candidate bands at the given rows."""
    return [LineBand(y=y, x_start=100, x_end=700, thickness=thickness) for y in ys]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


def test_a_staff_has_five_lines():
    assert LINES_PER_STAFF == 5


def test_the_top_line_sits_at_position_eight():
    """Bottom line at 0, one position per half space, five lines."""
    assert STAFF_TOP_POSITION == 2 * (LINES_PER_STAFF - 1)
    assert STAFF_TOP_POSITION == 8


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizontal_kernel_ratio", 0.0),
        ("horizontal_kernel_ratio", 1.5),
        ("row_threshold_ratio", 0.0),
        ("row_threshold_ratio", 1.5),
        ("spacing_tolerance", -0.1),
        ("spacing_tolerance", 1.0),
        ("max_thickness_ratio", 0.0),
        ("max_thickness_ratio", 1.5),
        ("min_line_spacing", 0.0),
        ("min_line_spacing", -4.0),
    ],
)
def test_invalid_config_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        StaffDetectionConfig(**{field: value})


def test_default_config_is_accepted():
    config = StaffDetectionConfig()

    assert 0.0 < config.horizontal_kernel_ratio <= 1.0
    assert 0.0 < config.row_threshold_ratio <= 1.0


# --------------------------------------------------------------------------- #
# Grayscale and thresholding
# --------------------------------------------------------------------------- #


def test_grayscale_passes_a_two_dimensional_page_through():
    page = blank_page(40, 60)

    gray = to_grayscale(page)

    assert gray.shape == (40, 60)
    assert gray.dtype == np.uint8


def test_grayscale_reduces_a_colour_page():
    page = np.full((40, 60, 3), 255, dtype=np.uint8)

    assert to_grayscale(page).shape == (40, 60)


def test_grayscale_reduces_a_page_with_an_alpha_channel():
    page = np.full((40, 60, 4), 255, dtype=np.uint8)

    assert to_grayscale(page).shape == (40, 60)


def test_grayscale_accepts_a_single_channel_page():
    page = np.full((40, 60, 1), 255, dtype=np.uint8)

    assert to_grayscale(page).shape == (40, 60)


def test_grayscale_rejects_an_unrecognised_shape():
    page = np.full((40, 60, 2), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="unsupported image shape"):
        to_grayscale(page)


def test_binarize_marks_ink_as_255():
    """Otsu inverts: dark ink becomes the foreground."""
    page = blank_page(100, 100)
    page[40:60, :] = 0

    binary = binarize(page)

    assert binary[50, 50] == 255
    assert binary[10, 50] == 0


# --------------------------------------------------------------------------- #
# Horizontal isolation
# --------------------------------------------------------------------------- #


def test_isolation_keeps_staff_lines():
    page = blank_page(600, 800)
    draw_staff(page, top_row=200, spacing=20)

    horizontal = isolate_horizontal_runs(binarize(to_grayscale(page)))

    assert horizontal[200, 400] == 255


def test_isolation_erases_a_stem():
    """The whole point of the opening: vertical ink must not survive."""
    page = blank_page(600, 800)
    draw_stem(page, x=400, top_row=150, bottom_row=300, width=3)

    horizontal = isolate_horizontal_runs(binarize(to_grayscale(page)))

    assert not horizontal.any()


def test_isolation_erases_a_notehead():
    page = blank_page(600, 800)
    draw_notehead(page, center_x=400, center_y=300, radius=10)

    horizontal = isolate_horizontal_runs(binarize(to_grayscale(page)))

    assert not horizontal.any()


def test_a_wider_kernel_erases_a_short_rule():
    """A rule spanning a tenth of the page survives a lax kernel, not a strict one."""
    page = blank_page(600, 800)
    page[300:302, 100:180] = 0
    binary = binarize(to_grayscale(page))

    lax = isolate_horizontal_runs(binary, StaffDetectionConfig(horizontal_kernel_ratio=0.05))
    strict = isolate_horizontal_runs(binary, StaffDetectionConfig(horizontal_kernel_ratio=0.35))

    assert lax.any()
    assert not strict.any()


# --------------------------------------------------------------------------- #
# Band extraction
# --------------------------------------------------------------------------- #


def test_bands_are_found_one_per_drawn_line():
    page = blank_page(600, 800)
    draw_staff(page, top_row=200, spacing=20)

    found = extract_line_bands(isolate_horizontal_runs(binarize(to_grayscale(page))))

    assert len(found) == LINES_PER_STAFF


def test_band_centroids_recover_the_drawn_rows():
    page = blank_page(600, 800)
    draw_staff(page, top_row=200, spacing=20)

    found = extract_line_bands(isolate_horizontal_runs(binarize(to_grayscale(page))))

    assert [band.y for band in found] == pytest.approx([200, 220, 240, 260, 280], abs=0.5)


def test_bands_are_ordered_top_to_bottom():
    page = blank_page(600, 800)
    draw_staff(page, top_row=200, spacing=20)

    found = extract_line_bands(isolate_horizontal_runs(binarize(to_grayscale(page))))

    assert found == sorted(found, key=lambda band: band.y)


def test_bands_report_the_horizontal_extent():
    page = blank_page(600, 800)
    draw_staff(page, top_row=200, spacing=20, x_start=150, x_end=650)

    found = extract_line_bands(isolate_horizontal_runs(binarize(to_grayscale(page))))

    assert found[0].x_start == pytest.approx(150, abs=X_TOL)
    assert found[0].x_end == pytest.approx(650, abs=X_TOL)


def test_band_width_spans_both_endpoints():
    band = LineBand(y=100.0, x_start=10, x_end=19, thickness=2)

    assert band.width == 10


def test_no_bands_on_a_blank_page():
    page = blank_page(600, 800)

    assert extract_line_bands(isolate_horizontal_runs(binarize(to_grayscale(page)))) == []


# --------------------------------------------------------------------------- #
# Grouping bands into staves
# --------------------------------------------------------------------------- #


def test_five_evenly_spaced_bands_form_one_staff():
    grids = group_bands_into_grids(bands(200, 220, 240, 260, 280))

    assert len(grids) == 1
    assert grids[0].line_spacing == pytest.approx(20.0)


def test_four_bands_are_not_a_staff():
    assert group_bands_into_grids(bands(200, 220, 240, 260)) == []


def test_ten_bands_form_two_staves():
    grids = group_bands_into_grids(bands(100, 120, 140, 160, 180, 400, 420, 440, 460, 480))

    assert len(grids) == 2


def test_grouped_staves_are_indexed_in_page_order():
    grids = group_bands_into_grids(bands(100, 120, 140, 160, 180, 400, 420, 440, 460, 480))

    assert [grid.index for grid in grids] == [0, 1]
    assert grids[0].top_line_y < grids[1].top_line_y


def test_bands_in_any_order_are_sorted_before_grouping():
    grids = group_bands_into_grids(bands(260, 200, 280, 220, 240))

    assert len(grids) == 1
    assert grids[0].line_ys == (200.0, 220.0, 240.0, 260.0, 280.0)


def test_uneven_spacing_is_rejected():
    """One gap at quadruple the median takes the window outside tolerance."""
    assert group_bands_into_grids(bands(200, 220, 240, 260, 340)) == []


def test_spacing_tolerance_can_be_loosened():
    warped = bands(200, 222, 240, 259, 280)
    strict = StaffDetectionConfig(spacing_tolerance=0.01)

    assert group_bands_into_grids(warped, strict) == []
    assert len(group_bands_into_grids(warped, StaffDetectionConfig(spacing_tolerance=0.25))) == 1


def test_bands_packed_tighter_than_the_floor_are_rejected():
    """Noise rows a pixel or two apart must not form a degenerate staff."""
    assert group_bands_into_grids(bands(200, 202, 204, 206, 208)) == []


def test_a_line_too_thick_for_its_spacing_is_rejected():
    """A blob taller than 60% of the spacing means two lines have merged."""
    blobs = bands(200, 210, 220, 230, 240, thickness=9)

    assert group_bands_into_grids(blobs) == []


def test_a_stray_rule_above_a_staff_costs_an_offset_not_the_staff():
    """Rejecting a window advances one band, so the real staff is still found."""
    grids = group_bands_into_grids(bands(50, 200, 220, 240, 260, 280))

    assert len(grids) == 1
    assert grids[0].line_ys == (200.0, 220.0, 240.0, 260.0, 280.0)


# --------------------------------------------------------------------------- #
# End-to-end detection
# --------------------------------------------------------------------------- #


def test_one_drawn_staff_is_detected():
    grid = rendered_staff(top_row=200, spacing=20)

    assert grid.line_ys == pytest.approx((200, 220, 240, 260, 280), abs=0.5)


@pytest.mark.parametrize("spacing", [10, 14, 20, 28])
def test_spacing_is_recovered_across_engraving_sizes(spacing):
    grid = rendered_staff(top_row=150, spacing=spacing)

    assert grid.line_spacing == pytest.approx(spacing, abs=0.5)


def test_two_staves_on_a_page_are_both_detected():
    page = blank_page(700, 800)
    draw_staff(page, top_row=100, spacing=20)
    draw_staff(page, top_row=400, spacing=20)

    grids = detect_staff_grids(page)

    assert len(grids) == 2
    assert [grid.index for grid in grids] == [0, 1]


def test_detection_survives_notes_on_the_staff():
    """Noteheads and stems must not disturb the line rows."""
    page = blank_page(600, 800)
    draw_staff(page, top_row=200, spacing=20)
    for x in (250, 350, 450):
        draw_notehead(page, center_x=x, center_y=240, radius=8)
        draw_stem(page, x=x + 10, top_row=180, bottom_row=240, width=3)

    grids = detect_staff_grids(page)

    assert len(grids) == 1
    assert grids[0].line_ys == pytest.approx((200, 220, 240, 260, 280), abs=0.5)


def test_detection_survives_barlines():
    page = blank_page(600, 800)
    draw_staff(page, top_row=200, spacing=20)
    for x in (100, 300, 500, 700):
        draw_stem(page, x=x, top_row=200, bottom_row=280, width=3)

    assert len(detect_staff_grids(page)) == 1


def test_a_blank_page_yields_no_staves():
    assert detect_staff_grids(blank_page(600, 800)) == []


def test_detection_on_an_empty_image_is_rejected():
    with pytest.raises(ValueError, match="empty image"):
        detect_staff_grids(np.zeros((0, 0), dtype=np.uint8))


def test_a_colour_page_is_detected():
    page = np.full((600, 800, 3), 255, dtype=np.uint8)
    for step in range(LINES_PER_STAFF):
        page[200 + step * 20 : 202 + step * 20, 100:700] = 0

    assert len(detect_staff_grids(page)) == 1


# --------------------------------------------------------------------------- #
# StaffGrid validation
# --------------------------------------------------------------------------- #


def test_a_staff_needs_exactly_five_lines():
    lines = tuple(StaffLine(y=200.0 + i * 20, x_start=0, x_end=10, thickness=2) for i in range(4))

    with pytest.raises(ValueError, match="exactly 5 lines"):
        StaffGrid(lines=lines)


def test_lines_must_ascend_in_y():
    lines = tuple(StaffLine(y=280.0 - i * 20, x_start=0, x_end=10, thickness=2) for i in range(5))

    with pytest.raises(ValueError, match="top to bottom"):
        StaffGrid(lines=lines)


def test_duplicate_line_rows_are_rejected():
    lines = tuple(StaffLine(y=200.0, x_start=0, x_end=10, thickness=2) for _ in range(5))

    with pytest.raises(ValueError, match="top to bottom"):
        StaffGrid(lines=lines)


def test_a_negative_index_is_rejected():
    with pytest.raises(ValueError, match="index must be non-negative"):
        synthetic_grid(index=-1)


def test_with_index_returns_a_renumbered_copy():
    grid = synthetic_grid(index=0)

    moved = grid.with_index(3)

    assert moved.index == 3
    assert grid.index == 0
    assert moved.line_ys == grid.line_ys


# --------------------------------------------------------------------------- #
# StaffGrid measurements
# --------------------------------------------------------------------------- #


def test_top_and_bottom_line_rows():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.top_line_y == 200.0
    assert grid.bottom_line_y == 280.0


def test_height_spans_the_outer_lines():
    assert synthetic_grid(top=200.0, spacing=20.0).height == pytest.approx(80.0)


def test_centre_row_is_the_middle_line():
    assert synthetic_grid(top=200.0, spacing=20.0).center_y == pytest.approx(240.0)


def test_step_height_is_half_the_line_spacing():
    grid = synthetic_grid(spacing=20.0)

    assert grid.step_height == pytest.approx(grid.line_spacing / 2.0)
    assert grid.step_height == pytest.approx(10.0)


def test_line_spacing_takes_the_median_so_one_warped_gap_does_not_skew_it():
    warped = StaffGrid(
        lines=(
            StaffLine(y=200.0, x_start=0, x_end=10, thickness=2),
            StaffLine(y=220.0, x_start=0, x_end=10, thickness=2),
            StaffLine(y=240.0, x_start=0, x_end=10, thickness=2),
            StaffLine(y=260.0, x_start=0, x_end=10, thickness=2),
            StaffLine(y=294.0, x_start=0, x_end=10, thickness=2),
        )
    )

    assert warped.line_spacing == pytest.approx(20.0)


def test_horizontal_extent_is_the_union_across_lines():
    grid = StaffGrid(
        lines=(
            StaffLine(y=200.0, x_start=110, x_end=690, thickness=2),
            StaffLine(y=220.0, x_start=100, x_end=700, thickness=2),
            StaffLine(y=240.0, x_start=105, x_end=695, thickness=2),
            StaffLine(y=260.0, x_start=108, x_end=680, thickness=2),
            StaffLine(y=280.0, x_start=112, x_end=670, thickness=2),
        )
    )

    assert grid.x_start == 100
    assert grid.x_end == 700
    assert grid.width == 601


def test_detected_extent_matches_the_drawn_extent():
    grid = rendered_staff(x_start=150, x_end=650)

    assert grid.x_start == pytest.approx(150, abs=X_TOL)
    assert grid.x_end == pytest.approx(650, abs=X_TOL)


def test_line_width_spans_both_endpoints():
    line = StaffLine(y=1.0, x_start=10, x_end=19, thickness=2)

    assert line.width == 10


def test_a_band_promotes_to_a_line_unchanged():
    band = LineBand(y=200.5, x_start=100, x_end=700, thickness=3)

    line = StaffLine.from_band(band)

    assert (line.y, line.x_start, line.x_end, line.thickness) == (200.5, 100, 700, 3)


# --------------------------------------------------------------------------- #
# The 0-8 position mapping
# --------------------------------------------------------------------------- #


def test_the_bottom_line_is_position_zero():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.y_to_position(280.0) == pytest.approx(0.0)


def test_the_top_line_is_position_eight():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.y_to_position(200.0) == pytest.approx(float(STAFF_TOP_POSITION))


@pytest.mark.parametrize(
    ("position", "y"),
    [(0, 280.0), (2, 260.0), (4, 240.0), (6, 220.0), (8, 200.0)],
)
def test_every_line_position_maps_to_its_row(position, y):
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.position_to_y(position) == pytest.approx(y)
    assert grid.y_to_position(y) == pytest.approx(float(position))


@pytest.mark.parametrize(
    ("position", "y"),
    [(1, 270.0), (3, 250.0), (5, 230.0), (7, 210.0)],
)
def test_every_space_position_maps_to_its_row(position, y):
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.position_to_y(position) == pytest.approx(y)


def test_the_snare_space_sits_where_stage_three_expects_it():
    """Position 5 is space 3, the acoustic snare in the drum mapping."""
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.position_to_y(5) == pytest.approx(grid.center_y - grid.step_height)


def test_even_positions_are_lines_and_odd_positions_are_spaces():
    assert StaffGrid.is_line_position(0)
    assert StaffGrid.is_line_position(8)
    assert not StaffGrid.is_line_position(5)


@pytest.mark.parametrize("position", [-4, -2, 0, 3, 4, 8, 10, 12])
def test_position_and_row_round_trip(position):
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.y_to_position(grid.position_to_y(position)) == pytest.approx(float(position))


def test_positions_above_the_staff_extend_linearly():
    """One ledger line above the top line is simply position 10."""
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.position_to_y(10) == pytest.approx(180.0)
    assert grid.y_to_position(180.0) == pytest.approx(10.0)


def test_positions_below_the_staff_are_negative():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.position_to_y(-2) == pytest.approx(300.0)
    assert grid.y_to_position(300.0) == pytest.approx(-2.0)


def test_a_row_between_a_line_and_a_space_is_fractional():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.y_to_position(275.0) == pytest.approx(0.5)


def test_interpolation_follows_measured_rows_not_assumed_spacing():
    """A warped staff maps by its own lines, so error does not accumulate."""
    warped = StaffGrid(
        lines=(
            StaffLine(y=200.0, x_start=0, x_end=10, thickness=2),
            StaffLine(y=224.0, x_start=0, x_end=10, thickness=2),
            StaffLine(y=244.0, x_start=0, x_end=10, thickness=2),
            StaffLine(y=262.0, x_start=0, x_end=10, thickness=2),
            StaffLine(y=280.0, x_start=0, x_end=10, thickness=2),
        )
    )

    assert warped.y_to_position(244.0) == pytest.approx(4.0)
    assert warped.y_to_position(224.0) == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# Snapping
# --------------------------------------------------------------------------- #


def test_nearest_position_rounds_to_an_integer():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.nearest_position(241.0) == 4
    assert isinstance(grid.nearest_position(241.0), int)


def test_snap_accepts_a_row_on_a_line():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.snap(240.0) == 4


def test_snap_accepts_a_row_slightly_off_a_line():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.snap(242.0) == 4


def test_snap_refuses_a_row_halfway_between_positions():
    """Stage 3 must not guess: an ambiguous centroid is a detection error."""
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.snap(245.0) is None


def test_a_wider_tolerance_admits_what_the_default_refuses():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.snap(245.0) is None
    assert grid.snap(245.0, tolerance=0.5) == 4


@pytest.mark.parametrize("tolerance", [0.0, -0.1, 0.6, 1.0])
def test_snap_rejects_a_tolerance_outside_the_permitted_range(tolerance):
    grid = synthetic_grid()

    with pytest.raises(ValueError, match="tolerance"):
        grid.snap(240.0, tolerance=tolerance)


# --------------------------------------------------------------------------- #
# Vertical reach
# --------------------------------------------------------------------------- #


def test_a_row_on_the_staff_is_contained():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.contains_y(240.0)
    assert grid.contains_y(200.0)
    assert grid.contains_y(280.0)


def test_a_row_above_the_staff_is_not_contained():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert not grid.contains_y(190.0)


def test_ledger_steps_widen_the_reach():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert not grid.contains_y(190.0)
    assert grid.contains_y(190.0, ledger_steps=2)


def test_two_ledger_lines_are_four_steps():
    grid = synthetic_grid(top=200.0, spacing=20.0)

    assert grid.contains_y(grid.position_to_y(12), ledger_steps=4)
    assert not grid.contains_y(grid.position_to_y(12), ledger_steps=2)
