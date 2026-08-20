"""Unit tests for :mod:`melodix.geometry.systems`.

The load-bearing assertion in this file is
:func:`test_margin_barlines_do_not_fuse_separate_systems`. Barlines at the left
and right margins align vertically down the whole page, so a grouping pass that
took column alignment as evidence of a system would collapse every system into
one — and the failure is silent, producing a plausible-looking result with the
measure count of one system and the staff count of the page. Systems must be
established from spacing and spanning strokes first, columns only afterwards.

Most tests build :class:`StaffGrid` and :class:`VerticalSegment` objects
directly. This module is pure geometry over the earlier passes' output, so
feeding it exact inputs isolates it from thresholding behaviour upstream; the
few end-to-end tests at the bottom confirm the modules still meet.
"""

from __future__ import annotations

import pytest

from melodix.geometry.barlines import VerticalSegment, detect_vertical_segments
from melodix.geometry.staff import LINES_PER_STAFF, StaffGrid, StaffLine, detect_staff_grids
from melodix.geometry.systems import (
    BarlineColumn,
    Measure,
    System,
    SystemGroupingConfig,
    assign_systems,
    build_columns,
    build_systems,
    classify_barlines,
    slice_measures,
)

from tests.helpers import blank_page, draw_notehead, draw_staff, draw_stem


def staff(top: float, spacing: float = 20.0, x_start: int = 100, x_end: int = 700, index: int = 0):
    """Build a staff at an exact row, bypassing detection."""
    return StaffGrid(
        lines=tuple(
            StaffLine(y=top + step * spacing, x_start=x_start, x_end=x_end, thickness=2)
            for step in range(LINES_PER_STAFF)
        ),
        index=index,
    )


def barline(x: float, top: float, bottom: float, thickness: int = 2) -> VerticalSegment:
    """A stroke running exactly from ``top`` to ``bottom``."""
    return VerticalSegment(x=x, y_start=int(top), y_end=int(bottom), thickness=thickness)


def column(x: float, staff_indices: tuple[int, ...] = (0,)) -> BarlineColumn:
    """A column with no member strokes, for slicing tests."""
    return BarlineColumn(x=x, segments=(), staff_indices=staff_indices)


# A page-shaped fixture: two systems, the first of two staves, the second of one.
# Staff height is 80px, so the 1.75 gap ratio keeps staves within 140px together.
SYSTEM_ONE = (100.0, 260.0)  # gap 80: same system
SYSTEM_TWO = (500.0,)  # gap 160 from the staff above: a new system


def page_staves():
    """The staves of the two-system fixture, in page order."""
    tops = [*SYSTEM_ONE, *SYSTEM_TWO]
    return [staff(top, index=i) for i, top in enumerate(tops)]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("span_tolerance_px", -1.0),
        ("column_tolerance_px", -1.0),
        ("system_gap_ratio", 0.0),
        ("system_gap_ratio", -1.0),
        ("min_measure_width_px", 0),
        ("min_measure_width_px", -5),
    ],
)
def test_invalid_config_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        SystemGroupingConfig(**{field: value})


def test_zero_tolerances_are_permitted():
    config = SystemGroupingConfig(span_tolerance_px=0.0, column_tolerance_px=0.0)

    assert config.span_tolerance_px == 0.0


def test_a_full_span_is_required_by_default():
    assert SystemGroupingConfig().require_full_span


# --------------------------------------------------------------------------- #
# Measure
# --------------------------------------------------------------------------- #


def test_an_inverted_measure_is_rejected():
    with pytest.raises(ValueError, match="x_end must be at least x_start"):
        Measure(0, 0, 0, x_start=500, x_end=100, y_top=200.0, y_bottom=280.0)


@pytest.mark.parametrize("indices", [(-1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_negative_indices_are_rejected(indices):
    with pytest.raises(ValueError, match="indices must be non-negative"):
        Measure(*indices, x_start=100, x_end=200, y_top=200.0, y_bottom=280.0)


def test_the_key_is_the_full_address():
    measure = Measure(1, 2, 3, x_start=100, x_end=200, y_top=200.0, y_bottom=280.0)

    assert measure.key == (1, 2, 3)


def test_measure_width_counts_both_edges():
    assert Measure(0, 0, 0, x_start=100, x_end=199, y_top=0.0, y_bottom=8.0).width == 100


def test_a_point_inside_the_measure_is_contained():
    measure = Measure(0, 0, 0, x_start=100, x_end=200, y_top=200.0, y_bottom=280.0)

    assert measure.contains(150, 240)


def test_the_corners_are_contained():
    measure = Measure(0, 0, 0, x_start=100, x_end=200, y_top=200.0, y_bottom=280.0)

    assert measure.contains(100, 200)
    assert measure.contains(200, 280)


def test_a_point_in_the_next_measure_is_not_contained():
    measure = Measure(0, 0, 0, x_start=100, x_end=200, y_top=200.0, y_bottom=280.0)

    assert not measure.contains(250, 240)


def test_a_notehead_above_the_top_line_is_outside():
    """Vertical bounds are the staff lines; ledger notes need a widened test."""
    measure = Measure(0, 0, 0, x_start=100, x_end=200, y_top=200.0, y_bottom=280.0)

    assert not measure.contains(150, 190)


# --------------------------------------------------------------------------- #
# BarlineColumn
# --------------------------------------------------------------------------- #


def test_a_column_rounds_its_centre_to_a_pixel():
    assert column(300.6).x_int == 301


def test_a_column_reaching_every_staff_spans_the_system():
    assert column(300.0, staff_indices=(0, 1)).spans_staves(2)


def test_a_partial_column_is_a_staff_local_division():
    assert not column(300.0, staff_indices=(0,)).spans_staves(2)


def test_repeated_staff_indices_do_not_inflate_the_span():
    assert not column(300.0, staff_indices=(0, 0)).spans_staves(2)


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #


def test_a_system_needs_at_least_one_staff():
    with pytest.raises(ValueError, match="at least one staff"):
        System(index=0, staves=())


def test_a_negative_system_index_is_rejected():
    with pytest.raises(ValueError, match="index must be non-negative"):
        System(index=-1, staves=(staff(100.0),))


def test_system_bounds_come_from_the_outer_staves():
    system = System(index=0, staves=(staff(100.0), staff(260.0)))

    assert system.y_top == 100.0
    assert system.y_bottom == 340.0
    assert system.staff_count == 2


def test_a_system_without_measures_counts_none():
    assert System(index=0, staves=(staff(100.0),)).measure_count == 0


def test_measure_count_is_per_staff_not_a_total():
    """Two staves of three bars each is a three-measure system."""
    measures = tuple(
        Measure(0, s, m, x_start=100 + m * 100, x_end=200 + m * 100, y_top=0.0, y_bottom=8.0)
        for s in range(2)
        for m in range(3)
    )
    system = System(index=0, staves=(staff(100.0), staff(260.0)), measures=measures)

    assert len(system.measures) == 6
    assert system.measure_count == 3


def test_one_staff_can_be_addressed_alone():
    measures = tuple(
        Measure(0, s, m, x_start=100 + m * 100, x_end=200 + m * 100, y_top=0.0, y_bottom=8.0)
        for s in range(2)
        for m in range(3)
    )
    system = System(index=0, staves=(staff(100.0), staff(260.0)), measures=measures)

    assert [m.key for m in system.measures_for_staff(1)] == [(0, 1, 0), (0, 1, 1), (0, 1, 2)]


def test_addressing_a_staff_out_of_range_is_rejected():
    system = System(index=0, staves=(staff(100.0),))

    with pytest.raises(IndexError, match="out of range"):
        system.measures_for_staff(5)


def test_the_ensemble_view_returns_every_players_bar():
    """Measure n of every staff sounds at the same moment."""
    measures = tuple(
        Measure(0, s, m, x_start=100 + m * 100, x_end=200 + m * 100, y_top=0.0, y_bottom=8.0)
        for s in range(3)
        for m in range(2)
    )
    system = System(index=0, staves=(staff(100.0), staff(260.0), staff(420.0)), measures=measures)

    simultaneous = system.measures_at(1)

    assert len(simultaneous) == 3
    assert {m.staff_index for m in simultaneous} == {0, 1, 2}
    assert {m.measure_index for m in simultaneous} == {1}


# --------------------------------------------------------------------------- #
# Tier 1: classifying strokes against staves
# --------------------------------------------------------------------------- #


def test_a_stroke_spanning_the_staff_is_a_barline():
    one = staff(200.0)

    found = classify_barlines([barline(300, 200, 280)], [one])

    assert len(found[0]) == 1


def test_a_stem_stopping_short_is_not_a_barline():
    """It hangs off a notehead and never reaches one end of the staff."""
    one = staff(200.0)

    assert classify_barlines([barline(300, 230, 280)], [one])[0] == []


def test_every_staff_appears_even_without_barlines():
    """Callers index the result directly, so no key may be missing."""
    found = classify_barlines([], [staff(100.0), staff(260.0)])

    assert set(found) == {0, 1}
    assert found[0] == []
    assert found[1] == []


def test_a_stroke_through_a_whole_system_counts_for_every_staff():
    """Why the two engraving styles converge here rather than downstream."""
    staves = [staff(100.0), staff(260.0)]

    found = classify_barlines([barline(300, 100, 340)], staves)

    assert len(found[0]) == 1
    assert len(found[1]) == 1


def test_a_stroke_is_not_a_barline_for_a_staff_it_misses():
    staves = [staff(100.0), staff(260.0)]

    found = classify_barlines([barline(300, 100, 180)], staves)

    assert len(found[0]) == 1
    assert found[1] == []


def test_the_span_tolerance_admits_a_stroke_ending_a_pixel_short():
    one = staff(200.0)
    short = [barline(300, 202, 278)]

    assert classify_barlines(short, [one], SystemGroupingConfig(span_tolerance_px=0.0))[0] == []
    assert len(classify_barlines(short, [one], SystemGroupingConfig(span_tolerance_px=3.0))[0]) == 1


def test_partial_strokes_can_be_admitted_for_damaged_scans():
    one = staff(200.0)
    partial = [barline(300, 230, 280)]

    lenient = SystemGroupingConfig(require_full_span=False)

    assert classify_barlines(partial, [one])[0] == []
    assert len(classify_barlines(partial, [one], lenient)[0]) == 1


def test_barlines_are_ordered_left_to_right():
    one = staff(200.0)
    scrambled = [barline(x, 200, 280) for x in (500, 100, 300)]

    found = classify_barlines(scrambled, [one])

    assert [b.x for b in found[0]] == [100, 300, 500]


# --------------------------------------------------------------------------- #
# Tier 2: grouping staves into systems
# --------------------------------------------------------------------------- #


def test_closely_spaced_staves_share_a_system():
    assert assign_systems([staff(100.0), staff(260.0)]) == [(0, 1)]


def test_widely_spaced_staves_are_separate_systems():
    assert assign_systems([staff(100.0), staff(500.0)]) == [(0,), (1,)]


def test_the_gap_ratio_decides_where_the_split_falls():
    staves = [staff(100.0), staff(300.0)]

    assert assign_systems(staves, config=SystemGroupingConfig(system_gap_ratio=0.5)) == [(0,), (1,)]
    assert assign_systems(staves, config=SystemGroupingConfig(system_gap_ratio=3.0)) == [(0, 1)]


def test_a_stroke_crossing_two_staves_ties_them_together():
    """No engraver draws a rule through the gap between systems."""
    staves = [staff(100.0), staff(500.0)]
    spanning = [barline(300, 100, 580)]

    assert assign_systems(staves) == [(0,), (1,)]
    assert assign_systems(staves, spanning) == [(0, 1)]


def test_systems_are_ordered_down_the_page():
    groups = assign_systems(page_staves())

    assert groups == [(0, 1), (2,)]


def test_staves_within_a_system_are_ordered_top_to_bottom():
    staves = [staff(260.0, index=0), staff(100.0, index=1)]

    assert assign_systems(staves) == [(1, 0)]


def test_no_staves_yield_no_systems():
    assert assign_systems([]) == []


def test_a_lone_staff_is_its_own_system():
    assert assign_systems([staff(200.0)]) == [(0,)]


def test_three_staves_of_one_system_stay_together():
    assert assign_systems([staff(100.0), staff(260.0), staff(420.0)]) == [(0, 1, 2)]


def test_margin_barlines_do_not_fuse_separate_systems():
    """The ordering guarantee. These columns align down the entire page; if
    alignment were taken as evidence of a system, both systems would collapse
    into one and the error would be silent."""
    staves = page_staves()
    margins = [
        barline(100, 100, 180),
        barline(700, 100, 180),
        barline(100, 260, 340),
        barline(700, 260, 340),
        barline(100, 500, 580),
        barline(700, 500, 580),
    ]

    assert assign_systems(staves, margins) == [(0, 1), (2,)]


# --------------------------------------------------------------------------- #
# Tier 2: column alignment
# --------------------------------------------------------------------------- #


def test_barlines_at_one_column_across_staves_form_one_column():
    staves = [staff(100.0), staff(260.0)]
    by_staff = {0: [barline(300, 100, 180)], 1: [barline(300, 260, 340)]}

    columns = build_columns(staves, by_staff, (0, 1))

    assert len(columns) == 1
    assert columns[0].staff_indices == (0, 1)


def test_a_small_column_drift_is_absorbed():
    staves = [staff(100.0), staff(260.0)]
    by_staff = {0: [barline(300, 100, 180)], 1: [barline(302, 260, 340)]}

    columns = build_columns(staves, by_staff, (0, 1))

    assert len(columns) == 1
    assert columns[0].x == pytest.approx(301.0)


def test_barlines_too_far_apart_stay_in_separate_columns():
    staves = [staff(100.0), staff(260.0)]
    by_staff = {0: [barline(300, 100, 180)], 1: [barline(310, 260, 340)]}

    columns = build_columns(staves, by_staff, (0, 1))

    assert len(columns) == 2
    assert not any(c.spans_staves(2) for c in columns)


def test_the_column_tolerance_can_be_widened():
    staves = [staff(100.0), staff(260.0)]
    by_staff = {0: [barline(300, 100, 180)], 1: [barline(310, 260, 340)]}
    loose = SystemGroupingConfig(column_tolerance_px=12.0)

    assert len(build_columns(staves, by_staff, (0, 1), loose)) == 1


def test_columns_are_ordered_left_to_right():
    staves = [staff(100.0)]
    by_staff = {0: [barline(x, 100, 180) for x in (500, 100, 300)]}

    columns = build_columns(staves, by_staff, (0,))

    assert [c.x for c in columns] == [100.0, 300.0, 500.0]


def test_a_column_records_its_strokes_top_to_bottom():
    staves = [staff(100.0), staff(260.0)]
    by_staff = {0: [barline(300, 100, 180)], 1: [barline(300, 260, 340)]}

    columns = build_columns(staves, by_staff, (0, 1))

    assert [s.y_start for s in columns[0].segments] == [100, 260]


def test_staff_indices_are_local_to_the_system():
    """The second system's first staff is index 0, not its page ordinal."""
    staves = page_staves()
    by_staff = {0: [], 1: [], 2: [barline(300, 500, 580)]}

    columns = build_columns(staves, by_staff, (2,))

    assert columns[0].staff_indices == (0,)


def test_a_continuous_stroke_contributes_once_per_staff():
    """One object, two staves: the column still records both."""
    staves = [staff(100.0), staff(260.0)]
    spanning = barline(300, 100, 340)
    by_staff = {0: [spanning], 1: [spanning]}

    columns = build_columns(staves, by_staff, (0, 1))

    assert len(columns) == 1
    assert columns[0].spans_staves(2)


def test_no_barlines_yield_no_columns():
    assert build_columns([staff(100.0)], {0: []}, (0,)) == []


# --------------------------------------------------------------------------- #
# Tier 3: measure slicing
# --------------------------------------------------------------------------- #


def test_two_interior_columns_cut_three_measures():
    one = staff(200.0, x_start=100, x_end=700)

    measures = slice_measures(one, [column(300.0), column(500.0)], 0, 0)

    assert [(m.x_start, m.x_end) for m in measures] == [(100, 300), (300, 500), (500, 700)]


def test_measures_are_numbered_from_zero():
    one = staff(200.0)

    measures = slice_measures(one, [column(300.0), column(500.0)], 0, 0)

    assert [m.measure_index for m in measures] == [0, 1, 2]


def test_measures_carry_their_system_and_staff():
    one = staff(200.0)

    measures = slice_measures(one, [column(400.0)], system_index=2, staff_index=1)

    assert all(m.system_index == 2 and m.staff_index == 1 for m in measures)


def test_measures_inherit_the_staff_rows():
    one = staff(200.0)

    measures = slice_measures(one, [column(400.0)], 0, 0)

    assert measures[0].y_top == 200.0
    assert measures[0].y_bottom == 280.0


def test_a_barline_at_the_staff_edge_makes_no_empty_measure():
    one = staff(200.0, x_start=100, x_end=700)

    measures = slice_measures(one, [column(100.0), column(400.0), column(700.0)], 0, 0)

    assert len(measures) == 2


def test_the_sliver_between_repeat_rules_is_dropped():
    """Two rules 5px apart bound nothing playable."""
    one = staff(200.0, x_start=100, x_end=700)

    measures = slice_measures(one, [column(300.0), column(305.0)], 0, 0)

    assert [(m.x_start, m.x_end) for m in measures] == [(100, 300), (305, 700)]
    assert [m.measure_index for m in measures] == [0, 1]


def test_the_minimum_width_can_be_raised():
    one = staff(200.0, x_start=100, x_end=700)
    columns = [column(300.0), column(340.0)]

    wide = SystemGroupingConfig(min_measure_width_px=60)

    assert len(slice_measures(one, columns, 0, 0)) == 3
    assert len(slice_measures(one, columns, 0, 0, wide)) == 2


def test_columns_beyond_the_staff_are_ignored():
    one = staff(200.0, x_start=100, x_end=700)

    measures = slice_measures(one, [column(50.0), column(400.0), column(750.0)], 0, 0)

    assert [(m.x_start, m.x_end) for m in measures] == [(100, 400), (400, 700)]


def test_a_staff_with_no_columns_is_one_measure():
    one = staff(200.0, x_start=100, x_end=700)

    measures = slice_measures(one, [], 0, 0)

    assert [(m.x_start, m.x_end) for m in measures] == [(100, 700)]


def test_a_staff_narrower_than_the_minimum_yields_nothing():
    narrow = staff(200.0, x_start=100, x_end=105)

    assert slice_measures(narrow, [], 0, 0) == []


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #


def test_no_staves_yield_no_systems_end_to_end():
    assert build_systems([], []) == []


def test_a_page_of_two_systems_is_assembled():
    staves = page_staves()
    segments = [
        *[barline(x, 100, 340) for x in (100, 300, 500, 700)],
        *[barline(x, 500, 580) for x in (100, 400, 700)],
    ]

    systems = build_systems(staves, segments)

    assert [s.index for s in systems] == [0, 1]
    assert [s.staff_count for s in systems] == [2, 1]
    assert [s.measure_count for s in systems] == [3, 2]


def test_every_staff_of_a_system_gets_the_same_measure_count():
    """The ensemble invariant: players share a timeline."""
    staves = page_staves()
    segments = [barline(x, 100, 340) for x in (100, 300, 500, 700)]

    system = build_systems(staves, segments)[0]

    assert len(system.measures_for_staff(0)) == len(system.measures_for_staff(1)) == 3


def test_measure_addresses_are_unique_across_a_page():
    staves = page_staves()
    segments = [
        *[barline(x, 100, 340) for x in (100, 300, 500, 700)],
        *[barline(x, 500, 580) for x in (100, 400, 700)],
    ]

    keys = [m.key for s in build_systems(staves, segments) for m in s.measures]

    assert len(keys) == len(set(keys))


def test_stems_do_not_create_measures():
    """A page of notes but no barlines is one measure per staff."""
    staves = [staff(200.0)]
    stems = [barline(x, 230, 280) for x in (200, 300, 400, 500)]

    system = build_systems(staves, stems)[0]

    assert system.columns == ()
    assert system.measure_count == 1


def test_both_engraving_styles_produce_the_same_structure():
    """Continuous system rules versus per-staff rules, same measure grid."""
    staves = page_staves()[:2]
    continuous = [barline(x, 100, 340) for x in (100, 300, 500, 700)]
    per_staff = [
        *[barline(x, 100, 180) for x in (100, 300, 500, 700)],
        *[barline(x, 260, 340) for x in (100, 300, 500, 700)],
    ]

    one = build_systems(staves, continuous)[0]
    other = build_systems(staves, per_staff)[0]

    assert [c.x for c in one.columns] == [c.x for c in other.columns]
    assert [m.key for m in one.measures] == [m.key for m in other.measures]


# --------------------------------------------------------------------------- #
# Against the earlier passes
# --------------------------------------------------------------------------- #


def test_a_rendered_page_flows_through_all_three_passes():
    page = blank_page(700, 800)
    for top in (100, 260):
        draw_staff(page, top_row=top, spacing=20, x_start=100, x_end=700)
    draw_staff(page, top_row=500, spacing=20, x_start=100, x_end=700)
    for x in (100, 300, 500, 700):
        draw_stem(page, x=x, top_row=100, bottom_row=340, width=3)
    for x in (100, 400, 700):
        draw_stem(page, x=x, top_row=500, bottom_row=580, width=3)

    staves = detect_staff_grids(page)
    segments = detect_vertical_segments(page, staff_spacing=staves[0].line_spacing)
    systems = build_systems(staves, segments)

    assert [s.staff_count for s in systems] == [2, 1]
    assert [s.measure_count for s in systems] == [3, 2]


def test_noteheads_on_a_rendered_page_land_in_the_right_measures():
    """The Stage 3 handoff: a detection centroid resolves to one address."""
    page = blank_page(600, 800)
    draw_staff(page, top_row=200, spacing=20, x_start=100, x_end=700)
    for x in (100, 300, 500, 700):
        draw_stem(page, x=x, top_row=200, bottom_row=280, width=3)
    for x in (200, 400, 600):
        draw_notehead(page, center_x=x, center_y=240, radius=8)

    staves = detect_staff_grids(page)
    segments = detect_vertical_segments(page, staff_spacing=staves[0].line_spacing)
    system = build_systems(staves, segments)[0]

    hits = [
        [m.measure_index for m in system.measures if m.contains(x, 240)] for x in (200, 400, 600)
    ]

    assert hits == [[0], [1], [2]]
