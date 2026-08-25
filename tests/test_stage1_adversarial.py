"""Stage 1 against ink that is not perfect.

This file exists because of a specific failure, and the failure was not in the
code alone. `staff.py` assumed unbroken runs and every one of its tests drew
solid, perfect lines, so the implementation and the corpus shared an assumption
and the suite could report green forever without ever touching it. A single
background pixel in a staff line erased the whole line, and 102 passing tests
said nothing about it.

Mutation testing cannot find that class of gap. It checks that tests detect
changes to the code; it cannot tell you the corpus never contained the input
that matters. The only thing that finds it is data drawn from different
assumptions than the code — which is what these fixtures are.

Every fixture here damages ink the way a real page does: interrupted lines,
faint sections, crossing ink, speckle. Several are pinned against the *unfixed*
behaviour, so the regression is anchored to something observed rather than to a
belief about what should happen.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from melodix.geometry.barlines import (
    BarlineDetectionConfig,
    detect_vertical_segments,
)
from melodix.geometry.staff import (
    StaffDetectionConfig,
    detect_staff_grids,
)
from melodix.geometry.systems import build_systems

#: The behaviour before gap tolerance was added. Used to pin regressions
#: against something measured rather than assumed.
NO_CLOSING = StaffDetectionConfig(gap_closing_ratio=0.0)
NO_CLOSING_VERTICAL = BarlineDetectionConfig(gap_closing_ratio=0.0)

PAGE_W = 800
PAGE_H = 600
SPACING = 20
TOP = 200


def clean_page(staves: int = 1, spacing: int = SPACING) -> np.ndarray:
    """A page with solid five-line staves, drawn as the original tests drew them."""
    page = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    for staff in range(staves):
        top = TOP + staff * (spacing * 4 + 120)
        for step in range(5):
            row = top + step * spacing
            page[row : row + 2, 100:700] = 0
    return page


def punch(page: np.ndarray, columns: list[tuple[int, int]], rows: list[int]) -> np.ndarray:
    """Erase short horizontal runs from given rows, simulating interruptions."""
    out = page.copy()
    for row in rows:
        for start, width in columns:
            out[row : row + 3, start : start + width] = 255
    return out


def speckle(page: np.ndarray, density: float, seed: int) -> np.ndarray:
    """Scatter salt-and-pepper noise, as dust and scanner grain do."""
    out = page.copy()
    mask = np.random.default_rng(seed).random(out.shape)
    out[mask < density / 2] = 0
    out[mask > 1 - density / 2] = 255
    return out


def staff_rows(spacing: int = SPACING) -> list[int]:
    """The five drawn line rows."""
    return [TOP + step * spacing for step in range(5)]


# --------------------------------------------------------------------------- #
# The mechanism: how many breaks it takes
# --------------------------------------------------------------------------- #


def test_one_break_is_survivable_and_two_are_not():
    """The mechanism, measured rather than assumed.

    An opening needs an unbroken run at least as long as its kernel, which is
    35% of page width. A line spanning most of the page therefore survives one
    break — the two halves are still ~50% of width each — and dies at two,
    where the three fragments are ~33%.

    Worth pinning precisely: the loose claim is that a single pixel destroys a
    line, and on this fixture that is not what happens. Two specks of dust are
    enough, which is still fragile, but the number matters for anyone reasoning
    about it later.
    """
    one = punch(clean_page(), columns=[(400, 1)], rows=staff_rows())
    two = punch(clean_page(), columns=[(300, 1), (500, 1)], rows=staff_rows())

    assert len(detect_staff_grids(one, NO_CLOSING)) == 1
    assert detect_staff_grids(two, NO_CLOSING) == []

    # Both recover once short gaps are bridged.
    assert len(detect_staff_grids(one)) == 1
    assert len(detect_staff_grids(two)) == 1


@pytest.mark.parametrize("gap", [1, 2])
def test_short_interruptions_are_bridged(gap):
    """A closing of length k bridges a gap of up to k-1 px, and at this page
    width the kernel is 3. Two breaks, so the unfixed code fails and the test
    is measuring the bridging rather than the line's own margin.
    """
    damaged = punch(clean_page(), columns=[(300, gap), (500, gap)], rows=staff_rows())

    assert detect_staff_grids(damaged, NO_CLOSING) == []
    assert len(detect_staff_grids(damaged)) == 1


def test_a_gap_wider_than_the_kernel_is_not_bridged():
    """The closing is bounded on purpose, and the bound is k-1 px."""
    damaged = punch(clean_page(), columns=[(300, 6), (500, 6)], rows=staff_rows())

    assert detect_staff_grids(damaged) == []


def test_many_interruptions_along_one_line_are_bridged():
    """A streaky line, not a single speck."""
    damaged = punch(
        clean_page(), columns=[(200, 2), (350, 1), (500, 2), (620, 1)], rows=staff_rows()
    )

    assert detect_staff_grids(damaged, NO_CLOSING) == []
    assert len(detect_staff_grids(damaged)) == 1


def test_a_staple_sized_hole_is_not_invented_over():
    """A genuine absence of ink — a staple, a tear — must not be filled in.

    Punched mid-line so no fragment reaches the opening kernel, which is what
    makes this a test of the closing's bound rather than of the line's margin.
    """
    damaged = punch(clean_page(), columns=[(350, 100)], rows=staff_rows())

    assert detect_staff_grids(damaged) == []


def test_a_short_faint_section_is_bridged():
    """Toner dropout leaves grey, and global Otsu on a mostly-white page reads
    mid-grey as background — so a faint section becomes a gap rather than
    surviving as ink. Recorded because it is a real limitation of the
    binarisation and separate from what the closing does: short faint runs are
    then bridged as gaps, long ones are not.
    """
    page = clean_page()
    for row in staff_rows():
        page[row : row + 2, 300:302] = 150      # 2 px of grey, twice
        page[row : row + 2, 500:502] = 150

    assert detect_staff_grids(page, NO_CLOSING) == []
    assert len(detect_staff_grids(page)) == 1


def test_a_long_faint_section_is_still_lost():
    """The other side of it: the closing does not resurrect faint ink, it only
    bridges short gaps. A long faint run is a hole.
    """
    page = clean_page()
    for row in staff_rows():
        page[row : row + 2, 350:450] = 150

    assert detect_staff_grids(page) == []


def test_ink_crossing_the_staff_does_not_break_it():
    """A stem, a slur, a pencil mark. Adding ink must not remove a line."""
    page = clean_page()
    for x in (250, 400, 550):
        cv2.line(page, (x, TOP - 40), (x, TOP + 4 * SPACING + 40), 0, 3)

    assert len(detect_staff_grids(page)) == 1


# --------------------------------------------------------------------------- #
# Speckle, reported as a distribution
# --------------------------------------------------------------------------- #

# Measured on this fixture over 12 seeds, staves recovered out of 1 expected.
# Recorded so the numbers in the report are reproducible from the test file.
#
#   density   without closing   with closing
#   0.0010        12/12             12/12
#   0.0025         9/12             12/12
#   0.0050         2/12             12/12
#   0.0100         0/12             12/12
SPECKLE_SEEDS = range(12)


def recovered(density: float, config: StaffDetectionConfig | None = None) -> int:
    """How many of the seeds recover the staff at this noise density."""
    page = clean_page()
    settings = config if config is not None else StaffDetectionConfig()
    return sum(
        len(detect_staff_grids(speckle(page, density, seed), settings)) == 1
        for seed in SPECKLE_SEEDS
    )


@pytest.mark.parametrize("density", [0.001, 0.0025, 0.005, 0.01])
def test_speckled_pages_recover_at_every_density(density):
    """Reported as a distribution, not a single lucky seed."""
    assert recovered(density) == len(SPECKLE_SEEDS)


def test_heavy_speckle_defeated_the_unfixed_code():
    """Anchors the fix to observed behaviour: this density must be one the old
    code demonstrably fails, or the test above proves nothing.
    """
    before = recovered(0.01, NO_CLOSING)

    assert before < len(SPECKLE_SEEDS) / 2, f"{before}/12 recovered without closing"


def test_the_fix_strictly_improves_recovery():
    for density in (0.0025, 0.005, 0.01):
        assert recovered(density) >= recovered(density, NO_CLOSING)


# --------------------------------------------------------------------------- #
# The same for barlines, where the consequence is worse
# --------------------------------------------------------------------------- #


def page_with_barlines(columns=(150, 350, 550), interrupt: int = 0) -> np.ndarray:
    """A staff with barlines, optionally interrupted mid-stroke."""
    page = clean_page()
    for x in columns:
        cv2.line(page, (x, TOP), (x, TOP + 4 * SPACING), 0, 3)
        if interrupt:
            middle = TOP + 2 * SPACING
            page[middle : middle + interrupt, x - 3 : x + 4] = 255
    return page


def test_an_interrupted_barline_used_to_vanish():
    """Worse than a lost staff line, because it is silent and it propagates:
    a dropped barline merges two measures and shifts every measure index after
    it on that staff.
    """
    damaged = page_with_barlines(interrupt=1)

    before = detect_vertical_segments(damaged, NO_CLOSING_VERTICAL, staff_spacing=SPACING)
    after = detect_vertical_segments(damaged, staff_spacing=SPACING)

    assert len(before) < 3
    assert len(after) == 3


@pytest.mark.parametrize("interrupt", [1, 2])
def test_short_barline_interruptions_are_bridged(interrupt):
    damaged = page_with_barlines(interrupt=interrupt)

    assert len(detect_vertical_segments(damaged, staff_spacing=SPACING)) == 3


def test_a_dropped_barline_shifts_every_later_measure_index():
    """The propagation, demonstrated rather than asserted in prose."""
    damaged = page_with_barlines(interrupt=1)
    grids = detect_staff_grids(damaged)

    broken = build_systems(
        grids, detect_vertical_segments(damaged, NO_CLOSING_VERTICAL, staff_spacing=SPACING)
    )
    whole = build_systems(
        grids, detect_vertical_segments(damaged, staff_spacing=SPACING)
    )

    assert broken[0].measure_count < whole[0].measure_count


def test_the_closing_does_not_invent_a_barline_from_a_stem():
    """A (1,k) kernel spans rows only, so it cannot pull a neighbouring stem
    sideways into a barline. Confirmed rather than assumed, since the
    consequence would be a phantom measure boundary.
    """
    page = clean_page()
    cv2.line(page, (300, TOP), (300, TOP + 4 * SPACING), 0, 3)      # real barline
    for dx in (4, 8, 12):
        cv2.line(page, (300 + dx, TOP - 30), (300 + dx, TOP + 30), 0, 3)  # stems beside

    found = detect_vertical_segments(page, staff_spacing=SPACING)
    grids = detect_staff_grids(page)
    systems = build_systems(grids, found)

    # The stems do not span the staff, so none of them becomes a barline.
    assert len(systems[0].columns) == 1


def test_a_speckled_barline_survives():
    page = speckle(page_with_barlines(), density=0.005, seed=3)

    assert len(detect_vertical_segments(page, staff_spacing=SPACING)) >= 3


# --------------------------------------------------------------------------- #
# The closing must not change clean pages
# --------------------------------------------------------------------------- #


def test_a_clean_page_is_unaffected():
    """The fix must be free on undamaged input, or it is not a fix."""
    page = clean_page(staves=2)

    assert len(detect_staff_grids(page)) == len(detect_staff_grids(page, NO_CLOSING)) == 2


def test_line_rows_are_not_shifted_by_the_closing():
    """An even-length kernel anchors off-centre and shifts every line one pixel.
    That bug cost two rounds in Phase 1.1; the closing must not reintroduce it.
    """
    page = clean_page()

    with_closing = detect_staff_grids(page)[0]
    without = detect_staff_grids(page, NO_CLOSING)[0]

    assert with_closing.line_ys == pytest.approx(without.line_ys, abs=0.01)


def test_horizontal_extent_is_not_shifted_by_the_closing():
    """The other half of the off-centre symptom: a sideways shift of the ends."""
    page = clean_page()

    with_closing = detect_staff_grids(page)[0]
    without = detect_staff_grids(page, NO_CLOSING)[0]

    assert with_closing.x_start == without.x_start
    assert with_closing.x_end == without.x_end


@pytest.mark.parametrize("spacing", [4, 6, 10, 20, 30])
def test_the_closing_cannot_fuse_adjacent_staff_lines(spacing):
    """A one-pixel-tall kernel spans columns only, so it cannot bridge
    vertically however long it is. Fusing two lines would corrupt every staff
    position on the page.

    Asserted unconditionally on purpose. An earlier version guarded this with
    ``if grids:``, which meant a kernel that fused the lines — and so found no
    staff at all — passed the test silently. A conditional that skips the
    assertion when the code fails is not a test.
    """
    page = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    for step in range(5):
        page[TOP + step * spacing, 100:700] = 0

    grids = detect_staff_grids(page)

    assert len(grids) == 1, f"staff lost at spacing {spacing}"
    assert len(grids[0].lines) == 5
    assert grids[0].line_spacing == pytest.approx(spacing, abs=0.6)


def test_the_closing_kernel_is_one_pixel_tall():
    """The property the test above depends on, asserted directly.

    A kernel taller than one pixel could bridge between staff lines at small
    engraving sizes, which is the one way this fix could do real damage.
    """
    import cv2

    from melodix.geometry import staff as staff_module

    captured: list[tuple[int, int]] = []
    real = cv2.getStructuringElement

    def spy(shape, size, *args, **kwargs):
        captured.append(size)
        return real(shape, size, *args, **kwargs)

    original = staff_module.cv2.getStructuringElement
    staff_module.cv2.getStructuringElement = spy
    try:
        detect_staff_grids(clean_page())
    finally:
        staff_module.cv2.getStructuringElement = original

    assert captured, "no structuring element was built"
    assert all(size[1] == 1 for size in captured), f"a kernel spans rows: {captured}"
