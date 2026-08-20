"""Barline classification, ensemble system grouping, and measure slicing.

The structural layer of Stage 1. It takes the two independent products of the
earlier passes — :class:`~melodix.geometry.staff.StaffGrid` objects from
:mod:`melodix.geometry.staff` and unclassified
:class:`~melodix.geometry.barlines.VerticalSegment` strokes from
:mod:`melodix.geometry.barlines` — and works out what the page actually means:
which staves are played simultaneously, where the measures fall, and which
vertical strokes are barlines rather than note stems.

Three tiers, in order.

**Tier 1, per-staff span check.** A stroke is a barline for a given staff when
it reaches both that staff's top and bottom line, within tolerance. Note stems
fail this: they hang off a notehead and stop short of one end. A single
continuous stroke drawn through an entire system passes for every staff it
crosses, which is what makes the two engraving styles converge here rather than
needing separate handling downstream.

**Tier 2, system grouping and column alignment.** Staves that sound together
are grouped into a :class:`System`, and their barlines are gathered into
:class:`BarlineColumn` objects sharing an x position.

The ordering matters and is easy to get backwards: **column alignment cannot
define a system.** Barlines at the left and right margins align vertically down
the whole page, so grouping staves by shared columns fuses every system on the
page into one. Systems are therefore established first, from vertical spacing
and from strokes that physically span more than one staff, and only then are
columns computed within each system.

**Tier 3, measure slicing.** Consecutive columns bound :class:`Measure`
regions, each addressed by ``(system_index, staff_index, measure_index)``.

Everything here is geometry. Which drum a notehead represents is Stage 3's
problem; this module only says where to look.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from melodix.geometry.barlines import VerticalSegment
from melodix.geometry.staff import StaffGrid

__all__ = [
    "BarlineColumn",
    "Measure",
    "System",
    "SystemGroupingConfig",
    "assign_systems",
    "build_columns",
    "build_systems",
    "classify_barlines",
    "slice_measures",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SystemGroupingConfig:
    """Tunable thresholds for structural analysis.

    Attributes:
        span_tolerance_px: Slack in pixels allowed at each end when testing
            whether a stroke spans a staff. Absorbs the stroke ending a pixel
            short of the line it meets.
        column_tolerance_px: Two barlines belong to the same column when their
            centre columns differ by no more than this.
        system_gap_ratio: Staves separated by a vertical gap no larger than
            this multiple of the taller staff's height belong to the same
            system. The gap is measured from one staff's bottom line to the
            next staff's top line.
        min_measure_width_px: Slices narrower than this are discarded. Filters
            the sliver between a repeat sign's two adjacent rules.
        require_full_span: When true, a stroke must span the staff to count as
            that staff's barline. Turning it off admits partial strokes and is
            only useful for damaged scans.
    """

    span_tolerance_px: float = 3.0
    column_tolerance_px: float = 3.0
    system_gap_ratio: float = 1.75
    min_measure_width_px: int = 12
    require_full_span: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any parameter falls outside its permitted range.
        """
        if self.span_tolerance_px < 0.0:
            raise ValueError(
                f"span_tolerance_px must be non-negative, got {self.span_tolerance_px}"
            )
        if self.column_tolerance_px < 0.0:
            raise ValueError(
                f"column_tolerance_px must be non-negative, got {self.column_tolerance_px}"
            )
        if self.system_gap_ratio <= 0.0:
            raise ValueError(f"system_gap_ratio must be positive, got {self.system_gap_ratio}")
        if self.min_measure_width_px < 1:
            raise ValueError(
                f"min_measure_width_px must be positive, got {self.min_measure_width_px}"
            )


DEFAULT_CONFIG: Final[SystemGroupingConfig] = SystemGroupingConfig()


# --------------------------------------------------------------------------- #
# Data objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Measure:
    """One measure of one staff.

    The addressable unit Stage 3 iterates over. ``x_start`` and ``x_end`` bound
    the region horizontally; the vertical bounds come from the owning staff.

    Attributes:
        system_index: Index of the owning system, top to bottom on the page.
        staff_index: Index of the staff within its system, top to bottom.
        measure_index: Index of the measure within its staff, left to right.
        x_start: Left edge in pixels, inclusive.
        x_end: Right edge in pixels, inclusive.
        y_top: Top line row of the owning staff.
        y_bottom: Bottom line row of the owning staff.
    """

    system_index: int
    staff_index: int
    measure_index: int
    x_start: int
    x_end: int
    y_top: float
    y_bottom: float

    def __post_init__(self) -> None:
        """Validate the measure.

        Raises:
            ValueError: If the horizontal extent is inverted or any index is
                negative.
        """
        if self.x_end < self.x_start:
            raise ValueError(f"x_end must be at least x_start, got {self.x_start}..{self.x_end}")
        if min(self.system_index, self.staff_index, self.measure_index) < 0:
            raise ValueError("indices must be non-negative")

    @property
    def key(self) -> tuple[int, int, int]:
        """The ``(system, staff, measure)`` address of this measure."""
        return (self.system_index, self.staff_index, self.measure_index)

    @property
    def width(self) -> int:
        """Horizontal extent in pixels."""
        return self.x_end - self.x_start + 1

    def contains(self, x: float, y: float) -> bool:
        """Report whether a point falls inside this measure.

        Args:
            x: Column to test.
            y: Row to test.

        Returns:
            ``True`` when the point lies within both bounds. Vertical bounds
            are the staff lines themselves, so a notehead above the top line
            is outside; widen with the staff grid if ledger notes must match.
        """
        return self.x_start <= x <= self.x_end and self.y_top <= y <= self.y_bottom


@dataclass(frozen=True, slots=True)
class BarlineColumn:
    """A vertical alignment of barlines across the staves of one system.

    Attributes:
        x: Mean centre column of the member strokes.
        segments: The strokes forming this column, top to bottom.
        staff_indices: Indices, within the system, of the staves the column
            crosses. A column reaching every staff is a system barline; a
            partial one is a staff-local division.
    """

    x: float
    segments: tuple[VerticalSegment, ...]
    staff_indices: tuple[int, ...]

    @property
    def x_int(self) -> int:
        """Centre column rounded to the nearest pixel."""
        return int(round(self.x))

    def spans_staves(self, count: int) -> bool:
        """Report whether this column crosses every staff of a system.

        Args:
            count: Number of staves in the system.

        Returns:
            ``True`` when the column reaches all of them.
        """
        return len(set(self.staff_indices)) >= count


@dataclass(frozen=True, slots=True)
class System:
    """A group of staves performed simultaneously.

    In a drum ensemble score one system holds one staff per player, so the
    staves of a system share a timeline: measure *n* of staff 0 sounds at the
    same moment as measure *n* of staff 1.

    Attributes:
        index: Position of this system on the page, top to bottom.
        staves: Member staves, top to bottom.
        columns: Barline columns within this system, left to right.
        measures: Measures of every staff, ordered by staff then position.
    """

    index: int
    staves: tuple[StaffGrid, ...]
    columns: tuple[BarlineColumn, ...] = ()
    measures: tuple[Measure, ...] = field(default=())

    def __post_init__(self) -> None:
        """Validate the system.

        Raises:
            ValueError: If the system holds no staves or the index is
                negative.
        """
        if not self.staves:
            raise ValueError("a system must hold at least one staff")
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}")

    @property
    def staff_count(self) -> int:
        """Number of staves, i.e. of simultaneous players."""
        return len(self.staves)

    @property
    def y_top(self) -> float:
        """Top line row of the highest staff."""
        return self.staves[0].top_line_y

    @property
    def y_bottom(self) -> float:
        """Bottom line row of the lowest staff."""
        return self.staves[-1].bottom_line_y

    @property
    def measure_count(self) -> int:
        """Number of measures per staff, or 0 when none were sliced."""
        if not self.measures:
            return 0
        return max(measure.measure_index for measure in self.measures) + 1

    def measures_for_staff(self, staff_index: int) -> tuple[Measure, ...]:
        """Return one staff's measures, left to right.

        Args:
            staff_index: Index of the staff within this system.

        Returns:
            The measures belonging to that staff.

        Raises:
            IndexError: If the staff index is out of range.
        """
        if not 0 <= staff_index < self.staff_count:
            raise IndexError(f"staff_index {staff_index} out of range for {self.staff_count}")
        return tuple(m for m in self.measures if m.staff_index == staff_index)

    def measures_at(self, measure_index: int) -> tuple[Measure, ...]:
        """Return the simultaneous measures across all staves.

        The ensemble view: every player's bar *n*, which sound together.

        Args:
            measure_index: Position of the measure within each staff.

        Returns:
            One measure per staff that has one, ordered by staff.
        """
        return tuple(m for m in self.measures if m.measure_index == measure_index)


# --------------------------------------------------------------------------- #
# Tier 1: per-staff span check
# --------------------------------------------------------------------------- #


def classify_barlines(
    segments: list[VerticalSegment],
    staves: list[StaffGrid],
    config: SystemGroupingConfig = DEFAULT_CONFIG,
) -> dict[int, list[VerticalSegment]]:
    """Identify which strokes are barlines for which staves.

    A stroke qualifies when it reaches both the top and bottom line of the
    staff, within ``span_tolerance_px``. Stems fail because they stop short of
    one end. A continuous stroke drawn through a whole system qualifies for
    every staff it crosses, so both engraving styles produce the same result.

    Args:
        segments: Unclassified strokes from
            :func:`~melodix.geometry.barlines.detect_vertical_segments`.
        staves: Detected staves, in page order.
        config: Span tolerance and whether a full span is required.

    Returns:
        A mapping from index into ``staves`` to that staff's barlines, ordered
        left to right. Staves with no barlines map to an empty list, so every
        index is present.
    """
    result: dict[int, list[VerticalSegment]] = {index: [] for index in range(len(staves))}

    for staff_index, staff in enumerate(staves):
        for segment in segments:
            if config.require_full_span:
                qualifies = segment.spans(
                    staff.top_line_y, staff.bottom_line_y, tolerance=config.span_tolerance_px
                )
            else:
                qualifies = (
                    segment.y_start <= staff.bottom_line_y
                    and segment.y_end >= staff.top_line_y
                )
            if qualifies:
                result[staff_index].append(segment)

    for staff_index in result:
        result[staff_index].sort(key=lambda segment: segment.x)

    return result


# --------------------------------------------------------------------------- #
# Tier 2: system grouping and column alignment
# --------------------------------------------------------------------------- #


def assign_systems(
    staves: list[StaffGrid],
    segments: list[VerticalSegment] | None = None,
    config: SystemGroupingConfig = DEFAULT_CONFIG,
) -> list[tuple[int, ...]]:
    """Group staves into systems.

    Two independent sources of evidence, combined:

    1. **Vertical spacing.** Staves of one system sit closer together than
       consecutive systems do. A gap no larger than ``system_gap_ratio`` times
       the taller staff's height keeps them together.
    2. **Physical spanning strokes.** A single stroke running continuously
       through two staves ties them together regardless of spacing, since no
       engraver draws one through the gap between systems.

    Column alignment is deliberately *not* used. Margin barlines align down the
    entire page, so grouping by shared columns would fuse every system into
    one.

    Args:
        staves: Detected staves, in page order.
        segments: Strokes, used only for the spanning evidence. Optional;
            without them grouping relies on spacing alone.
        config: Gap ratio and span tolerance.

    Returns:
        Groups of indices into ``staves``, each ordered top to bottom, the
        groups themselves ordered by page position.
    """
    if not staves:
        return []

    order = sorted(range(len(staves)), key=lambda index: staves[index].top_line_y)
    parent = list(range(len(staves)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    # Evidence 1: vertical proximity between page-adjacent staves.
    for upper_index, lower_index in zip(order, order[1:], strict=False):
        upper, lower = staves[upper_index], staves[lower_index]
        gap = lower.top_line_y - upper.bottom_line_y
        reference = max(upper.height, lower.height)
        if gap <= reference * config.system_gap_ratio:
            union(upper_index, lower_index)

    # Evidence 2: a single stroke physically crossing two staves.
    for segment in segments or []:
        crossed = [
            index
            for index, staff in enumerate(staves)
            if segment.spans(
                staff.top_line_y, staff.bottom_line_y, tolerance=config.span_tolerance_px
            )
        ]
        for other in crossed[1:]:
            union(crossed[0], other)

    groups: dict[int, list[int]] = {}
    for index in order:
        groups.setdefault(find(index), []).append(index)

    ordered_roots = sorted(groups, key=lambda root: staves[groups[root][0]].top_line_y)
    return [tuple(groups[root]) for root in ordered_roots]


def build_columns(
    staves: list[StaffGrid],
    barlines_by_staff: dict[int, list[VerticalSegment]],
    staff_indices: tuple[int, ...],
    config: SystemGroupingConfig = DEFAULT_CONFIG,
) -> list[BarlineColumn]:
    """Gather one system's barlines into aligned columns.

    Barlines whose centre columns agree to within ``column_tolerance_px`` form
    one column. This is what unifies the two engraving styles: a continuous
    system stroke contributes the same object once per staff, while per-staff
    strokes contribute distinct objects — either way the column records which
    staves it crosses.

    Args:
        staves: All detected staves, in page order.
        barlines_by_staff: Output of :func:`classify_barlines`.
        staff_indices: Indices of the staves in this system, top to bottom.
        config: Column tolerance.

    Returns:
        Columns ordered left to right.
    """
    entries: list[tuple[float, int, VerticalSegment]] = []
    for local_index, staff_index in enumerate(staff_indices):
        for segment in barlines_by_staff.get(staff_index, []):
            entries.append((segment.x, local_index, segment))
    if not entries:
        return []

    entries.sort(key=lambda entry: entry[0])

    columns: list[BarlineColumn] = []
    bucket: list[tuple[float, int, VerticalSegment]] = [entries[0]]

    def flush(group: list[tuple[float, int, VerticalSegment]]) -> BarlineColumn:
        xs = [item[0] for item in group]
        members = sorted(group, key=lambda item: item[2].y_start)
        return BarlineColumn(
            x=float(np.mean(xs)),
            segments=tuple(item[2] for item in members),
            staff_indices=tuple(item[1] for item in members),
        )

    for entry in entries[1:]:
        if entry[0] - bucket[-1][0] <= config.column_tolerance_px:
            bucket.append(entry)
        else:
            columns.append(flush(bucket))
            bucket = [entry]
    columns.append(flush(bucket))

    return columns


# --------------------------------------------------------------------------- #
# Tier 3: measure slicing
# --------------------------------------------------------------------------- #


def slice_measures(
    staff: StaffGrid,
    columns: list[BarlineColumn],
    system_index: int,
    staff_index: int,
    config: SystemGroupingConfig = DEFAULT_CONFIG,
) -> list[Measure]:
    """Cut one staff into measures between consecutive barline columns.

    Boundaries are the column positions plus the staff's own horizontal
    extent. A barline sitting at the staff edge does not create an empty
    leading or trailing measure, because slices narrower than
    ``min_measure_width_px`` are dropped — which also removes the sliver
    between the two rules of a repeat sign.

    Args:
        staff: The staff to slice.
        columns: Barline columns of the owning system, left to right.
        system_index: Index of the owning system.
        staff_index: Index of this staff within its system.
        config: Minimum measure width.

    Returns:
        Measures ordered left to right, renumbered from zero after filtering.
    """
    boundaries = sorted(
        {staff.x_start, staff.x_end, *(column.x_int for column in columns)}
    )
    boundaries = [b for b in boundaries if staff.x_start <= b <= staff.x_end]
    if len(boundaries) < 2:
        return []

    measures: list[Measure] = []
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        if right - left + 1 < config.min_measure_width_px:
            continue
        measures.append(
            Measure(
                system_index=system_index,
                staff_index=staff_index,
                measure_index=len(measures),
                x_start=int(left),
                x_end=int(right),
                y_top=staff.top_line_y,
                y_bottom=staff.bottom_line_y,
            )
        )
    return measures


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_systems(
    staves: list[StaffGrid],
    segments: list[VerticalSegment],
    config: SystemGroupingConfig = DEFAULT_CONFIG,
) -> list[System]:
    """Assemble the full structural description of a page.

    Runs all three tiers: classify strokes against staves, group staves into
    systems and their barlines into columns, then slice measures.

    Args:
        staves: Detected staves from
            :func:`~melodix.geometry.staff.detect_staff_grids`.
        segments: Unclassified strokes from
            :func:`~melodix.geometry.barlines.detect_vertical_segments`.
        config: Structural thresholds.

    Returns:
        Systems ordered top to bottom, each carrying its staves, barline
        columns and measures.
    """
    if not staves:
        return []

    barlines_by_staff = classify_barlines(segments, staves, config)
    groups = assign_systems(staves, segments, config)

    systems: list[System] = []
    for system_index, staff_indices in enumerate(groups):
        columns = build_columns(staves, barlines_by_staff, staff_indices, config)

        measures: list[Measure] = []
        for local_index, staff_index in enumerate(staff_indices):
            measures.extend(
                slice_measures(
                    staves[staff_index], columns, system_index, local_index, config
                )
            )

        systems.append(
            System(
                index=system_index,
                staves=tuple(staves[index] for index in staff_indices),
                columns=tuple(columns),
                measures=tuple(measures),
            )
        )

    return systems