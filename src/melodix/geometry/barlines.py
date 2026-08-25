"""Vertical stroke detection.

Finds every tall vertical stroke on a page and reports its exact geometry. It
does **not** decide what those strokes mean.

That restraint is deliberate, and it is not merely a layering preference: a
detector with no access to the staff grid genuinely cannot tell a barline from
a note stem. Both are thin vertical strokes; a barline spans the staff's four
spaces, a stem typically runs three to three and a half, and beamed sixteenth
runs routinely produce stems longer than a barline. Any height threshold that
excluded stems would also discard barlines on the next system engraved a size
smaller. The discriminating evidence — does this stroke start on the top line
and end on the bottom line of a known staff, and does it continue across the
gap into the staff below — lives in :class:`~melodix.geometry.staff.StaffGrid`
objects that this module never sees.

So :func:`detect_vertical_segments` returns barlines, stems, bracket spines and
the vertical edges of any box on the page, all undifferentiated.
:mod:`melodix.geometry.systems` consumes them alongside the detected staves and
performs the classification. Callers wanting only barlines should go through
that module, not this one.

Coordinates follow the rest of Stage 1: pixel space is y-down, ``y_start`` is
the topmost ink row and ``y_end`` the bottommost, both inclusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

from melodix.geometry.staff import BinaryImage, binarize, to_grayscale

__all__ = [
    "BarlineDetectionConfig",
    "VerticalSegment",
    "detect_vertical_segments",
    "extract_vertical_segments",
    "isolate_vertical_runs",
    "merge_collinear",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BarlineDetectionConfig:
    """Tunable thresholds for :func:`detect_vertical_segments`.

    Three ways of setting the minimum stroke height are supported, in
    descending priority: an explicit pixel count, a multiple of the staff line
    spacing, or a fraction of the image height. Prefer the staff-spacing route
    whenever Stage 1 has already run — it is the only one that adapts to a page
    carrying two differently sized ensembles.

    Attributes:
        vertical_kernel_ratio: Height of the vertical opening kernel as a
            fraction of the resolved minimum height. Below 1.0 so that a
            stroke slightly shorter than the threshold still survives the
            opening and can be rejected explicitly, rather than being erased
            and silently disappearing.
        min_height_px: Absolute minimum stroke height. Overrides everything
            else when set.
        min_height_staff_spans: Minimum height as a multiple of staff line
            spacing, used when a spacing hint is supplied.
        min_height_ratio: Minimum height as a fraction of image height, used
            when no spacing hint is available.
        max_thickness_px: Strokes wider than this are treated as blobs, not
            rules.
        min_aspect_ratio: Minimum height-to-width ratio. Rejects square-ish
            components that clear the height bar by being large overall.
        merge_x_tolerance_px: Two fragments count as collinear when their
            centre columns differ by no more than this.
        merge_max_gap_px: Largest vertical gap bridged when merging fragments.
            Keep this well below the gap between staves, or a barline on one
            staff will fuse with the barline beneath it.
        gap_closing_ratio: Length of the gap-bridging closing kernel, as a
            fraction of image height. See :func:`isolate_vertical_runs`. Set to
            0 to disable the closing.
    """

    vertical_kernel_ratio: float = 0.60
    min_height_px: int | None = None
    min_height_staff_spans: float = 3.0
    min_height_ratio: float = 0.02
    max_thickness_px: int = 20
    min_aspect_ratio: float = 3.0
    merge_x_tolerance_px: float = 2.0
    merge_max_gap_px: int = 3
    gap_closing_ratio: float = 0.0023

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any parameter falls outside its permitted range.
        """
        if not 0.0 < self.vertical_kernel_ratio <= 1.0:
            raise ValueError(
                f"vertical_kernel_ratio must be in (0, 1], got {self.vertical_kernel_ratio}"
            )
        if self.min_height_px is not None and self.min_height_px < 1:
            raise ValueError(f"min_height_px must be positive, got {self.min_height_px}")
        if self.min_height_staff_spans <= 0.0:
            raise ValueError(
                f"min_height_staff_spans must be positive, got {self.min_height_staff_spans}"
            )
        if not 0.0 < self.min_height_ratio <= 1.0:
            raise ValueError(f"min_height_ratio must be in (0, 1], got {self.min_height_ratio}")
        if self.max_thickness_px < 1:
            raise ValueError(f"max_thickness_px must be positive, got {self.max_thickness_px}")
        if self.min_aspect_ratio <= 0.0:
            raise ValueError(f"min_aspect_ratio must be positive, got {self.min_aspect_ratio}")
        if self.merge_x_tolerance_px < 0.0:
            raise ValueError(
                f"merge_x_tolerance_px must be non-negative, got {self.merge_x_tolerance_px}"
            )
        if self.merge_max_gap_px < 0:
            raise ValueError(
                f"merge_max_gap_px must be non-negative, got {self.merge_max_gap_px}"
            )
        if not 0.0 <= self.gap_closing_ratio < 1.0:
            raise ValueError(
                f"gap_closing_ratio must be in [0, 1), got {self.gap_closing_ratio}"
            )


DEFAULT_CONFIG: Final[BarlineDetectionConfig] = BarlineDetectionConfig()


# --------------------------------------------------------------------------- #
# Data object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class VerticalSegment:
    """A single tall vertical stroke, unclassified.

    Attributes:
        x: Ink-weighted centre column, sub-pixel.
        y_start: Topmost ink row, inclusive.
        y_end: Bottommost ink row, inclusive.
        thickness: Stroke width in pixels.
    """

    x: float
    y_start: int
    y_end: int
    thickness: int

    def __post_init__(self) -> None:
        """Validate the segment.

        Raises:
            ValueError: If the vertical extent is inverted or the thickness is
                not positive.
        """
        if self.y_end < self.y_start:
            raise ValueError(
                f"y_end must be at least y_start, got {self.y_start}..{self.y_end}"
            )
        if self.thickness < 1:
            raise ValueError(f"thickness must be positive, got {self.thickness}")

    @property
    def height(self) -> int:
        """Vertical extent in pixels, inclusive of both endpoints."""
        return self.y_end - self.y_start + 1

    @property
    def y_center(self) -> float:
        """Midpoint row of the stroke."""
        return (self.y_start + self.y_end) / 2.0

    @property
    def aspect_ratio(self) -> float:
        """Height divided by thickness."""
        return self.height / self.thickness

    def spans(self, y_top: float, y_bottom: float, tolerance: float = 0.0) -> bool:
        """Report whether this stroke covers a vertical range.

        The predicate Phase 1.4 uses to ask "does this stroke run the full
        height of that staff?".

        Args:
            y_top: Upper row of the range to cover.
            y_bottom: Lower row of the range to cover.
            tolerance: Slack in pixels permitted at each end.

        Returns:
            ``True`` when the stroke reaches both ends of the range.

        Raises:
            ValueError: If ``y_bottom`` is above ``y_top``.
        """
        if y_bottom < y_top:
            raise ValueError(f"y_bottom must be at least y_top, got {y_top}..{y_bottom}")
        return self.y_start <= y_top + tolerance and self.y_end >= y_bottom - tolerance

    def overlaps(self, other: VerticalSegment) -> bool:
        """Report whether two strokes share any rows."""
        return self.y_start <= other.y_end and other.y_start <= self.y_end


# --------------------------------------------------------------------------- #
# Detection stages
# --------------------------------------------------------------------------- #


def isolate_vertical_runs(
    binary: BinaryImage,
    kernel_height: int,
    config: BarlineDetectionConfig = DEFAULT_CONFIG,
) -> BinaryImage:
    """Erase everything that is not a tall vertical stroke.

    The mirror of
    :func:`~melodix.geometry.staff.isolate_horizontal_runs`: a morphological
    opening with a tall, one-pixel-wide kernel keeps only ink that survives
    erosion by it. Staff lines, beams, noteheads and text vanish.

    Why a closing runs first
    ------------------------
    Being a deliberate mirror of the horizontal pass, this inherited its
    fragility by construction. An opening needs an **unbroken** run at least as
    long as its kernel, so interruptions that leave no fragment that long erase
    the stroke entirely.

    The margin is thinner here than for a staff line. The vertical kernel is
    ``vertical_kernel_ratio`` (0.60) of the minimum stroke height, and a barline
    is only a little taller than that minimum, so a barline can be lost to a
    *single* interruption near its middle where a page-width staff line needs
    two.

    The consequence is worse here than for a staff line, because it is silent
    and it propagates. A dropped barline removes a measure boundary, so two
    measures merge into one and **every subsequent measure index on that staff
    shifts by one**. Stage 3 addresses notes by
    ``(system, staff, measure)``, so a page that loses one barline near the top
    produces a plausible, complete, wrongly-timed transcription rather than an
    error.

    The justification is the mechanism, not any particular noise model: one
    interrupted pixel destroys an entire stroke, and real pages interrupt
    strokes for many reasons — fold creases, staple holes, faint toner, pencil
    marks, JPEG ringing.

    A one-pixel-wide closing kernel spans rows only, so it cannot pull a
    neighbouring stem sideways into a barline. It can only bridge ink already
    sharing a column.

    Args:
        binary: Ink mask from :func:`~melodix.geometry.staff.binarize`.
        kernel_height: Height of the opening kernel in pixels. Forced odd, so
            that the anchor sits at the true centre.
        config: Detection thresholds, for the gap-closing length.

    Returns:
        A mask containing only tall vertical structures.

    Raises:
        ValueError: If ``kernel_height`` is less than 1.
    """
    if kernel_height < 1:
        raise ValueError(f"kernel_height must be positive, got {kernel_height}")

    if config.gap_closing_ratio > 0.0:
        # Same derivation as the horizontal pass: a closing of length k bridges
        # a gap of up to k-1 px, and the ratio targets 0.5 mm of page, the
        # scale of a speck or a toner dropout. Measured against image height,
        # since that is the axis the gap runs along.
        gap = max(1, int(round(binary.shape[0] * config.gap_closing_ratio)))
        closing_height = max(3, gap + 1) | 1
        closing = cv2.getStructuringElement(cv2.MORPH_RECT, (1, closing_height))
        binary = np.ascontiguousarray(
            cv2.morphologyEx(binary, cv2.MORPH_CLOSE, closing), dtype=np.uint8
        )

    # An even-height kernel anchors off-centre, so erosion and dilation are not
    # symmetric and the opening shifts every stroke one pixel down. Force odd,
    # exactly as the horizontal pass does.
    height = max(3, kernel_height) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # type: ignore[no-any-return]


def extract_vertical_segments(
    vertical: BinaryImage,
    min_height: int,
    config: BarlineDetectionConfig = DEFAULT_CONFIG,
) -> list[VerticalSegment]:
    """Reduce a vertical-run mask to discrete strokes.

    Connected components rather than a column projection: two barlines at the
    same column on different staves must stay separate, and a projection would
    fuse them into one impossible stroke spanning the gap between systems.

    Args:
        vertical: Mask from :func:`isolate_vertical_runs`.
        min_height: Minimum stroke height in pixels.
        config: Thickness and aspect thresholds.

    Returns:
        Surviving strokes, ordered left to right then top to bottom.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(vertical, connectivity=8)

    segments: list[VerticalSegment] = []
    for label in range(1, count):  # 0 is the background
        top = int(stats[label, cv2.CC_STAT_TOP])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        thickness = int(stats[label, cv2.CC_STAT_WIDTH])

        if height < min_height:
            continue
        if thickness > config.max_thickness_px:
            continue
        if height / thickness < config.min_aspect_ratio:
            continue

        segments.append(
            VerticalSegment(
                x=float(centroids[label][0]),
                y_start=top,
                y_end=top + height - 1,
                thickness=thickness,
            )
        )

    return _sorted(segments)


def merge_collinear(
    segments: list[VerticalSegment],
    config: BarlineDetectionConfig = DEFAULT_CONFIG,
) -> list[VerticalSegment]:
    """Rejoin stroke fragments split by scan noise or a faint print.

    Only fragments sharing a column to within ``merge_x_tolerance_px`` and
    separated by at most ``merge_max_gap_px`` rows are merged, so a barline on
    one staff never fuses with the barline on the staff below.

    Args:
        segments: Strokes to consider.
        config: Merge tolerances.

    Returns:
        Merged strokes, ordered left to right then top to bottom.
    """
    if not segments:
        return []

    merged: list[VerticalSegment] = []
    for segment in _sorted(segments):
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and abs(segment.x - previous.x) <= config.merge_x_tolerance_px
            and segment.y_start - previous.y_end - 1 <= config.merge_max_gap_px
            and segment.y_start >= previous.y_start
        ):
            total = previous.height + segment.height
            merged[-1] = VerticalSegment(
                x=(previous.x * previous.height + segment.x * segment.height) / total,
                y_start=previous.y_start,
                y_end=max(previous.y_end, segment.y_end),
                thickness=max(previous.thickness, segment.thickness),
            )
        else:
            merged.append(segment)

    return merged


def detect_vertical_segments(
    image: npt.NDArray[np.uint8],
    config: BarlineDetectionConfig = DEFAULT_CONFIG,
    staff_spacing: float | None = None,
) -> list[VerticalSegment]:
    """Detect every tall vertical stroke on a page.

    Returns barlines, stems, brackets and box edges alike — see the module
    docstring for why separating them requires the staff grid and therefore
    belongs to :mod:`melodix.geometry.systems`.

    Args:
        image: Page as a grayscale or colour ``uint8`` array. Deskew it first;
            a tilted page defeats the vertical kernel exactly as it defeats the
            horizontal one.
        config: Detection thresholds.
        staff_spacing: Line spacing of the detected staves, if Stage 1 has
            already run. Strongly preferred over the image-height fallback,
            since it adapts to the engraving size actually on the page.

    Returns:
        Strokes ordered left to right then top to bottom. Empty if the page
        holds none.

    Raises:
        ValueError: If the image is empty or not a recognised shape.
    """
    if image.size == 0:
        raise ValueError("cannot detect vertical segments in an empty image")

    gray = to_grayscale(image)
    binary = binarize(gray)

    min_height = _resolve_min_height(gray.shape[0], config, staff_spacing)
    kernel_height = max(3, int(round(min_height * config.vertical_kernel_ratio)))

    vertical = isolate_vertical_runs(binary, kernel_height, config)
    segments = extract_vertical_segments(vertical, min_height, config)
    return merge_collinear(segments, config)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _resolve_min_height(
    image_height: int,
    config: BarlineDetectionConfig,
    staff_spacing: float | None,
) -> int:
    """Resolve the minimum stroke height from the configured sources."""
    if config.min_height_px is not None:
        return config.min_height_px
    if staff_spacing is not None:
        if staff_spacing <= 0.0:
            raise ValueError(f"staff_spacing must be positive, got {staff_spacing}")
        return max(3, int(round(staff_spacing * config.min_height_staff_spans)))
    return max(3, int(round(image_height * config.min_height_ratio)))


def _sorted(segments: list[VerticalSegment]) -> list[VerticalSegment]:
    """Order strokes left to right, then top to bottom."""
    return sorted(segments, key=lambda segment: (segment.x, segment.y_start))
