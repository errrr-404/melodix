"""Staff-line detection and the staff coordinate grid.

This is the foundation of Pipeline Stage 1. Everything downstream — barline
segmentation, system grouping, and above all the Stage 3 translation of YOLO
bounding-box centroids into General MIDI percussion numbers — is expressed in
terms of the :class:`StaffGrid` produced here.

Coordinate conventions
----------------------
Pixel space is **y-down**, matching OpenCV and every image format we ingest.

Staff space is measured in *positions*: integer half-space steps, with the
**bottom line at position 0** and each step moving one half-space **upward**:

    position 8  ────────────  top line
    position 7                space 4
    position 6  ────────────  line 4
    position 5                space 3    <- Acoustic Snare [38]
    position 4  ────────────  line 3
    position 3                space 2
    position 2  ────────────  line 2
    position 1                space 1
    position 0  ────────────  bottom line

Even positions are lines, odd positions are spaces. Positions below the staff
are negative and positions above it exceed 8, so ledger-line notes need no
special handling — a note one ledger line above the staff is simply position
10.

Skew
----
Staff isolation relies on a long horizontal morphological kernel, which
degrades once a page is rotated beyond roughly one degree. Deskewing is
deliberately *not* performed here; it belongs to ``melodix.geometry.deskew``
and runs before detection. Keeping the two separate means each is provable on
its own.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

__all__ = [
    "LINES_PER_STAFF",
    "STAFF_TOP_POSITION",
    "BinaryImage",
    "GrayImage",
    "LineBand",
    "StaffDetectionConfig",
    "StaffGrid",
    "StaffLine",
    "binarize",
    "detect_staff_grids",
    "extract_line_bands",
    "group_bands_into_grids",
    "isolate_horizontal_runs",
    "to_grayscale",
]

#: A five-line staff. Percussion clefs use five lines exactly.
LINES_PER_STAFF: Final[int] = 5

#: Staff position of the top line, given a bottom line at position 0.
STAFF_TOP_POSITION: Final[int] = 2 * (LINES_PER_STAFF - 1)

GrayImage = npt.NDArray[np.uint8]
"""A 2-D ``uint8`` array. 0 is black, 255 is white."""

BinaryImage = npt.NDArray[np.uint8]
"""A 2-D ``uint8`` array where 255 marks ink and 0 marks background."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StaffDetectionConfig:
    """Tunable thresholds for :func:`detect_staff_grids`.

    Defaults target 150-300 DPI scans of printed drum notation. Every value is
    a ratio rather than an absolute pixel count so that behaviour is stable
    across scan resolutions.

    Attributes:
        horizontal_kernel_ratio: Length of the horizontal opening kernel as a
            fraction of image width. Structures shorter than this are erased,
            which is what separates staff lines from stems, beams, noteheads
            and text.
        row_threshold_ratio: A pixel row belongs to a staff line when its ink
            count reaches this fraction of the densest row on the page.
        spacing_tolerance: Maximum relative deviation of any one inter-line gap
            from the median gap of a candidate staff. Absorbs scanner warp;
            tighten it to reject false groupings on dense pages.
        max_thickness_ratio: A staff line thicker than this fraction of the
            line spacing is treated as a merged blob and rejects the candidate.
        min_line_spacing: Absolute floor, in pixels, on the gap between
            adjacent staff lines. Guards against noise rows grouping into a
            degenerate "staff" a few pixels tall.
    """

    horizontal_kernel_ratio: float = 0.35
    row_threshold_ratio: float = 0.40
    spacing_tolerance: float = 0.25
    max_thickness_ratio: float = 0.60
    min_line_spacing: float = 3.0

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any ratio falls outside its permitted range.
        """
        if not 0.0 < self.horizontal_kernel_ratio <= 1.0:
            raise ValueError(
                f"horizontal_kernel_ratio must be in (0, 1], got {self.horizontal_kernel_ratio}"
            )
        if not 0.0 < self.row_threshold_ratio <= 1.0:
            raise ValueError(
                f"row_threshold_ratio must be in (0, 1], got {self.row_threshold_ratio}"
            )
        if not 0.0 <= self.spacing_tolerance < 1.0:
            raise ValueError(
                f"spacing_tolerance must be in [0, 1), got {self.spacing_tolerance}"
            )
        if not 0.0 < self.max_thickness_ratio <= 1.0:
            raise ValueError(
                f"max_thickness_ratio must be in (0, 1], got {self.max_thickness_ratio}"
            )
        if self.min_line_spacing <= 0.0:
            raise ValueError(f"min_line_spacing must be positive, got {self.min_line_spacing}")


DEFAULT_CONFIG: Final[StaffDetectionConfig] = StaffDetectionConfig()


# --------------------------------------------------------------------------- #
# Data objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LineBand:
    """A contiguous run of ink rows that survived horizontal isolation.

    An intermediate product of detection: a band is a *candidate* staff line
    that has not yet been grouped into a staff.

    Attributes:
        y: Ink-weighted centroid row of the band, in pixels.
        x_start: Leftmost ink column, inclusive.
        x_end: Rightmost ink column, inclusive.
        thickness: Height of the band in pixels.
    """

    y: float
    x_start: int
    x_end: int
    thickness: int

    @property
    def width(self) -> int:
        """Horizontal extent of the band in pixels."""
        return self.x_end - self.x_start + 1


@dataclass(frozen=True, slots=True)
class StaffLine:
    """One of the five lines of a staff.

    Attributes:
        y: Sub-pixel row of the line's centre.
        x_start: Leftmost ink column, inclusive.
        x_end: Rightmost ink column, inclusive.
        thickness: Measured stroke height in pixels.
    """

    y: float
    x_start: int
    x_end: int
    thickness: int

    @property
    def width(self) -> int:
        """Horizontal extent of the line in pixels."""
        return self.x_end - self.x_start + 1

    @classmethod
    def from_band(cls, band: LineBand) -> StaffLine:
        """Promote a detected band to a confirmed staff line."""
        return cls(y=band.y, x_start=band.x_start, x_end=band.x_end, thickness=band.thickness)


@dataclass(frozen=True, slots=True)
class StaffGrid:
    """A single five-line staff and its pixel/staff coordinate mapping.

    The grid is the contract between Stage 1 and Stage 3: hand it the centroid
    of a YOLO detection and it answers which line or space that symbol sits on.

    Position conversions interpolate between the *measured* line rows rather
    than assuming a uniform spacing, so mild scanner warp does not accumulate
    error toward the top of the staff. Outside the staff, the model extends
    linearly using the outermost measured gap.

    Attributes:
        lines: Exactly five lines ordered top to bottom (ascending ``y``).
        index: Zero-based ordinal of this staff on its page, top to bottom.
    """

    lines: tuple[StaffLine, ...]
    index: int = 0

    def __post_init__(self) -> None:
        """Validate structural invariants.

        Raises:
            ValueError: If the staff does not hold exactly five lines ordered
                strictly top to bottom, or if ``index`` is negative.
        """
        if len(self.lines) != LINES_PER_STAFF:
            raise ValueError(
                f"a staff needs exactly {LINES_PER_STAFF} lines, got {len(self.lines)}"
            )
        ys = [line.y for line in self.lines]
        if any(later <= earlier for earlier, later in pairwise(ys)):
            raise ValueError(f"lines must be ordered top to bottom by strictly increasing y: {ys}")
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}")

    # -- measurements ------------------------------------------------------ #

    @property
    def line_ys(self) -> tuple[float, ...]:
        """Row of each line, top to bottom."""
        return tuple(line.y for line in self.lines)

    @property
    def top_line_y(self) -> float:
        """Row of the top line (staff position 8)."""
        return self.lines[0].y

    @property
    def bottom_line_y(self) -> float:
        """Row of the bottom line (staff position 0)."""
        return self.lines[-1].y

    @property
    def height(self) -> float:
        """Distance in pixels from the top line to the bottom line."""
        return self.bottom_line_y - self.top_line_y

    @property
    def line_spacing(self) -> float:
        """Median distance in pixels between adjacent staff lines.

        This is the natural unit of measure for the whole pipeline: notehead
        size, stem length, and ledger-line reach are all conventionally
        expressed as multiples of it.
        """
        gaps = np.diff(np.asarray(self.line_ys, dtype=np.float64))
        return float(np.median(gaps))

    @property
    def step_height(self) -> float:
        """Pixel height of one staff position, i.e. half the line spacing."""
        return self.line_spacing / 2.0

    @property
    def center_y(self) -> float:
        """Row of the middle line (staff position 4)."""
        return self.lines[LINES_PER_STAFF // 2].y

    @property
    def x_start(self) -> int:
        """Leftmost ink column across all five lines."""
        return min(line.x_start for line in self.lines)

    @property
    def x_end(self) -> int:
        """Rightmost ink column across all five lines."""
        return max(line.x_end for line in self.lines)

    @property
    def width(self) -> int:
        """Horizontal extent of the staff in pixels."""
        return self.x_end - self.x_start + 1

    # -- coordinate conversion --------------------------------------------- #

    @property
    def _ascending(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Line positions ascending, paired with their rows (bottom to top)."""
        positions = np.arange(0, 2 * LINES_PER_STAFF, 2, dtype=np.float64)
        ys = np.asarray(self.line_ys[::-1], dtype=np.float64)
        return positions, ys

    def position_to_y(self, position: float) -> float:
        """Convert a staff position to a pixel row.

        Args:
            position: Staff position in half-space steps. 0 is the bottom
                line, 8 the top line; values outside that range address ledger
                territory.

        Returns:
            The pixel row, possibly fractional and possibly outside the image.
        """
        positions, ys = self._ascending
        if 0.0 <= position <= STAFF_TOP_POSITION:
            return float(np.interp(position, positions, ys))
        if position < 0.0:
            slope = (ys[1] - ys[0]) / 2.0
            return float(ys[0] + position * slope)
        slope = (ys[-1] - ys[-2]) / 2.0
        return float(ys[-1] + (position - STAFF_TOP_POSITION) * slope)

    def y_to_position(self, y: float) -> float:
        """Convert a pixel row to a fractional staff position.

        Args:
            y: Pixel row, typically the vertical centroid of a detected
                notehead.

        Returns:
            The staff position; 5.0 means "centred in space 3", 5.4 means
            "slightly above centre in space 3".
        """
        positions, ys = self._ascending
        descending_y = ys[::-1]  # top to bottom, i.e. ascending y
        descending_positions = positions[::-1]
        if descending_y[0] <= y <= descending_y[-1]:
            return float(np.interp(y, descending_y, descending_positions))
        if y < descending_y[0]:
            slope = 2.0 / (ys[-1] - ys[-2])
            return float(STAFF_TOP_POSITION + (y - descending_y[0]) * slope)
        slope = 2.0 / (ys[1] - ys[0])
        return float((y - descending_y[-1]) * slope)

    def nearest_position(self, y: float) -> int:
        """Snap a pixel row to the closest integer staff position."""
        return int(round(self.y_to_position(y)))

    def snap(self, y: float, tolerance: float = 0.4) -> int | None:
        """Snap a pixel row to a staff position, or reject it as ambiguous.

        Stage 3 uses this to refuse to guess: a centroid landing halfway
        between a line and a space is more likely a detection error than a
        note, and silently rounding it would inject a wrong drum voice into
        the MIDI.

        Args:
            y: Pixel row to snap.
            tolerance: Maximum distance in staff positions between ``y`` and
                the nearest integer position for the snap to be accepted.

        Returns:
            The integer staff position, or ``None`` if ``y`` is further than
            ``tolerance`` from every integer position.

        Raises:
            ValueError: If ``tolerance`` is not in ``(0, 0.5]``.
        """
        if not 0.0 < tolerance <= 0.5:
            raise ValueError(f"tolerance must be in (0, 0.5], got {tolerance}")
        exact = self.y_to_position(y)
        snapped = round(exact)
        if abs(exact - snapped) > tolerance:
            return None
        return int(snapped)

    def contains_y(self, y: float, ledger_steps: int = 0) -> bool:
        """Report whether a row falls within the staff's vertical reach.

        Args:
            y: Pixel row to test.
            ledger_steps: Extra staff positions of tolerance above and below,
                for notes on ledger lines. Four steps is two ledger lines.

        Returns:
            ``True`` if the row lies inside the extended band.
        """
        position = self.y_to_position(y)
        return -ledger_steps <= position <= STAFF_TOP_POSITION + ledger_steps

    @staticmethod
    def is_line_position(position: int) -> bool:
        """Report whether an integer position names a line rather than a space."""
        return position % 2 == 0

    def with_index(self, index: int) -> StaffGrid:
        """Return a copy of this grid carrying a new page ordinal."""
        return replace(self, index=index)


# --------------------------------------------------------------------------- #
# Detection stages
# --------------------------------------------------------------------------- #


def to_grayscale(image: npt.NDArray[np.uint8]) -> GrayImage:
    """Reduce an image to a single 8-bit channel.

    Args:
        image: A 2-D grayscale array, or a 3-D array with 1, 3 (BGR) or 4
            (BGRA) channels.

    Returns:
        A 2-D ``uint8`` array.

    Raises:
        ValueError: If the array is not a recognised image shape.
    """
    if image.ndim == 2:
        return np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return np.ascontiguousarray(image[:, :, 0], dtype=np.uint8)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # type: ignore[no-any-return]
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)  # type: ignore[no-any-return]
    raise ValueError(f"unsupported image shape {image.shape}; expected 2-D or H*W*{{1,3,4}}")


def binarize(gray: GrayImage) -> BinaryImage:
    """Threshold a grayscale page so that ink reads as 255.

    Uses Otsu's method, which suits the bimodal histogram of printed sheet
    music. Photographs with uneven lighting should be flat-fielded upstream.

    Args:
        gray: 2-D ``uint8`` image.

    Returns:
        A 2-D ``uint8`` mask where 255 marks ink.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary  # type: ignore[no-any-return]


def isolate_horizontal_runs(
    binary: BinaryImage,
    config: StaffDetectionConfig = DEFAULT_CONFIG,
) -> BinaryImage:
    """Erase everything that is not a long horizontal stroke.

    A morphological opening with a wide, one-pixel-tall kernel keeps only ink
    that survives erosion by that kernel — staff lines and the occasional long
    beam or underline. Noteheads, stems, flags, text and dynamics vanish.

    Args:
        binary: Ink mask from :func:`binarize`.
        config: Detection thresholds.

    Returns:
        A mask containing only long horizontal structures.
    """
    width = binary.shape[1]
    kernel_length = max(3, int(round(width * config.horizontal_kernel_ratio)))
    # An even-width kernel anchors off-centre, so erosion and dilation are not
    # symmetric and the opening shifts every line one pixel right. Force odd.
    kernel_length |= 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # type: ignore[no-any-return]


def extract_line_bands(
    horizontal: BinaryImage,
    config: StaffDetectionConfig = DEFAULT_CONFIG,
) -> list[LineBand]:
    """Collapse a horizontal-run mask into candidate staff lines.

    Sums ink per row, thresholds relative to the densest row, then merges
    contiguous surviving rows into bands. The reported ``y`` is the
    ink-weighted centroid, which recovers sub-pixel accuracy on strokes two or
    three pixels thick.

    Args:
        horizontal: Mask from :func:`isolate_horizontal_runs`.
        config: Detection thresholds.

    Returns:
        Candidate bands ordered top to bottom.
    """
    profile = (horizontal > 0).sum(axis=1).astype(np.float64)
    peak = float(profile.max()) if profile.size else 0.0
    if peak <= 0.0:
        return []

    threshold = max(1.0, peak * config.row_threshold_ratio)
    active = profile >= threshold

    bands: list[LineBand] = []
    for first, last in _contiguous_runs(active):
        rows = np.arange(first, last + 1, dtype=np.float64)
        weights = profile[first : last + 1]
        centroid = float((rows * weights).sum() / weights.sum())

        columns = np.flatnonzero(horizontal[first : last + 1].any(axis=0))
        if columns.size == 0:  # pragma: no cover - implied by the row threshold
            continue

        bands.append(
            LineBand(
                y=centroid,
                x_start=int(columns[0]),
                x_end=int(columns[-1]),
                thickness=last - first + 1,
            )
        )
    return bands


def group_bands_into_grids(
    bands: list[LineBand],
    config: StaffDetectionConfig = DEFAULT_CONFIG,
) -> list[StaffGrid]:
    """Assemble candidate bands into five-line staves.

    Walks top to bottom taking windows of five. A window is accepted when its
    four gaps agree to within ``spacing_tolerance`` and no line is implausibly
    thick relative to that spacing. Accepting a window consumes all five bands;
    rejecting one advances by a single band, so stray horizontal rules between
    staves cost an offset rather than a missed staff.

    Args:
        bands: Candidate bands in any order.
        config: Detection thresholds.

    Returns:
        Staves ordered top to bottom, with ``index`` assigned in that order.
    """
    ordered = sorted(bands, key=lambda band: band.y)
    grids: list[StaffGrid] = []

    cursor = 0
    while cursor + LINES_PER_STAFF <= len(ordered):
        window = ordered[cursor : cursor + LINES_PER_STAFF]
        if _is_plausible_staff(window, config):
            grids.append(
                StaffGrid(
                    lines=tuple(StaffLine.from_band(band) for band in window),
                    index=len(grids),
                )
            )
            cursor += LINES_PER_STAFF
        else:
            cursor += 1

    return grids


def detect_staff_grids(
    image: npt.NDArray[np.uint8],
    config: StaffDetectionConfig = DEFAULT_CONFIG,
) -> list[StaffGrid]:
    """Detect every five-line staff on a page.

    The full Stage 1 staff pass: grayscale, threshold, isolate horizontal
    strokes, reduce to candidate lines, group into staves.

    Args:
        image: Page as a grayscale or colour ``uint8`` array. Deskew it first
            if the scan is visibly rotated.
        config: Detection thresholds.

    Returns:
        Staves ordered top to bottom. Empty if the page holds none.

    Raises:
        ValueError: If the image is empty or not a recognised shape.
    """
    if image.size == 0:
        raise ValueError("cannot detect staves in an empty image")

    gray = to_grayscale(image)
    binary = binarize(gray)
    horizontal = isolate_horizontal_runs(binary, config)
    bands = extract_line_bands(horizontal, config)
    return group_bands_into_grids(bands, config)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _contiguous_runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return inclusive ``(first, last)`` index pairs for each run of ``True``."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _is_plausible_staff(window: list[LineBand], config: StaffDetectionConfig) -> bool:
    """Judge whether five consecutive bands form a real staff."""
    gaps = [later.y - earlier.y for earlier, later in pairwise(window)]
    spacing = float(np.median(gaps))

    if spacing < config.min_line_spacing:
        return False
    if any(abs(gap - spacing) > config.spacing_tolerance * spacing for gap in gaps):
        return False
    return all(band.thickness <= config.max_thickness_ratio * spacing for band in window)