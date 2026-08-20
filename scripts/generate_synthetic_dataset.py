"""Generate a synthetic drum-notation dataset with exact ground truth.

Bootstrap half of the hybrid strategy: pretrain on synthetic pages where every
box is known by construction, then fine-tune on a much smaller corpus of real
scans. The expensive part of an OMR dataset is annotation, and this sidesteps
it entirely — the generator places each symbol, so it already knows where the
box goes. No annotator, no LilyPond bounding-box archaeology, no drift between
what was drawn and what was labelled.

Ground truth is exact, realism is not
------------------------------------
Symbols here are drawn procedurally with OpenCV, not engraved. They are the
right shapes in the right places at plausible sizes, and the augmentation pass
puts them through the same rotation Stage 1 corrects for, plus blur, noise and
ink-weight variation. That is enough to teach a detector what a cross notehead
is and roughly where to look for one. It is **not** enough on its own: a model
trained only on this will overfit to the synthetic glyph shapes. The fine-tune
pass on real scans is not optional.

For realism, :func:`emit_lilypond_source` writes a genuine ``.ly`` file for the
same pattern, which LilyPond will engrave properly. Note that this path gives
no ground truth — recovering boxes from LilyPond means dumping grob extents
from Scheme during engraving, which is not implemented here.

Usage::

    python scripts/generate_synthetic_dataset.py --out data/synthetic --pages 500
    python scripts/generate_synthetic_dataset.py --out /tmp/probe --pages 4 --preview
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from melodix.geometry.deskew import rotate_image  # noqa: E402
from melodix.vision.dataset import (  # noqa: E402
    Annotation,
    BoundingBox,
    LabeledImage,
    class_distribution,
    split_dataset,
    write_data_yaml,
    write_label_file,
)
from melodix.vision.labels import NUM_CLASSES, SymbolClass, label_for_id  # noqa: E402

Page = npt.NDArray[np.uint8]
"""A grayscale page being drawn into, modified in place."""

INK = 0
PAPER = 255

# Staff positions, in half-steps from the bottom line, for the common voices.
# The generator only needs these to place heads plausibly; nothing downstream
# reads them, because position is Stage 1's job at inference time.
POSITION_HIHAT = 8
POSITION_SNARE = 5
POSITION_KICK = 1
POSITION_TOM = 3


@dataclass(frozen=True, slots=True)
class PageStyle:
    """Engraving parameters for one synthetic page.

    Attributes:
        width: Page width in pixels.
        height: Page height in pixels.
        line_spacing: Gap between staff lines. Sets the scale of everything
            else, exactly as it does in real engraving.
        line_thickness: Staff line stroke width.
        margin: Blank border in pixels.
        system_gap: Vertical gap between the bottom line of one staff and the
            top line of the next.
    """

    width: int = 1000
    height: int = 1400
    line_spacing: int = 16
    line_thickness: int = 2
    margin: int = 80
    system_gap: int = 150


@dataclass(frozen=True, slots=True)
class PlacedSymbol:
    """One symbol drawn at a known place.

    Attributes:
        symbol: Its class.
        x_min: Left edge in pixels.
        y_min: Top edge in pixels.
        x_max: Right edge in pixels.
        y_max: Bottom edge in pixels.
    """

    symbol: SymbolClass
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_annotation(self, width: int, height: int) -> Annotation:
        """Convert to a normalised annotation, clamped to the page."""
        return Annotation(
            symbol=self.symbol,
            box=BoundingBox.from_pixels(
                max(0.0, self.x_min),
                max(0.0, self.y_min),
                min(float(width), self.x_max),
                min(float(height), self.y_max),
                width,
                height,
            ),
        )


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def position_to_y(top_line_y: float, spacing: float, position: int) -> float:
    """Convert a staff position to a pixel row.

    Mirrors :meth:`melodix.geometry.staff.StaffGrid.position_to_y` for a staff
    of uniform spacing: position 0 is the bottom line, 8 the top.
    """
    bottom_line_y = top_line_y + 4 * spacing
    return bottom_line_y - position * (spacing / 2.0)


def _pad(
    x_min: float, y_min: float, x_max: float, y_max: float, pad: float
) -> tuple[float, float, float, float]:
    """Widen a box by a uniform margin, so a stroke's width is inside it."""
    return (x_min - pad, y_min - pad, x_max + pad, y_max + pad)


# --------------------------------------------------------------------------- #
# Symbol drawing. Each returns the exact box it drew into.
# --------------------------------------------------------------------------- #


def draw_round_head(
    page: Page, x: float, y: float, spacing: float, filled: bool = True
) -> PlacedSymbol:
    """Draw a filled or hollow oval notehead."""
    rx, ry = spacing * 0.62, spacing * 0.46
    cv2.ellipse(
        page,
        (int(round(x)), int(round(y))),
        (int(round(rx)), int(round(ry))),
        -20,
        0,
        360,
        INK,
        -1 if filled else max(1, int(spacing * 0.14)),
    )
    symbol = SymbolClass.ROUND_NOTEHEAD if filled else SymbolClass.HOLLOW_NOTEHEAD
    return PlacedSymbol(symbol, *_pad(x - rx, y - ry, x + rx, y + ry, 1.0))


def draw_cross_head(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw an X notehead: cymbals and hi-hat."""
    reach = spacing * 0.46
    weight = max(1, int(spacing * 0.14))
    for dx, dy in ((-1, -1), (-1, 1)):
        cv2.line(
            page,
            (int(x + dx * reach), int(y + dy * reach)),
            (int(x - dx * reach), int(y - dy * reach)),
            INK,
            weight,
        )
    return PlacedSymbol(
        SymbolClass.CROSS_NOTEHEAD, *_pad(x - reach, y - reach, x + reach, y + reach, weight)
    )


def draw_circle_cross_head(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw an X notehead inside a circle: ride bell."""
    draw_cross_head(page, x, y, spacing * 0.75)
    radius = spacing * 0.62
    cv2.circle(page, (int(x), int(y)), int(radius), INK, max(1, int(spacing * 0.11)))
    return PlacedSymbol(
        SymbolClass.CIRCLE_CROSS_NOTEHEAD,
        *_pad(x - radius, y - radius, x + radius, y + radius, 2.0),
    )


def draw_diamond_head(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw a diamond notehead."""
    rx, ry = spacing * 0.55, spacing * 0.5
    points = np.array(
        [[x - rx, y], [x, y - ry], [x + rx, y], [x, y + ry]], dtype=np.int32
    )
    cv2.fillPoly(page, [points], INK)
    return PlacedSymbol(SymbolClass.DIAMOND_NOTEHEAD, *_pad(x - rx, y - ry, x + rx, y + ry, 1.0))


def draw_triangle_head(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw a triangular notehead."""
    rx, ry = spacing * 0.55, spacing * 0.5
    points = np.array([[x - rx, y + ry], [x, y - ry], [x + rx, y + ry]], dtype=np.int32)
    cv2.fillPoly(page, [points], INK)
    return PlacedSymbol(SymbolClass.TRIANGLE_NOTEHEAD, *_pad(x - rx, y - ry, x + rx, y + ry, 1.0))


def draw_slash_head(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw a rhythm slash."""
    rx, ry = spacing * 0.5, spacing * 0.6
    points = np.array(
        [[x - rx, y + ry], [x, y + ry], [x + rx, y - ry], [x, y - ry]], dtype=np.int32
    )
    cv2.fillPoly(page, [points], INK)
    return PlacedSymbol(SymbolClass.SLASH_NOTEHEAD, *_pad(x - rx, y - ry, x + rx, y + ry, 1.0))


def draw_stem(page: Page, x: float, y_from: float, y_to: float, spacing: float) -> None:
    """Draw a stem. Deliberately unlabelled: stems are not in the schema."""
    cv2.line(
        page,
        (int(x), int(y_from)),
        (int(x), int(y_to)),
        INK,
        max(1, int(spacing * 0.12)),
    )


def draw_beam(page: Page, x_from: float, x_to: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw a beam joining two stems."""
    thickness = max(2, int(spacing * 0.42))
    cv2.rectangle(
        page, (int(x_from), int(y)), (int(x_to), int(y + thickness)), INK, -1
    )
    return PlacedSymbol(SymbolClass.BEAM, *_pad(x_from, y, x_to, y + thickness, 1.0))


def draw_accent(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw a horizontal wedge above a notehead."""
    reach = spacing * 0.55
    weight = max(1, int(spacing * 0.12))
    cv2.line(page, (int(x - reach), int(y - reach * 0.6)), (int(x + reach), int(y)), INK, weight)
    cv2.line(page, (int(x + reach), int(y)), (int(x - reach), int(y + reach * 0.6)), INK, weight)
    return PlacedSymbol(
        SymbolClass.ACCENT, *_pad(x - reach, y - reach * 0.6, x + reach, y + reach * 0.6, weight)
    )


def draw_marcato(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw a vertical wedge above a notehead."""
    reach = spacing * 0.5
    weight = max(1, int(spacing * 0.12))
    cv2.line(page, (int(x - reach * 0.7), int(y + reach)), (int(x), int(y - reach)), INK, weight)
    cv2.line(page, (int(x), int(y - reach)), (int(x + reach * 0.7), int(y + reach)), INK, weight)
    return PlacedSymbol(
        SymbolClass.MARCATO, *_pad(x - reach * 0.7, y - reach, x + reach * 0.7, y + reach, weight)
    )


def draw_dot(page: Page, x: float, y: float, spacing: float, symbol: SymbolClass) -> PlacedSymbol:
    """Draw a small filled dot, used for staccato and augmentation."""
    radius = max(1, int(spacing * 0.16))
    cv2.circle(page, (int(x), int(y)), radius, INK, -1)
    return PlacedSymbol(symbol, *_pad(x - radius, y - radius, x + radius, y + radius, 1.0))


def draw_quarter_rest(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw a quarter rest as a zigzag."""
    weight = max(1, int(spacing * 0.18))
    points = [
        (x - spacing * 0.3, y - spacing * 1.0),
        (x + spacing * 0.3, y - spacing * 0.3),
        (x - spacing * 0.25, y + spacing * 0.3),
        (x + spacing * 0.3, y + spacing * 1.0),
    ]
    for start, end in zip(points, points[1:], strict=False):
        cv2.line(page, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), INK, weight)
    return PlacedSymbol(
        SymbolClass.REST_QUARTER,
        *_pad(x - spacing * 0.35, y - spacing * 1.05, x + spacing * 0.35, y + spacing * 1.05, 1.0),
    )


def draw_hook_rest(
    page: Page, x: float, y: float, spacing: float, symbol: SymbolClass
) -> PlacedSymbol:
    """Draw an eighth or sixteenth rest: a diagonal with one or two hooks."""
    weight = max(1, int(spacing * 0.16))
    hooks = 1 if symbol is SymbolClass.REST_EIGHTH else 2
    height = spacing * (0.7 if hooks == 1 else 1.05)
    cv2.line(
        page,
        (int(x + spacing * 0.28), int(y - height * 0.85)),
        (int(x - spacing * 0.2), int(y + height)),
        INK,
        weight,
    )
    for index in range(hooks):
        cv2.circle(
            page,
            (int(x - spacing * 0.1), int(y - height * 0.8 + index * spacing * 0.45)),
            max(1, weight),
            INK,
            -1,
        )
    return PlacedSymbol(
        symbol,
        *_pad(x - spacing * 0.3, y - height * 0.95, x + spacing * 0.35, y + height * 1.05, 1.0),
    )


def draw_bar_rest(
    page: Page, x: float, y: float, spacing: float, symbol: SymbolClass
) -> PlacedSymbol:
    """Draw a whole or half rest: a filled bar against a staff line."""
    half_w = spacing * 0.5
    height = spacing * 0.4
    top = y - height if symbol is SymbolClass.REST_WHOLE else y
    cv2.rectangle(
        page, (int(x - half_w), int(top)), (int(x + half_w), int(top + height)), INK, -1
    )
    return PlacedSymbol(symbol, *_pad(x - half_w, top, x + half_w, top + height, 1.0))


def draw_percussion_clef(page: Page, x: float, top_line_y: float, spacing: float) -> PlacedSymbol:
    """Draw the two thick vertical bars that open a percussion staff."""
    weight = max(2, int(spacing * 0.3))
    y_top = top_line_y + spacing * 0.6
    y_bottom = top_line_y + spacing * 3.4
    for offset in (0, spacing * 0.66):
        cv2.line(
            page,
            (int(x + offset), int(y_top)),
            (int(x + offset), int(y_bottom)),
            INK,
            weight,
        )
    return PlacedSymbol(
        SymbolClass.PERCUSSION_CLEF,
        *_pad(x - weight, y_top, x + spacing * 0.66 + weight, y_bottom, 1.0),
    )


def draw_time_signature(
    page: Page, x: float, top_line_y: float, spacing: float, upper: int, lower: int
) -> PlacedSymbol:
    """Draw a stacked time signature."""
    scale = spacing / 14.0
    thickness = max(1, int(spacing * 0.16))
    positions = ((upper, top_line_y + spacing * 0.95), (lower, top_line_y + spacing * 2.95))
    for value, baseline in positions:
        cv2.putText(
            page,
            str(value),
            (int(x), int(baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            INK,
            thickness,
            cv2.LINE_AA,
        )
    return PlacedSymbol(
        SymbolClass.TIME_SIGNATURE,
        *_pad(x - 2, top_line_y, x + spacing * 1.3, top_line_y + spacing * 3.2, 1.0),
    )


def draw_repeat_dots(page: Page, x: float, top_line_y: float, spacing: float) -> PlacedSymbol:
    """Draw the two dots that sit beside a repeat barline."""
    radius = max(1, int(spacing * 0.17))
    rows = (
        position_to_y(top_line_y, spacing, 3),
        position_to_y(top_line_y, spacing, 5),
    )
    for row in rows:
        cv2.circle(page, (int(x), int(row)), radius, INK, -1)
    return PlacedSymbol(
        SymbolClass.REPEAT_DOTS,
        *_pad(x - radius, min(rows) - radius, x + radius, max(rows) + radius, 1.0),
    )


def draw_repeat_measure(page: Page, x: float, top_line_y: float, spacing: float) -> PlacedSymbol:
    """Draw the percent sign meaning "repeat the previous bar"."""
    weight = max(1, int(spacing * 0.14))
    centre = position_to_y(top_line_y, spacing, 4)
    reach = spacing * 0.75
    cv2.line(
        page,
        (int(x - reach), int(centre + reach * 0.8)),
        (int(x + reach), int(centre - reach * 0.8)),
        INK,
        weight,
    )
    radius = max(1, int(spacing * 0.15))
    cv2.circle(page, (int(x - reach * 0.55), int(centre - reach * 0.5)), radius, INK, -1)
    cv2.circle(page, (int(x + reach * 0.55), int(centre + reach * 0.5)), radius, INK, -1)
    return PlacedSymbol(
        SymbolClass.REPEAT_MEASURE,
        *_pad(x - reach - radius, centre - reach * 0.9, x + reach + radius,
              centre + reach * 0.9, 1.0),
    )


def draw_flag(page: Page, x: float, y: float, spacing: float, symbol: SymbolClass) -> PlacedSymbol:
    """Draw one or two flags hanging off a stem end."""
    weight = max(1, int(spacing * 0.16))
    count = 1 if symbol is SymbolClass.FLAG_EIGHTH else 2
    for index in range(count):
        offset = index * spacing * 0.45
        cv2.line(
            page,
            (int(x), int(y + offset)),
            (int(x + spacing * 0.6), int(y + offset + spacing * 0.8)),
            INK,
            weight,
        )
    span = (count - 1) * spacing * 0.45
    return PlacedSymbol(
        symbol, *_pad(x, y, x + spacing * 0.6, y + span + spacing * 0.8, weight)
    )


def draw_grace_note(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw a small slashed head: a flam or drag."""
    small = spacing * 0.6
    rx, ry = small * 0.6, small * 0.45
    cv2.ellipse(page, (int(x), int(y)), (int(rx), int(ry)), -20, 0, 360, INK, -1)
    cv2.line(
        page,
        (int(x - rx), int(y - small * 1.1)),
        (int(x + rx * 1.6), int(y - small * 2.0)),
        INK,
        max(1, int(spacing * 0.1)),
    )
    draw_stem(page, x + rx, y, y - small * 2.2, spacing * 0.7)
    return PlacedSymbol(
        SymbolClass.GRACE_NOTE, *_pad(x - rx, y - small * 2.2, x + rx * 1.6, y + ry, 1.0)
    )


def draw_letter_modifier(
    page: Page, x: float, y: float, spacing: float, symbol: SymbolClass
) -> PlacedSymbol:
    """Draw the small ``o`` or ``+`` that sits above a hi-hat notehead."""
    radius = spacing * 0.28
    weight = max(1, int(spacing * 0.12))
    if symbol is SymbolClass.OPEN_MODIFIER:
        cv2.circle(page, (int(x), int(y)), int(radius), INK, weight)
    else:
        cv2.line(page, (int(x - radius), int(y)), (int(x + radius), int(y)), INK, weight)
        cv2.line(page, (int(x), int(y - radius)), (int(x), int(y + radius)), INK, weight)
    return PlacedSymbol(symbol, *_pad(x - radius, y - radius, x + radius, y + radius, weight))


def draw_ghost_note(page: Page, x: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw a notehead wrapped in parentheses."""
    draw_round_head(page, x, y, spacing * 0.9)
    reach = spacing * 0.85
    weight = max(1, int(spacing * 0.11))
    for sign in (-1, 1):
        cv2.ellipse(
            page,
            (int(x + sign * reach), int(y)),
            (int(spacing * 0.32), int(spacing * 0.72)),
            0,
            100 if sign > 0 else -80,
            260 if sign > 0 else 80,
            INK,
            weight,
        )
    return PlacedSymbol(
        SymbolClass.GHOST_NOTE,
        *_pad(x - reach - spacing * 0.3, y - spacing * 0.75, x + reach + spacing * 0.3,
              y + spacing * 0.75, 1.0),
    )


def draw_tie(page: Page, x_from: float, x_to: float, y: float, spacing: float) -> PlacedSymbol:
    """Draw an arc joining two noteheads."""
    weight = max(1, int(spacing * 0.11))
    centre = ((x_from + x_to) / 2.0, y)
    axes = (max(2, int((x_to - x_from) / 2.0)), max(2, int(spacing * 0.5)))
    cv2.ellipse(page, (int(centre[0]), int(centre[1])), axes, 0, 180, 360, INK, weight)
    return PlacedSymbol(
        SymbolClass.TIE_SLUR, *_pad(x_from, y - axes[1], x_to, y + weight, 1.0)
    )


NOTEHEAD_DRAWERS = {
    SymbolClass.ROUND_NOTEHEAD: lambda p, x, y, s: draw_round_head(p, x, y, s, filled=True),
    SymbolClass.HOLLOW_NOTEHEAD: lambda p, x, y, s: draw_round_head(p, x, y, s, filled=False),
    SymbolClass.CROSS_NOTEHEAD: draw_cross_head,
    SymbolClass.CIRCLE_CROSS_NOTEHEAD: draw_circle_cross_head,
    SymbolClass.DIAMOND_NOTEHEAD: draw_diamond_head,
    SymbolClass.TRIANGLE_NOTEHEAD: draw_triangle_head,
    SymbolClass.SLASH_NOTEHEAD: draw_slash_head,
}


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #


def draw_staff_lines(
    page: Page, top_line_y: float, x_start: int, x_end: int, style: PageStyle
) -> None:
    """Draw the five lines of one staff."""
    for step in range(5):
        row = int(round(top_line_y + step * style.line_spacing))
        cv2.line(page, (x_start, row), (x_end, row), INK, style.line_thickness)


def draw_barline(page: Page, x: float, top_line_y: float, style: PageStyle) -> None:
    """Draw a barline spanning one staff. Not labelled: Stage 1 finds these."""
    cv2.line(
        page,
        (int(x), int(top_line_y)),
        (int(x), int(top_line_y + 4 * style.line_spacing)),
        INK,
        max(1, int(style.line_spacing * 0.13)),
    )


def fill_measure(
    page: Page,
    rng: random.Random,
    x_start: float,
    x_end: float,
    top_line_y: float,
    style: PageStyle,
) -> list[PlacedSymbol]:
    """Populate one measure with a plausible drum pattern.

    Not a musically correct bar — durations are not made to add up. The
    detector learns shapes and their spatial habits, and inventing a valid
    rhythm would constrain the symbol mix without teaching it anything.
    """
    spacing = float(style.line_spacing)
    placed: list[PlacedSymbol] = []

    slots = rng.choice((4, 6, 8))
    step = (x_end - x_start) / (slots + 1)
    stem_top = top_line_y - spacing * 1.4
    beam_group: list[float] = []
    lower_heads: list[tuple[float, float]] = []

    for index in range(slots):
        x = x_start + step * (index + 1)

        if rng.random() < 0.10:
            rest = rng.choice(
                (
                    SymbolClass.REST_QUARTER,
                    SymbolClass.REST_EIGHTH,
                    SymbolClass.REST_SIXTEENTH,
                    SymbolClass.REST_HALF,
                    SymbolClass.REST_WHOLE,
                )
            )
            centre = position_to_y(top_line_y, spacing, 4)
            if rest is SymbolClass.REST_QUARTER:
                placed.append(draw_quarter_rest(page, x, centre, spacing))
            elif rest in (SymbolClass.REST_EIGHTH, SymbolClass.REST_SIXTEENTH):
                placed.append(draw_hook_rest(page, x, centre, spacing, rest))
            else:
                placed.append(draw_bar_rest(page, x, centre, spacing, rest))
            continue

        # Upper voice: hi-hat or ride, usually a cross head.
        upper = rng.choices(
            [
                SymbolClass.CROSS_NOTEHEAD,
                SymbolClass.CIRCLE_CROSS_NOTEHEAD,
                SymbolClass.DIAMOND_NOTEHEAD,
                SymbolClass.TRIANGLE_NOTEHEAD,
                SymbolClass.SLASH_NOTEHEAD,
            ],
            weights=[70, 12, 8, 5, 5],
        )[0]
        y_upper = position_to_y(top_line_y, spacing, POSITION_HIHAT)
        placed.append(NOTEHEAD_DRAWERS[upper](page, x, y_upper, spacing))
        draw_stem(page, x + spacing * 0.55, y_upper, stem_top, spacing)
        beam_group.append(x + spacing * 0.55)

        if rng.random() < 0.22:
            placed.append(
                draw_letter_modifier(
                    page,
                    x,
                    y_upper - spacing * 1.0,
                    spacing,
                    rng.choice((SymbolClass.OPEN_MODIFIER, SymbolClass.CLOSED_MODIFIER)),
                )
            )
        if rng.random() < 0.18:
            placed.append(draw_accent(page, x, y_upper - spacing * 1.9, spacing))
        elif rng.random() < 0.08:
            placed.append(draw_marcato(page, x, y_upper - spacing * 1.9, spacing))

        # Lower voice: snare, kick or a tom.
        if rng.random() < 0.55:
            position = rng.choice((POSITION_SNARE, POSITION_KICK, POSITION_TOM))
            y_lower = position_to_y(top_line_y, spacing, position)
            if rng.random() < 0.12:
                placed.append(draw_ghost_note(page, x, y_lower, spacing))
            else:
                filled = rng.random() > 0.12
                placed.append(draw_round_head(page, x, y_lower, spacing, filled=filled))
                lower_heads.append((x, y_lower))
                if rng.random() < 0.10:
                    placed.append(
                        draw_dot(
                            page,
                            x + spacing * 1.0,
                            y_lower,
                            spacing,
                            SymbolClass.AUGMENTATION_DOT,
                        )
                    )
                if rng.random() < 0.08:
                    placed.append(
                        draw_dot(
                            page,
                            x,
                            y_lower + spacing * 0.9,
                            spacing,
                            SymbolClass.STACCATO,
                        )
                    )
            if rng.random() < 0.10:
                placed.append(draw_grace_note(page, x - spacing * 1.5, y_lower, spacing))

    for (x_left, y_left), (x_right, y_right) in zip(lower_heads, lower_heads[1:], strict=False):
        if abs(y_left - y_right) < 1.0 and x_right - x_left < spacing * 6 and rng.random() < 0.18:
            placed.append(
                draw_tie(page, x_left, x_right, y_left + spacing * 0.75, spacing)
            )

    if rng.random() < 0.06:
        placed.append(
            draw_repeat_measure(page, (x_start + x_end) / 2.0, top_line_y, spacing)
        )

    if len(beam_group) >= 2 and rng.random() < 0.7:
        placed.append(draw_beam(page, beam_group[0], beam_group[-1], stem_top, spacing))
    elif beam_group and rng.random() < 0.4:
        placed.append(
            draw_flag(
                page,
                beam_group[-1],
                stem_top,
                spacing,
                rng.choice((SymbolClass.FLAG_EIGHTH, SymbolClass.FLAG_SIXTEENTH)),
            )
        )

    return placed


def render_page(
    rng: random.Random, style: PageStyle | None = None
) -> tuple[npt.NDArray[np.uint8], list[PlacedSymbol]]:
    """Draw one synthetic page and return it with its exact ground truth.

    Args:
        rng: Seeded source of randomness.
        style: Engraving parameters. Defaults to a letter-ish portrait page.

    Returns:
        The page as a grayscale array, and every symbol placed on it.
    """
    style = style or PageStyle()
    page = np.full((style.height, style.width), PAPER, dtype=np.uint8)
    placed: list[PlacedSymbol] = []

    spacing = float(style.line_spacing)
    x_start = style.margin
    x_end = style.width - style.margin
    staff_height = 4 * style.line_spacing

    top = float(style.margin)
    while top + staff_height + style.system_gap < style.height - style.margin:
        draw_staff_lines(page, top, x_start, x_end, style)

        cursor = float(x_start)
        placed.append(draw_percussion_clef(page, cursor + spacing * 0.4, top, spacing))
        cursor += spacing * 2.6

        if rng.random() < 0.6:
            placed.append(
                draw_time_signature(page, cursor, top, spacing, rng.choice((4, 3, 6)), 4)
            )
            cursor += spacing * 2.2

        measures = rng.choice((2, 3, 4))
        width = (x_end - cursor) / measures
        for index in range(measures):
            left = cursor + index * width
            right = left + width
            placed.extend(fill_measure(page, rng, left, right, top, style))
            draw_barline(page, right, top, style)

        draw_barline(page, cursor, top, style)
        if rng.random() < 0.25:
            placed.append(draw_repeat_dots(page, x_end - spacing * 0.8, top, spacing))

        top += staff_height + style.system_gap

    return page, placed


# --------------------------------------------------------------------------- #
# Scan simulation
# --------------------------------------------------------------------------- #


def rotate_symbols(
    placed: list[PlacedSymbol], angle_deg: float, width: int, height: int
) -> list[PlacedSymbol]:
    """Carry ground-truth boxes through the same rotation the page took.

    Rotating the image without rotating the boxes ships a dataset whose labels
    sit beside their symbols — which trains perfectly happily to a plausible
    loss and produces a detector that is systematically off. The matrix here is
    built exactly as :func:`~melodix.geometry.deskew.rotate_image` builds it,
    so the two cannot drift apart.

    A rotated rectangle is not axis-aligned, so the result is the axis-aligned
    hull of the rotated corners. At the couple of degrees used here that
    inflates a box by well under a pixel.

    Args:
        placed: Symbols in pre-rotation pixel coordinates.
        angle_deg: Rotation applied to the page, counter-clockwise positive.
        width: Page width in pixels.
        height: Page height in pixels.

    Returns:
        Symbols in post-rotation coordinates, dropping any rotated off-page.
    """
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    rotated: list[PlacedSymbol] = []

    for symbol in placed:
        corners = np.array(
            [
                [symbol.x_min, symbol.y_min, 1.0],
                [symbol.x_max, symbol.y_min, 1.0],
                [symbol.x_max, symbol.y_max, 1.0],
                [symbol.x_min, symbol.y_max, 1.0],
            ]
        )
        moved = corners @ matrix.T
        x_min, y_min = moved.min(axis=0)
        x_max, y_max = moved.max(axis=0)

        x_min, x_max = max(0.0, x_min), min(float(width), x_max)
        y_min, y_max = max(0.0, y_min), min(float(height), y_max)
        if x_max - x_min < 1.0 or y_max - y_min < 1.0:
            continue  # rotated off the page

        rotated.append(PlacedSymbol(symbol.symbol, x_min, y_min, x_max, y_max))

    return rotated


def augment(
    page: npt.NDArray[np.uint8], placed: list[PlacedSymbol], rng: random.Random
) -> tuple[npt.NDArray[np.uint8], list[PlacedSymbol], float]:
    """Rough up a clean render so it looks more like a scan.

    Rotation reuses :func:`melodix.geometry.deskew.rotate_image`, the same
    function Stage 1 uses to correct skew, so the synthetic tilt distribution
    matches what deskew is built to undo. Ground-truth boxes are carried
    through the same rotation.

    Args:
        page: A clean render.
        placed: Its ground truth, in pre-rotation coordinates.
        rng: Seeded source of randomness.

    Returns:
        The roughened page, its ground truth, and the tilt applied.
    """
    height, width = page.shape[:2]
    angle = rng.uniform(-2.0, 2.0)
    out = rotate_image(page, angle, border_value=PAPER)
    placed = rotate_symbols(placed, angle, width, height)

    if rng.random() < 0.7:
        radius = rng.choice((3, 5))
        out = cv2.GaussianBlur(out, (radius, radius), 0)

    if rng.random() < 0.8:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(
            0.0, rng.uniform(2.0, 9.0), out.shape
        )
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        gain = rng.uniform(0.85, 1.12)
        out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    return out, placed, angle


# --------------------------------------------------------------------------- #
# LilyPond source, for the realism path
# --------------------------------------------------------------------------- #


def emit_lilypond_source(measures: int = 4, seed: int = 0) -> str:
    """Return a LilyPond source string for a drum pattern.

    Engraving this gives a genuinely realistic page, which is what the
    fine-tune corpus should look like. It gives **no ground truth**: recovering
    boxes means dumping grob extents from Scheme during engraving, which is not
    implemented here. Use it to eyeball how far the procedural renderer is from
    real notation, or as the basis for a properly engraved corpus you annotate
    by other means.

    Args:
        measures: How many bars to write.
        seed: Seeds the pattern choice.

    Returns:
        LilyPond source, ready to write to a ``.ly`` file.
    """
    rng = random.Random(seed)
    patterns = ("hh8 hh hh hh hh hh hh hh", "hh8 hh sn hh hh hh sn hh", "cymc4 hh8 sn hh bd4 sn")
    up = " |\n    ".join(rng.choice(patterns) for _ in range(measures))
    down = " |\n    ".join("bd4 sn bd sn" for _ in range(measures))
    return (
        '\\version "2.24.0"\n'
        "\\score {\n"
        "  \\new DrumStaff <<\n"
        "    \\new DrumVoice { \\voiceOne \\drummode {\n"
        f"    {up} |\n"
        "    } }\n"
        "    \\new DrumVoice { \\voiceTwo \\drummode {\n"
        f"    {down} |\n"
        "    } }\n"
        "  >>\n"
        "  \\layout { }\n"
        "}\n"
    )


# --------------------------------------------------------------------------- #
# Dataset writing
# --------------------------------------------------------------------------- #


def generate(
    out_root: Path,
    pages: int,
    seed: int = 0,
    val_fraction: float = 0.2,
    style: PageStyle | None = None,
    clean: bool = False,
) -> dict[str, int]:
    """Render pages and write a complete YOLO dataset.

    Args:
        out_root: Directory to build the dataset in.
        pages: How many pages to render.
        seed: Seeds both rendering and the train/val split.
        val_fraction: Portion held out for validation.
        style: Engraving parameters.
        clean: Skip the scan-simulation pass, leaving crisp renders.

    Returns:
        A summary: page and annotation counts, and classes covered.
    """
    rng = random.Random(seed)
    style = style or PageStyle()

    staged: list[tuple[str, npt.NDArray[np.uint8], list[Annotation]]] = []
    for index in range(pages):
        page, placed = render_page(rng, style)
        if not clean:
            page, placed, _ = augment(page, placed, rng)
        height, width = page.shape[:2]
        staged.append(
            (
                f"synthetic_{index:05d}",
                page,
                [symbol.to_annotation(width, height) for symbol in placed],
            )
        )

    split = split_dataset(
        [
            LabeledImage(image_path=Path(f"images/{stem}.png"), annotations=tuple(annotations))
            for stem, _, annotations in staged
        ],
        val_fraction=val_fraction,
        seed=seed,
    )
    by_stem = {stem: image for stem, image, _ in staged}
    assigned = {image.stem: "val" for image in split.val}

    written = 0
    for group, images in (("train", split.train), ("val", split.val)):
        for labelled in images:
            stem = labelled.stem
            image_path = out_root / "images" / group / f"{stem}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), by_stem[stem])
            write_label_file(
                out_root / "labels" / group / f"{stem}.txt", labelled.annotations
            )
            written += 1

    write_data_yaml(out_root / "data.yaml", out_root)

    counts = class_distribution([*split.train, *split.val])
    return {
        "pages": written,
        "train": len(split.train),
        "val": len(split.val),
        "annotations": sum(counts.values()),
        "classes_covered": len(counts),
        "classes_total": NUM_CLASSES,
        "val_assigned": len(assigned),
    }


def report_coverage(out_root: Path, counts: dict[SymbolClass, int]) -> None:
    """Print per-class counts, loudest gap first."""
    print("\nclass coverage:")
    for class_id in range(NUM_CLASSES):
        symbol = SymbolClass(class_id)
        count = counts.get(symbol, 0)
        flag = "  " if count else "!!"
        print(f"  {flag} {class_id:2d} {label_for_id(class_id).name:24s} {count:6d}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="dataset root to build")
    parser.add_argument("--pages", type=int, default=200, help="pages to render")
    parser.add_argument("--seed", type=int, default=0, help="seeds rendering and the split")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="portion held out")
    parser.add_argument("--spacing", type=int, default=16, help="staff line spacing in pixels")
    parser.add_argument("--clean", action="store_true", help="skip scan simulation")
    parser.add_argument(
        "--preview", action="store_true", help="also write a boxed preview of the first page"
    )
    parser.add_argument(
        "--lilypond", type=Path, help="write a LilyPond source file for comparison and exit"
    )
    args = parser.parse_args(argv)

    if args.lilypond is not None:
        args.lilypond.parent.mkdir(parents=True, exist_ok=True)
        args.lilypond.write_text(emit_lilypond_source(seed=args.seed), encoding="utf-8")
        print(f"wrote {args.lilypond}")
        return 0

    style = PageStyle(line_spacing=args.spacing)
    summary = generate(
        args.out,
        pages=args.pages,
        seed=args.seed,
        val_fraction=args.val_fraction,
        style=style,
        clean=args.clean,
    )

    print(f"wrote {summary['pages']} pages to {args.out}")
    print(f"  train {summary['train']}, val {summary['val']}")
    print(f"  {summary['annotations']} annotations")
    print(f"  {summary['classes_covered']}/{summary['classes_total']} classes represented")

    if args.preview:
        rng = random.Random(args.seed)
        page, placed = render_page(rng, style)
        preview = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        for symbol in placed:
            cv2.rectangle(
                preview,
                (int(symbol.x_min), int(symbol.y_min)),
                (int(symbol.x_max), int(symbol.y_max)),
                (0, 0, 255),
                1,
            )
        path = args.out / "preview_boxes.png"
        cv2.imwrite(str(path), preview)
        print(f"  preview written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
