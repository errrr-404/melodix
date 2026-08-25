"""The symbol class schema: what the detector is trained to find.

This module is the contract between a trained YOLO checkpoint and Stage 3's
percussion mapping, and it is the most expensive file in the project to change.
A class id is baked into every checkpoint weight file and every annotated
image on disk, so reordering :class:`SymbolClass` silently invalidates both:
the detector keeps running and starts reporting hi-hats as snares. Append new
members at the end, never renumber.

Classes name **shapes, not voices**
-----------------------------------
A cross notehead is ``CROSS_NOTEHEAD`` whatever line it sits on. It is not
"closed hi-hat", even though a cross notehead on the top line nearly always is
one. The vertical position comes from Stage 1::

    position = grid.snap(box.center_pixels(width, height)[1])

and Stage 3 combines the two into a General MIDI voice. Folding position into
the label instead would multiply the class count by the nine staff positions
plus ledger territory, discard the sub-pixel precision
:class:`~melodix.geometry.staff.StaffGrid` already computes, and force a full
retrain whenever a kit is re-voiced.

The same reasoning splits detection from interpretation elsewhere in the
pipeline: :mod:`melodix.geometry.barlines` reports vertical strokes without
deciding which are barlines, for exactly this reason.

Nothing here imports torch or ultralytics. The schema is plain data, so it
stays importable — and testable — without the ``vision`` extra installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final

__all__ = [
    "LABELS",
    "NUM_CLASSES",
    "ClassMismatchError",
    "NoteheadShape",
    "SymbolCategory",
    "SymbolClass",
    "SymbolLabel",
    "class_names",
    "label_for_id",
    "label_for_name",
    "labels_in_category",
    "labels_with_shape",
    "verify_model_classes",
]


class ClassMismatchError(ValueError):
    """Raised when a checkpoint's class list disagrees with this schema.

    The failure this guards is silent: YOLO stores class *indices*, so a
    checkpoint trained against a different ordering keeps running and simply
    reports the wrong symbol for every detection. Nothing downstream can notice
    — a cross notehead reported as a diamond is still a plausible detection.
    """


class SymbolCategory(StrEnum):
    """What role a symbol plays once Stage 3 reads it.

    The category drives interpretation, not detection. Two symbols in the same
    category are handled by the same downstream branch.

    Attributes:
        NOTEHEAD: Sounds a drum voice. Carries a staff position.
        REST: Silence for a notated duration. Position is not meaningful.
        DURATION: Evidence of note length — flags, beams, augmentation dots.
        MODIFIER: Attaches to a nearby notehead and alters how it sounds.
        STRUCTURE: Page furniture: clefs, time signatures, repeat marks.
    """

    NOTEHEAD = "notehead"
    REST = "rest"
    DURATION = "duration"
    MODIFIER = "modifier"
    STRUCTURE = "structure"


class NoteheadShape(StrEnum):
    """The head shape Stage 3 pairs with a staff position to pick a drum.

    Shape and position together name a voice: an ``X`` head on position 8 is a
    closed hi-hat, a ``ROUND`` head on position 5 is an acoustic snare. Neither
    alone is enough, which is why the detector reports shape and Stage 1
    reports position.

    Two mappings here are decisions rather than transcriptions, and both matter
    downstream:

    ``HOLLOW_NOTEHEAD`` carries shape ``ROUND``. A hollow head differs from a
    filled one in *duration*, not in which drum is struck — a half-note snare
    is still a snare. Giving it its own shape would split one voice in two.

    ``CIRCLE_CROSS_NOTEHEAD`` carries shape ``CIRCLE_X``, not ``X``. A circled
    cross on the top line is a ride bell where a bare cross is a closed hi-hat,
    so collapsing the two would make those voices indistinguishable at exactly
    the position where they collide.

    Attributes:
        ROUND: Oval head, filled or hollow.
        X: Cross head.
        CIRCLE_X: Cross head inside a circle.
        TRIANGLE: Triangular head.
        DIAMOND: Diamond head.
        SLASH: Rhythm slash.
        NONE: Not a notehead. Rests, modifiers and page furniture.
    """

    ROUND = "round"
    X = "x"
    CIRCLE_X = "circle_x"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"
    SLASH = "slash"
    NONE = "none"


class SymbolClass(IntEnum):
    """A detectable symbol shape, and its YOLO class id.

    Values are a permanent part of the on-disk format. Append only.
    """

    # -- noteheads ---------------------------------------------------------- #
    ROUND_NOTEHEAD = 0
    HOLLOW_NOTEHEAD = 1
    CROSS_NOTEHEAD = 2
    CIRCLE_CROSS_NOTEHEAD = 3
    DIAMOND_NOTEHEAD = 4
    TRIANGLE_NOTEHEAD = 5
    SLASH_NOTEHEAD = 6

    # -- rests -------------------------------------------------------------- #
    REST_WHOLE = 7
    REST_HALF = 8
    REST_QUARTER = 9
    REST_EIGHTH = 10
    REST_SIXTEENTH = 11

    # -- duration evidence -------------------------------------------------- #
    FLAG_EIGHTH = 12
    FLAG_SIXTEENTH = 13
    BEAM = 14
    AUGMENTATION_DOT = 15

    # -- modifiers ---------------------------------------------------------- #
    ACCENT = 16
    MARCATO = 17
    STACCATO = 18
    GHOST_NOTE = 19
    OPEN_MODIFIER = 20
    CLOSED_MODIFIER = 21
    GRACE_NOTE = 22

    # -- structure ---------------------------------------------------------- #
    PERCUSSION_CLEF = 23
    TIME_SIGNATURE = 24
    REPEAT_DOTS = 25
    REPEAT_MEASURE = 26
    TIE_SLUR = 27

    @property
    def label(self) -> SymbolLabel:
        """The full record for this class."""
        return LABELS[int(self)]


@dataclass(frozen=True, slots=True)
class SymbolLabel:
    """One row of the class schema.

    Attributes:
        symbol: The class itself, which also carries its id.
        category: Downstream handling for this shape.
        shape: Head shape, which Stage 3 pairs with a staff position to pick a
            percussion voice. :attr:`NoteheadShape.NONE` for anything that is
            not a notehead.
        description: What an annotator should draw a box around.
    """

    symbol: SymbolClass
    category: SymbolCategory
    shape: NoteheadShape
    description: str

    @property
    def class_id(self) -> int:
        """The integer YOLO trains against."""
        return int(self.symbol)

    @property
    def name(self) -> str:
        """The snake_case name written into ``data.yaml``."""
        return self.symbol.name.lower()

    @property
    def carries_position(self) -> bool:
        """Whether Stage 3 must resolve a staff position for this symbol.

        True for noteheads only. A rest's vertical placement is an engraving
        convention rather than a pitch, and a modifier takes its voice from the
        notehead it attaches to.
        """
        return self.category is SymbolCategory.NOTEHEAD

    @property
    def attaches_to_notehead(self) -> bool:
        """Whether this symbol is meaningless on its own.

        An accent floating in whitespace is a detection error; an accent above
        a notehead makes that note louder.
        """
        return self.category is SymbolCategory.MODIFIER


def _label(
    symbol: SymbolClass,
    category: SymbolCategory,
    description: str,
    shape: NoteheadShape = NoteheadShape.NONE,
) -> SymbolLabel:
    """Build one schema row.

    ``shape`` defaults to :attr:`NoteheadShape.NONE`, so only the notehead rows
    below name one. A rest or an accent has no head shape, and defaulting keeps
    that from being restated twenty-one times.
    """
    return SymbolLabel(
        symbol=symbol, category=category, shape=shape, description=description
    )


_NOTEHEAD = SymbolCategory.NOTEHEAD
_REST = SymbolCategory.REST
_DURATION = SymbolCategory.DURATION
_MODIFIER = SymbolCategory.MODIFIER
_STRUCTURE = SymbolCategory.STRUCTURE

#: The schema, ordered by class id. Index equals :attr:`SymbolLabel.class_id`.
LABELS: Final[tuple[SymbolLabel, ...]] = (
    _label(
        SymbolClass.ROUND_NOTEHEAD, _NOTEHEAD, "Filled oval head: snare, toms, kick",
        NoteheadShape.ROUND,
    ),
    _label(
        SymbolClass.HOLLOW_NOTEHEAD, _NOTEHEAD, "Unfilled oval head, half or whole duration",
        NoteheadShape.ROUND,  # same voice as filled; the difference is duration
    ),
    _label(
        SymbolClass.CROSS_NOTEHEAD, _NOTEHEAD, "X head: hi-hat, ride, crash",
        NoteheadShape.X,
    ),
    _label(
        SymbolClass.CIRCLE_CROSS_NOTEHEAD, _NOTEHEAD, "X head inside a circle: ride bell",
        NoteheadShape.CIRCLE_X,  # distinct from X: collides with hi-hat at position 8
    ),
    _label(
        SymbolClass.DIAMOND_NOTEHEAD, _NOTEHEAD, "Diamond head: cymbal bell, harmonic",
        NoteheadShape.DIAMOND,
    ),
    _label(
        SymbolClass.TRIANGLE_NOTEHEAD, _NOTEHEAD, "Triangular head, kit-specific voice",
        NoteheadShape.TRIANGLE,
    ),
    _label(
        SymbolClass.SLASH_NOTEHEAD, _NOTEHEAD, "Rhythm slash: play the written groove",
        NoteheadShape.SLASH,
    ),
    _label(SymbolClass.REST_WHOLE, _REST, "Whole rest: filled bar hanging below a line"),
    _label(SymbolClass.REST_HALF, _REST, "Half rest: filled bar sitting on a line"),
    _label(SymbolClass.REST_QUARTER, _REST, "Quarter rest"),
    _label(SymbolClass.REST_EIGHTH, _REST, "Eighth rest, one hook"),
    _label(SymbolClass.REST_SIXTEENTH, _REST, "Sixteenth rest, two hooks"),
    _label(SymbolClass.FLAG_EIGHTH, _DURATION, "Single flag on an unbeamed stem"),
    _label(SymbolClass.FLAG_SIXTEENTH, _DURATION, "Double flag on an unbeamed stem"),
    _label(SymbolClass.BEAM, _DURATION, "Thick bar joining stems; count for duration"),
    _label(SymbolClass.AUGMENTATION_DOT, _DURATION, "Dot after a head: add half its value"),
    _label(SymbolClass.ACCENT, _MODIFIER, "Horizontal wedge: strike harder"),
    _label(SymbolClass.MARCATO, _MODIFIER, "Vertical wedge: strongest accent"),
    _label(SymbolClass.STACCATO, _MODIFIER, "Dot above or below a head: shorten"),
    _label(SymbolClass.GHOST_NOTE, _MODIFIER, "Head in parentheses: play very quietly"),
    _label(SymbolClass.OPEN_MODIFIER, _MODIFIER, "Small o above a head: let it ring"),
    _label(SymbolClass.CLOSED_MODIFIER, _MODIFIER, "Plus above a head: choke it"),
    _label(SymbolClass.GRACE_NOTE, _MODIFIER, "Small slashed head: flam or drag"),
    _label(SymbolClass.PERCUSSION_CLEF, _STRUCTURE, "Two vertical bars opening a staff"),
    _label(SymbolClass.TIME_SIGNATURE, _STRUCTURE, "Stacked numerals after the clef"),
    _label(SymbolClass.REPEAT_DOTS, _STRUCTURE, "Two dots beside a barline"),
    _label(SymbolClass.REPEAT_MEASURE, _STRUCTURE, "Percent sign: repeat the previous bar"),
    _label(SymbolClass.TIE_SLUR, _STRUCTURE, "Arc joining two heads"),
)

#: How many classes the detector head must predict.
NUM_CLASSES: Final[int] = len(LABELS)

_BY_NAME: Final[dict[str, SymbolLabel]] = {label.name: label for label in LABELS}


def label_for_id(class_id: int) -> SymbolLabel:
    """Look up a class by the integer the detector emits.

    Args:
        class_id: A YOLO class id.

    Returns:
        The matching schema row.

    Raises:
        KeyError: If no class carries that id. Raised rather than returning
            ``None`` because an unknown id means the checkpoint and this schema
            have diverged, which cannot be recovered from downstream.
    """
    if not 0 <= class_id < NUM_CLASSES:
        raise KeyError(f"no symbol class with id {class_id}; expected 0..{NUM_CLASSES - 1}")
    return LABELS[class_id]


def label_for_name(name: str) -> SymbolLabel:
    """Look up a class by its snake_case name.

    Args:
        name: A name as written in ``data.yaml``. Case-insensitive.

    Returns:
        The matching schema row.

    Raises:
        KeyError: If the name is not in the schema.
    """
    try:
        return _BY_NAME[name.lower()]
    except KeyError:
        raise KeyError(f"no symbol class named {name!r}") from None


def class_names() -> list[str]:
    """Return every class name ordered by id, for ``data.yaml``.

    Ultralytics reads this list positionally, so the order *is* the id
    assignment. Never sort it.
    """
    return [label.name for label in LABELS]


def labels_with_shape(shape: NoteheadShape) -> tuple[SymbolLabel, ...]:
    """Return every class carrying one head shape, ordered by id.

    Args:
        shape: The shape to filter by.

    Returns:
        Matching schema rows, possibly empty.
    """
    return tuple(label for label in LABELS if label.shape is shape)


def verify_model_classes(names: Mapping[int, str] | Sequence[str]) -> None:
    """Check a checkpoint's class list against this schema, or raise.

    Call this immediately after loading a checkpoint, before any detection is
    trusted. YOLO stores class *indices*, so a checkpoint trained against a
    different ordering does not fail — it silently reports the wrong symbol for
    every detection, and Stage 3 turns that into the wrong drum.

    Takes the name mapping rather than the model itself, so this module stays
    free of torch::

        verify_model_classes(model.names)

    Args:
        names: Either ultralytics' ``{index: name}`` mapping or a plain
            sequence of names in index order.

    Raises:
        ClassMismatchError: If the count differs or any index carries a
            different name. The message lists every disagreement, since a
            checkpoint that has drifted usually differs in more than one place.
    """
    if isinstance(names, Mapping):
        try:
            declared = [names[index] for index in sorted(names)]
        except (KeyError, TypeError) as error:
            raise ClassMismatchError(f"class-name mapping is not indexed by int: {error}") from None
    else:
        declared = list(names)

    expected = class_names()
    if declared == expected:
        return

    problems: list[str] = []
    if len(declared) != len(expected):
        problems.append(f"checkpoint has {len(declared)} classes, schema has {len(expected)}")
    for index in range(max(len(declared), len(expected))):
        left = declared[index] if index < len(declared) else "<missing>"
        right = expected[index] if index < len(expected) else "<missing>"
        if left != right:
            problems.append(f"index {index}: checkpoint {left!r} != schema {right!r}")

    listed = "\n  ".join(problems)
    raise ClassMismatchError(
        "checkpoint class list does not match melodix.vision.labels:\n  "
        f"{listed}\n"
        "Every detection would be mapped to the wrong symbol, and nothing "
        "downstream can detect this. Retrain against a data.yaml generated by "
        "melodix.vision.dataset.write_data_yaml, or pin the schema to the "
        "checkpoint."
    )


def labels_in_category(category: SymbolCategory) -> tuple[SymbolLabel, ...]:
    """Return every class in one category, ordered by id.

    Args:
        category: The category to filter by.

    Returns:
        Matching schema rows, possibly empty.
    """
    return tuple(label for label in LABELS if label.category is category)
