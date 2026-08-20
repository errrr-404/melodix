"""Unit tests for :mod:`melodix.vision.labels`.

The schema is the contract between a trained checkpoint and Stage 3, so the
tests that matter most here are the boring structural ones. A reordered
:class:`SymbolClass` breaks nothing at import time and raises no exception at
inference time — the detector simply starts reporting the wrong drum. The
pinned mapping below is what turns that silent corruption into a red test.
"""

from __future__ import annotations

import pytest

from melodix.vision.labels import (
    LABELS,
    NUM_CLASSES,
    SymbolCategory,
    SymbolClass,
    SymbolLabel,
    class_names,
    label_for_id,
    label_for_name,
    labels_in_category,
)

# The published class ids. Append to this list when adding a class; never
# reorder it. Changing a line here invalidates every checkpoint and every
# annotated page already on disk.
PINNED_SCHEMA = [
    (0, "round_notehead"),
    (1, "hollow_notehead"),
    (2, "cross_notehead"),
    (3, "circle_cross_notehead"),
    (4, "diamond_notehead"),
    (5, "triangle_notehead"),
    (6, "slash_notehead"),
    (7, "rest_whole"),
    (8, "rest_half"),
    (9, "rest_quarter"),
    (10, "rest_eighth"),
    (11, "rest_sixteenth"),
    (12, "flag_eighth"),
    (13, "flag_sixteenth"),
    (14, "beam"),
    (15, "augmentation_dot"),
    (16, "accent"),
    (17, "marcato"),
    (18, "staccato"),
    (19, "ghost_note"),
    (20, "open_modifier"),
    (21, "closed_modifier"),
    (22, "grace_note"),
    (23, "percussion_clef"),
    (24, "time_signature"),
    (25, "repeat_dots"),
    (26, "repeat_measure"),
    (27, "tie_slur"),
]


# --------------------------------------------------------------------------- #
# The pinned schema
# --------------------------------------------------------------------------- #


def test_the_schema_matches_the_published_class_ids():
    """If this fails, every existing checkpoint and label file is now wrong."""
    assert [(label.class_id, label.name) for label in LABELS] == PINNED_SCHEMA


def test_the_class_count_matches_the_schema():
    assert len(PINNED_SCHEMA) == NUM_CLASSES


def test_ids_run_contiguously_from_zero():
    """YOLO indexes its output head positionally; a gap would shift every
    class above it.
    """
    assert [label.class_id for label in LABELS] == list(range(NUM_CLASSES))


def test_labels_are_ordered_by_id():
    assert tuple(sorted(LABELS, key=lambda label: label.class_id)) == LABELS


def test_ids_are_unique():
    assert len({label.class_id for label in LABELS}) == NUM_CLASSES


def test_names_are_unique():
    assert len({label.name for label in LABELS}) == NUM_CLASSES


def test_every_enum_member_has_a_schema_row():
    assert {label.symbol for label in LABELS} == set(SymbolClass)


def test_the_registry_is_indexed_by_class_id():
    for label in LABELS:
        assert LABELS[label.class_id] is label


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", LABELS, ids=lambda label: label.name)
def test_names_are_lowercase_snake_case(label):
    assert label.name.replace("_", "").isalnum()
    assert label.name == label.name.lower()
    assert not label.name.startswith("_")
    assert not label.name.endswith("_")


@pytest.mark.parametrize("label", LABELS, ids=lambda label: label.name)
def test_the_name_derives_from_the_enum_member(label):
    """One source of truth: no separate name table to drift out of sync."""
    assert label.name == label.symbol.name.lower()


@pytest.mark.parametrize("label", LABELS, ids=lambda label: label.name)
def test_every_class_carries_a_description(label):
    """Annotators read these; an empty one produces inconsistent boxes."""
    assert label.description.strip()


# --------------------------------------------------------------------------- #
# Shapes, not voices
# --------------------------------------------------------------------------- #


def test_no_class_name_mentions_a_drum_voice():
    """The design rule, asserted mechanically. A class named for a drum has
    folded position into the label and will need a retrain to re-voice.
    """
    voices = ("snare", "kick", "hihat", "hi_hat", "tom", "ride", "crash", "cymbal", "bell")
    offenders = [label.name for label in LABELS if any(v in label.name for v in voices)]

    assert offenders == []


def test_no_class_name_mentions_a_staff_position():
    positions = ("line", "space", "position", "ledger", "above", "below")
    offenders = [label.name for label in LABELS if any(p in label.name for p in positions)]

    assert offenders == []


def test_notehead_shapes_are_distinguished_not_their_placement():
    """Cross versus round is a shape difference and belongs in the schema."""
    names = {label.name for label in labels_in_category(SymbolCategory.NOTEHEAD)}

    assert {"round_notehead", "cross_notehead"} <= names


# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", LABELS, ids=lambda label: label.name)
def test_every_class_has_a_known_category(label):
    assert label.category in set(SymbolCategory)


def test_every_category_is_populated():
    """An empty category means a planned group was never filled in."""
    empty = [c.value for c in SymbolCategory if not labels_in_category(c)]

    assert empty == []


def test_categories_partition_the_schema():
    counted = sum(len(labels_in_category(category)) for category in SymbolCategory)

    assert counted == NUM_CLASSES


def test_labels_in_a_category_are_ordered_by_id():
    rests = labels_in_category(SymbolCategory.REST)

    assert [label.class_id for label in rests] == sorted(label.class_id for label in rests)


def test_the_rest_category_covers_the_common_durations():
    names = {label.name for label in labels_in_category(SymbolCategory.REST)}

    assert {"rest_whole", "rest_half", "rest_quarter", "rest_eighth"} <= names


def test_a_category_with_no_members_returns_empty_not_an_error():
    """Guards the filter itself, using a category that is currently populated
    only so the call shape is exercised.
    """
    assert isinstance(labels_in_category(SymbolCategory.STRUCTURE), tuple)


# --------------------------------------------------------------------------- #
# Downstream behaviour flags
# --------------------------------------------------------------------------- #


def test_only_noteheads_carry_a_staff_position():
    """Stage 3 calls grid.snap() for exactly these."""
    carriers = {label.name for label in LABELS if label.carries_position}
    noteheads = {label.name for label in labels_in_category(SymbolCategory.NOTEHEAD)}

    assert carriers == noteheads


def test_a_rest_does_not_carry_a_position():
    """Its vertical placement is engraving convention, not pitch."""
    assert not label_for_name("rest_quarter").carries_position


def test_a_modifier_does_not_carry_a_position():
    assert not label_for_name("accent").carries_position


def test_only_modifiers_attach_to_a_notehead():
    attaching = {label.name for label in LABELS if label.attaches_to_notehead}
    modifiers = {label.name for label in labels_in_category(SymbolCategory.MODIFIER)}

    assert attaching == modifiers


def test_a_notehead_does_not_attach_to_another_notehead():
    assert not label_for_name("round_notehead").attaches_to_notehead


def test_no_class_both_carries_a_position_and_attaches():
    """The two flags select disjoint sets; a symbol cannot be both."""
    both = [label.name for label in LABELS if label.carries_position and label.attaches_to_notehead]

    assert both == []


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("class_id", "name"), PINNED_SCHEMA)
def test_lookup_by_id_returns_the_right_class(class_id, name):
    assert label_for_id(class_id).name == name


@pytest.mark.parametrize(("class_id", "name"), PINNED_SCHEMA)
def test_lookup_by_name_returns_the_right_id(class_id, name):
    assert label_for_name(name).class_id == class_id


def test_lookup_by_name_is_case_insensitive():
    assert label_for_name("CROSS_NOTEHEAD") is label_for_name("cross_notehead")


@pytest.mark.parametrize("class_id", [-1, NUM_CLASSES, 999])
def test_an_unknown_id_is_rejected(class_id):
    """A checkpoint emitting an id outside the schema has diverged from it;
    returning None would push a silent failure downstream.
    """
    with pytest.raises(KeyError, match="no symbol class with id"):
        label_for_id(class_id)


def test_an_unknown_name_is_rejected():
    with pytest.raises(KeyError, match="no symbol class named"):
        label_for_name("cowbell")


def test_the_enum_reaches_its_own_label():
    assert SymbolClass.ACCENT.label.name == "accent"
    assert SymbolClass.ACCENT.label.category is SymbolCategory.MODIFIER


def test_a_class_is_its_own_id():
    """IntEnum, so it can be written straight into a label file."""
    assert int(SymbolClass.CROSS_NOTEHEAD) == 2
    assert SymbolClass.CROSS_NOTEHEAD == 2


# --------------------------------------------------------------------------- #
# data.yaml ordering
# --------------------------------------------------------------------------- #


def test_class_names_are_ordered_by_id():
    """Ultralytics reads this list positionally, so order is the id
    assignment. Sorting it alphabetically would silently remap every class.
    """
    assert class_names() == [name for _, name in PINNED_SCHEMA]


def test_class_names_are_not_alphabetical():
    """A guard against someone helpfully sorting them."""
    assert class_names() != sorted(class_names())


def test_class_names_length_matches_the_class_count():
    assert len(class_names()) == NUM_CLASSES


def test_class_names_returns_a_fresh_list():
    """Callers hand this to ultralytics; a shared list could be mutated."""
    first = class_names()
    first.append("bogus")

    assert len(class_names()) == NUM_CLASSES


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #


def test_a_label_cannot_be_mutated():
    with pytest.raises(AttributeError):
        LABELS[0].description = "changed"  # type: ignore[misc]


def test_the_registry_is_a_tuple():
    assert isinstance(LABELS, tuple)


def test_a_label_is_hashable():
    """So schema rows can key a lookup table in Stage 3."""
    assert len({LABELS[0], LABELS[0], LABELS[1]}) == 2


def test_the_category_is_a_string_enum():
    """So it serialises into a sync map without conversion."""
    assert SymbolCategory.NOTEHEAD == "notehead"
    assert isinstance(SymbolLabel(SymbolClass.ACCENT, SymbolCategory.MODIFIER, "x").category, str)
