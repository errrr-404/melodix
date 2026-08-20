"""Unit tests for ``scripts/generate_synthetic_dataset.py``.

The generator is a script rather than a package module, so it is loaded here by
path. That is worth the small amount of ceremony: its ground truth is the
training signal for all of Stage 2, and a quiet error in it produces a dataset
that trains to a plausible loss and yields a detector that is systematically
wrong.

The rotation tests are the ones that matter. Augmentation tilts the page, and
if the boxes do not tilt with it the labels sit beside their symbols — which is
invisible in every summary statistic the generator prints.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from melodix.geometry.staff import StaffGrid, StaffLine
from melodix.vision.dataset import parse_label_file
from melodix.vision.labels import NUM_CLASSES, SymbolCategory, SymbolClass

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_synthetic_dataset.py"


def _load():
    """Import the generator script as a module."""
    spec = importlib.util.spec_from_file_location("generate_synthetic_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclass(slots=True) rebuilds the class and looks itself up in
    # sys.modules, so the module must be registered before it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load()


@pytest.fixture
def clean_page():
    """A deterministic clean render and its ground truth."""
    return gen.render_page(random.Random(5), gen.PageStyle())


def ink_fraction(page, placed, category=SymbolCategory.NOTEHEAD) -> float:
    """Mean fraction of dark pixels inside each box of one category.

    The measure that distinguishes aligned labels from misaligned ones: a box
    sitting on its notehead is mostly ink, one sitting beside it is mostly
    paper.
    """
    values = []
    for symbol in placed:
        if symbol.symbol.label.category is not category:
            continue
        x0, y0 = int(max(0, symbol.x_min)), int(max(0, symbol.y_min))
        x1 = int(min(page.shape[1], symbol.x_max))
        y1 = int(min(page.shape[0], symbol.y_max))
        if x1 <= x0 or y1 <= y0:
            continue
        values.append(float((page[y0:y1, x0:x1] < 128).mean()))
    return float(np.mean(values)) if values else 0.0


# --------------------------------------------------------------------------- #
# Agreement with Stage 1
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("position", [0, 1, 4, 5, 8])
def test_the_position_mapping_matches_the_staff_grid(position):
    """The generator places heads by staff position, and Stage 1 reads them
    back the same way. A disagreement would put every head half a space off.
    """
    top, spacing = 200.0, 20.0
    grid = StaffGrid(
        lines=tuple(
            StaffLine(y=top + step * spacing, x_start=100, x_end=700, thickness=2)
            for step in range(5)
        )
    )

    assert gen.position_to_y(top, spacing, position) == pytest.approx(
        grid.position_to_y(position)
    )


def test_the_bottom_line_is_position_zero():
    assert gen.position_to_y(100.0, 20.0, 0) == pytest.approx(180.0)


def test_the_top_line_is_position_eight():
    assert gen.position_to_y(100.0, 20.0, 8) == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_a_page_renders_at_the_configured_size():
    style = gen.PageStyle(width=640, height=900)

    page, _ = gen.render_page(random.Random(1), style)

    assert page.shape == (900, 640)
    assert page.dtype == np.uint8


def test_a_page_carries_symbols(clean_page):
    _, placed = clean_page

    assert len(placed) > 20


def test_rendering_is_deterministic():
    """A dataset must be reproducible from its seed alone."""
    first, first_placed = gen.render_page(random.Random(11), gen.PageStyle())
    second, second_placed = gen.render_page(random.Random(11), gen.PageStyle())

    assert np.array_equal(first, second)
    assert [s.symbol for s in first_placed] == [s.symbol for s in second_placed]


def test_different_seeds_render_different_pages():
    first, _ = gen.render_page(random.Random(1), gen.PageStyle())
    second, _ = gen.render_page(random.Random(2), gen.PageStyle())

    assert not np.array_equal(first, second)


def test_every_box_has_positive_area(clean_page):
    _, placed = clean_page

    assert all(s.x_max > s.x_min and s.y_max > s.y_min for s in placed)


def test_every_box_lands_on_ink(clean_page):
    """Ground truth is exact by construction; this catches a drawer whose
    reported box drifts from what it actually drew.
    """
    page, placed = clean_page

    assert ink_fraction(page, placed) > 0.3


def test_stems_are_not_labelled(clean_page):
    """Stems are not in the schema — Stage 1 finds vertical strokes."""
    _, placed = clean_page
    names = {s.symbol.name for s in placed}

    assert "STEM" not in names


def test_a_circled_cross_is_labelled_once():
    """It draws a cross internally; that inner box must not also be reported."""
    page = np.full((200, 200), 255, dtype=np.uint8)

    placed = gen.draw_circle_cross_head(page, 100, 100, 16.0)

    assert placed.symbol is SymbolClass.CIRCLE_CROSS_NOTEHEAD


def test_a_ghost_note_is_labelled_once():
    page = np.full((200, 200), 255, dtype=np.uint8)

    placed = gen.draw_ghost_note(page, 100, 100, 16.0)

    assert placed.symbol is SymbolClass.GHOST_NOTE


# --------------------------------------------------------------------------- #
# Carrying boxes through rotation
# --------------------------------------------------------------------------- #


def test_a_zero_rotation_leaves_boxes_alone(clean_page):
    _, placed = clean_page

    moved = gen.rotate_symbols(placed, 0.0, 1000, 1400)

    assert len(moved) == len(placed)
    assert moved[0].x_min == pytest.approx(placed[0].x_min, abs=0.01)


@pytest.mark.parametrize("angle", [-4.0, -2.0, 1.0, 2.0, 4.0])
def test_rotated_boxes_still_land_on_their_symbols(angle, clean_page):
    """The regression this file exists for. Rotating the page without
    rotating the boxes is invisible in every summary the generator prints.
    """
    from melodix.geometry.deskew import rotate_image

    page, placed = clean_page
    height, width = page.shape[:2]

    rotated_page = rotate_image(page, angle, border_value=255)
    moved = gen.rotate_symbols(placed, angle, width, height)

    assert ink_fraction(rotated_page, moved) > 0.4


@pytest.mark.parametrize("angle", [2.0, 4.0])
def test_leaving_boxes_behind_measurably_degrades_them(angle, clean_page):
    """Proves the check above can actually tell the two apart, rather than
    passing because the measure is insensitive.
    """
    from melodix.geometry.deskew import rotate_image

    page, placed = clean_page
    height, width = page.shape[:2]

    rotated_page = rotate_image(page, angle, border_value=255)
    carried = gen.rotate_symbols(placed, angle, width, height)

    assert ink_fraction(rotated_page, carried) > ink_fraction(rotated_page, placed) * 1.3


def test_rotation_uses_the_same_convention_as_deskew():
    """Both build the matrix about the image centre; a sign flip here would
    rotate the boxes the wrong way.
    """
    import cv2

    symbol = gen.PlacedSymbol(SymbolClass.ROUND_NOTEHEAD, 100.0, 100.0, 120.0, 120.0)
    matrix = cv2.getRotationMatrix2D((500.0, 500.0), 3.0, 1.0)
    expected = np.array([110.0, 110.0, 1.0]) @ matrix.T

    moved = gen.rotate_symbols([symbol], 3.0, 1000, 1000)[0]

    assert (moved.x_min + moved.x_max) / 2 == pytest.approx(expected[0], abs=0.5)
    assert (moved.y_min + moved.y_max) / 2 == pytest.approx(expected[1], abs=0.5)


def test_a_symbol_rotated_off_the_page_is_dropped():
    corner = gen.PlacedSymbol(SymbolClass.ROUND_NOTEHEAD, 0.0, 0.0, 6.0, 6.0)

    assert gen.rotate_symbols([corner], 40.0, 1000, 1400) == []


def test_rotated_boxes_stay_inside_the_page(clean_page):
    _, placed = clean_page

    moved = gen.rotate_symbols(placed, 3.0, 1000, 1400)

    assert all(s.x_min >= 0 and s.x_max <= 1000 for s in moved)
    assert all(s.y_min >= 0 and s.y_max <= 1400 for s in moved)


# --------------------------------------------------------------------------- #
# Scan simulation
# --------------------------------------------------------------------------- #


def test_augmentation_returns_boxes_alongside_the_page(clean_page):
    page, placed = clean_page

    out, moved, angle = gen.augment(page.copy(), placed, random.Random(3))

    assert out.shape == page.shape
    assert len(moved) <= len(placed)
    assert -2.0 <= angle <= 2.0


def test_augmentation_changes_the_page(clean_page):
    page, placed = clean_page

    out, _, _ = gen.augment(page.copy(), placed, random.Random(3))

    assert not np.array_equal(out, page)


def test_augmentation_is_deterministic(clean_page):
    page, placed = clean_page

    first, _, _ = gen.augment(page.copy(), placed, random.Random(7))
    second, _, _ = gen.augment(page.copy(), placed, random.Random(7))

    assert np.array_equal(first, second)


# Seed 2 draws a tilt near the top of the augmentation range. A seed drawing a
# fraction of a degree cannot tell a carried box from an abandoned one.
LARGE_TILT_SEED = 2


def test_the_chosen_seed_really_does_tilt_the_page():
    """Guards the test below: if the seed stopped producing a large angle,
    the wiring test would start passing for the wrong reason.
    """
    page, placed = gen.render_page(random.Random(5), gen.PageStyle())

    _, _, angle = gen.augment(page, placed, random.Random(LARGE_TILT_SEED))

    assert abs(angle) > 1.5


def test_augmented_labels_still_land_on_ink(clean_page):
    page, placed = clean_page

    out, moved, _ = gen.augment(page.copy(), placed, random.Random(LARGE_TILT_SEED))

    assert ink_fraction(out, moved) > 0.35


def test_augmentation_carries_the_boxes_through_its_own_rotation(clean_page):
    """The wiring, not just the helper. Rotating the page while leaving the
    boxes behind passes every other check in this file.
    """
    page, placed = clean_page

    out, moved, _ = gen.augment(page.copy(), placed, random.Random(LARGE_TILT_SEED))

    assert ink_fraction(out, moved) > ink_fraction(out, placed) * 1.3


# --------------------------------------------------------------------------- #
# Annotation conversion
# --------------------------------------------------------------------------- #


def test_a_placed_symbol_normalises_to_the_page():
    symbol = gen.PlacedSymbol(SymbolClass.ACCENT, 100.0, 200.0, 140.0, 230.0)

    annotation = symbol.to_annotation(800, 600)

    assert annotation.symbol is SymbolClass.ACCENT
    assert annotation.box.cx == pytest.approx(120 / 800)
    assert annotation.box.cy == pytest.approx(215 / 600)


def test_a_box_spilling_off_the_page_is_clamped():
    symbol = gen.PlacedSymbol(SymbolClass.ACCENT, -10.0, -5.0, 40.0, 30.0)

    annotation = symbol.to_annotation(800, 600)

    assert annotation.box.x_min >= 0.0
    assert annotation.box.y_min >= 0.0


def test_every_rendered_symbol_converts(clean_page):
    page, placed = clean_page
    height, width = page.shape[:2]

    converted = [symbol.to_annotation(width, height) for symbol in placed]

    assert len(converted) == len(placed)


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


def test_a_dataset_is_written_in_the_ultralytics_layout(tmp_path):
    gen.generate(tmp_path, pages=4, seed=1, val_fraction=0.25)

    assert (tmp_path / "data.yaml").exists()
    assert list((tmp_path / "images" / "train").glob("*.png"))
    assert list((tmp_path / "labels" / "train").glob("*.txt"))


def test_every_image_has_a_matching_label_file(tmp_path):
    gen.generate(tmp_path, pages=6, seed=2, val_fraction=0.34)

    for group in ("train", "val"):
        images = {p.stem for p in (tmp_path / "images" / group).glob("*.png")}
        labels = {p.stem for p in (tmp_path / "labels" / group).glob("*.txt")}
        assert images == labels


def test_the_summary_counts_what_was_written(tmp_path):
    summary = gen.generate(tmp_path, pages=5, seed=1, val_fraction=0.2)

    written = len(list((tmp_path / "images").rglob("*.png")))
    assert summary["pages"] == written == 5
    assert summary["train"] + summary["val"] == 5


def test_written_labels_parse_back(tmp_path):
    """Round trip through the same reader training will use."""
    gen.generate(tmp_path, pages=3, seed=1, val_fraction=0.34)

    for path in (tmp_path / "labels").rglob("*.txt"):
        assert parse_label_file(path) is not None


def test_generation_is_reproducible_from_its_seed(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"

    gen.generate(first, pages=3, seed=42, val_fraction=0.34)
    gen.generate(second, pages=3, seed=42, val_fraction=0.34)

    for path in (first / "labels").rglob("*.txt"):
        twin = second / path.relative_to(first)
        assert path.read_text() == twin.read_text()


def test_a_clean_run_skips_augmentation(tmp_path):
    summary = gen.generate(tmp_path, pages=2, seed=1, val_fraction=0.0, clean=True)

    assert summary["pages"] == 2


def test_the_dataset_covers_every_class(tmp_path):
    """A class with no examples cannot be learned, and the schema promises 28."""
    summary = gen.generate(tmp_path, pages=12, seed=1, val_fraction=0.2)

    assert summary["classes_total"] == NUM_CLASSES
    assert summary["classes_covered"] == NUM_CLASSES


def test_the_data_yaml_matches_the_schema(tmp_path):
    gen.generate(tmp_path, pages=2, seed=1, val_fraction=0.0)

    body = (tmp_path / "data.yaml").read_text(encoding="utf-8")
    assert f"nc: {NUM_CLASSES}" in body
    assert "0: round_notehead" in body


# --------------------------------------------------------------------------- #
# LilyPond source
# --------------------------------------------------------------------------- #


def test_lilypond_source_declares_a_version():
    assert "\\version" in gen.emit_lilypond_source()


def test_lilypond_source_uses_a_drum_staff():
    source = gen.emit_lilypond_source(measures=2)

    assert "DrumStaff" in source
    assert "drummode" in source


def test_lilypond_source_is_deterministic():
    assert gen.emit_lilypond_source(seed=3) == gen.emit_lilypond_source(seed=3)


def test_more_measures_make_a_longer_source():
    short = gen.emit_lilypond_source(measures=1)
    long = gen.emit_lilypond_source(measures=8)

    assert len(long) > len(short)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_the_cli_builds_a_dataset(tmp_path):
    code = gen.main(["--out", str(tmp_path), "--pages", "2", "--seed", "1"])

    assert code == 0
    assert (tmp_path / "data.yaml").exists()


def test_the_cli_can_write_a_preview(tmp_path):
    gen.main(["--out", str(tmp_path), "--pages", "1", "--seed", "1", "--preview"])

    assert (tmp_path / "preview_boxes.png").exists()


def test_the_cli_can_emit_lilypond_and_stop(tmp_path):
    target = tmp_path / "pattern.ly"

    code = gen.main(["--out", str(tmp_path), "--lilypond", str(target)])

    assert code == 0
    assert "DrumStaff" in target.read_text(encoding="utf-8")
    assert not (tmp_path / "data.yaml").exists()
