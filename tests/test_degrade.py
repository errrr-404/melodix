"""Unit tests for ``scripts/degrade.py``.

The load-bearing tests here are the geometric ones. A degraded page carrying
stale labels trains a model to place boxes slightly beside symbols, converges to
a perfectly plausible loss, and yields a detector that is confidently and
systematically wrong. The generator shipped exactly that bug once, so the checks
below measure *ink inside the box* rather than trusting that a transform was
called — a box that stopped tracking its glyph shows up as paper.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from melodix.vision.dataset import Annotation, BoundingBox, parse_label_file, write_label_file
from melodix.vision.labels import SymbolClass

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "degrade.py"


def _load():
    """Import the degradation script as a module."""
    spec = importlib.util.spec_from_file_location("degrade", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


deg = _load()

PAGE_W = 600
PAGE_H = 800


def page_with_blobs(centres=((150, 200), (300, 400), (450, 600)), radius=18):
    """A white page carrying solid black discs, and exact boxes for them.

    Discs rather than glyphs: a filled shape makes "is the box still on the
    symbol" a clean measurement, which is the property under test.
    """
    image = np.full((PAGE_H, PAGE_W), 255, dtype=np.uint8)
    boxes = []
    for x, y in centres:
        cv2.circle(image, (x, y), radius, 0, -1)
        boxes.append(
            Annotation(
                symbol=SymbolClass.ROUND_NOTEHEAD,
                box=BoundingBox.from_pixels(
                    x - radius, y - radius, x + radius, y + radius, PAGE_W, PAGE_H
                ),
            )
        )
    return image, tuple(boxes)


def ink_in_boxes(image, boxes) -> float:
    """Mean fraction of dark pixels inside each box.

    On the fixture above a correctly placed box is roughly pi/4 ink. A box that
    has drifted off its disc drops sharply toward zero.
    """
    values = []
    height, width = image.shape[:2]
    for annotation in boxes:
        x0, y0, x1, y1 = annotation.box.to_pixels(width, height)
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(width, x1)), int(min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        values.append(float((image[y0:y1, x0:x1] < 128).mean()))
    return float(np.mean(values)) if values else 0.0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_every_named_effect_is_known():
    assert set(deg.EFFECT_ORDER) >= deg.GEOMETRIC_EFFECTS


def test_the_geometric_effects_are_the_ones_that_move_pixels():
    """Only these two need to carry boxes; the rest change what a pixel is,
    not where it is.
    """
    assert {"rotate", "perspective"} == deg.GEOMETRIC_EFFECTS


def test_an_unknown_effect_is_rejected():
    with pytest.raises(ValueError, match="unknown effect"):
        deg.DegradationConfig(enabled=frozenset({"sepia"}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rotate_degrees", -1.0),
        ("perspective_strength", -0.1),
        ("perspective_strength", 0.9),
        ("speckle_density", 1.5),
    ],
)
def test_invalid_config_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        deg.DegradationConfig(**{field: value})


def test_an_inverted_range_is_rejected():
    with pytest.raises(ValueError, match="blur_sigma is inverted"):
        deg.DegradationConfig(blur_sigma=(2.0, 0.5))


def test_effects_can_be_disabled():
    config = deg.DegradationConfig().without("jpeg", "blur")

    assert "jpeg" not in config.enabled
    assert "rotate" in config.enabled


def test_effects_can_be_narrowed_to_one():
    config = deg.DegradationConfig().only("rotate")

    assert config.enabled == {"rotate"}


# --------------------------------------------------------------------------- #
# Boxes travel with geometry
# --------------------------------------------------------------------------- #


def test_rotation_keeps_boxes_on_their_glyphs():
    """The hard requirement. Rotate a page with known boxes and assert each box
    still bounds the same disc.
    """
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig().only("rotate")

    out, moved = deg.degrade(image, boxes, config, seed=3)

    assert ink_in_boxes(out, moved) > 0.6


def test_leaving_boxes_behind_after_rotation_is_measurably_worse():
    """Proves the check above can tell the two apart, rather than passing
    because the measure is insensitive.
    """
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig(rotate_degrees=8.0).only("rotate")

    out, moved = deg.degrade(image, boxes, config, seed=3)

    assert ink_in_boxes(out, moved) > ink_in_boxes(out, boxes) * 1.2


def test_perspective_keeps_boxes_on_their_glyphs():
    """Run at a strength where the check can actually fail.

    At the default perspective_strength (0.012) the corner displacement is so
    small that a box still covers its disc even when it is not transformed at
    all, so this assertion would pass whether or not the code works. Mutation
    testing caught exactly that.
    """
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig(perspective_strength=0.05).only("perspective")

    out, moved = deg.degrade(image, boxes, config, seed=5)

    assert ink_in_boxes(out, moved) > 0.6


def test_the_default_perspective_is_deliberately_mild():
    """Nothing in the pipeline corrects perspective — Stage 1 deskews rotation
    only — so a heavily warped page is one the pipeline cannot straighten.
    """
    assert deg.DegradationConfig().perspective_strength <= 0.02


def test_leaving_boxes_behind_after_perspective_is_measurably_worse():
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig(perspective_strength=0.06).only("perspective")

    out, moved = deg.degrade(image, boxes, config, seed=5)

    assert ink_in_boxes(out, moved) > ink_in_boxes(out, boxes) * 1.2


def test_both_geometric_effects_together_keep_boxes_on_glyphs():
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig().only("rotate", "perspective")

    out, moved = deg.degrade(image, boxes, config, seed=11)

    assert ink_in_boxes(out, moved) > 0.55


def test_the_full_pipeline_keeps_boxes_on_glyphs():
    """Every effect at once, which is how the CLI runs it."""
    image, boxes = page_with_blobs()

    out, moved = deg.degrade(image, boxes, deg.DegradationConfig(), seed=17)

    assert ink_in_boxes(out, moved) > 0.5


def test_zero_rotation_leaves_boxes_untouched():
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig(rotate_degrees=0.0).only("rotate")

    _, moved = deg.degrade(image, boxes, config, seed=1)

    assert moved == boxes


def test_a_box_rotated_off_the_page_is_dropped():
    image = np.full((PAGE_H, PAGE_W), 255, dtype=np.uint8)
    corner = (
        Annotation(
            symbol=SymbolClass.ROUND_NOTEHEAD,
            box=BoundingBox.from_pixels(0, 0, 6, 6, PAGE_W, PAGE_H),
        ),
    )
    config = deg.DegradationConfig(rotate_degrees=45.0).only("rotate")

    _, moved = deg.degrade(image, corner, config, seed=2)

    assert moved == ()


def test_moved_boxes_stay_inside_the_page():
    image, boxes = page_with_blobs()

    _, moved = deg.degrade(image, boxes, deg.DegradationConfig(), seed=4)

    for annotation in moved:
        assert annotation.box.x_min >= 0.0
        assert annotation.box.x_max <= 1.0
        assert annotation.box.y_min >= 0.0
        assert annotation.box.y_max <= 1.0


# --------------------------------------------------------------------------- #
# Photometric effects leave boxes alone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "effect", ["jpeg", "blur", "brightness", "gamma", "texture", "speckle", "ink"]
)
def test_a_photometric_effect_does_not_move_boxes(effect):
    """These change what a pixel is, not where it is."""
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig().only(effect)

    _, moved = deg.degrade(image, boxes, config, seed=6)

    assert moved == boxes


@pytest.mark.parametrize(
    "effect", ["jpeg", "blur", "brightness", "gamma", "texture", "speckle"]
)
def test_a_photometric_effect_changes_the_page(effect):
    """A no-op effect would silently contribute nothing to the corpus.

    Measured on a page carrying midtones. See the two tests below for why that
    matters: on pure black and white, two of these effects genuinely cannot do
    anything.
    """
    image, boxes = page_with_blobs()
    image = cv2.GaussianBlur(image, (0, 0), 1.2)  # give it a grey ramp to act on
    config = deg.DegradationConfig().only(effect)

    out, _ = deg.degrade(image, boxes, config, seed=6)

    assert not np.array_equal(out, image)


def test_gamma_cannot_alter_a_pure_bilevel_page():
    """Worth recording rather than working around: a gamma curve fixes both
    endpoints, so 0 stays 0 and 255 stays 255. Gamma contributes nothing until
    something has introduced midtones — which is why EFFECT_ORDER runs blur and
    texture before it.
    """
    image, boxes = page_with_blobs()
    assert set(np.unique(image)) == {0, 255}

    out, _ = deg.degrade(image, boxes, deg.DegradationConfig().only("gamma"), seed=6)

    assert np.array_equal(out, image)


def test_the_effect_order_creates_midtones_before_using_them():
    """The ordering that makes gamma and brightness meaningful."""
    order = list(deg.EFFECT_ORDER)

    assert order.index("blur") < order.index("gamma")
    assert order.index("texture") < order.index("brightness")
    assert order.index("jpeg") == len(order) - 1  # last, as on a real scan


def test_every_effect_preserves_shape_and_dtype():
    image, boxes = page_with_blobs()

    for effect in deg.EFFECT_ORDER:
        out, _ = deg.degrade(image, boxes, deg.DegradationConfig().only(effect), seed=8)
        assert out.shape == image.shape, effect
        assert out.dtype == np.uint8, effect


def test_jpeg_leaves_compression_artefacts():
    """A JPEG round trip must not be lossless, or it is contributing nothing."""
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig(jpeg_quality=(20, 20)).only("jpeg")

    out, _ = deg.degrade(image, boxes, config, seed=1)

    assert not np.array_equal(out, image)


def test_heavier_ink_darkens_the_page():
    image, boxes = page_with_blobs()
    heavier = deg.DegradationConfig(ink_range=(1, 1)).only("ink")

    out, _ = deg.degrade(image, boxes, heavier, seed=1)

    assert (out < 128).sum() > (image < 128).sum()


def test_fainter_ink_lightens_the_page():
    image, boxes = page_with_blobs()
    fainter = deg.DegradationConfig(ink_range=(-1, -1)).only("ink")

    out, _ = deg.degrade(image, boxes, fainter, seed=1)

    assert (out < 128).sum() < (image < 128).sum()


def test_neutral_ink_is_a_no_op():
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig(ink_range=(0, 0)).only("ink")

    out, _ = deg.degrade(image, boxes, config, seed=1)

    assert np.array_equal(out, image)


def test_a_colour_page_survives_the_pipeline():
    image, boxes = page_with_blobs()
    colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    out, _ = deg.degrade(colour, boxes, deg.DegradationConfig(), seed=2)

    assert out.shape == colour.shape


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_the_same_seed_reproduces_the_page():
    """A dataset must be regenerable exactly, or two runs are incomparable."""
    image, boxes = page_with_blobs()

    first, first_boxes = deg.degrade(image, boxes, seed=42)
    second, second_boxes = deg.degrade(image, boxes, seed=42)

    assert np.array_equal(first, second)
    assert first_boxes == second_boxes


def test_a_different_seed_produces_a_different_page():
    image, boxes = page_with_blobs()

    first, _ = deg.degrade(image, boxes, seed=1)
    second, _ = deg.degrade(image, boxes, seed=2)

    assert not np.array_equal(first, second)


@pytest.mark.parametrize("effect", list(deg.EFFECT_ORDER))
def test_each_effect_is_individually_reproducible(effect):
    image, boxes = page_with_blobs()
    config = deg.DegradationConfig().only(effect)

    first, _ = deg.degrade(image, boxes, config, seed=9)
    second, _ = deg.degrade(image, boxes, config, seed=9)

    assert np.array_equal(first, second)


def test_degrading_does_not_mutate_the_input():
    image, boxes = page_with_blobs()
    before = image.copy()

    deg.degrade(image, boxes, deg.DegradationConfig(), seed=3)

    assert np.array_equal(image, before)


# --------------------------------------------------------------------------- #
# Walking a dataset
# --------------------------------------------------------------------------- #


def build_dataset(root: Path, train: int = 3, val: int = 2) -> Path:
    """Write a small YOLO dataset of blob pages."""
    for split, count in (("train", train), ("val", val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image, boxes = page_with_blobs()
            cv2.imwrite(str(root / "images" / split / f"page_{index}.png"), image)
            write_label_file(root / "labels" / split / f"page_{index}.txt", boxes)
    (root / "data.yaml").write_text(
        f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )
    return root


def test_a_dataset_is_copied_with_its_layout(tmp_path):
    source = build_dataset(tmp_path / "src")
    destination = tmp_path / "out"

    counts = deg.degrade_dataset(source, destination, seed=1)

    assert counts == {"train": 3, "val": 2}
    assert len(list((destination / "images" / "train").glob("*.png"))) == 3
    assert len(list((destination / "labels" / "val").glob("*.txt"))) == 2


def test_every_copied_image_keeps_a_label_file(tmp_path):
    source = build_dataset(tmp_path / "src")
    destination = tmp_path / "out"

    deg.degrade_dataset(source, destination, seed=1)

    for split in ("train", "val"):
        images = {p.stem for p in (destination / "images" / split).glob("*.png")}
        labels = {p.stem for p in (destination / "labels" / split).glob("*.txt")}
        assert images == labels


def test_copied_labels_still_land_on_their_glyphs(tmp_path):
    """End to end through the CLI path, not just the in-memory function."""
    source = build_dataset(tmp_path / "src", train=2, val=0)
    destination = tmp_path / "out"

    deg.degrade_dataset(source, destination, seed=1, splits=("train",))

    path = next((destination / "images" / "train").glob("*.png"))
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    boxes = parse_label_file(destination / "labels" / "train" / f"{path.stem}.txt")

    assert ink_in_boxes(image, boxes) > 0.5


def test_the_descriptor_is_repointed_at_the_copy(tmp_path):
    source = build_dataset(tmp_path / "src")
    destination = tmp_path / "out"

    deg.degrade_dataset(source, destination, seed=1)

    body = (destination / "data.yaml").read_text(encoding="utf-8")
    assert destination.resolve().as_posix() in body
    assert source.resolve().as_posix() not in body


def test_an_unlabelled_image_stops_the_walk(tmp_path):
    """Degrading a page with no ground truth silently adds a page of nothing."""
    source = build_dataset(tmp_path / "src")
    (source / "labels" / "train" / "page_0.txt").unlink()

    with pytest.raises(FileNotFoundError, match="have no label file"):
        deg.degrade_dataset(source, tmp_path / "out", seed=1)


def test_a_missing_split_is_reported(tmp_path):
    source = build_dataset(tmp_path / "src")

    with pytest.raises(FileNotFoundError, match="no test images"):
        deg.degrade_dataset(source, tmp_path / "out", seed=1, splits=("test",))


def test_the_walk_is_reproducible(tmp_path):
    source = build_dataset(tmp_path / "src", train=2, val=0)

    deg.degrade_dataset(source, tmp_path / "a", seed=5, splits=("train",))
    deg.degrade_dataset(source, tmp_path / "b", seed=5, splits=("train",))

    for path in (tmp_path / "a" / "labels" / "train").glob("*.txt"):
        twin = tmp_path / "b" / "labels" / "train" / path.name
        assert path.read_text() == twin.read_text()


def test_pages_are_degraded_differently_from_one_another(tmp_path):
    """A per-page derived seed, so every page is not the same tilt."""
    source = build_dataset(tmp_path / "src", train=3, val=0)
    destination = tmp_path / "out"

    deg.degrade_dataset(source, destination, seed=1, splits=("train",))

    bodies = {
        path.read_text() for path in (destination / "labels" / "train").glob("*.txt")
    }
    assert len(bodies) > 1


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_the_cli_degrades_a_dataset(tmp_path, capsys):
    source = build_dataset(tmp_path / "src", train=2, val=1)

    code = deg.main(["--in", str(source), "--out", str(tmp_path / "out"), "--seed", "3"])

    assert code == 0
    assert "train: 2" in capsys.readouterr().out


def test_the_cli_can_disable_an_effect(tmp_path, capsys):
    source = build_dataset(tmp_path / "src", train=1, val=1)

    deg.main(
        ["--in", str(source), "--out", str(tmp_path / "out"), "--disable", "jpeg", "blur"]
    )

    effects = capsys.readouterr().out
    assert "jpeg" not in effects.split("effects:")[1]


def test_the_cli_can_select_one_effect(tmp_path, capsys):
    source = build_dataset(tmp_path / "src", train=1, val=1)

    deg.main(["--in", str(source), "--out", str(tmp_path / "out"), "--only", "rotate"])

    line = capsys.readouterr().out.split("effects:")[1].strip()
    assert line == "rotate"
