"""Scan-realistic degradation for synthetic pages.

The cheapest lever available on the synthetic-to-real gap. Annotating real pages
costs hours per page; degrading a synthetic one costs milliseconds and reuses
ground truth that is already exact. It does not close the gap — the glyphs are
still procedurally drawn — but it removes the easiest tell, which is that a
clean render has no noise, no compression, no tilt and perfectly uniform ink.

Boxes travel with the image
---------------------------
Every geometric effect here transforms the ground-truth boxes with the same
matrix it applies to the pixels. This is the whole correctness burden of the
module. A degraded page carrying stale labels is worse than no data at all: it
trains the model to put boxes slightly beside symbols, converges to a perfectly
plausible loss, and produces a detector that is confidently and systematically
off. The generator had exactly this bug once already.

Non-geometric effects (compression, blur, brightness, noise, ink weight) leave
boxes alone, correctly — they change what a pixel is, not where it is.

Reproducibility
---------------
Every effect draws from a seeded :class:`random.Random`. The same seed and
config reproduce a dataset byte for byte, which is what makes a training run
comparable to the one before it.

Usage::

    python scripts/degrade.py --in datasets/melodix_synth --out datasets/melodix_degraded
    python scripts/degrade.py --in a --out b --seed 7 --disable perspective jpeg
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from melodix.vision.dataset import (  # noqa: E402
    IMAGE_SUFFIXES,
    Annotation,
    BoundingBox,
    find_unlabelled_images,
    parse_label_file,
    write_label_file,
)

Image = npt.NDArray[np.uint8]
"""A grayscale or colour ``uint8`` page."""

Boxes = tuple[Annotation, ...]
"""Ground truth travelling alongside a page."""

#: Effects that move pixels around, and therefore must move boxes too.
GEOMETRIC_EFFECTS: frozenset[str] = frozenset({"rotate", "perspective"})

#: Every effect this module can apply, in the order it applies them.
EFFECT_ORDER: tuple[str, ...] = (
    "ink",
    "rotate",
    "perspective",
    "blur",
    "texture",
    "speckle",
    "brightness",
    "gamma",
    "jpeg",
)


@dataclass(frozen=True, slots=True)
class DegradationConfig:
    """Which effects to apply and how hard.

    Each effect carries a probability and a strength range. A probability of
    0.0 disables the effect entirely, which is what ``--disable`` sets.

    Ordering is fixed by :data:`EFFECT_ORDER` and is not arbitrary: ink weight
    is applied to clean geometry before anything blurs it, geometry moves
    before optics, and JPEG runs last because on a real scan it is the last
    thing to touch the image.

    Attributes:
        enabled: Effect names to consider. Anything absent is skipped.
        rotate_degrees: Maximum absolute tilt. Kept small because Stage 1
            deskews before detection, so this exercises the real path instead
            of breaking it.
        perspective_strength: Corner displacement as a fraction of page size.
        blur_sigma: Gaussian blur sigma range.
        jpeg_quality: Compression quality range, low being worse.
        brightness_delta: Additive shift range, in 0-255 units.
        contrast_scale: Multiplicative contrast range around the midpoint.
        gamma_range: Gamma exponent range. Below 1 lightens, above darkens.
        texture_strength: Amplitude of the low-frequency paper mottling.
        speckle_density: Fraction of pixels hit by salt-and-pepper specks.
        ink_range: Morphological ink weight. Negative erodes (faint print),
            positive dilates (heavy print), zero leaves it alone.
    """

    enabled: frozenset[str] = field(default_factory=lambda: frozenset(EFFECT_ORDER))
    rotate_degrees: float = 2.0
    perspective_strength: float = 0.012
    blur_sigma: tuple[float, float] = (0.4, 1.6)
    jpeg_quality: tuple[int, int] = (35, 92)
    brightness_delta: tuple[float, float] = (-28.0, 22.0)
    contrast_scale: tuple[float, float] = (0.82, 1.18)
    gamma_range: tuple[float, float] = (0.75, 1.35)
    texture_strength: float = 14.0
    speckle_density: float = 0.0025
    ink_range: tuple[int, int] = (-1, 1)

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If an unknown effect is named or a range is inverted.
        """
        unknown = sorted(self.enabled - set(EFFECT_ORDER))
        if unknown:
            raise ValueError(f"unknown effect(s): {unknown}; expected {list(EFFECT_ORDER)}")
        if self.rotate_degrees < 0.0:
            raise ValueError(f"rotate_degrees must be non-negative, got {self.rotate_degrees}")
        if not 0.0 <= self.perspective_strength < 0.5:
            raise ValueError(
                f"perspective_strength must be in [0, 0.5), got {self.perspective_strength}"
            )
        if not 0.0 <= self.speckle_density <= 1.0:
            raise ValueError(f"speckle_density must be in [0, 1], got {self.speckle_density}")
        for name, (low, high) in (
            ("blur_sigma", self.blur_sigma),
            ("jpeg_quality", self.jpeg_quality),
            ("brightness_delta", self.brightness_delta),
            ("contrast_scale", self.contrast_scale),
            ("gamma_range", self.gamma_range),
            ("ink_range", self.ink_range),
        ):
            if low > high:
                raise ValueError(f"{name} is inverted: {low} > {high}")

    def without(self, *effects: str) -> DegradationConfig:
        """Return a copy with the named effects disabled."""
        return replace(self, enabled=self.enabled - set(effects))

    def only(self, *effects: str) -> DegradationConfig:
        """Return a copy with only the named effects enabled."""
        return replace(self, enabled=frozenset(effects))


# --------------------------------------------------------------------------- #
# Box transport
# --------------------------------------------------------------------------- #


def _corners(annotation: Annotation, width: int, height: int) -> npt.NDArray[np.float64]:
    """Return a box's four pixel corners, ready for a matrix multiply."""
    x0, y0, x1, y1 = annotation.box.to_pixels(width, height)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _rebuild(
    annotation: Annotation,
    moved: npt.NDArray[np.float64],
    width: int,
    height: int,
) -> Annotation | None:
    """Rebuild an annotation from transformed corners, clamped to the page.

    A rotated rectangle is not axis-aligned, so the result is the axis-aligned
    hull of the moved corners. At the small angles used here that inflates a box
    by well under a pixel.

    Returns:
        The moved annotation, or ``None`` if it left the page entirely.
    """
    x_min, y_min = moved.min(axis=0)
    x_max, y_max = moved.max(axis=0)

    x_min, x_max = max(0.0, float(x_min)), min(float(width), float(x_max))
    y_min, y_max = max(0.0, float(y_min)), min(float(height), float(y_max))
    if x_max - x_min < 1.0 or y_max - y_min < 1.0:
        return None

    return Annotation(
        symbol=annotation.symbol,
        box=BoundingBox.from_pixels(x_min, y_min, x_max, y_max, width, height),
    )


def transform_boxes_affine(
    boxes: Boxes, matrix: npt.NDArray[np.float64], width: int, height: int
) -> Boxes:
    """Carry boxes through the same 2x3 affine the image took.

    Args:
        boxes: Ground truth before the transform.
        matrix: The 2x3 matrix passed to ``cv2.warpAffine``.
        width: Page width in pixels.
        height: Page height in pixels.

    Returns:
        Moved boxes, dropping any that left the page.
    """
    moved: list[Annotation] = []
    for annotation in boxes:
        corners = _corners(annotation, width, height)
        homogeneous = np.hstack([corners, np.ones((4, 1))])
        result = _rebuild(annotation, homogeneous @ matrix.T, width, height)
        if result is not None:
            moved.append(result)
    return tuple(moved)


def transform_boxes_perspective(
    boxes: Boxes, matrix: npt.NDArray[np.float64], width: int, height: int
) -> Boxes:
    """Carry boxes through the same 3x3 perspective the image took.

    Args:
        boxes: Ground truth before the transform.
        matrix: The 3x3 matrix passed to ``cv2.warpPerspective``.
        width: Page width in pixels.
        height: Page height in pixels.

    Returns:
        Moved boxes, dropping any that left the page.
    """
    moved: list[Annotation] = []
    for annotation in boxes:
        corners = _corners(annotation, width, height).reshape(-1, 1, 2).astype(np.float32)
        warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
        result = _rebuild(annotation, warped.astype(np.float64), width, height)
        if result is not None:
            moved.append(result)
    return tuple(moved)


# --------------------------------------------------------------------------- #
# Geometric effects: these move boxes
# --------------------------------------------------------------------------- #


def apply_rotation(
    image: Image, boxes: Boxes, rng: random.Random, config: DegradationConfig
) -> tuple[Image, Boxes]:
    """Tilt the page a degree or two, carrying the boxes with it.

    Stage 1 deskews before anything else runs, so a small tilt exercises the
    real pipeline rather than defeating it.
    """
    if config.rotate_degrees <= 0.0:
        return image, boxes

    angle = rng.uniform(-config.rotate_degrees, config.rotate_degrees)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    warped = cv2.warpAffine(
        image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    return warped, transform_boxes_affine(boxes, matrix, width, height)


def apply_perspective(
    image: Image, boxes: Boxes, rng: random.Random, config: DegradationConfig
) -> tuple[Image, Boxes]:
    """Nudge the page corners, as a phone held slightly off square would."""
    if config.perspective_strength <= 0.0:
        return image, boxes

    height, width = image.shape[:2]
    reach = config.perspective_strength
    source = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    destination = np.array(
        [
            [rng.uniform(0, reach) * width, rng.uniform(0, reach) * height],
            [width - rng.uniform(0, reach) * width, rng.uniform(0, reach) * height],
            [width - rng.uniform(0, reach) * width, height - rng.uniform(0, reach) * height],
            [rng.uniform(0, reach) * width, height - rng.uniform(0, reach) * height],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    return warped, transform_boxes_perspective(boxes, matrix.astype(np.float64), width, height)


# --------------------------------------------------------------------------- #
# Photometric effects: these leave boxes alone
# --------------------------------------------------------------------------- #


def apply_ink(image: Image, rng: random.Random, config: DegradationConfig) -> Image:
    """Thicken or thin the ink, as heavy or faint printing would.

    Dilation on a dark-ink page thickens strokes; erosion thins them. Both are
    common on real scans and both change what a notehead looks like at the
    edges without moving it.
    """
    low, high = config.ink_range
    weight = rng.randint(low, high)
    if weight == 0:
        return image

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Ink is dark, so eroding the image spreads dark: erode = heavier print.
    if weight > 0:
        return cv2.erode(image, kernel, iterations=weight)
    return cv2.dilate(image, kernel, iterations=-weight)


def apply_blur(image: Image, rng: random.Random, config: DegradationConfig) -> Image:
    """Soften the page, as a slightly out-of-focus capture would."""
    sigma = rng.uniform(*config.blur_sigma)
    if sigma <= 0.0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def apply_texture(image: Image, rng: random.Random, config: DegradationConfig) -> Image:
    """Add low-frequency paper mottling.

    Built by upscaling a small noise field, so it varies over centimetres
    rather than pixels — which is what paper does and what speckle does not.
    """
    if config.texture_strength <= 0.0:
        return image

    height, width = image.shape[:2]
    generator = np.random.default_rng(rng.randrange(2**32))
    shape = (max(2, height // 64), max(2, width // 64))
    coarse = generator.normal(0.0, config.texture_strength, shape)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    if image.ndim == 3:
        field = field[:, :, np.newaxis]
    return np.clip(image.astype(np.float32) + field, 0, 255).astype(np.uint8)


def apply_speckle(image: Image, rng: random.Random, config: DegradationConfig) -> Image:
    """Scatter salt-and-pepper specks, as dust and scanner noise would."""
    if config.speckle_density <= 0.0:
        return image

    generator = np.random.default_rng(rng.randrange(2**32))
    out = image.copy()
    mask = generator.random(image.shape[:2])
    dark = mask < config.speckle_density / 2.0
    light = mask > 1.0 - config.speckle_density / 2.0
    out[dark] = 0
    out[light] = 255
    return out


def apply_brightness(image: Image, rng: random.Random, config: DegradationConfig) -> Image:
    """Shift exposure and contrast, as uneven lighting would."""
    scale = rng.uniform(*config.contrast_scale)
    delta = rng.uniform(*config.brightness_delta)
    shifted = (image.astype(np.float32) - 128.0) * scale + 128.0 + delta
    return np.clip(shifted, 0, 255).astype(np.uint8)


def apply_gamma(image: Image, rng: random.Random, config: DegradationConfig) -> Image:
    """Apply a gamma curve, as a different scanner profile would."""
    gamma = rng.uniform(*config.gamma_range)
    if gamma <= 0.0:
        return image
    table = np.array(
        [((value / 255.0) ** gamma) * 255 for value in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, table)


def apply_jpeg(image: Image, rng: random.Random, config: DegradationConfig) -> Image:
    """Round-trip through JPEG, leaving its blocking artefacts behind.

    The single most characteristic difference between a synthetic render and a
    page that has been through a phone camera or a web upload.
    """
    quality = rng.randint(*config.jpeg_quality)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - encoder failure on a valid array
        return image
    flag = cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR
    return cv2.imdecode(buffer, flag)


_PHOTOMETRIC = {
    "ink": apply_ink,
    "blur": apply_blur,
    "texture": apply_texture,
    "speckle": apply_speckle,
    "brightness": apply_brightness,
    "gamma": apply_gamma,
    "jpeg": apply_jpeg,
}

_GEOMETRIC = {
    "rotate": apply_rotation,
    "perspective": apply_perspective,
}


def degrade(
    image: Image,
    boxes: Boxes,
    config: DegradationConfig | None = None,
    seed: int = 0,
) -> tuple[Image, Boxes]:
    """Apply the configured effects to one page and its ground truth.

    Args:
        image: A clean page.
        boxes: Its ground truth.
        config: Which effects and how hard. Defaults to everything on.
        seed: Seeds every effect. The same seed reproduces the page exactly.

    Returns:
        The degraded page and its boxes, which have been carried through every
        geometric effect.
    """
    config = config or DegradationConfig()
    rng = random.Random(seed)

    out, moved = image, boxes
    for name in EFFECT_ORDER:
        if name not in config.enabled:
            continue
        if name in _GEOMETRIC:
            out, moved = _GEOMETRIC[name](out, moved, rng, config)
        else:
            out = _PHOTOMETRIC[name](out, rng, config)
    return out, moved


# --------------------------------------------------------------------------- #
# Walking a dataset
# --------------------------------------------------------------------------- #


def degrade_dataset(
    source: Path,
    destination: Path,
    config: DegradationConfig | None = None,
    seed: int = 0,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, int]:
    """Write a degraded copy of a YOLO dataset, preserving its layout.

    Each page gets its own derived seed, so adding a page does not change the
    degradation of every other page.

    Args:
        source: Dataset root holding ``images/<split>`` and ``labels/<split>``.
        destination: Where to write the copy.
        config: Effects to apply.
        seed: Base seed.
        splits: Split directories to walk.

    Returns:
        Page counts per split.

    Raises:
        FileNotFoundError: If a split's image directory is missing.
    """
    config = config or DegradationConfig()
    counts: dict[str, int] = {}

    for split in splits:
        images_dir = source / "images" / split
        if not images_dir.is_dir():
            raise FileNotFoundError(f"no {split} images at {images_dir}")

        unlabelled = find_unlabelled_images(images_dir)
        if unlabelled:
            names = ", ".join(path.name for path in unlabelled[:3])
            raise FileNotFoundError(
                f"{len(unlabelled)} {split} image(s) have no label file (e.g. {names}). "
                f"Degrading them would produce pages with no ground truth."
            )

        out_images = destination / "images" / split
        out_labels = destination / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        written = 0
        for index, path in enumerate(sorted(images_dir.iterdir())):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"could not decode {path}")

            label_path = source / "labels" / split / f"{path.stem}.txt"
            boxes = parse_label_file(label_path)

            # Derived per page so the dataset is stable under additions.
            out, moved = degrade(image, boxes, config, seed=seed * 1_000_003 + index)
            cv2.imwrite(str(out_images / path.name), out)
            write_label_file(out_labels / f"{path.stem}.txt", moved)
            written += 1

        counts[split] = written

    descriptor = source / "data.yaml"
    if descriptor.exists():
        body = descriptor.read_text(encoding="utf-8")
        body = body.replace(source.resolve().as_posix(), destination.resolve().as_posix())
        (destination / "data.yaml").write_text(body, encoding="utf-8")
    else:  # pragma: no cover - only when the source lacks a descriptor
        shutil.rmtree(destination / "data.yaml", ignore_errors=True)

    return counts


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", dest="destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable", nargs="*", default=[], choices=EFFECT_ORDER, help="effects to skip"
    )
    parser.add_argument(
        "--only", nargs="*", default=None, choices=EFFECT_ORDER, help="apply only these"
    )
    parser.add_argument("--rotate-degrees", type=float, default=2.0)
    args = parser.parse_args(argv)

    config = DegradationConfig(rotate_degrees=args.rotate_degrees)
    if args.only is not None:
        config = config.only(*args.only)
    if args.disable:
        config = config.without(*args.disable)

    counts = degrade_dataset(args.source, args.destination, config, seed=args.seed)

    print(f"degraded {args.source} -> {args.destination}")
    for split, count in counts.items():
        print(f"  {split}: {count} pages")
    print(f"  effects: {', '.join(n for n in EFFECT_ORDER if n in config.enabled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
