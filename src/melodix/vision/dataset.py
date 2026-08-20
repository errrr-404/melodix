"""Annotation storage, YOLO-format I/O, and dataset splitting.

Ultralytics expects a rigid layout on disk: images under ``images/train`` and
``images/val``, one ``.txt`` of labels per image under ``labels/train`` and
``labels/val`` with matching stems, and a ``data.yaml`` naming the classes in
id order. This module builds and reads that layout without importing torch or
ultralytics, so annotations can be validated long before the ``vision`` extra
is installed.

Boxes are normalised
--------------------
YOLO stores ``class_id cx cy w h`` with all four geometry values as fractions
of image width and height. Normalisation is what lets one checkpoint read pages
scanned at 150 and 300 DPI, so :class:`BoundingBox` holds fractions and
converts to pixels only on request.

The seam with Stage 1 is :meth:`BoundingBox.center_pixels`. Stage 3 takes that
centre, hands the row to :meth:`~melodix.geometry.staff.StaffGrid.snap`, and
pairs the resulting staff position with the detected shape to get a drum voice.
Nothing in this module needs to know that, which is the point: detection stores
pixels, geometry owns positions.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from melodix.vision.labels import NUM_CLASSES, SymbolClass, class_names

__all__ = [
    "IMAGE_SUFFIXES",
    "Annotation",
    "BoundingBox",
    "DatasetSplit",
    "LabeledImage",
    "class_distribution",
    "label_path_for_image",
    "parse_label_file",
    "split_dataset",
    "write_data_yaml",
    "write_label_file",
]

#: Image extensions ultralytics reads, lowercase.
IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)

#: Decimal places written per coordinate. Six is well past the precision of a
#: hand-drawn box on a 300 DPI page and keeps label files diffable.
_COORD_PRECISION: Final[int] = 6


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A YOLO box: centre and size as fractions of the image.

    Attributes:
        cx: Centre column, 0.0 at the left edge and 1.0 at the right.
        cy: Centre row, 0.0 at the top edge and 1.0 at the bottom.
        w: Width as a fraction of image width. Strictly positive.
        h: Height as a fraction of image height. Strictly positive.
    """

    cx: float
    cy: float
    w: float
    h: float

    def __post_init__(self) -> None:
        """Validate the box.

        Raises:
            ValueError: If the centre falls outside the image or either extent
                is not strictly positive. A zero-area box is rejected because
                it usually means an annotator clicked without dragging, and it
                would contribute a degenerate target to the loss.
        """
        for field, value in (("cx", self.cx), ("cy", self.cy)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {value}")
        for field, value in (("w", self.w), ("h", self.h)):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{field} must be in (0, 1], got {value}")

    # -- normalised edges --------------------------------------------------- #

    @property
    def x_min(self) -> float:
        """Left edge as a fraction of image width, possibly negative."""
        return self.cx - self.w / 2.0

    @property
    def x_max(self) -> float:
        """Right edge as a fraction of image width, possibly above 1."""
        return self.cx + self.w / 2.0

    @property
    def y_min(self) -> float:
        """Top edge as a fraction of image height, possibly negative."""
        return self.cy - self.h / 2.0

    @property
    def y_max(self) -> float:
        """Bottom edge as a fraction of image height, possibly above 1."""
        return self.cy + self.h / 2.0

    @property
    def area(self) -> float:
        """Fraction of the page this box covers."""
        return self.w * self.h

    @property
    def is_clipped(self) -> bool:
        """Whether the box runs off the edge of the page.

        Legal — a symbol cut off by the scanner is still a symbol — but worth
        reporting, since a dataset full of clipped boxes usually means the
        annotations were exported against a different crop than the images.
        """
        return self.x_min < 0.0 or self.y_min < 0.0 or self.x_max > 1.0 or self.y_max > 1.0

    # -- conversion --------------------------------------------------------- #

    @classmethod
    def from_pixels(
        cls,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        image_width: int,
        image_height: int,
    ) -> BoundingBox:
        """Build a box from pixel corners.

        Args:
            x_min: Left edge in pixels.
            y_min: Top edge in pixels.
            x_max: Right edge in pixels.
            y_max: Bottom edge in pixels.
            image_width: Page width in pixels.
            image_height: Page height in pixels.

        Returns:
            The equivalent normalised box.

        Raises:
            ValueError: If the image dimensions are not positive or the corners
                are inverted.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"image size must be positive, got {image_width}x{image_height}")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"corners must be ordered top-left to bottom-right, got "
                f"({x_min}, {y_min})..({x_max}, {y_max})"
            )
        return cls(
            cx=(x_min + x_max) / 2.0 / image_width,
            cy=(y_min + y_max) / 2.0 / image_height,
            w=(x_max - x_min) / image_width,
            h=(y_max - y_min) / image_height,
        )

    def to_pixels(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        """Return pixel corners ``(x_min, y_min, x_max, y_max)``.

        Args:
            image_width: Page width in pixels.
            image_height: Page height in pixels.

        Returns:
            The four corners in pixels, unrounded.

        Raises:
            ValueError: If the image dimensions are not positive.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"image size must be positive, got {image_width}x{image_height}")
        return (
            self.x_min * image_width,
            self.y_min * image_height,
            self.x_max * image_width,
            self.y_max * image_height,
        )

    def center_pixels(self, image_width: int, image_height: int) -> tuple[float, float]:
        """Return the centre in pixels, as ``(x, y)``.

        The handoff to Stage 1: pass the row to
        :meth:`~melodix.geometry.staff.StaffGrid.snap` to get a staff position.

        Args:
            image_width: Page width in pixels.
            image_height: Page height in pixels.

        Returns:
            Centre column and row in pixels.

        Raises:
            ValueError: If the image dimensions are not positive.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"image size must be positive, got {image_width}x{image_height}")
        return (self.cx * image_width, self.cy * image_height)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with another box.

        Args:
            other: The box to compare against.

        Returns:
            Overlap in ``[0, 1]``. Zero when the boxes are disjoint.
        """
        overlap_w = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        overlap_h = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        if overlap_w <= 0.0 or overlap_h <= 0.0:
            return 0.0
        intersection = overlap_w * overlap_h
        union = self.area + other.area - intersection
        # Clamped: accumulated float error can push a box's overlap with
        # itself a few ulps above 1.0, and callers threshold on this.
        return min(1.0, intersection / union)


@dataclass(frozen=True, slots=True)
class Annotation:
    """One labelled symbol on one page.

    Attributes:
        symbol: Which shape was drawn around.
        box: Where it sits, normalised.
    """

    symbol: SymbolClass
    box: BoundingBox

    def to_yolo_line(self) -> str:
        """Serialise to one line of a YOLO label file.

        Lossy: coordinates are written to six decimals, so a round trip
        through :meth:`from_yolo_line` agrees to about 1e-6 rather than
        exactly. That is far finer than a hand-drawn box on a 300 DPI page,
        and it keeps label files readable in a diff.
        """
        coords = " ".join(
            f"{value:.{_COORD_PRECISION}f}"
            for value in (self.box.cx, self.box.cy, self.box.w, self.box.h)
        )
        return f"{int(self.symbol)} {coords}"

    @classmethod
    def from_yolo_line(cls, line: str) -> Annotation:
        """Parse one line of a YOLO label file.

        Args:
            line: A line of the form ``class_id cx cy w h``.

        Returns:
            The parsed annotation.

        Raises:
            ValueError: If the line has the wrong number of fields, any field
                does not parse as a number, the class id is not in the schema,
                or the geometry is out of range.
        """
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"expected 5 fields in a YOLO label line, got {len(fields)}: {line!r}")

        raw_id, *raw_coords = fields
        try:
            class_id = int(raw_id)
        except ValueError:
            raise ValueError(f"class id must be an integer, got {raw_id!r}") from None
        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(f"class id {class_id} outside the schema (0..{NUM_CLASSES - 1})")

        try:
            cx, cy, w, h = (float(value) for value in raw_coords)
        except ValueError:
            raise ValueError(f"box coordinates must be numbers, got {raw_coords!r}") from None

        return cls(symbol=SymbolClass(class_id), box=BoundingBox(cx=cx, cy=cy, w=w, h=h))


@dataclass(frozen=True, slots=True)
class LabeledImage:
    """One page and everything annotated on it.

    Attributes:
        image_path: Where the page lives.
        annotations: Its symbols, in file order. May be empty: a page with no
            symbols is a valid negative example and ultralytics trains on it.
    """

    image_path: Path
    annotations: tuple[Annotation, ...] = ()

    @property
    def stem(self) -> str:
        """Filename without its extension, which pairs image to label file."""
        return self.image_path.stem

    def symbols(self) -> tuple[SymbolClass, ...]:
        """Every annotated class on this page, in file order."""
        return tuple(annotation.symbol for annotation in self.annotations)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A train/validation partition.

    Attributes:
        train: Pages to fit on.
        val: Pages held out for evaluation.
    """

    train: tuple[LabeledImage, ...]
    val: tuple[LabeledImage, ...]

    @property
    def total(self) -> int:
        """Pages across both halves."""
        return len(self.train) + len(self.val)


def label_path_for_image(image_path: Path, labels_root: Path | None = None) -> Path:
    """Return the label file that pairs with an image.

    Ultralytics pairs by stem, swapping an ``images`` path component for
    ``labels``. Passing ``labels_root`` explicitly avoids that substitution,
    which is worth doing whenever a path might contain ``images`` more than
    once.

    Args:
        image_path: Path to the page.
        labels_root: Directory holding label files. When omitted, the last
            ``images`` component of ``image_path`` is swapped for ``labels``.

    Returns:
        The ``.txt`` path for this image.

    Raises:
        ValueError: If ``labels_root`` is omitted and the path has no
            ``images`` component to substitute.
    """
    if labels_root is not None:
        return labels_root / f"{image_path.stem}.txt"

    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(
        f"cannot derive a label path from {image_path}: no 'images' component. "
        f"Pass labels_root explicitly."
    )


def parse_label_file(path: Path) -> tuple[Annotation, ...]:
    """Read every annotation from a YOLO label file.

    Blank lines are skipped, so a file of only whitespace reads as a negative
    example rather than an error.

    Args:
        path: The ``.txt`` to read.

    Returns:
        Annotations in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line is malformed. The message names the line
            number, because a bad annotation in a thousand-page dataset is
            otherwise very hard to find.
    """
    annotations: list[Annotation] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            annotations.append(Annotation.from_yolo_line(line))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return tuple(annotations)


def write_label_file(path: Path, annotations: tuple[Annotation, ...]) -> None:
    """Write annotations to a YOLO label file, creating parent directories.

    An empty sequence writes an empty file rather than skipping it, because
    ultralytics reads a missing label file and an empty one differently: the
    empty file marks a deliberate negative example.

    Args:
        path: The ``.txt`` to write.
        annotations: What to record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{annotation.to_yolo_line()}\n" for annotation in annotations)
    path.write_text(body, encoding="utf-8")


def split_dataset(
    images: list[LabeledImage],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> DatasetSplit:
    """Partition pages into training and validation sets.

    Deterministic given a seed: pages are sorted by path before shuffling, so
    the split does not depend on filesystem ordering. Re-running training with
    the same seed evaluates against the same held-out pages, which is what
    makes two runs comparable.

    The validation half is rounded down, and at least one page is held out
    whenever the fraction is positive and more than one page exists — a split
    that silently validates on nothing is worse than a small validation set.

    Args:
        images: Every annotated page.
        val_fraction: Portion to hold out, in ``[0, 1)``.
        seed: Shuffle seed.

    Returns:
        The partition. Every input page appears in exactly one half.

    Raises:
        ValueError: If ``val_fraction`` is outside ``[0, 1)``.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not images:
        return DatasetSplit(train=(), val=())

    ordered = sorted(images, key=lambda image: str(image.image_path))
    random.Random(seed).shuffle(ordered)

    val_size = int(len(ordered) * val_fraction)
    if val_fraction > 0.0 and val_size == 0 and len(ordered) > 1:
        val_size = 1

    return DatasetSplit(train=tuple(ordered[val_size:]), val=tuple(ordered[:val_size]))


def class_distribution(images: list[LabeledImage]) -> Counter[SymbolClass]:
    """Count annotations per class across pages.

    Drum notation is heavily imbalanced — cross noteheads outnumber triangle
    noteheads by orders of magnitude — so this is worth looking at before
    training rather than after.

    Args:
        images: Pages to tally.

    Returns:
        A counter over classes. Classes with no annotations are absent rather
        than zero, which :meth:`Counter.__getitem__` reports as 0 anyway.
    """
    return Counter(
        annotation.symbol for image in images for annotation in image.annotations
    )


def write_data_yaml(
    path: Path,
    dataset_root: Path,
    train_dir: str = "images/train",
    val_dir: str = "images/val",
) -> None:
    """Write the ``data.yaml`` ultralytics reads.

    Class names come from :mod:`melodix.vision.labels` in id order, so the file
    cannot drift from the schema. It is written by hand rather than through a
    YAML library to keep the base install free of a dependency used in exactly
    one place.

    Args:
        path: Where to write the file.
        dataset_root: Directory the split paths are relative to. Written out
            resolved to an absolute path: ultralytics resolves a relative
            ``path`` against its own configured datasets directory rather than
            the working directory, which would send training elsewhere.
        train_dir: Training images, relative to ``dataset_root``.
        val_dir: Validation images, relative to ``dataset_root``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names()))
    body = (
        "# Generated by melodix.vision.dataset. Do not edit by hand:\n"
        "# class ids come from melodix.vision.labels and must match the checkpoint.\n"
        f"path: {dataset_root.resolve().as_posix()}\n"
        f"train: {train_dir}\n"
        f"val: {val_dir}\n"
        f"nc: {NUM_CLASSES}\n"
        "names:\n"
        f"{names}\n"
    )
    path.write_text(body, encoding="utf-8")
