"""The YOLO symbol detector, wrapped so importing it costs nothing.

Ultralytics pulls in torch, which is a multi-gigabyte dependency and several
seconds of import time. Stage 1 and Stage 3 never need it, and neither does
annotation tooling, so this module imports it **lazily**: the import happens
inside :meth:`SymbolDetector.load`, not at module scope. Importing
:mod:`melodix.vision.detector` on a machine without the ``vision`` extra
succeeds, and only calling :meth:`~SymbolDetector.detect` raises — with a
message naming the install command.

What comes back
---------------
Detections carry a **shape class and a normalised box**, never a drum voice.
Resolving a voice needs the staff grid, and follows the same handoff the rest
of the pipeline uses::

    detections = detector.detect(page)
    for hit in detections.noteheads():
        x, y = hit.center_pixels(detections.image_width, detections.image_height)
        position = grid.snap(y)          # None when ambiguous
        # Stage 3 pairs (hit.symbol, position) into a GM percussion note

Defaults are tuned for sheet music rather than photographs. Two of them differ
sharply from the ultralytics defaults and matter enough to state plainly:
``image_size`` is 1280 because a notehead on a letter-size page scanned at
300 DPI is only a few pixels across once downscaled to 640, and
``max_detections`` is 3000 because ultralytics stops at 300, which a dense
ensemble page exceeds before the second system.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from melodix.vision.dataset import BoundingBox
from melodix.vision.labels import NUM_CLASSES, SymbolCategory, SymbolClass, SymbolLabel

__all__ = [
    "Detection",
    "DetectorConfig",
    "DetectorNotAvailableError",
    "PageDetections",
    "SymbolDetector",
]

#: Smallest box side, as a fraction of the page, that is worth keeping. Below
#: this a "detection" is a rounding artefact rather than an engraved symbol.
_MIN_BOX_EXTENT: Final[float] = 1e-6

_INSTALL_HINT: Final[str] = (
    'ultralytics is not installed. Install the vision extra:\n'
    '    pip install -e ".[dev,vision]"'
)


class DetectorNotAvailableError(ImportError):
    """Raised when the detector is used without the ``vision`` extra.

    Subclasses :class:`ImportError` so that callers wanting to degrade
    gracefully can catch either.
    """


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Inference settings.

    Attributes:
        weights: Path to a ``.pt`` checkpoint trained against
            :mod:`melodix.vision.labels`.
        confidence_threshold: Detections scoring below this are discarded.
        iou_threshold: Overlap above which non-maximum suppression treats two
            boxes as the same symbol. Kept lower than a photographic default
            because engraved symbols sit close together but rarely overlap.
        image_size: Longest side, in pixels, the page is resized to before
            inference. Sheet music needs far more than the ultralytics default
            of 640; a notehead must survive the downscale.
        max_detections: Cap on boxes returned per page. The ultralytics default
            of 300 truncates a dense ensemble page mid-system.
        device: ``"cpu"``, ``"cuda"``, ``"cuda:1"``, or ``"auto"`` to let
            ultralytics choose.
        half: Use float16 inference. Ignored on CPU.
    """

    weights: Path
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 1280
    max_detections: int = 3000
    device: str = "auto"
    half: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any parameter falls outside its permitted range.
        """
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {self.iou_threshold}")
        if self.image_size < 32:
            raise ValueError(f"image_size must be at least 32, got {self.image_size}")
        if self.image_size % 32 != 0:
            # YOLO downsamples by 32; a non-multiple is silently rounded, which
            # makes the reported and actual inference size disagree.
            raise ValueError(f"image_size must be a multiple of 32, got {self.image_size}")
        if self.max_detections < 1:
            raise ValueError(f"max_detections must be positive, got {self.max_detections}")


@dataclass(frozen=True, slots=True)
class Detection:
    """One symbol the detector found.

    Attributes:
        symbol: The shape class. Never a drum voice — see the module docstring.
        box: Where it sits, normalised to the page.
        confidence: Model score in ``[0, 1]``.
    """

    symbol: SymbolClass
    box: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        """Validate the detection.

        Raises:
            ValueError: If the confidence is outside ``[0, 1]``.
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def label(self) -> SymbolLabel:
        """The schema row for this detection's class."""
        return self.symbol.label

    @property
    def category(self) -> SymbolCategory:
        """How Stage 3 should handle this shape."""
        return self.label.category

    @property
    def carries_position(self) -> bool:
        """Whether a staff position must be resolved for this detection."""
        return self.label.carries_position

    def center_pixels(self, image_width: int, image_height: int) -> tuple[float, float]:
        """Return the centre in pixels, as ``(x, y)``.

        The row is what
        :meth:`~melodix.geometry.staff.StaffGrid.snap` converts to a staff
        position.

        Args:
            image_width: Page width in pixels.
            image_height: Page height in pixels.

        Returns:
            Centre column and row in pixels.
        """
        return self.box.center_pixels(image_width, image_height)


@dataclass(frozen=True, slots=True)
class PageDetections:
    """Everything found on one page, with the size it was found at.

    Carrying the page dimensions alongside the boxes means callers never have
    to keep the array around to convert a normalised box back to pixels.

    Attributes:
        detections: Symbols found, ordered by descending confidence.
        image_width: Page width in pixels.
        image_height: Page height in pixels.
    """

    detections: tuple[Detection, ...]
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        """Validate the page size.

        Raises:
            ValueError: If either dimension is not positive.
        """
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                f"image size must be positive, got {self.image_width}x{self.image_height}"
            )

    def __len__(self) -> int:
        """Number of detections."""
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        """Iterate detections in confidence order."""
        return iter(self.detections)

    def of_category(self, category: SymbolCategory) -> tuple[Detection, ...]:
        """Return detections in one category, preserving order.

        Args:
            category: The category to filter by.

        Returns:
            Matching detections, possibly empty.
        """
        return tuple(hit for hit in self.detections if hit.category is category)

    def noteheads(self) -> tuple[Detection, ...]:
        """Return the detections that carry a staff position."""
        return self.of_category(SymbolCategory.NOTEHEAD)

    def above_confidence(self, threshold: float) -> tuple[Detection, ...]:
        """Return detections scoring at or above a threshold.

        Args:
            threshold: Minimum confidence, inclusive.

        Returns:
            Matching detections, in confidence order.
        """
        return tuple(hit for hit in self.detections if hit.confidence >= threshold)

    def in_reading_order(self) -> tuple[Detection, ...]:
        """Return detections ordered top to bottom, then left to right.

        Detections arrive in confidence order, which is what a detector
        naturally produces and what a caller thresholding wants. Stage 3 reads
        a page instead, so it wants this.
        """
        return tuple(
            sorted(self.detections, key=lambda hit: (hit.box.cy, hit.box.cx))
        )


def _default_model_factory(weights: Path, config: DetectorConfig) -> Any:
    """Load an ultralytics model, importing torch only when called.

    Args:
        weights: Checkpoint to load.
        config: Settings, used to place the model on a device.

    Returns:
        The loaded ``YOLO`` model.

    Raises:
        DetectorNotAvailableError: If ultralytics is not installed.
    """
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - requires the extra absent
        raise DetectorNotAvailableError(_INSTALL_HINT) from error

    model = YOLO(str(weights))
    if config.device != "auto":
        model.to(config.device)
    return model


class SymbolDetector:
    """Runs a trained YOLO checkpoint over pages and returns typed detections.

    The model is loaded on first use rather than in ``__init__``, so building a
    detector is free and a program that never detects never pays for torch.

    Args:
        config: Inference settings.
        model_factory: Builds the underlying model from a checkpoint path.
            Defaults to loading ultralytics. Injecting a factory is the seam
            the test suite uses to exercise this class without torch, and it
            also allows wrapping the model for profiling.
    """

    def __init__(
        self,
        config: DetectorConfig,
        model_factory: Callable[[Path, DetectorConfig], Any] | None = None,
    ) -> None:
        self._config = config
        self._factory = model_factory or _default_model_factory
        self._model: Any | None = None

    @property
    def config(self) -> DetectorConfig:
        """The settings this detector runs with."""
        return self._config

    @property
    def is_loaded(self) -> bool:
        """Whether the checkpoint has been read yet."""
        return self._model is not None

    def load(self) -> None:
        """Load the checkpoint now, rather than on first detection.

        Idempotent. Worth calling explicitly at startup so that a missing
        checkpoint or a missing torch install fails before a user has waited
        through a page ingest.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            DetectorNotAvailableError: If ultralytics is not installed.
        """
        self._ensure_model()

    def _ensure_model(self) -> Any:
        """Return the loaded model, loading it on first call."""
        if self._model is None:
            if not self._config.weights.exists():
                raise FileNotFoundError(f"no checkpoint at {self._config.weights}")
            self._model = self._factory(self._config.weights, self._config)
        return self._model

    def detect(
        self,
        image: npt.NDArray[np.uint8],
        min_confidence: float | None = None,
    ) -> PageDetections:
        """Detect every symbol on one page or crop.

        Args:
            image: A grayscale or colour ``uint8`` array. Deskew it first —
                the detector has no idea the page is crooked, and a tilted
                notehead is a different shape as far as it is concerned.
            min_confidence: Overrides the configured threshold for this call.

        Returns:
            Detections ordered by descending confidence, with the page size.

        Raises:
            ValueError: If the image is empty or not a recognised shape.
            FileNotFoundError: If the checkpoint does not exist.
            DetectorNotAvailableError: If ultralytics is not installed.
        """
        prepared = _prepare_image(image)
        height, width = prepared.shape[:2]

        model = self._ensure_model()

        threshold = (
            self._config.confidence_threshold if min_confidence is None else min_confidence
        )
        results = model.predict(
            prepared,
            conf=threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.image_size,
            max_det=self._config.max_detections,
            half=self._config.half,
            verbose=False,
            **({} if self._config.device == "auto" else {"device": self._config.device}),
        )

        detections = _detections_from_results(results, threshold)
        return PageDetections(
            detections=tuple(
                sorted(detections, key=lambda hit: hit.confidence, reverse=True)
            ),
            image_width=width,
            image_height=height,
        )

    def detect_batch(
        self,
        images: Sequence[npt.NDArray[np.uint8]],
        min_confidence: float | None = None,
    ) -> list[PageDetections]:
        """Detect over several pages or crops.

        Runs them one at a time rather than as a batch tensor: pages of a
        scanned book vary in size, and ultralytics would letterbox them to a
        common shape, which shifts every normalised box.

        Args:
            images: Pages or crops.
            min_confidence: Overrides the configured threshold.

        Returns:
            One result per input, in order.
        """
        return [self.detect(image, min_confidence) for image in images]


# --------------------------------------------------------------------------- #
# Result parsing
# --------------------------------------------------------------------------- #


def _prepare_image(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Coerce a page into the 3-channel array ultralytics expects.

    Args:
        image: 2-D grayscale, or 3-D with 1, 3 or 4 channels.

    Returns:
        A contiguous ``H*W*3`` ``uint8`` array.

    Raises:
        ValueError: If the image is empty or not a recognised shape.
    """
    if image.size == 0:
        raise ValueError("cannot detect symbols in an empty image")

    if image.ndim == 2:
        stacked = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        return np.ascontiguousarray(stacked, dtype=np.uint8)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            stacked = np.repeat(image, 3, axis=2)
            return np.ascontiguousarray(stacked, dtype=np.uint8)
        if channels == 3:
            return np.ascontiguousarray(image, dtype=np.uint8)
        if channels == 4:
            return np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
    raise ValueError(f"unsupported image shape {image.shape}; expected 2-D or H*W*{{1,3,4}}")


def _detections_from_results(results: Iterable[Any], threshold: float) -> list[Detection]:
    """Convert ultralytics results into typed detections.

    Args:
        results: What ``model.predict`` returned.
        threshold: Confidence floor, applied again here because a caller can
            pass a model that ignores the ``conf`` argument.

    Returns:
        Detections, unordered.

    Raises:
        ValueError: If a result carries a class id outside the schema, which
            means the checkpoint was trained against a different label set.
    """
    detections: list[Detection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for x1, y1, x2, y2, confidence, class_id in _rows(boxes):
            if confidence < threshold:
                continue
            box = _box_from_xyxyn(x1, y1, x2, y2)
            if box is None:
                continue
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"detector emitted class id {class_id}, outside the schema "
                    f"(0..{NUM_CLASSES - 1}). The checkpoint was trained against a "
                    f"different label set than melodix.vision.labels defines."
                )
            detections.append(
                Detection(
                    symbol=SymbolClass(class_id),
                    box=box,
                    confidence=min(1.0, max(0.0, confidence)),
                )
            )
    return detections


def _rows(boxes: Any) -> Iterator[tuple[float, float, float, float, float, int]]:
    """Yield ``(x1, y1, x2, y2, confidence, class_id)`` per detected box.

    Reads the ultralytics ``Boxes`` attributes positionally rather than
    converting tensors wholesale, so a plain object exposing ``xyxyn``,
    ``conf`` and ``cls`` works identically. That keeps the parsing testable
    without torch.

    Args:
        boxes: An ultralytics ``Boxes`` or anything with the same three
            attributes.

    Raises:
        ValueError: If the three attributes disagree on length, or a box does
            not carry exactly four coordinates.
    """
    coordinates = list(getattr(boxes, "xyxyn", []))
    confidences = list(getattr(boxes, "conf", []))
    classes = list(getattr(boxes, "cls", []))

    if not (len(coordinates) == len(confidences) == len(classes)):
        raise ValueError(
            f"detector result is inconsistent: {len(coordinates)} boxes, "
            f"{len(confidences)} scores, {len(classes)} classes"
        )

    for corner, confidence, class_id in zip(coordinates, confidences, classes, strict=True):
        values = [float(value) for value in corner]
        if len(values) != 4:
            raise ValueError(f"expected 4 box coordinates, got {len(values)}")
        x1, y1, x2, y2 = values
        yield x1, y1, x2, y2, float(confidence), int(class_id)


def _box_from_xyxyn(x1: float, y1: float, x2: float, y2: float) -> BoundingBox | None:
    """Convert normalised corners to a box, clamping to the page.

    YOLO routinely emits corners a hair outside ``[0, 1]`` for a symbol at the
    page edge, and :class:`~melodix.vision.dataset.BoundingBox` validates
    strictly. Clamping here keeps a legitimate edge detection rather than
    letting valid inference raise.

    Args:
        x1: Left edge, normalised.
        y1: Top edge, normalised.
        x2: Right edge, normalised.
        y2: Bottom edge, normalised.

    Returns:
        The box, or ``None`` when clamping collapsed it to no area — which
        happens for a detection entirely outside the page.
    """
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    left = min(1.0, max(0.0, left))
    right = min(1.0, max(0.0, right))
    top = min(1.0, max(0.0, top))
    bottom = min(1.0, max(0.0, bottom))

    width = right - left
    height = bottom - top
    if width < _MIN_BOX_EXTENT or height < _MIN_BOX_EXTENT:
        return None

    return BoundingBox(
        cx=(left + right) / 2.0,
        cy=(top + bottom) / 2.0,
        w=width,
        h=height,
    )
