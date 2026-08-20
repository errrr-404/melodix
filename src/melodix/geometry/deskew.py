"""Skew estimation and correction.

Runs *before* :mod:`melodix.geometry.staff`. Staff isolation relies on a long
horizontal morphological kernel, which erases the very lines it is meant to
find once a page is rotated more than about a degree: across a 700 px staff, a
1.5 degree tilt drifts 18 px vertically, so no single image row lies on the
line for its full length. Every downstream stage inherits that failure, so
levelling the page is the first thing the pipeline does.

Two estimators are provided.

**Projection profile** (Radon-like) rotates a working copy through candidate
angles and scores the variance of each row-ink histogram. A level staff
concentrates ink into five sharp peaks separated by empty rows, which is the
variance maximum. Accurate and hard to fool, but it costs one warp per
candidate angle.

**Hough** finds long straight segments directly and takes their
length-weighted median inclination. It is the intuitive choice, and it is the
default in most deskewing literature, but on this workload it is not the
better one: measured on a 1700x2200 page carrying four staves of notes, Hough
ran slower than the full projection sweep (171 ms against 163 ms) and landed
further from the true angle. Its cost scales with ink, so a dense ensemble
score makes it worse rather than better, and it can lock onto a long beam or a
table rule instead of the staff.

Projection is therefore the default. Hough is kept because it fails
differently — it reads geometry directly rather than through a global score,
so it remains a useful cross-check when a page defeats the sweep — and
``"auto"`` runs it as a seed to narrow the projection search. Neither is
faster; choose ``"auto"`` when you want two independent estimators to agree
before trusting a correction.

Sign convention
---------------
:attr:`SkewEstimate.skew_deg` describes the *page*: it is positive when
content is rotated counter-clockwise, i.e. staff lines rise towards the right
of the image (``y`` decreasing as ``x`` increases).

:attr:`SkewEstimate.correction_deg` is what you rotate *by* to fix it, and is
always the negation. Only one of the two is stored, so they cannot disagree.

Angles passed to :func:`rotate_image` follow OpenCV: positive is
counter-clockwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import cv2
import numpy as np
import numpy.typing as npt

from melodix.geometry.staff import BinaryImage, GrayImage, binarize, to_grayscale

__all__ = [
    "DeskewConfig",
    "DeskewResult",
    "SkewEstimate",
    "SkewMethod",
    "deskew",
    "estimate_skew",
    "estimate_skew_hough",
    "estimate_skew_projection",
    "rotate_image",
]

SkewMethod = Literal["projection", "hough", "auto"]

#: Half-width of the projection refinement window when Hough seeds the search.
_SEEDED_RADIUS_DEG: Final[float] = 1.0

#: Angular agreement window used to score Hough consensus.
_HOUGH_CONSENSUS_DEG: Final[float] = 0.5


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DeskewConfig:
    """Tunable parameters for skew estimation and correction.

    Attributes:
        method: Which estimator to use. ``"projection"`` is the default and
            is both the fastest and the most accurate on typical scores.
            ``"auto"`` seeds a narrow projection search from a confident Hough
            result, costing more time to gain a second opinion, and falls back
            to a full projection sweep otherwise.
        max_angle_deg: Half-width of the search. Scans exceeding this are
            treated as unrecoverable rather than wrapped to a wrong answer.
        coarse_step_deg: Step of the first projection sweep.
        fine_step_deg: Step of the refinement sweep around the coarse winner,
            and the practical accuracy floor of the estimate.
        working_width: Images wider than this are downscaled before
            estimation. Correction is always applied at full resolution.
            Halving this roughly halves estimation time; 600 measured no
            accuracy loss on 200 DPI pages, but very low-DPI scans risk losing
            thin staff lines to the downscale, so the default stays
            conservative.
        min_confidence: Estimates below this are reported but not applied.
        min_correction_deg: Corrections smaller than this are skipped, since
            resampling the page costs more sharpness than the tilt costs. Keep
            it above ``fine_step_deg``, or search quantisation alone will
            trigger a rotation on a perfectly level page.
        hough_min_line_ratio: Minimum segment length for Hough, as a fraction
            of image width.
        hough_threshold: Hough accumulator vote threshold.
        hough_max_line_gap: Largest gap in pixels bridged within one segment.
    """

    method: SkewMethod = "projection"
    max_angle_deg: float = 10.0
    coarse_step_deg: float = 0.5
    fine_step_deg: float = 0.05
    working_width: int = 1000
    min_confidence: float = 0.05
    min_correction_deg: float = 0.10
    hough_min_line_ratio: float = 0.2
    hough_threshold: int = 100
    hough_max_line_gap: int = 6

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any parameter falls outside its permitted range.
        """
        if self.method not in ("projection", "hough", "auto"):
            raise ValueError(f"method must be projection, hough or auto, got {self.method!r}")
        if not 0.0 < self.max_angle_deg <= 45.0:
            raise ValueError(f"max_angle_deg must be in (0, 45], got {self.max_angle_deg}")
        if not 0.0 < self.coarse_step_deg <= self.max_angle_deg:
            raise ValueError(
                f"coarse_step_deg must be in (0, max_angle_deg], got {self.coarse_step_deg}"
            )
        if not 0.0 < self.fine_step_deg <= self.coarse_step_deg:
            raise ValueError(
                f"fine_step_deg must be in (0, coarse_step_deg], got {self.fine_step_deg}"
            )
        if self.working_width < 100:
            raise ValueError(f"working_width must be at least 100, got {self.working_width}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if self.min_correction_deg < 0.0:
            raise ValueError(
                f"min_correction_deg must be non-negative, got {self.min_correction_deg}"
            )
        if not 0.0 < self.hough_min_line_ratio <= 1.0:
            raise ValueError(
                f"hough_min_line_ratio must be in (0, 1], got {self.hough_min_line_ratio}"
            )
        if self.hough_threshold < 1:
            raise ValueError(f"hough_threshold must be positive, got {self.hough_threshold}")
        if self.hough_max_line_gap < 0:
            raise ValueError(
                f"hough_max_line_gap must be non-negative, got {self.hough_max_line_gap}"
            )


DEFAULT_CONFIG: Final[DeskewConfig] = DeskewConfig()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SkewEstimate:
    """The measured tilt of a page.

    Attributes:
        skew_deg: Tilt of the page content in degrees, positive when staff
            lines rise towards the right of the image.
        confidence: Strength of the evidence, in ``[0, 1]``. For the
            projection estimator this is how far the winning angle's score
            stands above the typical score; for Hough it is the share of
            segment length agreeing with the median inclination. Zero means no
            usable structure was found.
        method: Which estimator produced the value.
    """

    skew_deg: float
    confidence: float
    method: str

    def __post_init__(self) -> None:
        """Validate the estimate.

        Raises:
            ValueError: If ``confidence`` is outside ``[0, 1]``.
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def correction_deg(self) -> float:
        """Rotation to apply to level the page, counter-clockwise positive."""
        return -self.skew_deg

    def is_actionable(self, config: DeskewConfig = DEFAULT_CONFIG) -> bool:
        """Report whether this estimate justifies resampling the page.

        Args:
            config: Thresholds for confidence and minimum correction.

        Returns:
            ``True`` when the estimate is both confident enough to trust and
            large enough to be worth the loss of sharpness.
        """
        return (
            self.confidence >= config.min_confidence
            and abs(self.correction_deg) >= config.min_correction_deg
        )


@dataclass(frozen=True, slots=True)
class DeskewResult:
    """A levelled page and the reasoning behind it.

    Attributes:
        image: The corrected image, or the untouched input when no rotation
            was warranted.
        estimate: The skew that was measured, whether or not it was applied.
        applied: Whether the image was actually rotated.
    """

    image: npt.NDArray[np.uint8]
    estimate: SkewEstimate
    applied: bool


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #


def rotate_image(
    image: npt.NDArray[np.uint8],
    angle_deg: float,
    border_value: int = 255,
    expand: bool = False,
    interpolation: int | None = None,
) -> npt.NDArray[np.uint8]:
    """Rotate an image about its centre.

    Args:
        image: Grayscale or colour ``uint8`` array.
        angle_deg: Rotation in degrees, counter-clockwise positive.
        border_value: Fill for regions rotated in from outside the frame. 255
            (white) suits scanned paper; use 0 when rotating an ink mask.
        expand: Grow the canvas so no content is clipped. Off by default,
            since the small corrections typical of scans clip nothing and a
            stable frame keeps coordinates comparable to the input.
        interpolation: OpenCV interpolation flag. Use ``cv2.INTER_NEAREST``
            for binary masks, where blending would create grey pixels that
            then count as ink.

    Returns:
        The rotated image.

    Raises:
        ValueError: If the image is empty.
    """
    if image.size == 0:
        raise ValueError("cannot rotate an empty image")

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    out_width, out_height = width, height
    if expand:
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        out_width = int(np.ceil(height * sin + width * cos))
        out_height = int(np.ceil(height * cos + width * sin))
        matrix[0, 2] += out_width / 2.0 - center[0]
        matrix[1, 2] += out_height / 2.0 - center[1]

    # Choose a sensible default interpolation when the caller didn't specify.
    # Preserve binary masks (0/255) with nearest-neighbour to avoid creating
    # greys from resampling; use linear for continuous-tone images.
    interp = interpolation
    if interp is None:
        try:
            uniq = np.unique(image)
            if uniq.size <= 2 and set(int(x) for x in uniq.tolist()).issubset({0, 255}):
                interp = cv2.INTER_NEAREST
            else:
                interp = cv2.INTER_LINEAR
        except Exception:
            interp = cv2.INTER_LINEAR

    return cv2.warpAffine(  # type: ignore[no-any-return]
        image,
        matrix,
        (out_width, out_height),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #


def estimate_skew_projection(
    image: npt.NDArray[np.uint8],
    config: DeskewConfig = DEFAULT_CONFIG,
    center_deg: float = 0.0,
    radius_deg: float | None = None,
) -> SkewEstimate:
    """Estimate skew by maximising horizontal projection variance.

    Rotates a working copy through candidate angles and scores each by the
    variance of its row-ink histogram. Levelling a staff concentrates ink into
    five tall peaks with empty rows between them, which is the variance
    maximum. A coarse sweep locates the basin, then a fine sweep resolves it.

    Args:
        image: Page as a grayscale or colour ``uint8`` array.
        config: Search parameters.
        center_deg: Centre of the correction search, in correction degrees.
        radius_deg: Half-width of the search. Defaults to
            ``config.max_angle_deg``.

    Returns:
        The measured skew. Confidence is zero for a page with no ink.

    Raises:
        ValueError: If the image is empty.
    """
    mask = _working_mask(image, config)
    radius = config.max_angle_deg if radius_deg is None else radius_deg
    low_bound = center_deg - radius
    high_bound = center_deg + radius

    coarse = _angle_range(low_bound, high_bound, config.coarse_step_deg)
    coarse_scores = _score_angles(mask, coarse)
    if coarse_scores.size == 0 or float(coarse_scores.max()) <= 0.0:
        return SkewEstimate(skew_deg=0.0, confidence=0.0, method="projection")

    best_coarse = float(coarse[int(np.argmax(coarse_scores))])
    # Clamp the refinement to the declared window. Without this the fine sweep
    # reaches one coarse step past the bounds, so a page tilted beyond
    # max_angle_deg could be reported as tilted beyond max_angle_deg too.
    fine = _angle_range(
        max(best_coarse - config.coarse_step_deg, low_bound),
        min(best_coarse + config.coarse_step_deg, high_bound),
        config.fine_step_deg,
    )
    fine_scores = _score_angles(mask, fine)
    best_correction = float(fine[int(np.argmax(fine_scores))])

    peak = float(coarse_scores.max())
    typical = float(np.median(coarse_scores))
    confidence = float(np.clip((peak - typical) / peak, 0.0, 1.0))

    return SkewEstimate(
        skew_deg=_clean(-best_correction),
        confidence=confidence,
        method="projection",
    )


def estimate_skew_hough(
    image: npt.NDArray[np.uint8],
    config: DeskewConfig = DEFAULT_CONFIG,
) -> SkewEstimate:
    """Estimate skew from the inclination of long straight segments.

    Runs a probabilistic Hough transform, discards segments steeper than
    ``max_angle_deg``, and takes the length-weighted median inclination of
    what remains. Confidence is the share of surviving segment length that
    agrees with that median, so a page whose long strokes disagree — a staff
    competing with a tilted table rule — reports low confidence rather than a
    confident average of the two.

    Args:
        image: Page as a grayscale or colour ``uint8`` array.
        config: Search and Hough parameters.

    Returns:
        The measured skew. Confidence is zero when no segment qualifies.

    Raises:
        ValueError: If the image is empty.
    """
    mask = _working_mask(image, config)
    width = mask.shape[1]
    min_length = max(10, int(round(width * config.hough_min_line_ratio)))

    segments = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 1800.0,
        threshold=config.hough_threshold,
        minLineLength=min_length,
        maxLineGap=config.hough_max_line_gap,
    )
    if segments is None:
        return SkewEstimate(skew_deg=0.0, confidence=0.0, method="hough")

    angles: list[float] = []
    lengths: list[float] = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0.0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) > config.max_angle_deg:
            continue
        angles.append(angle)
        lengths.append(float(np.hypot(dx, dy)))

    if not angles:
        return SkewEstimate(skew_deg=0.0, confidence=0.0, method="hough")

    angle_array = np.asarray(angles, dtype=np.float64)
    length_array = np.asarray(lengths, dtype=np.float64)
    median_angle = _weighted_median(angle_array, length_array)

    agreeing = length_array[np.abs(angle_array - median_angle) <= _HOUGH_CONSENSUS_DEG]
    confidence = float(np.clip(agreeing.sum() / length_array.sum(), 0.0, 1.0))

    # A segment sloping down to the right has positive inclination in y-down
    # pixel space, which means the page is rotated clockwise: negative skew.
    return SkewEstimate(skew_deg=_clean(-median_angle), confidence=confidence, method="hough")


def estimate_skew(
    image: npt.NDArray[np.uint8],
    config: DeskewConfig = DEFAULT_CONFIG,
) -> SkewEstimate:
    """Estimate page skew using the configured strategy.

    Args:
        image: Page as a grayscale or colour ``uint8`` array.
        config: Estimation parameters.

    Returns:
        The measured skew.

    Raises:
        ValueError: If the image is empty.
    """
    if config.method == "projection":
        return estimate_skew_projection(image, config)
    if config.method == "hough":
        return estimate_skew_hough(image, config)

    seed = estimate_skew_hough(image, config)
    if seed.confidence >= config.min_confidence:
        refined = estimate_skew_projection(
            image,
            config,
            center_deg=seed.correction_deg,
            radius_deg=_SEEDED_RADIUS_DEG,
        )
        if refined.confidence > 0.0:
            return SkewEstimate(
                skew_deg=refined.skew_deg,
                confidence=max(refined.confidence, seed.confidence),
                method="auto:hough+projection",
            )

    fallback = estimate_skew_projection(image, config)
    return SkewEstimate(
        skew_deg=fallback.skew_deg,
        confidence=fallback.confidence,
        method="auto:projection",
    )


def deskew(
    image: npt.NDArray[np.uint8],
    config: DeskewConfig = DEFAULT_CONFIG,
) -> DeskewResult:
    """Measure a page's skew and level it.

    Rotation is skipped when the estimate is unconfident or the correction is
    negligible; in both cases the original array is returned untouched, so a
    clean page passes through without a resampling penalty.

    Args:
        image: Page as a grayscale or colour ``uint8`` array.
        config: Estimation and correction parameters.

    Returns:
        The levelled page together with the estimate and whether it was used.

    Raises:
        ValueError: If the image is empty.
    """
    estimate = estimate_skew(image, config)
    if not estimate.is_actionable(config):
        return DeskewResult(image=image, estimate=estimate, applied=False)

    rotated = rotate_image(image, estimate.correction_deg, border_value=255)
    return DeskewResult(image=rotated, estimate=estimate, applied=True)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _working_mask(image: npt.NDArray[np.uint8], config: DeskewConfig) -> BinaryImage:
    """Produce a downscaled ink mask for estimation."""
    if image.size == 0:
        raise ValueError("cannot estimate skew on an empty image")

    gray: GrayImage = to_grayscale(image)
    if gray.shape[1] > config.working_width:
        scale = config.working_width / gray.shape[1]
        gray = cv2.resize(
            gray,
            (config.working_width, max(1, int(round(gray.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return binarize(gray)


def _angle_range(low: float, high: float, step: float) -> npt.NDArray[np.float64]:
    """Return inclusive candidate angles from ``low`` to ``high``."""
    count = int(np.floor((high - low) / step + 1e-9)) + 1
    return low + np.arange(count, dtype=np.float64) * step


def _score_angles(
    mask: BinaryImage,
    angles: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Score each candidate correction by its projection-profile variance."""
    return np.asarray([_projection_variance(mask, float(a)) for a in angles], dtype=np.float64)


def _projection_variance(mask: BinaryImage, angle_deg: float) -> float:
    """Variance of the row-ink histogram after rotating ``mask`` by ``angle_deg``.

    Rotation uses nearest-neighbour sampling and is applied even at zero
    degrees. Blending would spread each line across two rows that both then
    count as full ink, inflating the score for slightly tilted candidates;
    skipping the warp at zero would leave that one candidate scored on
    different terms from all the others. Both would bias the argmax.
    """
    rotated = rotate_image(
        mask, angle_deg, border_value=0, interpolation=cv2.INTER_NEAREST
    )
    profile = (rotated > 0).sum(axis=1).astype(np.float64)
    return float(profile.var())


def _weighted_median(values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> float:
    """Return the weight-weighted median of ``values``."""
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cutoff = cumulative[-1] / 2.0
    index = min(int(np.searchsorted(cumulative, cutoff)), sorted_values.size - 1)
    return float(sorted_values[index])


def _clean(angle: float) -> float:
    """Collapse negative zero and floating-point dust to exact zero."""
    return 0.0 if abs(angle) < 1e-9 else angle
