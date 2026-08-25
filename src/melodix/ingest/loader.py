"""Turning source documents into page arrays.

The front door of the pipeline. Real drum sheets arrive as PDFs, occasionally as
photographs or scans, and everything downstream works on ``uint8`` numpy arrays,
so this is where the two meet.

Named ``ingest`` rather than ``io`` deliberately: a module called ``io`` inside a
package shadows the standard library one for every reader, if not for the
interpreter.

Why DPI is not cosmetic
-----------------------
Every threshold in Stage 1 is expressed as a ratio of **staff line spacing in
pixels**, and render resolution is what sets that::

    spacing_px = spacing_pt * dpi / 72

Real engravings run about 6-9 points between staff lines. Stage 1 rejects a
candidate staff below :attr:`StaffDetectionConfig.min_line_spacing`, 3.0 px, so
for a 7 pt engraving the hard floor sits near **31 DPI** — measured, not
estimated. Below it, :func:`~melodix.geometry.staff.detect_staff_grids` returns
an empty list rather than a poor result, which is a failure with no error
attached to it.

That floor is much lower than the useful working range, and it is worth being
precise about why. On clean synthetic input Stage 1 still finds staves at 36 DPI.
The reasons to render far above the floor are downstream of it:

- **Stage 2 detects noteheads**, not lines. A notehead spans roughly one staff
  space, so at 72 DPI it is about seven pixels across — too few for a shape
  classifier to separate a cross from a diamond, whatever the model.
- **:meth:`~melodix.geometry.staff.StaffGrid.snap` resolves half-space steps.**
  Fewer pixels per space means a notehead centroid lands closer to the tolerance
  boundary, and snap returns ``None`` rather than guessing.
- **Real scans carry noise** that clean fixtures do not, so the margin that
  survives at low DPI in a test disappears on paper.

:data:`DEFAULT_DPI` is 300. :data:`MIN_USEFUL_DPI` is 150, well above the
mechanical floor and chosen for the notehead-size reason rather than the
line-spacing one; below it this module warns.

Nothing here imports torch. A PDF can be turned into pages on any machine.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_DPI",
    "IMAGE_SUFFIXES",
    "MIN_USEFUL_DPI",
    "PDF_SUFFIXES",
    "EncryptedPdfError",
    "Page",
    "UnsupportedFormatError",
    "load_document",
    "load_image",
    "load_pdf",
    "page_count",
    "read_grayscale",
    "write_image",
]

#: Render resolution that puts roughly 20 px between staff lines on a
#: letter-size engraving, which is comfortably inside Stage 1's working range.
DEFAULT_DPI: Final[int] = 300

#: Below this a notehead is too few pixels across for Stage 2 to classify its
#: shape. Well above the ~31 DPI mechanical floor for staff detection itself.
MIN_USEFUL_DPI: Final[int] = 150

#: Raster formats read directly, lowercase.
IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)

#: Document formats rasterised page by page.
PDF_SUFFIXES: Final[frozenset[str]] = frozenset({".pdf"})


class UnsupportedFormatError(ValueError):
    """Raised for a file this module cannot turn into pages."""


class EncryptedPdfError(ValueError):
    """Raised for a password-protected PDF.

    Separate from :class:`UnsupportedFormatError` because the remedy is
    different: the format is fine, the file needs a password.
    """


@dataclass(frozen=True, slots=True)
class Page:
    """One page of a source document, with what it takes to cite it later.

    ``sync_map.json`` has to point a browser at a specific page of a specific
    file, so a bare array is not enough — the provenance travels with the
    pixels rather than being reconstructed downstream.

    Attributes:
        image: The page as a 2-D grayscale or 3-D BGR ``uint8`` array.
        source: Path of the file this came from.
        page_index: Zero-based page number within that file. Always 0 for a
            single image.
        dpi: Resolution this page was rendered at. ``None`` for a raster image
            loaded as-is, where the true DPI is whatever the scanner used and
            is not reliably recorded.
    """

    image: npt.NDArray[np.uint8]
    source: Path
    page_index: int
    dpi: int | None

    def __post_init__(self) -> None:
        """Validate the page.

        Raises:
            ValueError: If the array is empty, not ``uint8``, not a recognised
                shape, or the page index is negative.
        """
        if self.page_index < 0:
            raise ValueError(f"page_index must be non-negative, got {self.page_index}")
        if self.image.size == 0:
            raise ValueError(f"{self.source} page {self.page_index} is empty")
        if self.image.dtype != np.uint8:
            raise ValueError(f"page must be uint8, got {self.image.dtype}")
        if self.image.ndim not in (2, 3):
            raise ValueError(f"unsupported page shape {self.image.shape}")

    @property
    def height(self) -> int:
        """Page height in pixels."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Page width in pixels."""
        return int(self.image.shape[1])

    @property
    def is_colour(self) -> bool:
        """Whether the page carries colour channels."""
        return self.image.ndim == 3

    @property
    def label(self) -> str:
        """A short human-readable identifier, e.g. ``score.pdf#p3``."""
        return f"{self.source.name}#p{self.page_index}"

    def metadata(self) -> dict[str, object]:
        """Return the provenance fields, ready to embed in a sync map."""
        return {
            "source": str(self.source),
            "page_index": self.page_index,
            "dpi": self.dpi,
            "width": self.width,
            "height": self.height,
        }


def read_grayscale(path: Path) -> npt.NDArray[np.uint8]:
    """Read an image as a strictly 2-D grayscale array.

    Use this rather than ``cv2.imread(..., cv2.IMREAD_GRAYSCALE)`` anywhere the
    same process might also import ultralytics.

    Importing ultralytics replaces ``cv2.imread`` process-wide with
    ``ultralytics.utils.patches.imread``, which returns ``(H, W, 1)`` for a
    grayscale read where OpenCV returns ``(H, W)``. Nothing announces the
    substitution, and code that indexes ``shape[:2]`` keeps working while code
    that assumes two dimensions quietly gets a singleton channel axis. This
    normalises the result either way.

    Args:
        path: Image to read.

    Returns:
        A 2-D ``uint8`` array.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If it cannot be decoded.
    """
    import cv2

    _check_readable(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode {path}")

    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] in (3, 4):
            # The patched reader routes .tif/.tiff through imdecodemulti with
            # IMREAD_UNCHANGED, ignoring the flags it was given, so a colour
            # TIFF can come back three-channel from a grayscale request.
            code = cv2.COLOR_BGRA2GRAY if array.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            array = cv2.cvtColor(array, code)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.uint8)


def write_image(path: Path, image: npt.NDArray[np.uint8]) -> None:
    """Write an image, raising rather than failing silently.

    ``cv2.imwrite`` reports failure by returning ``False``, and almost every
    caller ignores the return value. Under ultralytics' Windows patch the write
    goes through ``imencode`` plus ``ndarray.tofile`` inside a ``try``, so an
    unwritable path or an unsupported extension produces ``False`` and no other
    signal — a dataset generation run would finish reporting success with
    missing pages.

    Args:
        path: Destination. Parent directories are created.
        image: The array to write.

    Raises:
        ValueError: If the write fails.
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    # Two failure modes to unify: OpenCV raises cv2.error for an unsupported
    # extension, while the patched writer swallows its exceptions and returns
    # False. Callers should have to handle one.
    try:
        written = cv2.imwrite(str(path), image)
    except cv2.error as error:
        raise ValueError(f"could not write {path}: {error}") from error
    if not written:
        raise ValueError(
            f"could not write {path}. The extension may be unsupported, the path "
            f"unwritable, or the array the wrong dtype ({image.dtype}, "
            f"shape {image.shape})."
        )


def _check_dpi(dpi: int) -> None:
    """Reject a nonsensical DPI and warn about a merely unwise one.

    Raises:
        ValueError: If ``dpi`` is not positive.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    if dpi < MIN_USEFUL_DPI:
        warnings.warn(
            f"rendering at {dpi} DPI. Staff detection survives well below this, but "
            f"a notehead spans about one staff space, so at this resolution it is "
            f"only a few pixels across and Stage 2 cannot tell one head shape from "
            f"another. {DEFAULT_DPI} is the default for that reason.",
            stacklevel=3,
        )


def _check_readable(path: Path) -> None:
    """Reject a path that is missing or not a file.

    Raises:
        FileNotFoundError: If nothing is there, or it is a directory.
    """
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")


def load_image(path: Path) -> Page:
    """Load a single raster image as one page.

    Args:
        path: The image to read.

    Returns:
        One page. ``dpi`` is ``None``, because a scanner's true resolution is
        not reliably recorded in the file and guessing it would produce a
        confident wrong number in the sync map.

    Raises:
        FileNotFoundError: If the file is missing.
        UnsupportedFormatError: If the suffix is not a raster format.
        ValueError: If the file cannot be decoded.
    """
    _check_readable(path)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise UnsupportedFormatError(
            f"{path.suffix or '<no suffix>'} is not a supported image format; "
            f"expected one of {sorted(IMAGE_SUFFIXES)}"
        )

    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(
            f"could not decode {path}. The suffix says {path.suffix} but the "
            f"contents did not parse; the file is probably truncated or corrupt."
        )

    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return Page(
        image=np.ascontiguousarray(image, dtype=np.uint8),
        source=path,
        page_index=0,
        dpi=None,
    )


def page_count(path: Path) -> int:
    """Return how many pages a document holds.

    Args:
        path: A PDF or raster image.

    Returns:
        The page count. Always 1 for a raster image.

    Raises:
        FileNotFoundError: If the file is missing.
        UnsupportedFormatError: If the format is not recognised.
        EncryptedPdfError: If the PDF needs a password.
    """
    _check_readable(path)
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return 1
    if suffix not in PDF_SUFFIXES:
        raise UnsupportedFormatError(f"{suffix or '<no suffix>'} is not a supported format")

    document = _open_pdf(path)
    try:
        return len(document)
    finally:
        document.close()


def _open_pdf(path: Path) -> Any:
    """Open a PDF, translating library errors into this module's vocabulary.

    Raises:
        EncryptedPdfError: If the file is password-protected.
        ValueError: If the file cannot be parsed as a PDF.
    """
    import pypdfium2 as pdfium

    try:
        return pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as error:
        message = str(error).lower()
        if "password" in message or "encrypt" in message:
            raise EncryptedPdfError(
                f"{path} is password-protected. Decrypt it before ingesting; this "
                f"loader deliberately does not accept passwords."
            ) from error
        raise ValueError(
            f"could not open {path} as a PDF: {error}. The file is probably "
            f"truncated or is not really a PDF."
        ) from error


def load_pdf(
    path: Path,
    dpi: int = DEFAULT_DPI,
    pages: range | list[int] | None = None,
) -> list[Page]:
    """Rasterise a PDF, one array per page.

    Args:
        path: The PDF to read.
        dpi: Render resolution. See the module docstring — this sets staff line
            spacing, which sets every Stage 1 threshold.
        pages: Which page indices to render, zero-based. All of them when
            omitted.

    Returns:
        Pages in document order.

    Raises:
        FileNotFoundError: If the file is missing.
        UnsupportedFormatError: If it is not a PDF.
        EncryptedPdfError: If it is password-protected.
        ValueError: If ``dpi`` is not positive, the PDF holds no pages, or a
            requested index is out of range.
    """
    _check_readable(path)
    if path.suffix.lower() not in PDF_SUFFIXES:
        raise UnsupportedFormatError(f"{path.suffix or '<no suffix>'} is not a PDF")
    _check_dpi(dpi)

    document = _open_pdf(path)
    try:
        total = len(document)
        if total == 0:
            raise ValueError(f"{path} holds no pages")

        wanted = list(range(total)) if pages is None else list(pages)
        out_of_range = [index for index in wanted if not 0 <= index < total]
        if out_of_range:
            raise ValueError(
                f"{path} has {total} page(s); requested index "
                f"{out_of_range[0]} is out of range"
            )

        # pypdfium2 renders at a scale factor relative to 72 DPI, the PDF unit.
        scale = dpi / 72.0

        rendered: list[Page] = []
        for index in wanted:
            page = document[index]
            bitmap = page.render(scale=scale)
            array = np.ascontiguousarray(bitmap.to_numpy(), dtype=np.uint8)
            if array.ndim == 3 and array.shape[2] in (3, 4):
                import cv2

                code = cv2.COLOR_RGBA2BGR if array.shape[2] == 4 else cv2.COLOR_RGB2BGR
                array = np.ascontiguousarray(cv2.cvtColor(array, code), dtype=np.uint8)
            rendered.append(Page(image=array, source=path, page_index=index, dpi=dpi))
        return rendered
    finally:
        document.close()


def load_document(
    path: Path,
    dpi: int = DEFAULT_DPI,
    pages: range | list[int] | None = None,
) -> list[Page]:
    """Load any supported source as a list of pages.

    The entry point callers should reach for: it dispatches on suffix so a
    caller handling a user-supplied file does not have to.

    Args:
        path: A PDF or raster image.
        dpi: Render resolution, used only for PDFs.
        pages: Page indices to load, zero-based. All when omitted.

    Returns:
        Pages in document order. A single-element list for a raster image.

    Raises:
        FileNotFoundError: If the file is missing.
        UnsupportedFormatError: If the format is not recognised.
        EncryptedPdfError: If a PDF is password-protected.
        ValueError: For a corrupt file, an empty PDF, or a bad page index.
    """
    _check_readable(path)
    suffix = path.suffix.lower()

    if suffix in PDF_SUFFIXES:
        return load_pdf(path, dpi=dpi, pages=pages)

    if suffix in IMAGE_SUFFIXES:
        if pages is not None and list(pages) not in ([], [0]):
            raise ValueError(
                f"{path} is a single image; requested pages {list(pages)} but only "
                f"index 0 exists"
            )
        return [] if pages is not None and not list(pages) else [load_image(path)]

    raise UnsupportedFormatError(
        f"{suffix or '<no suffix>'} is not a supported format; expected a PDF or "
        f"one of {sorted(IMAGE_SUFFIXES)}"
    )
