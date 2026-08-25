"""Source documents in, page arrays out.

Named ``ingest`` rather than ``io`` so it does not shadow the standard library
module for readers of this codebase::

    from melodix.ingest import load_document

    for page in load_document(Path("score.pdf"), dpi=300):
        grids = detect_staff_grids(page.image)

Render DPI is a correctness setting, not a quality knob: it determines staff
line spacing in pixels, which every Stage 1 threshold is expressed against. See
:mod:`melodix.ingest.loader`.
"""

from melodix.ingest.loader import (
    DEFAULT_DPI,
    IMAGE_SUFFIXES,
    MIN_USEFUL_DPI,
    PDF_SUFFIXES,
    EncryptedPdfError,
    Page,
    UnsupportedFormatError,
    load_document,
    load_image,
    load_pdf,
    page_count,
    read_grayscale,
    write_image,
)

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
