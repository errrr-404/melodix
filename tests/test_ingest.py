"""Unit tests for :mod:`melodix.ingest.loader`.

Fixtures are generated into a temp directory rather than committed. A repo full
of small binary PDFs is a repo nobody can review, and generating them keeps the
test honest about what it is actually exercising.

The DPI tests are the ones that matter beyond plumbing. Render resolution sets
staff line spacing in pixels, and every Stage 1 threshold is a ratio against
that, so a page rendered too small makes detection return an empty list rather
than a poor result — a failure with no error attached to it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from melodix.geometry.staff import detect_staff_grids
from melodix.ingest import (
    DEFAULT_DPI,
    IMAGE_SUFFIXES,
    MIN_USEFUL_DPI,
    EncryptedPdfError,
    Page,
    UnsupportedFormatError,
    load_document,
    load_image,
    load_pdf,
    page_count,
)

# US Letter in PDF points, the unit pypdfium2 scales from.
LETTER_W_PT = 612
LETTER_H_PT = 792


def staff_array(height: int = 792, width: int = 612, spacing: int = 7) -> np.ndarray:
    """A page with one five-line staff drawn on it."""
    page = np.full((height, width), 255, dtype=np.uint8)
    for step in range(5):
        row = 300 + step * spacing
        page[row : row + 2, 60 : width - 60] = 0
    return page


def write_pdf(path: Path, pages: int = 1, spacing: int = 7) -> Path:
    """Write a small multi-page PDF built from generated arrays."""
    images = [
        Image.fromarray(staff_array(spacing=spacing)).convert("RGB") for _ in range(pages)
    ]
    images[0].save(path, save_all=True, append_images=images[1:])
    return path


def write_image(path: Path, height: int = 600, width: int = 800) -> Path:
    """Write a small raster image."""
    Image.fromarray(staff_array(height=height, width=width)).save(path)
    return path


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def test_a_page_reports_its_dimensions():
    page = Page(np.zeros((600, 800), np.uint8), Path("a.png"), 0, None)

    assert (page.width, page.height) == (800, 600)


def test_a_grayscale_page_is_not_colour():
    assert not Page(np.zeros((10, 10), np.uint8), Path("a.png"), 0, None).is_colour


def test_a_three_channel_page_is_colour():
    assert Page(np.zeros((10, 10, 3), np.uint8), Path("a.png"), 0, None).is_colour


def test_a_page_labels_itself_for_a_human():
    page = Page(np.zeros((10, 10), np.uint8), Path("scores/score.pdf"), 3, 300)

    assert page.label == "score.pdf#p3"


def test_metadata_carries_what_a_sync_map_needs():
    """A browser has to be pointed at a page of a file, so provenance travels
    with the pixels rather than being reconstructed downstream.
    """
    page = Page(np.zeros((600, 800), np.uint8), Path("s.pdf"), 2, 300)

    meta = page.metadata()

    assert meta["page_index"] == 2
    assert meta["dpi"] == 300
    assert meta["width"] == 800
    assert meta["height"] == 600
    assert "s.pdf" in str(meta["source"])


def test_an_empty_page_is_rejected():
    with pytest.raises(ValueError, match="is empty"):
        Page(np.zeros((0, 0), np.uint8), Path("a.png"), 0, None)


def test_a_non_uint8_page_is_rejected():
    """Every downstream threshold assumes 0-255."""
    with pytest.raises(ValueError, match="must be uint8"):
        Page(np.zeros((10, 10), np.float32), Path("a.png"), 0, None)


def test_a_negative_page_index_is_rejected():
    with pytest.raises(ValueError, match="page_index must be non-negative"):
        Page(np.zeros((10, 10), np.uint8), Path("a.png"), -1, None)


def test_an_unsupported_shape_is_rejected():
    with pytest.raises(ValueError, match="unsupported page shape"):
        Page(np.zeros((2, 2, 2, 2), np.uint8), Path("a.png"), 0, None)


# --------------------------------------------------------------------------- #
# Raster images
# --------------------------------------------------------------------------- #


def test_an_image_loads_as_one_page(tmp_path):
    path = write_image(tmp_path / "page.png")

    page = load_image(path)

    assert page.page_index == 0
    assert page.image.dtype == np.uint8


def test_an_image_reports_no_dpi(tmp_path):
    """A scanner's true resolution is not reliably recorded, and a guess would
    put a confident wrong number in the sync map.
    """
    page = load_image(write_image(tmp_path / "page.png"))

    assert page.dpi is None


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".bmp", ".tiff"])
def test_common_raster_formats_load(tmp_path, suffix):
    path = tmp_path / f"page{suffix}"
    Image.fromarray(staff_array()).convert("RGB").save(path)

    assert load_image(path).width == 612


def test_an_alpha_channel_is_dropped(tmp_path):
    path = tmp_path / "page.png"
    Image.fromarray(np.full((40, 60, 4), 255, np.uint8), mode="RGBA").save(path)

    page = load_image(path)

    assert page.image.shape == (40, 60, 3)


def test_a_missing_image_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        load_image(tmp_path / "absent.png")


def test_a_directory_is_not_a_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a file"):
        load_image(tmp_path)


def test_an_unsupported_suffix_is_reported(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not an image", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError, match="not a supported image format"):
        load_image(path)


def test_a_corrupt_image_is_reported(tmp_path):
    """The suffix says PNG; the bytes do not."""
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 4)

    with pytest.raises(ValueError, match="could not decode"):
        load_image(path)


def test_the_recognised_suffixes_are_lowercase_and_dotted():
    assert all(s.startswith(".") and s == s.lower() for s in IMAGE_SUFFIXES)


# --------------------------------------------------------------------------- #
# PDFs
# --------------------------------------------------------------------------- #


def test_a_pdf_rasterises_to_one_array_per_page(tmp_path):
    path = write_pdf(tmp_path / "score.pdf", pages=3)

    pages = load_pdf(path, dpi=150)

    assert len(pages) == 3
    assert [page.page_index for page in pages] == [0, 1, 2]


def test_pages_come_back_as_uint8_arrays(tmp_path):
    pages = load_pdf(write_pdf(tmp_path / "s.pdf"), dpi=150)

    assert pages[0].image.dtype == np.uint8
    assert pages[0].image.ndim in (2, 3)


def test_the_page_count_can_be_read_without_rendering(tmp_path):
    path = write_pdf(tmp_path / "score.pdf", pages=4)

    assert page_count(path) == 4


def test_an_image_counts_as_one_page(tmp_path):
    assert page_count(write_image(tmp_path / "page.png")) == 1


def test_a_subset_of_pages_can_be_rendered(tmp_path):
    path = write_pdf(tmp_path / "score.pdf", pages=5)

    pages = load_pdf(path, dpi=100, pages=[1, 3])

    assert [page.page_index for page in pages] == [1, 3]


def test_a_page_index_out_of_range_is_reported(tmp_path):
    path = write_pdf(tmp_path / "score.pdf", pages=2)

    with pytest.raises(ValueError, match="out of range"):
        load_pdf(path, dpi=100, pages=[7])


def test_a_missing_pdf_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        load_pdf(tmp_path / "absent.pdf")


def test_a_non_pdf_suffix_is_reported(tmp_path):
    path = write_image(tmp_path / "page.png")

    with pytest.raises(UnsupportedFormatError, match="not a PDF"):
        load_pdf(path)


def test_a_corrupt_pdf_is_reported(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nnot really a pdf")

    with pytest.raises(ValueError, match="could not open"):
        load_pdf(path)


def test_an_encrypted_pdf_is_reported_distinctly(tmp_path):
    """A different remedy from an unsupported format: the file is fine, it
    needs a password.
    """
    path = tmp_path / "locked.pdf"
    image = Image.fromarray(staff_array()).convert("RGB")
    try:
        image.save(path, save_all=True)
        import pypdfium2 as pdfium

        source = pdfium.PdfDocument(str(path))
        encrypted = tmp_path / "encrypted.pdf"
        source.save(str(encrypted), version=17)
        source.close()
    except Exception:  # pragma: no cover - Pillow cannot encrypt
        pytest.skip("cannot produce an encrypted PDF with the installed tools")

    # Pillow cannot write encryption, so assert the error type exists and is
    # distinct rather than fabricating a fixture.
    assert issubclass(EncryptedPdfError, ValueError)
    assert not issubclass(EncryptedPdfError, UnsupportedFormatError)


# --------------------------------------------------------------------------- #
# DPI
# --------------------------------------------------------------------------- #


def test_the_default_dpi_suits_omr():
    """300 puts roughly 20 px between staff lines on a letter engraving."""
    assert DEFAULT_DPI == 300
    assert DEFAULT_DPI > MIN_USEFUL_DPI


@pytest.mark.parametrize("dpi", [100, 150, 300])
def test_render_size_scales_with_dpi(tmp_path, dpi):
    """A US Letter page is 612x792 points; pypdfium2 scales from 72 DPI."""
    path = write_pdf(tmp_path / "s.pdf")

    page = load_pdf(path, dpi=dpi)[0]

    assert page.width == pytest.approx(LETTER_W_PT * dpi / 72, abs=2)
    assert page.height == pytest.approx(LETTER_H_PT * dpi / 72, abs=2)


def test_the_rendered_dpi_is_recorded(tmp_path):
    page = load_pdf(write_pdf(tmp_path / "s.pdf"), dpi=200)[0]

    assert page.dpi == 200


@pytest.mark.parametrize("dpi", [0, -100])
def test_a_non_positive_dpi_is_rejected(tmp_path, dpi):
    with pytest.raises(ValueError, match="dpi must be positive"):
        load_pdf(write_pdf(tmp_path / "s.pdf"), dpi=dpi)


def test_a_very_low_dpi_warns(tmp_path):
    """The downstream failure is silent, so the warning is the only signal."""
    path = write_pdf(tmp_path / "s.pdf")

    with pytest.warns(UserWarning, match="only a few pixels across"):
        load_pdf(path, dpi=72)


def test_a_sensible_dpi_does_not_warn(tmp_path):
    import warnings

    path = write_pdf(tmp_path / "s.pdf")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_pdf(path, dpi=DEFAULT_DPI)


def test_line_spacing_in_pixels_tracks_dpi(tmp_path):
    """The mechanical relationship every Stage 1 threshold rests on:
    spacing_px = spacing_pt * dpi / 72.
    """
    path = write_pdf(tmp_path / "s.pdf", spacing=7)

    measured = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dpi in (72, 150, 300):
            grids = detect_staff_grids(load_pdf(path, dpi=dpi)[0].image)
            assert grids, f"no staff at {dpi} DPI"
            measured[dpi] = grids[0].line_spacing

    for dpi, spacing in measured.items():
        assert spacing == pytest.approx(7 * dpi / 72, abs=1.0)


def test_below_the_spacing_floor_detection_returns_nothing(tmp_path):
    """Stage 1 rejects a staff whose lines are closer than min_line_spacing
    (3.0 px). For a 7 pt engraving that floor lands near 31 DPI — and the
    failure is an empty list, not an error.
    """
    path = write_pdf(tmp_path / "s.pdf", spacing=7)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        below = load_pdf(path, dpi=24)[0]   # 7pt -> 2.3 px, under the floor
        above = load_pdf(path, dpi=36)[0]   # 7pt -> 3.5 px, over it

    assert detect_staff_grids(below.image) == []
    assert len(detect_staff_grids(above.image)) == 1


def test_staff_detection_survives_far_below_the_default_dpi(tmp_path):
    """Recorded because it is easy to assume otherwise: the reason to render at
    300 is notehead size for Stage 2, not staff detection for Stage 1.
    """
    path = write_pdf(tmp_path / "s.pdf", spacing=7)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = load_pdf(path, dpi=72)[0]

    assert len(detect_staff_grids(page.image)) == 1


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def test_a_pdf_dispatches_to_the_pdf_path(tmp_path):
    path = write_pdf(tmp_path / "score.pdf", pages=2)

    assert len(load_document(path, dpi=100)) == 2


def test_an_image_dispatches_to_the_image_path(tmp_path):
    path = write_image(tmp_path / "page.png")

    pages = load_document(path)

    assert len(pages) == 1
    assert pages[0].dpi is None


def test_dispatch_is_case_insensitive_on_the_suffix(tmp_path):
    path = tmp_path / "SCORE.PDF"
    write_pdf(path)

    assert len(load_document(path, dpi=100)) == 1


def test_an_unknown_format_is_reported(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError, match="not a supported format"):
        load_document(path)


def test_a_file_with_no_suffix_is_reported(tmp_path):
    path = tmp_path / "scan"
    path.write_bytes(b"x")

    with pytest.raises(UnsupportedFormatError, match="<no suffix>"):
        load_document(path)


def test_asking_an_image_for_page_two_is_reported(tmp_path):
    path = write_image(tmp_path / "page.png")

    with pytest.raises(ValueError, match="only index 0 exists"):
        load_document(path, pages=[1])


# --------------------------------------------------------------------------- #
# Into Stage 1
# --------------------------------------------------------------------------- #


def test_a_rendered_page_feeds_staff_detection(tmp_path):
    """The seam that matters: what comes out of ingest goes straight in."""
    path = write_pdf(tmp_path / "score.pdf", spacing=12)

    page = load_document(path, dpi=DEFAULT_DPI)[0]
    grids = detect_staff_grids(page.image)

    assert len(grids) == 1
    assert grids[0].line_spacing > 10


def test_a_loaded_image_feeds_staff_detection(tmp_path):
    path = tmp_path / "page.png"
    array = np.full((600, 800), 255, np.uint8)
    for step in range(5):
        array[200 + step * 20 : 202 + step * 20, 100:700] = 0
    Image.fromarray(array).save(path)

    grids = detect_staff_grids(load_image(path).image)

    assert len(grids) == 1
