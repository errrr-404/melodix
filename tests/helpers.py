"""Synthetic image generator helpers for geometry tests.

These helpers try to use OpenCV (`cv2`) when available; when it's not
installed they fall back to lightweight NumPy drawing so tests can run in
minimal environments.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised by CI without CV2
    cv2 = None


def blank_page(
    height: int = 600,
    width: int = 800,
    color: int = 255,
) -> npt.NDArray[np.uint8]:
    """Create a blank grayscale page (default white background)."""
    return np.full((height, width), color, dtype=np.uint8)


def _draw_line_np(out: npt.NDArray[np.uint8], p0: tuple[int, int], p1: tuple[int, int], thickness: int = 1) -> None:
    """Draw a simple anti-aliased line by rasterising endpoints and filling a
    square neighbourhood for thickness. Not perfect but fine for tests.
    """
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx // 2
        while x != x1:
            x0r = max(0, x - thickness)
            x1r = min(out.shape[1], x + thickness + 1)
            y0r = max(0, y - thickness)
            y1r = min(out.shape[0], y + thickness + 1)
            out[y0r:y1r, x0r:x1r] = 0
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while y != y1:
            x0r = max(0, x - thickness)
            x1r = min(out.shape[1], x + thickness + 1)
            y0r = max(0, y - thickness)
            y1r = min(out.shape[0], y + thickness + 1)
            out[y0r:y1r, x0r:x1r] = 0
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy


def draw_staff(
    image: npt.NDArray[np.uint8],
    top_row: int = 200,
    spacing: int = 20,
    thickness: int = 2,
    x_start: int = 100,
    x_end: int = 700,
    num_lines: int = 5,
) -> npt.NDArray[np.uint8]:
    """Draw horizontal staff lines on an image copy.

    Matches the call-style used in tests: `draw_staff(page, top_row=200, spacing=14)`.
    """
    for i in range(num_lines):
        y = int(round(top_row + i * spacing))
        xs = int(x_start)
        xe = int(x_end)
        if cv2 is not None:
            cv2.line(image, (xs, y), (xe, y), color=0, thickness=int(thickness))
        else:
            image[y : y + int(thickness), xs : xe + 1] = 0
    return image


def draw_notehead(
    image: npt.NDArray[np.uint8],
    center_x: int,
    center_y: int,
    radius: int = 8,
) -> npt.NDArray[np.uint8]:
    """Draw a filled notehead (elliptical) on an image copy."""
    if cv2 is not None:
        cv2.ellipse(image, (int(center_x), int(center_y)), (int(radius * 1.3), int(radius)), 0, 0, 360, color=0, thickness=-1)
    else:
        ys, xs = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = ((xs - center_x) / (radius * 1.3)) ** 2 + ((ys - center_y) / radius) ** 2 <= 1.0
        image[mask] = 0
    return image


def draw_stem(
    image: npt.NDArray[np.uint8],
    x: int,
    top_row: int,
    bottom_row: int,
    width: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draw a vertical stem on an image copy.

    Tests call this as `draw_stem(page, x=x, top_row=..., bottom_row=..., width=3)`.
    """
    if cv2 is not None:
        cv2.line(image, (int(x), int(top_row)), (int(x), int(bottom_row)), color=0, thickness=int(width))
    else:
        image[int(top_row) : int(bottom_row), int(x) : int(x + width)] = 0
    return image
