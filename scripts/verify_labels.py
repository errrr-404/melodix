"""Check that a dataset's boxes actually sit on their glyphs.

Label misalignment is the most expensive defect a dataset can carry, because it
does not look like a defect. Training converges, the loss curve is smooth, the
metrics are plausible, and the resulting detector is systematically offset. The
generator shipped exactly that bug once: :func:`augment` rotated the page while
leaving the ground truth at its pre-rotation coordinates.

Three measurements per box, none of which requires knowing what the glyph is:

**Ink fraction** — dark pixels inside the box over box area. A box on paper
reads near zero. Weak on its own, because a box that has drifted onto a staff
line still finds ink.

**Centroid offset** — distance from the box centre to the ink's centre of mass,
as a fraction of box size. This is the strong signal: a systematically shifted
label set shows a consistent non-zero offset, and a rotation-induced one shows
offsets that grow with distance from the page centre.

**Tightness** — how much of the box the ink's own bounding box fills. A correct
box hugs its glyph; a stale one contains a corner of it plus paper.

Centroid offset has one confound worth knowing before reading a report: in dense
notation a small symbol often sits against a large dark one, and that neighbour's
ink inside the box drags the centre of mass. An ``open_modifier`` tucked under a
beam measures an offset of 0.33 with a box that is pixel-exact. So a high offset
on a small class adjacent to beams or noteheads is weak evidence on its own.

:func:`radius_correlation` is the check that is not confounded. A label set left
behind by a rotation is displaced in proportion to distance from the page centre,
so offset rises with radius. Neighbouring ink does not do that. Measured on this
project's dataset the correlation is -0.06 and the offset is flat across radius
bands, which is what a correctly transformed label set looks like.

Usage::

    python scripts/verify_labels.py --data datasets/melodix_synth
    python scripts/verify_labels.py --data d --split train --sample 200 --json out.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from melodix.ingest import read_grayscale  # noqa: E402
from melodix.vision.dataset import (  # noqa: E402
    IMAGE_SUFFIXES,
    parse_label_file,
)
from melodix.vision.labels import NUM_CLASSES, SymbolClass, class_names  # noqa: E402

#: A box whose ink centre of mass sits further than this from the box centre,
#: as a fraction of box size, is suspicious. Well clear of the ~0.05 a correct
#: box shows from asymmetric glyphs like flags and grace notes.
CENTROID_TOLERANCE = 0.25

#: Below this ink fraction a box is very likely sitting on paper.
MIN_INK_FRACTION = 0.02

#: Smallest spread of mean offset across radius bands that counts as a real
#: rotation trend. A correct dataset measures around 0.005; a 6-degree stale
#: rotation measures an order of magnitude more.
MIN_RADIUS_EFFECT = 0.05

#: Fallback cutoff, used only for a page with no bimodal split to find.
#:
#: A fixed cutoff is not good enough and this tool originally used one. On a
#: page that augmentation has brightened and blurred, a one-pixel stroke fades
#: well above 128 and reads as paper — which made accents, marcato and open
#: modifiers look catastrophically misaligned when their boxes were exactly
#: right. Ink is found with Otsu instead, per page, the same way
#: :func:`melodix.geometry.staff.binarize` does it.
FALLBACK_INK_THRESHOLD = 128


@dataclass(frozen=True, slots=True)
class BoxReport:
    """What one box measured."""

    class_id: int
    ink_fraction: float
    centroid_offset: float
    tightness: float

    @property
    def suspicious(self) -> bool:
        """Whether this box looks misaligned."""
        return (
            self.ink_fraction < MIN_INK_FRACTION
            or self.centroid_offset > CENTROID_TOLERANCE
        )


def ink_mask(image: np.ndarray) -> np.ndarray:
    """Return a boolean ink mask for a whole page, thresholded by Otsu.

    Per page rather than per box: a box holding one thin stroke has no bimodal
    histogram of its own, so a local threshold would invent a split in what is
    almost all paper.
    """
    threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not 0 < threshold < 255:  # pragma: no cover - a blank page
        threshold = FALLBACK_INK_THRESHOLD
    return image <= threshold


def measure_box(
    image: np.ndarray, x0: float, y0: float, x1: float, y1: float, class_id: int
) -> BoxReport | None:
    """Measure one box against the ink underneath it.

    Args:
        image: Grayscale page.
        x0: Left edge in pixels.
        y0: Top edge in pixels.
        x1: Right edge in pixels.
        y1: Bottom edge in pixels.
        class_id: Which class the box claims.

    Returns:
        The measurement, or ``None`` if the box has no area on the page.
    """
    height, width = image.shape[:2]
    left, top = int(max(0, x0)), int(max(0, y0))
    right, bottom = int(min(width, x1)), int(min(height, y1))
    if right - left < 2 or bottom - top < 2:
        return None

    crop = image[top:bottom, left:right]
    mask = crop if crop.dtype == bool else ink_mask(image)[top:bottom, left:right]
    ink = int(mask.sum())
    area = mask.size
    if ink == 0:
        return BoxReport(class_id, 0.0, 1.0, 0.0)

    rows, cols = np.nonzero(mask)
    # Centre of mass of the ink, relative to the box centre, scaled by box size.
    centre_x, centre_y = (right - left) / 2.0, (bottom - top) / 2.0
    offset_x = abs(float(cols.mean()) - centre_x) / max(1.0, right - left)
    offset_y = abs(float(rows.mean()) - centre_y) / max(1.0, bottom - top)

    # How much of the box the ink's own bounding box fills.
    span_x = (cols.max() - cols.min() + 1) / max(1.0, right - left)
    span_y = (rows.max() - rows.min() + 1) / max(1.0, bottom - top)

    return BoxReport(
        class_id=class_id,
        ink_fraction=ink / area,
        centroid_offset=float(np.hypot(offset_x, offset_y)),
        tightness=float(min(1.0, (span_x + span_y) / 2.0)),
    )


def radius_correlation(
    root: Path, split: str = "train", pages: int = 60, classes: set[int] | None = None
) -> dict[str, Any]:
    """Test for the signature of a rotation-stale label set.

    A box left behind by an image rotation is displaced proportionally to its
    distance from the centre of rotation, so centroid offset climbs with radius.
    Nothing else in a dataset produces that pattern — in particular neighbouring
    ink, which is the main source of false positives elsewhere in this tool,
    is independent of where on the page a symbol sits.

    Args:
        root: Dataset root.
        split: Split to read.
        pages: How many pages to measure.
        classes: Class ids to include. Defaults to large, well-separated
            symbols, whose offsets are not polluted by neighbours.

    Correlation alone is not enough, and assuming otherwise produces false
    alarms. A correct label set can correlate strongly at a trivial magnitude:
    measured on a clean fixture, correlation 0.79 across offsets that ranged
    from 0.001 to 0.009 — statistically a trend, practically nothing. So the
    effect size is reported alongside it as ``offset_span``, the spread of mean
    offset across radius bands. A real rotation moves boxes by tens of pixels
    and produces a span an order of magnitude larger.

    Returns:
        The correlation, the effect size, and the mean offset per radius band.
    """
    if classes is None:
        classes = {
            int(SymbolClass.CROSS_NOTEHEAD),
            int(SymbolClass.ROUND_NOTEHEAD),
            int(SymbolClass.PERCUSSION_CLEF),
            int(SymbolClass.DIAMOND_NOTEHEAD),
        }

    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"no {split} images at {images_dir}")

    radii: list[float] = []
    offsets: list[float] = []
    listing = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    for path in listing[:pages]:
        label_path = labels_dir / f"{path.stem}.txt"
        if not label_path.exists():
            continue
        image = read_grayscale(path)
        mask = ink_mask(image)
        height, width = image.shape[:2]
        limit = float(np.hypot(width / 2, height / 2))
        for annotation in parse_label_file(label_path):
            if int(annotation.symbol) not in classes:
                continue
            x0, y0, x1, y1 = annotation.box.to_pixels(width, height)
            report = measure_box(mask, x0, y0, x1, y1, int(annotation.symbol))
            if report is None:
                continue
            centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
            radii.append(float(np.hypot(centre_x - width / 2, centre_y - height / 2)) / limit)
            offsets.append(report.centroid_offset)

    if len(radii) < 2:
        return {"boxes": len(radii), "correlation": 0.0, "offset_span": 0.0, "bands": []}

    array_r, array_o = np.array(radii), np.array(offsets)

    # A flawless label set has identical offsets everywhere, so the offset
    # series has zero variance and corrcoef divides by it, returning NaN. NaN
    # compares False against every threshold, which would make the verdict
    # silently treat a perfect dataset as unjudged. Zero variance means no
    # trend, which is exactly what a correlation of 0.0 says.
    if array_r.std() == 0.0 or array_o.std() == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(array_r, array_o)[0, 1])

    bands = []
    for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        selected = (array_r >= low) & (array_r < high)
        if selected.any():
            bands.append(
                {
                    "from": low,
                    "to": high,
                    "boxes": int(selected.sum()),
                    "mean_offset": float(array_o[selected].mean()),
                }
            )

    spans = [band["mean_offset"] for band in bands]
    return {
        "boxes": len(radii),
        "correlation": correlation,
        "offset_span": (max(spans) - min(spans)) if spans else 0.0,
        "bands": bands,
    }


def verify_split(
    root: Path, split: str = "train", sample: int | None = 100, seed: int = 0
) -> dict[str, Any]:
    """Measure every box on a sample of pages.

    Args:
        root: Dataset root.
        split: Split directory to read.
        sample: How many pages to measure. All of them when ``None``.
        seed: Seeds the page sample, so a report is reproducible.

    Returns:
        Per-class and overall statistics, plus the worst offenders.

    Raises:
        FileNotFoundError: If the split has no image directory.
    """
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"no {split} images at {images_dir}")

    pages = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if sample is not None and sample < len(pages):
        pages = random.Random(seed).sample(pages, sample)
        pages.sort()

    by_class: dict[int, list[BoxReport]] = defaultdict(list)
    worst: list[tuple[float, str, str]] = []

    for path in pages:
        image = read_grayscale(path)
        label_path = labels_dir / f"{path.stem}.txt"
        if not label_path.exists():
            continue

        # Threshold once per page, not once per box.
        mask = ink_mask(image)
        height, width = image.shape[:2]
        for annotation in parse_label_file(label_path):
            x0, y0, x1, y1 = annotation.box.to_pixels(width, height)
            report = measure_box(mask, x0, y0, x1, y1, int(annotation.symbol))
            if report is None:
                continue
            by_class[report.class_id].append(report)
            if report.suspicious:
                worst.append(
                    (report.centroid_offset, path.name, SymbolClass(report.class_id).name)
                )

    names = class_names()
    per_class = []
    for class_id in range(NUM_CLASSES):
        reports = by_class.get(class_id, [])
        if not reports:
            per_class.append({"index": class_id, "name": names[class_id], "boxes": 0})
            continue
        per_class.append(
            {
                "index": class_id,
                "name": names[class_id],
                "boxes": len(reports),
                "mean_ink": float(np.mean([r.ink_fraction for r in reports])),
                "mean_offset": float(np.mean([r.centroid_offset for r in reports])),
                "max_offset": float(np.max([r.centroid_offset for r in reports])),
                "mean_tightness": float(np.mean([r.tightness for r in reports])),
                "suspicious": sum(1 for r in reports if r.suspicious),
            }
        )

    total = sum(len(v) for v in by_class.values())
    flagged = sum(row.get("suspicious", 0) for row in per_class)
    all_reports = [r for reports in by_class.values() for r in reports]

    return {
        "data": str(root),
        "split": split,
        "pages": len(pages),
        "boxes": total,
        "suspicious": flagged,
        "suspicious_rate": flagged / total if total else 0.0,
        "mean_ink": float(np.mean([r.ink_fraction for r in all_reports])) if total else 0.0,
        "mean_offset": (
            float(np.mean([r.centroid_offset for r in all_reports])) if total else 0.0
        ),
        "per_class": per_class,
        "worst": [
            {"offset": offset, "page": page, "class": name}
            for offset, page, name in sorted(worst, reverse=True)[:15]
        ],
    }


def verdict(report: dict[str, Any]) -> str:
    """Judge a report, weighting the unconfounded signal highest.

    The radius correlation decides. A raised suspicious rate on its own does
    not mean misalignment: it is dominated by small symbols sitting against
    beams and noteheads, whose neighbours' ink drags the measured centroid
    while the box itself is exact.
    """
    radius = report.get("radius") or {}
    # Both, not either: a trend that moves boxes by a thousandth of their width
    # is not misalignment however cleanly it correlates.
    if (
        radius.get("boxes")
        and radius["correlation"] >= 0.3
        and radius.get("offset_span", 0.0) >= MIN_RADIUS_EFFECT
    ):
        return (
            "MISALIGNED - offset climbs with distance from the page centre, the "
            "signature of boxes left behind by a rotation. Regenerate before training."
        )
    if report["suspicious_rate"] >= 0.20:
        return (
            f"MISALIGNED - {report['suspicious_rate'] * 100:.1f}% of boxes flagged, "
            f"too many for the neighbouring-ink confound. Investigate."
        )
    if report["suspicious_rate"] >= 0.02:
        return (
            f"aligned. {report['suspicious_rate'] * 100:.1f}% flagged, but the radius "
            f"check is flat, so those are small symbols against dense ink rather than "
            f"stale boxes. No regeneration needed."
        )
    return "aligned. No regeneration needed."


def print_report(report: dict[str, Any]) -> None:
    """Print the human-readable summary."""
    print(f"{report['data']} [{report['split']}]")
    print(f"  {report['pages']} pages, {report['boxes']} boxes measured")
    print(f"  mean ink fraction  {report['mean_ink']:.3f}")
    print(f"  mean centroid offset {report['mean_offset']:.4f}  (0 = perfect)")
    print(
        f"  suspicious: {report['suspicious']} "
        f"({report['suspicious_rate'] * 100:.2f}%)"
    )

    print("\nPER CLASS, worst centroid offset first")
    rows = [row for row in report["per_class"] if row["boxes"]]
    for row in sorted(rows, key=lambda r: -r["mean_offset"]):
        print(
            f"  offset {row['mean_offset']:.4f} (max {row['max_offset']:.3f})  "
            f"ink {row['mean_ink']:.3f}  tight {row['mean_tightness']:.3f}  "
            f"n={row['boxes']:<6d} {row['name']}"
            + (f"  [{row['suspicious']} flagged]" if row["suspicious"] else "")
        )

    missing = [row["name"] for row in report["per_class"] if not row["boxes"]]
    if missing:
        print(f"\nno boxes sampled for: {', '.join(missing)}")

    if report["worst"]:
        print("\nWORST BOXES")
        for row in report["worst"][:10]:
            print(f"  offset {row['offset']:.3f}  {row['class']}  {row['page']}")

    radius = report.get("radius")
    if radius and radius["boxes"]:
        print("\nROTATION-STALENESS CHECK (offset vs distance from page centre)")
        print(
            f"  correlation {radius['correlation']:+.4f}  "
            f"effect size {radius.get('offset_span', 0.0):.4f}  "
            f"over {radius['boxes']} boxes"
        )
        for band in radius["bands"]:
            print(
                f"    radius {band['from']:.2f}-{band['to']:.2f}: "
                f"mean offset {band['mean_offset']:.4f}  (n={band['boxes']})"
            )
        rising = (
            radius["correlation"] >= 0.3
            and radius.get("offset_span", 0.0) >= MIN_RADIUS_EFFECT
        )
        reading = "RISING - investigate" if rising else "FLAT - boxes track the page"
        print(f"  a rotation-stale set climbs sharply with radius; this one is {reading}")

    print(f"\nVERDICT: {verdict(report)}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="dataset root")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--sample", type=int, default=100, help="pages to measure; 0 for all"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    report = verify_split(
        args.data, args.split, sample=args.sample or None, seed=args.seed
    )
    report["radius"] = radius_correlation(args.data, args.split)
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json}")

    return 0 if not verdict(report).startswith("MISALIGNED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
