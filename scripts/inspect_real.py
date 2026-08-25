"""Draw what the model sees on a real page, so a human can look at it.

No ground truth, no metrics, no mAP. Four phases of instruments have been built
for a measurement nobody has taken, and every number recorded so far is
synthetic scored against synthetic. This is the cheapest available signal about
whether any of it transfers, and it costs one PDF and no annotation time.

**Read the Stage 1 overlay first.** It is the difference between two failures
that look identical in a detection count:

- staves drawn, few boxes  — Stage 1 works, Stage 2 does not transfer
- no staves drawn          — Stage 1 failed, and Stage 2 never had a chance

Without that distinction a disappointing page tells you nothing about where to
look. Deskew runs first, as it must, so the overlay also shows whether the page
was straightened.

The confidence floor defaults deliberately low. On a first real page it matters
more to see what the model *nearly* found than to see a clean view: a page where
everything sits at 0.15 is a different diagnosis from one where nothing is
predicted at all.

Usage::

    python scripts/inspect_real.py --input scan.pdf --weights models/.../best.pt
    python scripts/inspect_real.py --input page.png --weights w.pt --conf 0.05
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from melodix.ingest import DEFAULT_DPI, Page, load_document, write_image  # noqa: E402
from melodix.vision.labels import SymbolCategory  # noqa: E402

#: One colour per category, BGR. Enough to tell at a glance whether the model is
#: finding noteheads or hallucinating structure.
CATEGORY_COLOURS: dict[SymbolCategory, tuple[int, int, int]] = {
    SymbolCategory.NOTEHEAD: (0, 0, 220),
    SymbolCategory.REST: (200, 0, 200),
    SymbolCategory.DURATION: (0, 140, 255),
    SymbolCategory.MODIFIER: (0, 170, 0),
    SymbolCategory.STRUCTURE: (200, 120, 0),
}

#: Stage 1 overlay, drawn under the detections.
STAFF_COLOUR = (255, 190, 0)
BARLINE_COLOUR = (255, 120, 60)

#: Low on purpose. See the module docstring.
DEFAULT_CONFIDENCE = 0.10


def overlay_stage_one(canvas: npt.NDArray[np.uint8], page: npt.NDArray[np.uint8]) -> dict:
    """Draw Stage 1's view onto the canvas and report what it found.

    Args:
        canvas: Colour image to draw on, modified in place.
        page: The grayscale page Stage 1 should read.

    Returns:
        A summary: staves, systems, measures, and the skew that was corrected.
    """
    from melodix.geometry import (
        build_systems,
        deskew,
        detect_staff_grids,
        detect_vertical_segments,
    )

    result = deskew(page)
    grids = detect_staff_grids(result.image)

    summary: dict = {
        "skew_deg": round(result.estimate.skew_deg, 3),
        "deskewed": bool(result.applied),
        "staves": len(grids),
        "systems": 0,
        "measures": 0,
    }
    if not grids:
        return summary

    for grid in grids:
        for line in grid.lines:
            cv2.line(
                canvas,
                (grid.x_start, int(round(line.y))),
                (grid.x_end, int(round(line.y))),
                STAFF_COLOUR,
                1,
            )

    strokes = detect_vertical_segments(result.image, staff_spacing=grids[0].line_spacing)
    systems = build_systems(grids, strokes)
    summary["systems"] = len(systems)
    summary["measures"] = sum(system.measure_count for system in systems)
    summary["line_spacing"] = round(grids[0].line_spacing, 2)

    for system in systems:
        for column in system.columns:
            cv2.line(
                canvas,
                (column.x_int, int(system.y_top)),
                (column.x_int, int(system.y_bottom)),
                BARLINE_COLOUR,
                1,
            )
    return summary


def draw_detections(canvas: npt.NDArray[np.uint8], detections, label_every: bool) -> None:
    """Draw each detection as a coloured box with its class and confidence."""
    for hit in detections:
        colour = CATEGORY_COLOURS.get(hit.category, (60, 60, 60))
        cv2.rectangle(
            canvas,
            (int(hit.x_min), int(hit.y_min)),
            (int(hit.x_max), int(hit.y_max)),
            colour,
            1,
        )
        if label_every:
            cv2.putText(
                canvas,
                f"{hit.class_name} {hit.confidence:.2f}",
                (int(hit.x_min), max(8, int(hit.y_min) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                colour,
                1,
                cv2.LINE_AA,
            )


def summarise(detections) -> dict:
    """Describe a page's detections without judging them."""
    if not detections:
        return {"count": 0, "by_class": {}, "confidence": {}}

    scores = np.array([hit.confidence for hit in detections])
    counts = Counter(hit.class_name for hit in detections)
    return {
        "count": len(detections),
        "by_class": dict(counts.most_common()),
        "confidence": {
            "min": round(float(scores.min()), 3),
            "p25": round(float(np.percentile(scores, 25)), 3),
            "median": round(float(np.median(scores)), 3),
            "p75": round(float(np.percentile(scores, 75)), 3),
            "max": round(float(scores.max()), 3),
            "above_0.5": int((scores >= 0.5).sum()),
        },
    }


def inspect_page(page: Page, detector, out_dir: Path, staves: bool, labels: bool) -> dict:
    """Annotate one page and return what was on it.

    Args:
        page: A page from :mod:`melodix.ingest`.
        detector: Anything with ``detect(image) -> PageDetections``.
        out_dir: Where to write the annotated image.
        staves: Whether to overlay Stage 1.
        labels: Whether to write a class name beside every box.

    Returns:
        The page summary, including the Stage 1 result when requested.
    """
    grey = page.image if not page.is_colour else cv2.cvtColor(page.image, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    report: dict = {"page": page.page_index, "size": [page.width, page.height]}
    if staves:
        report["stage1"] = overlay_stage_one(canvas, grey)

    found = detector.detect(grey)
    draw_detections(canvas, found, labels)
    report.update(summarise(list(found)))

    path = out_dir / f"{page.source.stem}_p{page.page_index:02d}.png"
    write_image(path, canvas)
    report["written"] = str(path)
    return report


def print_page_report(report: dict) -> None:
    """Print one page's summary."""
    print(f"\npage {report['page']}  ({report['size'][0]}x{report['size'][1]})")

    stage1 = report.get("stage1")
    if stage1 is not None:
        if stage1["staves"] == 0:
            print("  STAGE 1 FOUND NO STAVES - Stage 2 never had a chance on this page.")
            print(f"    estimated skew {stage1['skew_deg']:+.3f} deg, "
                  f"deskew applied: {stage1['deskewed']}")
        else:
            print(
                f"  stage 1: {stage1['staves']} staves, {stage1['systems']} systems, "
                f"{stage1['measures']} measures/staff, "
                f"line spacing {stage1.get('line_spacing')} px, "
                f"skew {stage1['skew_deg']:+.3f} deg"
            )

    print(f"  detections: {report['count']}")
    if not report["count"]:
        print("    nothing found. Try a lower --conf before concluding anything.")
        return

    confidence = report["confidence"]
    print(
        f"  confidence: min {confidence['min']}  p25 {confidence['p25']}  "
        f"median {confidence['median']}  p75 {confidence['p75']}  max {confidence['max']}"
    )
    print(f"    at or above 0.5: {confidence['above_0.5']} of {report['count']}")
    print("  by class:")
    for name, count in list(report["by_class"].items())[:12]:
        print(f"    {count:6d}  {name}")
    remaining = len(report["by_class"]) - 12
    if remaining > 0:
        print(f"    ... and {remaining} more classes")
    print(f"  written to {report['written']}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="a PDF or image")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("inspection"))
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="confidence floor; low by default so near-misses stay visible",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pages", type=int, nargs="*", help="page indices; all when omitted")
    parser.add_argument(
        "--no-staves", action="store_true", help="skip the Stage 1 overlay"
    )
    parser.add_argument(
        "--no-labels", action="store_true", help="boxes only, no class names"
    )
    args = parser.parse_args(argv)

    if not args.weights.exists():
        print(f"no checkpoint at {args.weights}", file=sys.stderr)
        return 2

    from melodix.vision.detector import DetectorConfig, SymbolDetector

    detector = SymbolDetector(
        DetectorConfig(
            weights=args.weights,
            confidence_threshold=args.conf,
            image_size=args.imgsz,
            device=args.device or "auto",
        )
    )

    pages = load_document(args.input, dpi=args.dpi, pages=args.pages)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"{args.input}  ->  {len(pages)} page(s) at {args.dpi} DPI, conf >= {args.conf}")
    for page in pages:
        print_page_report(
            inspect_page(page, detector, args.out, not args.no_staves, not args.no_labels)
        )

    print(f"\nannotated pages in {args.out}")
    print("Look at them. A count alone does not distinguish a model that is")
    print("wrong from one that is right about a page Stage 1 mangled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
