"""Measure a checkpoint against real annotated pages.

This produces the only number in the project that means anything about shipping.
Every metric recorded so far is synthetic against synthetic: the model was
trained on procedurally rendered pages and validated on more pages from the same
generator, which measures how well it learned that generator, not how well it
reads drum notation.

Three things here exist because an aggregate number hides them.

**The ensemble slice.** Multi-drummer scores are the product goal and will be a
small minority of any real corpus. Averaged in, a total failure on ensembles
disappears behind success on single-staff pages. Mark them (see
:func:`load_manifest`) and this prints that slice on its own, always.

**The confusion matrix.** ``augmentation_dot`` and ``staccato`` are the same
glyph — a small filled circle — separated only by where they sit relative to a
notehead. A detector that deliberately does not encode position cannot fully
separate them, and the pair is called out by name.

**The synthetic warning.** Pointing this script at the training generator's own
output produces a beautiful number that means nothing. It detects that case and
says so loudly.

Metrics are computed here rather than deferred to ultralytics, so the ensemble
slice and the confusion matrix come from the same matching pass and cannot
disagree with each other.

Usage::

    python scripts/evaluate_real.py --weights models/.../best.pt --data real/ --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from melodix.ingest import read_grayscale  # noqa: E402
from melodix.vision.dataset import (  # noqa: E402
    IMAGE_SUFFIXES,
    Annotation,
    parse_label_file,
)
from melodix.vision.labels import NUM_CLASSES, class_names  # noqa: E402

#: Filename listing the pages that carry more than one staff per system. One
#: stem per line, ``#`` for comments. Optional; a directory named ``ensemble``
#: anywhere in a page's path marks it too.
MANIFEST_NAME = "ensemble.txt"

#: The pair worth watching: identical glyphs, distinguished only by position.
CONFUSABLE_PAIR = ("augmentation_dot", "staccato")

#: IoU at which a prediction is credited to a ground-truth box.
DEFAULT_IOU = 0.5

#: A class whose median ground-truth box is smaller than this, on its shorter
#: side, is reported separately. Measured per corpus rather than listed, so the
#: set adapts when the engraving size does.
SMALL_CLASS_PX = 8.0

#: Typical staff line spacing in the corpus, in pixels. Only used to turn
#: snap's tolerance into a pixel budget; override it for a corpus engraved
#: at a different size.
DEFAULT_STAFF_SPACING_PX = 14.0


def snap_tolerance_px(line_spacing: float = DEFAULT_STAFF_SPACING_PX) -> float:
    """Vertical centroid error Stage 1 can absorb, in pixels.

    Read from :meth:`~melodix.geometry.staff.StaffGrid.snap` rather than
    hard-coded, so it tracks the geometry module. One staff position is half a
    line spacing, and snap accepts a centroid within ``tolerance`` positions of
    an integer before it gives up and returns ``None``.

    At the default 14 px spacing that is +/-2.8 px.
    """
    import inspect

    from melodix.geometry.staff import StaffGrid

    tolerance = inspect.signature(StaffGrid.snap).parameters["tolerance"].default
    return float(tolerance) * (line_spacing / 2.0)


@dataclass(frozen=True, slots=True)
class Box:
    """A pixel-space box with a class, used for both truth and prediction."""

    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def iou(self, other: Box) -> float:
        """Intersection over union with another box."""
        overlap_w = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        overlap_h = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        if overlap_w <= 0.0 or overlap_h <= 0.0:
            return 0.0
        intersection = overlap_w * overlap_h
        union = self.area + other.area - intersection
        return 0.0 if union <= 0.0 else min(1.0, intersection / union)


@dataclass
class ClassTally:
    """Running counts for one class."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    scores: list[tuple[float, bool]] = field(default_factory=list)
    #: Signed centroid error per matched detection, in pixels.
    dx: list[float] = field(default_factory=list)
    dy: list[float] = field(default_factory=list)
    #: Shorter side of each ground-truth box, for the small-class split.
    truth_sizes: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Fraction of predictions that were right."""
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """Fraction of ground truth that was found."""
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def support(self) -> int:
        """Ground-truth instances of this class."""
        return self.true_positives + self.false_negatives


def annotations_to_boxes(
    annotations: tuple[Annotation, ...], width: int, height: int
) -> list[Box]:
    """Convert stored annotations to pixel boxes."""
    out = []
    for annotation in annotations:
        x0, y0, x1, y1 = annotation.box.to_pixels(width, height)
        out.append(Box(int(annotation.symbol), x0, y0, x1, y1))
    return out


def match_boxes(
    truth: list[Box], predictions: list[Box], iou_threshold: float = DEFAULT_IOU
) -> tuple[list[tuple[Box, Box]], list[Box], list[Box]]:
    """Greedily match predictions to ground truth, highest confidence first.

    Greedy-by-confidence is what COCO and ultralytics do, so the numbers here
    are comparable to theirs. A prediction may claim at most one truth box, and
    only one of the same class.

    Args:
        truth: Ground-truth boxes for one page.
        predictions: Predicted boxes for the same page.
        iou_threshold: Minimum overlap to count as a match.

    Returns:
        ``(matched_pairs, unmatched_predictions, unmatched_truth)``.
    """
    unclaimed = list(range(len(truth)))
    pairs: list[tuple[Box, Box]] = []
    spare: list[Box] = []

    for prediction in sorted(predictions, key=lambda b: b.confidence, reverse=True):
        best_index, best_iou = -1, 0.0
        for index in unclaimed:
            if truth[index].class_id != prediction.class_id:
                continue
            overlap = prediction.iou(truth[index])
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0 and best_iou >= iou_threshold:
            unclaimed.remove(best_index)
            pairs.append((truth[best_index], prediction))
        else:
            spare.append(prediction)

    return pairs, spare, [truth[index] for index in unclaimed]


def average_precision(scores: list[tuple[float, bool]], support: int) -> float:
    """Area under the precision-recall curve for one class.

    Args:
        scores: ``(confidence, is_true_positive)`` for every prediction.
        support: Ground-truth instances.

    Returns:
        AP in ``[0, 1]``. Zero when the class has no ground truth.
    """
    if support == 0 or not scores:
        return 0.0

    ordered = sorted(scores, key=lambda item: item[0], reverse=True)
    hits = np.cumsum([1 if correct else 0 for _, correct in ordered])
    ranks = np.arange(1, len(ordered) + 1)
    precisions = hits / ranks
    recalls = hits / support

    # 101-point interpolation, as COCO defines it.
    total = 0.0
    for level in np.linspace(0.0, 1.0, 101):
        candidates = precisions[recalls >= level]
        total += float(candidates.max()) if candidates.size else 0.0
    return total / 101.0


@dataclass
class Evaluation:
    """Accumulated results over a set of pages."""

    tallies: dict[int, ClassTally] = field(
        default_factory=lambda: defaultdict(ClassTally)
    )
    confusion: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    pages: int = 0

    def add_page(
        self, truth: list[Box], predictions: list[Box], iou_threshold: float
    ) -> None:
        """Fold one page's matches into the totals."""
        self.pages += 1
        pairs, spare, missed = match_boxes(truth, predictions, iou_threshold)

        for actual, prediction in pairs:
            tally = self.tallies[prediction.class_id]
            tally.true_positives += 1
            tally.scores.append((prediction.confidence, True))
            # What the application actually consumes: how far the centroid moved.
            tally.dx.append(
                abs((prediction.x_min + prediction.x_max) / 2
                    - (actual.x_min + actual.x_max) / 2)
            )
            tally.dy.append(
                abs((prediction.y_min + prediction.y_max) / 2
                    - (actual.y_min + actual.y_max) / 2)
            )
            self.confusion[(prediction.class_id, prediction.class_id)] += 1

        for prediction in spare:
            tally = self.tallies[prediction.class_id]
            tally.false_positives += 1
            tally.scores.append((prediction.confidence, False))
            # Attribute the mistake to whichever truth box it overlaps most,
            # which is what makes the confusion matrix informative rather than
            # a column of unexplained background errors.
            best, best_iou = None, 0.0
            for box in truth:
                overlap = prediction.iou(box)
                if overlap > best_iou:
                    best, best_iou = box, overlap
            actual = best.class_id if best is not None and best_iou >= 0.1 else -1
            self.confusion[(actual, prediction.class_id)] += 1

        for box in missed:
            self.tallies[box.class_id].false_negatives += 1
            self.confusion[(box.class_id, -1)] += 1

        for box in truth:
            self.tallies[box.class_id].truth_sizes.append(
                min(box.x_max - box.x_min, box.y_max - box.y_min)
            )

    def per_class(self) -> list[dict[str, Any]]:
        """Return one row per class, ordered by class id."""
        names = class_names()
        rows = []
        for class_id in range(NUM_CLASSES):
            tally = self.tallies.get(class_id, ClassTally())
            rows.append(
                {
                    "index": class_id,
                    "name": names[class_id],
                    "support": tally.support,
                    "precision": tally.precision,
                    "recall": tally.recall,
                    "ap50": average_precision(tally.scores, tally.support),
                    "median_box_px": (
                        float(np.median(tally.truth_sizes)) if tally.truth_sizes else 0.0
                    ),
                    "is_small": bool(
                        tally.truth_sizes
                        and float(np.median(tally.truth_sizes)) < SMALL_CLASS_PX
                    ),
                    "centroid": _centroid_stats(tally),
                }
            )
        return rows

    def aggregate(self) -> dict[str, float]:
        """Return macro-averaged metrics over classes that have ground truth."""
        rows = [row for row in self.per_class() if row["support"] > 0]
        if not rows:
            return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "classes": 0}
        return {
            "precision": sum(r["precision"] for r in rows) / len(rows),
            "recall": sum(r["recall"] for r in rows) / len(rows),
            "map50": sum(r["ap50"] for r in rows) / len(rows),
            "classes": len(rows),
        }


def _centroid_verdict(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    """Judge vertical centroid error against what Stage 1 can absorb.

    This is the number the application cares about, and it is not mAP. Stage 3
    feeds a detection's row to snap, which accepts anything within the
    tolerance and returns None outside it. A class whose 90th-percentile
    vertical error clears that budget is good enough for the product however
    its IoU-based scores read.
    """
    judged = [r for r in rows if r["centroid"].get("matched")]
    if not judged:
        return {"classes": 0}

    failing = [
        {"name": r["name"], "dy_p90": r["centroid"]["dy_p90"]}
        for r in judged
        if r["centroid"]["dy_p90"] > tolerance
    ]
    return {
        "classes": len(judged),
        "tolerance_px": tolerance,
        "passing": len(judged) - len(failing),
        "failing": failing,
        "worst_dy_p90": max(r["centroid"]["dy_p90"] for r in judged),
    }


def _centroid_stats(tally: ClassTally) -> dict[str, float]:
    """Summarise how far matched centroids sat from truth.

    Vertical and horizontal are reported apart because the application uses
    them for different things with very different tolerances: the row goes to
    :meth:`~melodix.geometry.staff.StaffGrid.snap`, the column decides which
    notehead a modifier attaches to.
    """
    if not tally.dy:
        return {"matched": 0}
    return {
        "matched": len(tally.dy),
        "dy_median": float(np.median(tally.dy)),
        "dy_p90": float(np.percentile(tally.dy, 90)),
        "dx_median": float(np.median(tally.dx)),
        "dx_p90": float(np.percentile(tally.dx, 90)),
    }


def load_manifest(root: Path) -> set[str]:
    """Return the page stems marked as ensemble scores.

    Two ways to mark a page, either of which is enough:

    - list its stem in ``ensemble.txt`` at the dataset root, one per line
    - put it under a directory named ``ensemble``

    Args:
        root: Dataset root.

    Returns:
        Stems of the ensemble pages. Empty when nothing is marked.
    """
    stems: set[str] = set()

    manifest = root / MANIFEST_NAME
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                stems.add(Path(entry).stem)

    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES and "ensemble" in {
            part.lower() for part in path.parts
        }:
            stems.add(path.stem)

    return stems


def looks_synthetic(root: Path) -> bool:
    """Report whether this dataset came from the project's own generator.

    A page named ``synthetic_00042.png`` is a page the model may well have
    trained on, and scoring against it produces a number that says nothing
    about real performance.
    """
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.stem.startswith("synthetic_"):
            return True
    return False


def evaluate(
    weights: Path,
    root: Path,
    split: str = "val",
    iou_threshold: float = DEFAULT_IOU,
    confidence: float = 0.25,
    image_size: int = 1280,
    device: str | None = None,
    detector: Any = None,
    staff_spacing: float = DEFAULT_STAFF_SPACING_PX,
) -> dict[str, Any]:
    """Score a checkpoint over a directory of annotated pages.

    Args:
        weights: Checkpoint to evaluate.
        root: Dataset root holding ``images/<split>`` and ``labels/<split>``.
        split: Split directory to read.
        iou_threshold: Overlap at which a prediction counts as correct.
        confidence: Detections below this are discarded.
        image_size: Inference resolution.
        device: Passed through to the detector.
        staff_spacing: Typical staff line spacing in pixels, which sets the
            centroid pass threshold via :func:`snap_tolerance_px`.
        detector: An object with ``detect(image) -> PageDetections``. Injected
            by the tests so metric computation runs without a checkpoint; the
            real one is built here when omitted.

    Returns:
        A report: aggregate metrics, per class, the ensemble slice, and the
        confusion matrix.

    Raises:
        FileNotFoundError: If the split has no image directory.
    """
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"no {split} images at {images_dir}")

    if detector is None:
        from melodix.vision.detector import DetectorConfig, SymbolDetector

        detector = SymbolDetector(
            DetectorConfig(
                weights=weights,
                confidence_threshold=confidence,
                image_size=image_size,
                device=device or "auto",
            )
        )


    ensemble_stems = load_manifest(root)
    overall = Evaluation()
    ensemble = Evaluation()
    single = Evaluation()

    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = read_grayscale(path)

        label_path = labels_dir / f"{path.stem}.txt"
        annotations = parse_label_file(label_path) if label_path.exists() else ()
        height, width = image.shape[:2]
        truth = annotations_to_boxes(annotations, width, height)

        found = detector.detect(image)
        predictions = [
            Box(hit.class_id, hit.x_min, hit.y_min, hit.x_max, hit.y_max, hit.confidence)
            for hit in found
        ]

        overall.add_page(truth, predictions, iou_threshold)
        target = ensemble if path.stem in ensemble_stems else single
        target.add_page(truth, predictions, iou_threshold)

    names = class_names()
    rows = overall.per_class()
    small = [r for r in rows if r["is_small"] and r["support"]]
    tolerance = snap_tolerance_px(staff_spacing)
    return {
        "weights": str(weights),
        "data": str(root),
        "split": split,
        "iou_threshold": iou_threshold,
        "synthetic_source": looks_synthetic(root),
        "pages": overall.pages,
        "aggregate": overall.aggregate(),
        "per_class": rows,
        "staff_spacing_px": staff_spacing,
        "snap_tolerance_px": tolerance,
        "small_classes": {
            "threshold_px": SMALL_CLASS_PX,
            "classes": [r["name"] for r in small],
            "mean_ap50": (
                float(np.mean([r["ap50"] for r in small])) if small else 0.0
            ),
        },
        "centroid_pass": _centroid_verdict(rows, tolerance),
        "ensemble": {
            "pages": ensemble.pages,
            "aggregate": ensemble.aggregate(),
            "per_class": ensemble.per_class() if ensemble.pages else [],
        },
        "single_staff": {"pages": single.pages, "aggregate": single.aggregate()},
        "confusion": [
            {
                "actual": names[actual] if actual >= 0 else "<background>",
                "predicted": names[predicted] if predicted >= 0 else "<missed>",
                "count": count,
            }
            for (actual, predicted), count in sorted(
                overall.confusion.items(), key=lambda item: -item[1]
            )
            if actual != predicted
        ],
    }


def print_report(report: dict[str, Any]) -> None:
    """Print the human-readable summary."""
    if report["synthetic_source"]:
        print("=" * 72)
        print("WARNING: this dataset contains pages named synthetic_*, which means")
        print("it came from scripts/generate_synthetic_dataset.py -- the same")
        print("generator the model trained on. These numbers measure how well the")
        print("model learned that generator. They say nothing about real pages.")
        print("=" * 72)
        print()

    aggregate = report["aggregate"]
    print(f"{report['pages']} pages, IoU {report['iou_threshold']}")
    print(
        f"  precision {aggregate['precision']:.4f}  recall {aggregate['recall']:.4f}  "
        f"mAP50 {aggregate['map50']:.4f}  over {aggregate['classes']} classes with truth"
    )

    print("\nENSEMBLE SLICE (multi-drummer pages, the product goal)")
    ensemble = report["ensemble"]
    if ensemble["pages"] == 0:
        print("  none marked. Add ensemble.txt at the dataset root, one page stem")
        print("  per line, or put ensemble pages under a directory named 'ensemble'.")
        print("  An aggregate number hides ensemble failure behind single-staff success.")
    else:
        agg = ensemble["aggregate"]
        single = report["single_staff"]["aggregate"]
        print(
            f"  {ensemble['pages']} pages: precision {agg['precision']:.4f}  "
            f"recall {agg['recall']:.4f}  mAP50 {agg['map50']:.4f}"
        )
        print(
            f"  single-staff for comparison: precision {single['precision']:.4f}  "
            f"recall {single['recall']:.4f}  mAP50 {single['map50']:.4f}"
        )
        if agg["map50"] < single["map50"] * 0.9:
            print("  ENSEMBLE IS MEASURABLY WORSE than single-staff.")

    small = report.get("small_classes", {})
    if small.get("classes"):
        print(
            f"\nSMALL CLASSES (median box under {small['threshold_px']:.0f} px), "
            f"AP50 shown apart from the localisation ceiling"
        )
        print(f"  mean AP50 {small['mean_ap50']:.4f} over {len(small['classes'])} classes")
        for row in report["per_class"]:
            if row["is_small"] and row["support"]:
                print(
                    f"    AP50 {row['ap50']:.3f}  box {row['median_box_px']:.1f} px  "
                    f"n={row['support']:<6d} {row['name']}"
                )
        print("  mAP50-95 under-reports these. See PROVENANCE.md for the arithmetic.")

    passed = report.get("centroid_pass", {})
    if passed.get("classes"):
        print("\nCENTROID ERROR vs what the application needs")
        print(
            f"  Stage 1 snap absorbs +/-{passed['tolerance_px']:.2f} px vertically at "
            f"{report['staff_spacing_px']:.0f} px staff spacing"
        )
        print(
            f"  {passed['passing']} of {passed['classes']} matched classes keep their "
            f"90th-percentile vertical error inside it"
        )
        if passed["failing"]:
            print("  outside it:")
            for row in sorted(passed["failing"], key=lambda r: -r["dy_p90"])[:10]:
                print(f"    dy_p90 {row['dy_p90']:6.2f} px  {row['name']}")
        else:
            print("  every matched class is inside the budget.")

    print("\nPER CLASS, weakest first  (dy and dx shown as median/p90, in pixels)")
    for row in sorted(report["per_class"], key=lambda r: (r["support"] == 0, r["ap50"])):
        note = "  (no ground truth)" if row["support"] == 0 else ""
        centroid = row["centroid"]
        moved = (
            f"  dy {centroid['dy_median']:.2f}/{centroid['dy_p90']:.2f}"
            f"  dx {centroid['dx_median']:.2f}/{centroid['dx_p90']:.2f}"
            if centroid.get("matched")
            else ""
        )
        print(
            f"  AP50 {row['ap50']:.3f}  P {row['precision']:.3f}  R {row['recall']:.3f}  "
            f"n={row['support']:<5d} {row['name']}{note}{moved}"
        )

    print("\nCONFUSION, most frequent first")
    if not report["confusion"]:
        print("  none")
    for row in report["confusion"][:15]:
        print(f"  {row['count']:5d}  {row['actual']} -> {row['predicted']}")

    first, second = CONFUSABLE_PAIR
    pair = [
        row
        for row in report["confusion"]
        if {row["actual"], row["predicted"]} == {first, second}
    ]
    print(f"\n{first} <-> {second} (same glyph, position is the only difference)")
    if pair:
        for row in pair:
            print(f"  {row['count']:5d}  {row['actual']} -> {row['predicted']}")
        print("  Stage 3 should disambiguate these geometrically, not trust the class.")
    else:
        print("  not confused in this run")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="dataset root")
    parser.add_argument("--split", default="val")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--staff-spacing",
        type=float,
        default=DEFAULT_STAFF_SPACING_PX,
        help="typical staff line spacing in px; sets the centroid pass threshold",
    )
    parser.add_argument("--json", type=Path, help="write the report here")
    args = parser.parse_args(argv)

    report = evaluate(
        weights=args.weights,
        root=args.data,
        split=args.split,
        iou_threshold=args.iou,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
        staff_spacing=args.staff_spacing,
    )
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
