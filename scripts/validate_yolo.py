r"""Measure a trained checkpoint and report per-class numbers.

Aggregate mAP hides the failure that matters on drum notation. A page is
overwhelmingly cross and round noteheads, so a model that never once detects a
triangle head still scores well overall — and a beginner-facing app that
silently drops a symbol class is worse than one that scores lower evenly.

The class-order check runs first and is the more important of the two. YOLO
stores class *indices*; if the checkpoint's order differs from
:mod:`melodix.vision.labels`, every symbol maps to the wrong drum downstream
and nothing raises.

Usage::

    python scripts/validate_yolo.py --weights models/stage2_synth/weights/best.pt \\
        --data datasets/melodix_synth/data.yaml --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from melodix.vision.labels import NUM_CLASSES, class_names  # noqa: E402


def check_class_order(model: Any) -> list[tuple[int, str, str]]:
    """Compare a loaded model's class list against the schema.

    Args:
        model: A loaded ultralytics ``YOLO``.

    Returns:
        Mismatches as ``(index, checkpoint_name, schema_name)``. Empty when the
        two agree exactly.
    """
    names = model.names
    checkpoint = [names[index] for index in sorted(names)]
    schema = class_names()

    mismatches: list[tuple[int, str, str]] = []
    for index in range(max(len(checkpoint), len(schema))):
        left = checkpoint[index] if index < len(checkpoint) else "<missing>"
        right = schema[index] if index < len(schema) else "<missing>"
        if left != right:
            mismatches.append((index, left, right))
    return mismatches


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=3000)
    parser.add_argument("--json", type=Path, help="write the report here")
    args = parser.parse_args(argv)

    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(args.weights))

    mismatches = check_class_order(model)
    print(f"checkpoint classes: {len(model.names)}, schema: {NUM_CLASSES}")
    if mismatches:
        print("CLASS ORDER MISMATCH - every symbol would map to the wrong drum:")
        for index, checkpoint, schema in mismatches:
            print(f"  {index:3d}  checkpoint={checkpoint!r}  schema={schema!r}")
    else:
        print("class order matches melodix.vision.labels exactly")

    settings: dict[str, Any] = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "max_det": args.max_det,
        "verbose": False,
        "plots": False,
    }
    if args.device is not None:
        settings["device"] = args.device

    metrics = model.val(**settings)

    schema = class_names()
    evaluated = {int(c) for c in metrics.box.ap_class_index}
    per_class = []
    for index, name in enumerate(schema):
        score = float(metrics.box.maps[index]) if index < len(metrics.box.maps) else 0.0
        per_class.append(
            {"index": index, "name": name, "map50_95": score, "had_instances": index in evaluated}
        )

    report = {
        "weights": str(args.weights),
        "data": str(args.data),
        "class_order_matches": not mismatches,
        "mismatches": mismatches,
        "aggregate": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "per_class": per_class,
    }

    print("\nAGGREGATE")
    for key, value in report["aggregate"].items():
        print(f"  {key:10s} {value:.4f}")

    print("\nPER CLASS (mAP50-95), weakest first")
    for row in sorted(per_class, key=lambda r: r["map50_95"]):
        note = "" if row["had_instances"] else "   (no val instances)"
        print(f"  {row['map50_95']:6.3f}  {row['index']:2d} {row['name']}{note}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
