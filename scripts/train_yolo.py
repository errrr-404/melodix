"""Train the Stage 2 symbol detector.

A thin wrapper over ultralytics whose real content is its defaults. The stock
YOLO settings are tuned for photographs of everyday objects, and several of
them quietly destroy sheet music. This script changes those and leaves the rest
alone.

What differs from stock, and why
--------------------------------
``imgsz=1280``
    A notehead on a letter page scanned at 300 DPI is roughly 20 px across.
    The stock 640 halves that before the first convolution and the class
    difference between a round and a cross head stops being visible.

``mosaic=0.0``
    Mosaic augmentation stitches four images into one, so a page fragment sits
    against three unrelated fragments. It is excellent for scattered objects
    and actively harmful here: notation is a grid, and a detector benefits from
    learning that hi-hats sit in a row above snares. Mosaic destroys exactly
    that regularity, and the seams it creates look like barlines.

``degrees=2.0``
    Real scans are tilted by a degree or two, and Stage 1 corrects them before
    the detector ever runs. Training past that range wastes capacity on
    rotations the pipeline guarantees will not arrive.

``max_det=3000``
    Ultralytics stops at 300 detections. A dense ensemble page passes that
    inside the second system, so validation mAP would be measured against a
    truncated page and read far worse than the model deserves.

``fliplr=0.0``, ``flipud=0.0``
    Notation is not mirror-symmetric. A flipped page teaches the model that a
    reversed flag or an upside-down rest is a valid symbol, and nothing in the
    pipeline will ever show it one.

``scale`` is left fairly wide, because engraving size genuinely varies between
publishers and between systems on one page.

Usage::

    python scripts/train_yolo.py --data datasets/melodix_synth/data.yaml
    python scripts/train_yolo.py --data d.yaml --epochs 100 --batch 8 --resume
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from melodix.vision.labels import NUM_CLASSES, class_names  # noqa: E402

#: Base checkpoint. The small model is the right starting point for symbol
#: detection: the task needs resolution far more than it needs depth, and the
#: budget is better spent on imgsz than on parameters.
DEFAULT_MODEL = "yolov8s.pt"

#: Augmentation and inference settings that differ from the ultralytics stock
#: values. See the module docstring for the reasoning behind each.
SCORE_DEFAULTS: dict[str, object] = {
    "imgsz": 1280,
    "max_det": 3000,
    "degrees": 2.0,
    "mosaic": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "scale": 0.35,
    "translate": 0.05,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.25,
}


def verify_dataset(data_yaml: Path) -> dict[str, int]:
    """Check a dataset before spending an hour training against it.

    Catches the two failures that otherwise surface only as a bad mAP: a
    ``data.yaml`` whose class list has drifted from
    :mod:`melodix.vision.labels`, and a split whose images and labels do not
    pair up.

    Args:
        data_yaml: The dataset descriptor ultralytics will read.

    Returns:
        Image counts per split.

    Raises:
        FileNotFoundError: If the descriptor or a split directory is missing.
        ValueError: If the class list disagrees with the schema, or images and
            labels do not correspond.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"no dataset descriptor at {data_yaml}")

    body = data_yaml.read_text(encoding="utf-8")
    declared = [
        line.split(": ", 1)[1].strip()
        for line in body.splitlines()
        if line.startswith("  ") and ": " in line
    ]
    expected = class_names()
    if declared != expected:
        raise ValueError(
            f"{data_yaml} declares {len(declared)} classes that do not match "
            f"melodix.vision.labels ({NUM_CLASSES}). A checkpoint trained against "
            f"this file would report the wrong symbol for every class after the "
            f"first mismatch. Regenerate it with write_data_yaml()."
        )

    root = data_yaml.parent
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        if not images_dir.is_dir():
            raise FileNotFoundError(f"no {split} images at {images_dir}")

        images = {path.stem for path in images_dir.iterdir() if path.suffix != ".txt"}
        labels = {path.stem for path in labels_dir.glob("*.txt")} if labels_dir.is_dir() else set()

        missing = images - labels
        if missing:
            sample = ", ".join(sorted(missing)[:3])
            raise ValueError(
                f"{len(missing)} {split} images have no label file (e.g. {sample}). "
                f"Ultralytics reads a missing label file as an unlabelled image and "
                f"trains on it as a negative example, so this fails silently."
            )
        counts[split] = len(images)

    return counts


def train(
    data_yaml: Path,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    epochs: int = 30,
    batch: int = 8,
    device: str | None = None,
    workers: int = 4,
    patience: int = 20,
    resume: bool = False,
    name: str = "melodix_symbols",
    overrides: dict[str, object] | None = None,
) -> Any:
    """Run training and return the ultralytics results object.

    Args:
        data_yaml: Dataset descriptor.
        output_dir: Directory runs are written under.
        model: Base checkpoint or a ``.yaml`` architecture.
        epochs: Passes over the training set.
        batch: Images per step. Lower this first if CUDA runs out of memory —
            at 1280 px the activations, not the weights, dominate.
        device: ``"cpu"``, ``"0"``, ``"0,1"``. Left to ultralytics when unset.
        workers: Dataloader workers.
        patience: Stop after this many epochs without validation improvement.
        resume: Continue an interrupted run.
        name: Run name under ``output_dir``.
        overrides: Extra ultralytics arguments, applied last so they win over
            everything this script sets.

    Returns:
        Whatever ``YOLO.train`` returned.

    Raises:
        ImportError: If ultralytics is not installed.
    """
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - requires the extra absent
        raise ImportError(
            'ultralytics is not installed. Install the vision extra:\n'
            '    pip install -e ".[dev,vision]"'
        ) from error

    settings: dict[str, object] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "project": str(output_dir),
        "name": name,
        "workers": workers,
        "patience": patience,
        "resume": resume,
        "exist_ok": True,
        **SCORE_DEFAULTS,
    }
    if device is not None:
        settings["device"] = device
    if overrides:
        settings.update(overrides)

    return YOLO(model).train(**settings)


def parse_overrides(pairs: list[str]) -> dict[str, object]:
    """Parse ``key=value`` strings into typed ultralytics arguments.

    Args:
        pairs: Strings of the form ``lr0=0.001``.

    Returns:
        A mapping with values coerced to bool, int or float where they look
        like one, and left as strings otherwise.

    Raises:
        ValueError: If a string has no ``=``.
    """
    parsed: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        parsed[key.strip()] = _coerce(raw.strip())
    return parsed


def _coerce(raw: str) -> object:
    """Convert a command-line string to the type it looks like."""
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train the melodix Stage 2 symbol detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True, help="path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="training epochs")
    parser.add_argument("--batch", type=int, default=8, help="images per step")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/runs"), help="where runs are written"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="base checkpoint")
    parser.add_argument("--name", default="melodix_symbols", help="run name")
    parser.add_argument("--device", default=None, help='"cpu", "0", "0,1"')
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--patience", type=int, default=20, help="early-stopping patience")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the dataset and print the settings, then exit without training",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        metavar="KEY=VALUE",
        nargs="*",
        default=[],
        help="extra ultralytics arguments, applied last",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = build_parser().parse_args(argv)

    try:
        counts = verify_dataset(args.data)
    except (FileNotFoundError, ValueError) as error:
        print(f"dataset check failed: {error}", file=sys.stderr)
        return 2

    overrides = parse_overrides(args.overrides)
    print(f"dataset {args.data}")
    print(f"  train {counts['train']} images, val {counts['val']} images")
    print(f"  {NUM_CLASSES} classes, matching melodix.vision.labels")
    print(f"model {args.model}, {args.epochs} epochs, batch {args.batch}")
    for key, value in sorted({**SCORE_DEFAULTS, **overrides}.items()):
        print(f"  {key} = {value}")

    if args.check_only:
        print("\n--check-only: not training")
        return 0

    if shutil.disk_usage(".").free < 2 * 1024**3:
        print("warning: under 2 GB free; a run writes checkpoints each epoch", file=sys.stderr)

    results = train(
        data_yaml=args.data,
        output_dir=args.output_dir,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        resume=args.resume,
        name=args.name,
        overrides=overrides,
    )

    run_dir = args.output_dir / args.name
    print(f"\nrun written to {run_dir}")
    print(f"  best weights: {run_dir / 'weights' / 'best.pt'}")
    print("point a detector at them with:")
    print(f"    DetectorConfig(weights=Path({str(run_dir / 'weights' / 'best.pt')!r}))")
    return 0 if results is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
