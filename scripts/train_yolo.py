"""Train the Stage 2 symbol detector.

A thin wrapper over ultralytics whose real content is :data:`SCORE_DEFAULTS`.
Every augmentation and inference value is stated there explicitly with a
one-line reason; nothing is inherited. An ultralytics default is tuned for COCO
photographs, and several of those defaults are actively destructive on engraved
notation.

That is not hypothetical. The first checkpoint was trained by an ad-hoc
``model.train()`` call that bypassed this script entirely, so it ran on stock
values: ``fliplr=0.5`` (half the pages mirrored, teaching invariance to the
left-right asymmetry that distinguishes time signatures, accents and stem
direction), ``mosaic=1.0`` (four pages per frame, halving effective resolution
for the smallest symbols) and ``degrees=0.0`` (no exposure to the residual tilt
deskew leaves behind). See ``models/PROVENANCE.md``.

**Ad-hoc ``model.train()`` calls are not the supported path.** They drift, and
the drift is invisible until someone reads the checkpoint. Use this script, or
``notebooks/train_colab.ipynb``, which calls it. Every run writes its fully
resolved configuration to ``melodix_resolved_config.json`` beside the weights,
so a run's real settings are recoverable from the run itself — and a run
directory lacking that file is identifiable as one that did not come through
here.

Usage::

    python scripts/train_yolo.py --data datasets/melodix_synth/data.yaml
    python scripts/train_yolo.py --data d.yaml --degraded datasets/melodix_degraded
"""
from __future__ import annotations

import argparse
import json
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

#: Written into every run directory so a run's real settings are recoverable
#: from the run itself, not only from inside the checkpoint.
RESOLVED_CONFIG_NAME = "melodix_resolved_config.json"

#: Every augmentation and inference setting, stated explicitly with a reason.
#:
#: Nothing here is left to inherit. An ultralytics default is tuned for COCO
#: photographs of everyday objects, and several of those defaults are actively
#: destructive on engraved notation — which is not hypothetical: the first
#: checkpoint was trained on stock values and carries the damage.
SCORE_DEFAULTS: dict[str, object] = {
    # -- resolution ------------------------------------------------------- #
    # A notehead on a letter page at 300 DPI is ~20 px. The stock 640 halves
    # that before the first convolution and round-vs-cross stops being visible.
    "imgsz": 1280,
    # Ultralytics stops at 300 detections; a dense ensemble page passes that
    # inside the second system, so validation would score a truncated page.
    "max_det": 3000,
    # -- mirroring: the one that must never come back ---------------------- #
    # Music notation is not mirror-symmetric, and flipping does worse than
    # waste samples: it teaches invariance to left-right asymmetry, which is
    # the exact cue several glyphs are distinguished by. Time signature digits
    # mirror into non-glyphs; an accent (>) mirrors into (<), which is not in
    # the schema; stems attach right for stem-up and left for stem-down, so a
    # mirrored page shows stem-up-on-the-left, which no engraver produces.
    "fliplr": 0.0,
    # Same argument, more obviously. An upside-down page is not a page.
    "flipud": 0.0,
    # -- mosaic ------------------------------------------------------------ #
    # Four pages stitched into one frame puts each source page in roughly a
    # 640x640 region of a 1280 canvas, halving effective resolution for exactly
    # the small symbols that are already weakest, and cutting glyphs at the
    # seams. The seams themselves read as barlines. Off entirely rather than
    # relying on close_mosaic to undo it for the last few epochs.
    "mosaic": 0.0,
    "close_mosaic": 0,  # moot at mosaic=0, set so it cannot be inherited
    # -- geometry ---------------------------------------------------------- #
    # Derived from what Stage 1 actually leaves behind, not guessed. Deskew
    # quantises to fine_step_deg=0.05 and declines to rotate below
    # min_correction_deg=0.10, so up to 0.10 deg survives by design; measured
    # residual on synthetic pages is 0.050 deg, max 0.100. The projection
    # estimator's measured error is 0.12 deg and the Hough path's is 0.25, so a
    # realistic worst case is ~0.35 deg. 1.0 gives roughly 3x margin over that
    # and partial cover for the tail where deskew declines on low confidence
    # and the full tilt reaches the detector.
    "degrees": 1.0,
    # Derived, and the derivation matters because the stock 0.5 was starving the
    # two weakest classes. Ultralytics draws a factor from [1-s, 1+s], so what
    # counts is the minimum.
    #
    # Measured on this dataset: augmentation_dot and staccato render 6.0 px on a
    # 1000x1400 page, which letterboxes to 5.49 px at imgsz=1280. The stock
    # scale=0.5 takes that to 2.75 px.
    #
    # The failure is not that the glyph vanishes -- a 6 px disc downscaled 2x
    # still leaves nine dark pixels, measured. It is that the localisation
    # budget collapses. For equal boxes offset by d, IoU clears a threshold t
    # only while the offset stays under d(1-t)/(1+t), so:
    #
    #   dot at 2.75 px  ->  0.92 px to clear IoU 0.50,  0.39 px for IoU 0.75
    #   dot at 4.39 px  ->  1.46 px to clear IoU 0.50,  0.63 px for IoU 0.75
    #
    # YOLOv8's finest head is P3 at stride 8, so a sub-pixel budget asks for
    # accuracy finer than the head's own grid.
    #
    # 0.20 is chosen because two independent arguments agree on it. Real rastral
    # sizes span roughly 6-9 pt of staff height, about +/-20% around the middle,
    # so [0.8, 1.2] covers the engraving variation that actually exists rather
    # than one invented for augmentation. And it leaves the smallest classes at
    # 4.39 px, a 59% larger IoU-50 budget than the stock setting allows.
    "scale": 0.20,
    # Page margins vary between publishers, but a symbol's position inside its
    # system does not. A small shift teaches robustness to page registration
    # without pretending the layout itself moves.
    "translate": 0.05,
    # Nothing in the pipeline produces or corrects shear. A sheared page is not
    # a scan artefact — a scanner produces rotation and perspective, not skew
    # along one axis — so this trains for an input that cannot arrive.
    "shear": 0.0,
    # Deskew corrects rotation only; it has no perspective model. Training on
    # warped pages would teach the detector to read pages that Stage 1 will
    # then fail to straighten, moving the failure downstream rather than
    # removing it. Revisit only if real pages turn out to be phone photographs
    # (see scripts/degrade.py, which can produce them when that is known).
    "perspective": 0.0,
    # -- colour ------------------------------------------------------------ #
    # A rendered page is grayscale promoted to three channels, so every pixel
    # has zero saturation and undefined hue: both of these are inert on this
    # data. Set to zero to say so, rather than inheriting 0.015/0.7 and leaving
    # a reader to wonder whether they mattered.
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    # Brightness is the one colour axis that does vary on a real scan.
    "hsv_v": 0.25,
    # -- classification-only knobs ----------------------------------------- #
    # These apply to YOLO classification, not detection, so they are inert
    # here. Pinned anyway: inert-by-default is not the same as inert-by-intent,
    # and a future ultralytics could extend them to detect.
    "erasing": 0.0,
    "auto_augment": None,
}

#: Augmentations that must never be enabled, and why in one phrase. Guarded by
#: a test: this is the regression that would be invisible in every metric.
FORBIDDEN_AUGMENTATIONS: dict[str, str] = {
    "fliplr": "notation is not mirror-symmetric",
    "flipud": "notation is not mirror-symmetric",
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

    settings = resolve_settings(
        data_yaml=data_yaml,
        output_dir=output_dir,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        resume=resume,
        name=name,
        overrides=overrides,
    )
    check_forbidden(settings)

    run_dir = output_dir / name
    write_resolved_config(run_dir, model, settings)

    return YOLO(model).train(**settings)


def resolve_settings(
    data_yaml: Path,
    output_dir: Path,
    epochs: int = 30,
    batch: int = 8,
    device: str | None = None,
    workers: int = 4,
    patience: int = 20,
    resume: bool = False,
    name: str = "melodix_symbols",
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the exact argument dict that will be handed to ultralytics.

    Separated from :func:`train` so the resolved configuration can be printed,
    written to disk and asserted on without starting a run.

    Returns:
        Every setting, with ``overrides`` applied last.
    """
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
    return settings


def check_forbidden(settings: dict[str, object]) -> None:
    """Refuse to start a run with an augmentation known to be harmful.

    An override is allowed to set anything, including the mirror flips, so this
    is the backstop that turns a typo into a refusal rather than a wasted
    hundred minutes and a subtly worse model.

    Raises:
        ValueError: If a forbidden augmentation is enabled.
    """
    enabled = [
        f"{key}={settings[key]} ({reason})"
        for key, reason in FORBIDDEN_AUGMENTATIONS.items()
        if float(settings.get(key, 0.0) or 0.0) != 0.0
    ]
    if enabled:
        listed = "; ".join(enabled)
        raise ValueError(
            f"refusing to train with a forbidden augmentation: {listed}. "
            f"These do not merely waste samples, they teach invariance to a cue "
            f"the schema depends on. Override FORBIDDEN_AUGMENTATIONS only with "
            f"a written reason."
        )


def write_resolved_config(
    run_dir: Path, model: str, settings: dict[str, object]
) -> Path:
    """Record the resolved configuration next to the weights.

    The root cause of this project's one configuration drift was a run that
    bypassed this script, leaving its real settings recoverable only from
    inside the checkpoint. Writing them beside the weights means a run's
    settings can be read without loading torch — and a run whose directory
    lacks this file is identifiable as one that did not come through here.

    Args:
        run_dir: Directory ultralytics will write the run into.
        model: Base checkpoint or architecture.
        settings: The resolved arguments.

    Returns:
        Path to the file written.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RESOLVED_CONFIG_NAME
    payload = {
        "written_by": "scripts/train_yolo.py",
        "model": model,
        "melodix_classes": NUM_CLASSES,
        "settings": {key: settings[key] for key in sorted(settings)},
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


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
        "--degraded",
        type=Path,
        default=None,
        help=(
            "train against a degraded copy produced by scripts/degrade.py instead "
            "of --data. Deliberately unused: degradation simulates a distribution "
            "nobody has observed yet, so the effects cannot be tuned until real "
            "pages exist. The switch is here so a run can start the day they do."
        ),
    )
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

    data_yaml = args.data
    if args.degraded is not None:
        degraded_yaml = args.degraded / "data.yaml"
        if not degraded_yaml.exists():
            print(
                f"--degraded {args.degraded} has no data.yaml. Produce one with:\n"
                f"    python scripts/degrade.py --in {args.data.parent} "
                f"--out {args.degraded}",
                file=sys.stderr,
            )
            return 2
        data_yaml = degraded_yaml
        print(f"training against the degraded variant at {data_yaml}")

    try:
        counts = verify_dataset(data_yaml)
    except (FileNotFoundError, ValueError) as error:
        print(f"dataset check failed: {error}", file=sys.stderr)
        return 2

    overrides = parse_overrides(args.overrides)
    settings = resolve_settings(
        data_yaml=data_yaml,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        resume=args.resume,
        name=args.name,
        overrides=overrides,
    )

    print(f"dataset {data_yaml}")
    print(f"  train {counts['train']} images, val {counts['val']} images")
    print(f"  {NUM_CLASSES} classes, matching melodix.vision.labels")
    print(f"model {args.model}")
    print("\nRESOLVED CONFIG (every value; nothing inherited from ultralytics)")
    for key in sorted(settings):
        print(f"  {key} = {settings[key]}")

    try:
        check_forbidden(settings)
    except ValueError as error:
        print(f"\n{error}", file=sys.stderr)
        return 2

    if args.check_only:
        print("\n--check-only: not training")
        return 0

    if shutil.disk_usage(".").free < 2 * 1024**3:
        print("warning: under 2 GB free; a run writes checkpoints each epoch", file=sys.stderr)

    results = train(
        data_yaml=data_yaml,
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
