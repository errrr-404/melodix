"""Unit tests for ``scripts/train_yolo.py``.

Training itself is not exercised — that needs a GPU and an hour. What is
exercised is everything that runs *before* training, because those are the
checks that turn a silent eight-hour waste into an immediate error: a
``data.yaml`` whose class list has drifted from the schema, and a split whose
images and labels do not pair up.

The settings tests pin the handful of ultralytics defaults this project
overrides. Each of them is a decision with a reason recorded in the script's
docstring, and each would still train to a plausible-looking loss if silently
reverted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from melodix.vision.dataset import write_data_yaml
from melodix.vision.labels import NUM_CLASSES, class_names

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train_yolo.py"


def _load():
    """Import the training script as a module."""
    spec = importlib.util.spec_from_file_location("train_yolo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load()


def build_dataset(root: Path, train: int = 3, val: int = 2, with_labels: bool = True) -> Path:
    """Write a dataset skeleton: empty images and matching label files."""
    for split, count in (("train", train), ("val", val)):
        for index in range(count):
            image = root / "images" / split / f"page_{index}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"")
            if with_labels:
                label = root / "labels" / split / f"page_{index}.txt"
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    write_data_yaml(root / "data.yaml", root)
    return root / "data.yaml"


# --------------------------------------------------------------------------- #
# Settings that differ from stock ultralytics
# --------------------------------------------------------------------------- #


def test_mosaic_is_disabled():
    """Mosaic stitches four images together. Notation is a grid, and the seams
    it creates look like barlines.
    """
    assert runner.SCORE_DEFAULTS["mosaic"] == 0.0


def test_neither_flip_is_enabled():
    """Notation is not mirror-symmetric; a reversed flag is not a symbol."""
    assert runner.SCORE_DEFAULTS["fliplr"] == 0.0
    assert runner.SCORE_DEFAULTS["flipud"] == 0.0


def test_the_image_size_suits_engraving():
    """The stock 640 leaves a notehead a handful of pixels across."""
    assert runner.SCORE_DEFAULTS["imgsz"] == 1280


def test_the_image_size_is_a_multiple_of_the_stride():
    assert runner.SCORE_DEFAULTS["imgsz"] % 32 == 0


def test_the_detection_cap_clears_a_dense_page():
    """The ultralytics default of 300 truncates validation on a real page."""
    assert runner.SCORE_DEFAULTS["max_det"] == 3000


def test_rotation_matches_what_deskew_leaves_behind():
    """Stage 1 corrects skew first, so training past the residual it leaves
    spends capacity on inputs the pipeline guarantees will not arrive. The
    value is derived in SCORE_DEFAULTS from deskew's own constants; the two
    tests further down check that derivation rather than the literal.
    """
    assert runner.SCORE_DEFAULTS["degrees"] == 1.0


def test_hue_and_saturation_jitter_are_off():
    """Scans are effectively greyscale; only brightness varies."""
    assert runner.SCORE_DEFAULTS["hsv_h"] == 0.0
    assert runner.SCORE_DEFAULTS["hsv_s"] == 0.0
    assert runner.SCORE_DEFAULTS["hsv_v"] > 0.0


def test_the_base_model_is_the_small_checkpoint():
    assert runner.DEFAULT_MODEL == "yolov8s.pt"


# --------------------------------------------------------------------------- #
# Dataset verification
# --------------------------------------------------------------------------- #


def test_a_well_formed_dataset_passes(tmp_path):
    data_yaml = build_dataset(tmp_path, train=3, val=2)

    counts = runner.verify_dataset(data_yaml)

    assert counts == {"train": 3, "val": 2}


def test_a_missing_descriptor_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no dataset descriptor"):
        runner.verify_dataset(tmp_path / "absent.yaml")


def test_a_missing_split_is_reported(tmp_path):
    data_yaml = build_dataset(tmp_path)
    for path in (tmp_path / "images" / "val").iterdir():
        path.unlink()
    (tmp_path / "images" / "val").rmdir()

    with pytest.raises(FileNotFoundError, match="no val images"):
        runner.verify_dataset(data_yaml)


def test_an_image_without_a_label_is_reported(tmp_path):
    """Ultralytics reads a missing label file as a negative example and trains
    on it happily, so this has to be caught before training starts.
    """
    data_yaml = build_dataset(tmp_path)
    next((tmp_path / "labels" / "train").iterdir()).unlink()

    with pytest.raises(ValueError, match="have no label file"):
        runner.verify_dataset(data_yaml)


def test_a_drifted_class_list_is_reported(tmp_path):
    """A checkpoint trained against a stale data.yaml reports the wrong symbol
    for every class after the first mismatch.
    """
    data_yaml = build_dataset(tmp_path)
    body = data_yaml.read_text(encoding="utf-8")
    data_yaml.write_text(body.replace("0: round_notehead", "0: cowbell"), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        runner.verify_dataset(data_yaml)


def test_a_truncated_class_list_is_reported(tmp_path):
    data_yaml = build_dataset(tmp_path)
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    data_yaml.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        runner.verify_dataset(data_yaml)


def test_the_generated_dataset_descriptor_is_accepted(tmp_path):
    """What write_data_yaml emits must be what the trainer accepts."""
    data_yaml = build_dataset(tmp_path)

    assert runner.verify_dataset(data_yaml)["train"] == 3
    assert len(class_names()) == NUM_CLASSES


# --------------------------------------------------------------------------- #
# Override parsing
# --------------------------------------------------------------------------- #


def test_a_float_override_is_typed():
    assert runner.parse_overrides(["lr0=0.001"]) == {"lr0": 0.001}


def test_an_int_override_is_typed():
    assert runner.parse_overrides(["warmup_epochs=5"]) == {"warmup_epochs": 5}


def test_a_bool_override_is_typed():
    assert runner.parse_overrides(["cos_lr=true"]) == {"cos_lr": True}
    assert runner.parse_overrides(["cos_lr=False"]) == {"cos_lr": False}


def test_a_string_override_survives():
    assert runner.parse_overrides(["optimizer=AdamW"]) == {"optimizer": "AdamW"}


def test_several_overrides_parse_together():
    parsed = runner.parse_overrides(["lr0=0.01", "cos_lr=true", "optimizer=SGD"])

    assert parsed == {"lr0": 0.01, "cos_lr": True, "optimizer": "SGD"}


def test_an_override_without_a_value_is_rejected():
    with pytest.raises(ValueError, match="expected key=value"):
        runner.parse_overrides(["lr0"])


def test_no_overrides_parse_to_nothing():
    assert runner.parse_overrides([]) == {}


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_check_only_verifies_without_training(tmp_path, capsys):
    data_yaml = build_dataset(tmp_path)

    code = runner.main(["--data", str(data_yaml), "--check-only"])

    assert code == 0
    assert "not training" in capsys.readouterr().out


def test_check_only_reports_the_split_sizes(tmp_path, capsys):
    data_yaml = build_dataset(tmp_path, train=7, val=3)

    runner.main(["--data", str(data_yaml), "--check-only"])

    assert "train 7 images, val 3 images" in capsys.readouterr().out


def test_check_only_prints_the_overridden_settings(tmp_path, capsys):
    data_yaml = build_dataset(tmp_path)

    runner.main(["--data", str(data_yaml), "--check-only"])

    out = capsys.readouterr().out
    assert "mosaic = 0.0" in out
    assert "imgsz = 1280" in out


def test_a_bad_dataset_exits_non_zero_without_training(tmp_path, capsys):
    """The failure mode this guards: an eight-hour run against a broken split."""
    code = runner.main(["--data", str(tmp_path / "absent.yaml")])

    assert code == 2
    assert "dataset check failed" in capsys.readouterr().err


def test_command_line_overrides_reach_the_printed_settings(tmp_path, capsys):
    data_yaml = build_dataset(tmp_path)

    runner.main(["--data", str(data_yaml), "--check-only", "--set", "lr0=0.002"])

    assert "lr0 = 0.002" in capsys.readouterr().out


def test_an_override_beats_a_score_default(tmp_path, capsys):
    """Overrides are applied last so an experiment can contradict the script."""
    data_yaml = build_dataset(tmp_path)

    runner.main(["--data", str(data_yaml), "--check-only", "--set", "mosaic=0.5"])

    assert "mosaic = 0.5" in capsys.readouterr().out


def test_the_default_epoch_count(tmp_path):
    parser = runner.build_parser()

    assert parser.parse_args(["--data", "d.yaml"]).epochs == 30


def test_the_output_directory_can_be_set():
    parser = runner.build_parser()

    args = parser.parse_args(["--data", "d.yaml", "--output-dir", "runs/here"])

    assert args.output_dir == Path("runs/here")


def test_batch_size_can_be_set():
    parser = runner.build_parser()

    assert parser.parse_args(["--data", "d.yaml", "--batch", "16"]).batch == 16


def test_data_is_required():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([])


# --------------------------------------------------------------------------- #
# Mirror augmentation: the regression that must never come back
# --------------------------------------------------------------------------- #


def test_no_mirror_augmentation_is_configured():
    """The single most important assertion in this file.

    The first checkpoint trained with fliplr=0.5, mirroring half its pages.
    That does worse than waste samples: it teaches invariance to left-right
    asymmetry, which is the exact cue that distinguishes a time signature from
    a non-glyph, an accent from a shape not in the schema, and stem-up from
    stem-down. Nothing in any metric would show it coming back.
    """
    for name in runner.FORBIDDEN_AUGMENTATIONS:
        assert runner.SCORE_DEFAULTS[name] == 0.0, name


def test_both_mirror_axes_are_forbidden():
    assert set(runner.FORBIDDEN_AUGMENTATIONS) == {"fliplr", "flipud"}


def test_every_forbidden_augmentation_carries_a_reason():
    for name, reason in runner.FORBIDDEN_AUGMENTATIONS.items():
        assert reason.strip(), name


def test_a_run_with_mirroring_is_refused():
    """Not merely defaulted off but actively refused, because an override can
    set anything and a typo would otherwise cost a hundred minutes.
    """
    settings = {**runner.SCORE_DEFAULTS, "fliplr": 0.5}

    with pytest.raises(ValueError, match="forbidden augmentation"):
        runner.check_forbidden(settings)


def test_the_refusal_names_the_offending_parameter():
    with pytest.raises(ValueError, match="flipud"):
        runner.check_forbidden({**runner.SCORE_DEFAULTS, "flipud": 1.0})


def test_the_refusal_explains_why():
    with pytest.raises(ValueError, match="mirror-symmetric"):
        runner.check_forbidden({**runner.SCORE_DEFAULTS, "fliplr": 0.5})


def test_a_clean_config_is_accepted():
    runner.check_forbidden(dict(runner.SCORE_DEFAULTS))


def test_an_override_cannot_smuggle_mirroring_past_the_check(tmp_path):
    """Setting fliplr through --set must not start a run."""
    data_yaml = build_dataset(tmp_path)

    code = runner.main(["--data", str(data_yaml), "--check-only", "--set", "fliplr=0.5"])

    assert code == 2


# --------------------------------------------------------------------------- #
# Nothing is inherited from ultralytics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "imgsz",
        "max_det",
        "mosaic",
        "close_mosaic",
        "fliplr",
        "flipud",
        "degrees",
        "scale",
        "translate",
        "shear",
        "perspective",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "erasing",
        "auto_augment",
    ],
)
def test_every_augmentation_knob_is_stated_explicitly(name):
    """A knob left unset inherits a COCO-photograph default, and the first
    checkpoint is what that costs.
    """
    assert name in runner.SCORE_DEFAULTS


def test_mosaic_is_off_rather_than_merely_closed_early():
    """close_mosaic only stops mosaic for the last N epochs. The Colab run had
    close_mosaic=10, so 20 of its 30 epochs still ran mosaicked.
    """
    assert runner.SCORE_DEFAULTS["mosaic"] == 0.0
    assert runner.SCORE_DEFAULTS["close_mosaic"] == 0


def test_rotation_is_small_and_non_zero():
    """Matched to the residual tilt deskew leaves behind."""
    degrees = runner.SCORE_DEFAULTS["degrees"]

    assert 0.0 < degrees <= 2.0


def test_rotation_covers_the_measured_deskew_residual():
    """Derived from deskew's own constants rather than guessed.

    Worst case is the Hough estimator's measured error (0.25 deg) plus the dead
    zone below which deskew declines to rotate at all.
    """
    from melodix.geometry.deskew import DeskewConfig

    worst_case = 0.25 + DeskewConfig().min_correction_deg

    assert runner.SCORE_DEFAULTS["degrees"] > worst_case


def test_shear_and_perspective_are_off():
    """Deskew corrects rotation only. Training on warped pages teaches the
    detector to read inputs Stage 1 will then fail to straighten.
    """
    assert runner.SCORE_DEFAULTS["shear"] == 0.0
    assert runner.SCORE_DEFAULTS["perspective"] == 0.0


def test_hue_and_saturation_are_off_but_brightness_is_not():
    """A rendered page is grayscale in three channels, so saturation is zero
    and hue undefined. Brightness genuinely varies on a real scan.
    """
    assert runner.SCORE_DEFAULTS["hsv_h"] == 0.0
    assert runner.SCORE_DEFAULTS["hsv_s"] == 0.0
    assert runner.SCORE_DEFAULTS["hsv_v"] > 0.0


# --------------------------------------------------------------------------- #
# The resolved config, recorded next to the weights
# --------------------------------------------------------------------------- #


def test_the_resolved_config_contains_every_setting(tmp_path):
    settings = runner.resolve_settings(tmp_path / "d.yaml", tmp_path / "runs")

    for name in runner.SCORE_DEFAULTS:
        assert name in settings


def test_overrides_win_over_the_defaults(tmp_path):
    settings = runner.resolve_settings(
        tmp_path / "d.yaml", tmp_path / "runs", overrides={"degrees": 7.5}
    )

    assert settings["degrees"] == 7.5


def test_the_resolved_config_is_written_beside_the_weights(tmp_path):
    """The drift this fixes: the first run's real settings were recoverable
    only by loading the checkpoint.
    """
    settings = runner.resolve_settings(tmp_path / "d.yaml", tmp_path / "runs")

    path = runner.write_resolved_config(tmp_path / "runs" / "r", "yolov8s.pt", settings)

    assert path.name == runner.RESOLVED_CONFIG_NAME
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["settings"]["fliplr"] == 0.0
    assert written["model"] == "yolov8s.pt"


def test_the_resolved_config_records_the_class_count(tmp_path):
    """So a run can be matched to the schema it trained against."""
    settings = runner.resolve_settings(tmp_path / "d.yaml", tmp_path / "runs")

    path = runner.write_resolved_config(tmp_path / "runs" / "r", "m.pt", settings)

    assert json.loads(path.read_text(encoding="utf-8"))["melodix_classes"] == NUM_CLASSES


def test_check_only_prints_the_whole_resolved_config(tmp_path, capsys):
    data_yaml = build_dataset(tmp_path)

    runner.main(["--data", str(data_yaml), "--check-only"])

    out = capsys.readouterr().out
    assert "RESOLVED CONFIG" in out
    assert "fliplr = 0.0" in out
    assert "epochs = 30" in out


# --------------------------------------------------------------------------- #
# The degradation switch: built, deliberately unused
# --------------------------------------------------------------------------- #


def test_a_degraded_variant_can_be_selected(tmp_path, capsys):
    source = build_dataset(tmp_path / "clean")
    degraded_root = tmp_path / "degraded"
    build_dataset(degraded_root)

    code = runner.main(
        ["--data", str(source), "--degraded", str(degraded_root), "--check-only"]
    )

    assert code == 0
    assert str(degraded_root) in capsys.readouterr().out


def test_a_missing_degraded_variant_is_reported(tmp_path, capsys):
    source = build_dataset(tmp_path / "clean")

    code = runner.main(
        ["--data", str(source), "--degraded", str(tmp_path / "absent"), "--check-only"]
    )

    assert code == 2
    assert "scripts/degrade.py" in capsys.readouterr().err


def test_the_clean_dataset_is_used_by_default(tmp_path, capsys):
    """The switch stays off until real pages show what to tune it against."""
    data_yaml = build_dataset(tmp_path)

    runner.main(["--data", str(data_yaml), "--check-only"])

    preamble = capsys.readouterr().out.split("RESOLVED CONFIG")[0]
    assert "degraded" not in preamble


# --------------------------------------------------------------------------- #
# scale: derived, and pinned by the derivation rather than the literal
# --------------------------------------------------------------------------- #

# Measured on datasets/melodix_synth: augmentation_dot and staccato render
# 6.0 px on a 1000x1400 page. YOLO letterboxes the long side to imgsz, so at
# imgsz=1280 that is 6.0 * 1280/1400 = 5.49 px reaching the model.
SMALLEST_GLYPH_PX_ON_PAGE = 6.0
PAGE_LONG_SIDE = 1400

# Below roughly half of YOLOv8's finest stride (P3, stride 8) a box has no
# localisation budget left: clearing IoU 0.5 would need sub-pixel accuracy from
# a head that predicts on an 8 px grid.
MIN_GLYPH_PX_AT_INPUT = 4.0


def smallest_glyph_at_input(imgsz: int, scale: float) -> float:
    """Size of the smallest class at model input, at the harshest scale draw."""
    letterboxed = SMALLEST_GLYPH_PX_ON_PAGE * imgsz / PAGE_LONG_SIDE
    return letterboxed * (1 - scale)  # ultralytics draws from [1-s, 1+s]


def localisation_budget(size_px: float, iou: float) -> float:
    """How far a predicted box may sit from truth and still clear an IoU."""
    return size_px * (1 - iou) / (1 + iou)


def test_scale_leaves_the_smallest_classes_above_the_localisation_floor():
    """The reason scale is 0.20 and not the ultralytics default.

    At the stock 0.5 the smallest glyphs reach the model at 2.75 px, where
    clearing IoU 0.50 demands sub-pixel localisation from a stride-8 head.
    augmentation_dot at 0.503 mAP50-95 and staccato at 0.687 are the two
    weakest classes in the baseline, and both are that glyph.
    """
    size = smallest_glyph_at_input(
        int(runner.SCORE_DEFAULTS["imgsz"]), float(runner.SCORE_DEFAULTS["scale"])
    )

    assert size >= MIN_GLYPH_PX_AT_INPUT, f"smallest glyph falls to {size:.2f} px"


def test_the_stock_scale_would_starve_them():
    """Guards the test above against becoming vacuous: the floor must be one
    the stock setting actually violates.
    """
    stock = smallest_glyph_at_input(1280, 0.5)

    assert stock < MIN_GLYPH_PX_AT_INPUT


def test_scale_still_covers_real_engraving_variation():
    """The other half of the derivation. Rastral sizes span roughly 6-9 pt of
    staff height, about +/-20%, so scale must not be so small that genuine
    engraving-size variation goes untrained.
    """
    assert runner.SCORE_DEFAULTS["scale"] >= 0.15


def test_the_smallest_class_keeps_a_localisation_budget_over_one_pixel():
    """A budget under a pixel asks for accuracy finer than the P3 grid."""
    size = smallest_glyph_at_input(
        int(runner.SCORE_DEFAULTS["imgsz"]), float(runner.SCORE_DEFAULTS["scale"])
    )

    assert localisation_budget(size, 0.5) > 1.0


def test_lowering_imgsz_would_break_the_floor_too():
    """Scale is not the only lever on this: the derivation depends on imgsz,
    so halving the render resolution starves the same classes.
    """
    assert smallest_glyph_at_input(640, float(runner.SCORE_DEFAULTS["scale"])) < (
        MIN_GLYPH_PX_AT_INPUT
    )
