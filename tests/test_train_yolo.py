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
    """Stage 1 corrects skew first, so training past a couple of degrees
    spends capacity on inputs the pipeline guarantees will not arrive.
    """
    assert runner.SCORE_DEFAULTS["degrees"] == 2.0


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
