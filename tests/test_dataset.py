"""Unit tests for :mod:`melodix.vision.dataset`.

Two properties get the most attention here. Normalisation must survive a round
trip through pixels, because a systematic error there shifts every box on every
page by the same amount and still trains to a plausible-looking loss. And the
split must be deterministic, because a split that depends on filesystem
ordering makes two training runs incomparable without ever failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from melodix.vision.dataset import (
    IMAGE_SUFFIXES,
    Annotation,
    BoundingBox,
    LabeledImage,
    class_distribution,
    label_path_for_image,
    parse_label_file,
    split_dataset,
    write_data_yaml,
    write_label_file,
)
from melodix.vision.labels import NUM_CLASSES, SymbolClass, class_names

# Coordinates are written to six decimals, so a file round trip is lossy at
# roughly that scale.
COORD_TOL = 1e-6

PAGE_W = 800
PAGE_H = 600


def box(cx=0.5, cy=0.5, w=0.1, h=0.1) -> BoundingBox:
    """A valid box, overridable field by field."""
    return BoundingBox(cx=cx, cy=cy, w=w, h=h)


def annotation(symbol=SymbolClass.ROUND_NOTEHEAD, **kwargs) -> Annotation:
    """One annotation with a valid box."""
    return Annotation(symbol=symbol, box=box(**kwargs))


def page(name: str, *symbols: SymbolClass) -> LabeledImage:
    """A labelled page carrying one annotation per given symbol."""
    return LabeledImage(
        image_path=Path(f"images/{name}.png"),
        annotations=tuple(annotation(symbol) for symbol in symbols),
    )


def pages(count: int) -> list[LabeledImage]:
    """``count`` distinct pages, named with a stable zero-padded stem."""
    return [page(f"page_{index:03d}") for index in range(count)]


# --------------------------------------------------------------------------- #
# BoundingBox validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cx", [-0.1, 1.1, 2.0])
def test_a_centre_outside_the_page_is_rejected(cx):
    with pytest.raises(ValueError, match="cx must be in"):
        box(cx=cx)


@pytest.mark.parametrize("cy", [-0.1, 1.1])
def test_a_centre_row_outside_the_page_is_rejected(cy):
    with pytest.raises(ValueError, match="cy must be in"):
        box(cy=cy)


@pytest.mark.parametrize("w", [0.0, -0.1, 1.1])
def test_an_invalid_width_is_rejected(w):
    """Zero area usually means an annotator clicked without dragging."""
    with pytest.raises(ValueError, match="w must be in"):
        box(w=w)


@pytest.mark.parametrize("h", [0.0, -0.1, 1.1])
def test_an_invalid_height_is_rejected(h):
    with pytest.raises(ValueError, match="h must be in"):
        box(h=h)


def test_a_box_covering_the_whole_page_is_valid():
    assert box(cx=0.5, cy=0.5, w=1.0, h=1.0).area == pytest.approx(1.0)


def test_a_box_at_the_corner_is_valid():
    assert box(cx=0.0, cy=0.0, w=0.1, h=0.1).cx == 0.0


# --------------------------------------------------------------------------- #
# BoundingBox geometry
# --------------------------------------------------------------------------- #


def test_edges_straddle_the_centre():
    subject = box(cx=0.5, cy=0.4, w=0.2, h=0.1)

    assert subject.x_min == pytest.approx(0.4)
    assert subject.x_max == pytest.approx(0.6)
    assert subject.y_min == pytest.approx(0.35)
    assert subject.y_max == pytest.approx(0.45)


def test_area_is_the_product_of_the_extents():
    assert box(w=0.2, h=0.5).area == pytest.approx(0.1)


def test_a_box_inside_the_page_is_not_clipped():
    assert not box(cx=0.5, cy=0.5, w=0.2, h=0.2).is_clipped


@pytest.mark.parametrize(
    ("cx", "cy"),
    [(0.02, 0.5), (0.98, 0.5), (0.5, 0.02), (0.5, 0.98)],
)
def test_a_box_running_off_an_edge_is_clipped(cx, cy):
    """Legal, but worth reporting: a dataset full of these usually means the
    annotations were exported against a different crop than the images.
    """
    assert box(cx=cx, cy=cy, w=0.1, h=0.1).is_clipped


# --------------------------------------------------------------------------- #
# Pixel conversion
# --------------------------------------------------------------------------- #


def test_a_box_is_built_from_pixel_corners():
    subject = BoundingBox.from_pixels(100, 200, 300, 400, PAGE_W, PAGE_H)

    assert subject.cx == pytest.approx(200 / PAGE_W)
    assert subject.cy == pytest.approx(300 / PAGE_H)
    assert subject.w == pytest.approx(200 / PAGE_W)
    assert subject.h == pytest.approx(200 / PAGE_H)


def test_pixels_survive_a_round_trip():
    """A systematic error here shifts every box on every page equally and
    still trains to a plausible loss.
    """
    corners = (100.0, 200.0, 300.0, 400.0)

    subject = BoundingBox.from_pixels(*corners, PAGE_W, PAGE_H)

    assert subject.to_pixels(PAGE_W, PAGE_H) == pytest.approx(corners)


@pytest.mark.parametrize(("width", "height"), [(800, 600), (1600, 1200), (413, 977)])
def test_normalisation_is_resolution_independent(width, height):
    """One checkpoint must read 150 and 300 DPI scans alike."""
    subject = BoundingBox.from_pixels(
        0.1 * width, 0.2 * height, 0.3 * width, 0.4 * height, width, height
    )

    assert subject.cx == pytest.approx(0.2)
    assert subject.cy == pytest.approx(0.3)


def test_the_centre_in_pixels_is_what_stage_one_snaps():
    subject = BoundingBox.from_pixels(100, 200, 300, 400, PAGE_W, PAGE_H)

    assert subject.center_pixels(PAGE_W, PAGE_H) == pytest.approx((200.0, 300.0))


def test_inverted_corners_are_rejected():
    with pytest.raises(ValueError, match="corners must be ordered"):
        BoundingBox.from_pixels(300, 200, 100, 400, PAGE_W, PAGE_H)


def test_zero_width_corners_are_rejected():
    with pytest.raises(ValueError, match="corners must be ordered"):
        BoundingBox.from_pixels(100, 200, 100, 400, PAGE_W, PAGE_H)


@pytest.mark.parametrize(("width", "height"), [(0, 600), (800, 0), (-1, 600)])
def test_a_non_positive_image_size_is_rejected_when_building(width, height):
    with pytest.raises(ValueError, match="image size must be positive"):
        BoundingBox.from_pixels(10, 20, 30, 40, width, height)


@pytest.mark.parametrize(("width", "height"), [(0, 600), (800, 0)])
def test_a_non_positive_image_size_is_rejected_when_converting(width, height):
    with pytest.raises(ValueError, match="image size must be positive"):
        box().to_pixels(width, height)


@pytest.mark.parametrize(("width", "height"), [(0, 600), (800, 0)])
def test_a_non_positive_image_size_is_rejected_when_centring(width, height):
    with pytest.raises(ValueError, match="image size must be positive"):
        box().center_pixels(width, height)


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #


def test_a_box_fully_overlaps_itself():
    assert box().iou(box()) == pytest.approx(1.0)


def test_overlap_never_exceeds_one():
    """Float error can push a self-overlap a few ulps above 1.0, and callers
    threshold on this value.
    """
    subject = BoundingBox.from_pixels(100, 200, 140, 230, PAGE_W, PAGE_H)

    assert subject.iou(subject) <= 1.0


def test_disjoint_boxes_do_not_overlap():
    assert box(cx=0.2, cy=0.2).iou(box(cx=0.8, cy=0.8)) == 0.0


def test_boxes_touching_at_an_edge_do_not_overlap():
    left = box(cx=0.2, cy=0.5, w=0.2, h=0.2)
    right = box(cx=0.4, cy=0.5, w=0.2, h=0.2)

    assert left.iou(right) == 0.0


def test_half_overlapping_boxes_score_a_third():
    """Two equal boxes sharing half their area: 0.5 over 1.5."""
    left = box(cx=0.4, cy=0.5, w=0.2, h=0.2)
    right = box(cx=0.5, cy=0.5, w=0.2, h=0.2)

    assert left.iou(right) == pytest.approx(1 / 3)


def test_overlap_is_symmetric():
    left = box(cx=0.4, cy=0.5, w=0.2, h=0.3)
    right = box(cx=0.5, cy=0.55, w=0.25, h=0.2)

    assert left.iou(right) == pytest.approx(right.iou(left))


def test_a_contained_box_scores_the_area_ratio():
    outer = box(cx=0.5, cy=0.5, w=0.4, h=0.4)
    inner = box(cx=0.5, cy=0.5, w=0.2, h=0.2)

    assert outer.iou(inner) == pytest.approx(inner.area / outer.area)


# --------------------------------------------------------------------------- #
# YOLO line format
# --------------------------------------------------------------------------- #


def test_a_line_leads_with_the_class_id():
    line = Annotation(SymbolClass.CROSS_NOTEHEAD, box()).to_yolo_line()

    assert line.split()[0] == "2"


def test_a_line_carries_five_fields():
    assert len(annotation().to_yolo_line().split()) == 5


def test_a_line_round_trips_within_the_written_precision():
    original = Annotation(SymbolClass.ACCENT, BoundingBox(0.123456, 0.654321, 0.05, 0.07))

    parsed = Annotation.from_yolo_line(original.to_yolo_line())

    assert parsed.symbol is original.symbol
    assert parsed.box.cx == pytest.approx(original.box.cx, abs=COORD_TOL)
    assert parsed.box.cy == pytest.approx(original.box.cy, abs=COORD_TOL)


def test_a_line_is_parsed_into_the_right_class():
    parsed = Annotation.from_yolo_line("2 0.5 0.5 0.1 0.1")

    assert parsed.symbol is SymbolClass.CROSS_NOTEHEAD


def test_extra_whitespace_is_tolerated():
    parsed = Annotation.from_yolo_line("  2   0.5  0.5   0.1 0.1  ")

    assert parsed.symbol is SymbolClass.CROSS_NOTEHEAD


@pytest.mark.parametrize(
    "line",
    ["2 0.5 0.5 0.1", "2 0.5 0.5 0.1 0.1 0.1", "2", ""],
)
def test_a_line_with_the_wrong_field_count_is_rejected(line):
    with pytest.raises(ValueError, match="expected 5 fields"):
        Annotation.from_yolo_line(line)


def test_a_non_integer_class_id_is_rejected():
    with pytest.raises(ValueError, match="class id must be an integer"):
        Annotation.from_yolo_line("notehead 0.5 0.5 0.1 0.1")


@pytest.mark.parametrize("class_id", [-1, NUM_CLASSES, 999])
def test_a_class_id_outside_the_schema_is_rejected(class_id):
    """The label file and the schema have diverged; guessing would mislabel."""
    with pytest.raises(ValueError, match="outside the schema"):
        Annotation.from_yolo_line(f"{class_id} 0.5 0.5 0.1 0.1")


def test_non_numeric_coordinates_are_rejected():
    with pytest.raises(ValueError, match="coordinates must be numbers"):
        Annotation.from_yolo_line("2 left top 0.1 0.1")


def test_an_out_of_range_coordinate_is_rejected_by_the_box():
    with pytest.raises(ValueError, match="cx must be in"):
        Annotation.from_yolo_line("2 1.5 0.5 0.1 0.1")


# --------------------------------------------------------------------------- #
# Label files
# --------------------------------------------------------------------------- #


def test_a_written_file_reads_back(tmp_path):
    path = tmp_path / "page.txt"
    written = (
        Annotation(SymbolClass.CROSS_NOTEHEAD, box(cx=0.25, cy=0.25)),
        Annotation(SymbolClass.ROUND_NOTEHEAD, box(cx=0.75, cy=0.75)),
    )

    write_label_file(path, written)

    assert [a.symbol for a in parse_label_file(path)] == [a.symbol for a in written]


def test_annotation_order_is_preserved(tmp_path):
    path = tmp_path / "page.txt"
    symbols = [SymbolClass.ACCENT, SymbolClass.ROUND_NOTEHEAD, SymbolClass.BEAM]

    write_label_file(path, tuple(annotation(symbol) for symbol in symbols))

    assert [a.symbol for a in parse_label_file(path)] == symbols


def test_writing_creates_missing_directories(tmp_path):
    path = tmp_path / "labels" / "train" / "page.txt"

    write_label_file(path, (annotation(),))

    assert path.exists()


def test_an_empty_annotation_list_writes_an_empty_file(tmp_path):
    """Ultralytics distinguishes a missing label file from an empty one: the
    empty file marks a deliberate negative example.
    """
    path = tmp_path / "page.txt"

    write_label_file(path, ())

    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def test_an_empty_file_parses_as_no_annotations(tmp_path):
    path = tmp_path / "page.txt"
    path.write_text("", encoding="utf-8")

    assert parse_label_file(path) == ()


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "page.txt"
    path.write_text("2 0.5 0.5 0.1 0.1\n\n   \n0 0.2 0.2 0.1 0.1\n", encoding="utf-8")

    assert len(parse_label_file(path)) == 2


def test_every_line_ends_with_a_newline(tmp_path):
    path = tmp_path / "page.txt"

    write_label_file(path, (annotation(), annotation()))

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_a_malformed_line_names_the_file_and_line_number(tmp_path):
    """A bad annotation in a thousand-page dataset is otherwise hard to find."""
    path = tmp_path / "page.txt"
    path.write_text("2 0.5 0.5 0.1 0.1\n0 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"page\.txt:2:"):
        parse_label_file(path)


def test_a_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_label_file(tmp_path / "absent.txt")


# --------------------------------------------------------------------------- #
# Image and label pairing
# --------------------------------------------------------------------------- #


def test_a_label_path_swaps_images_for_labels():
    found = label_path_for_image(Path("data/images/train/page_001.png"))

    assert found == Path("data/labels/train/page_001.txt")


def test_pairing_uses_the_last_images_component():
    """A dataset stored under a directory called 'images' would otherwise
    have its root rewritten instead of its split directory.
    """
    found = label_path_for_image(Path("images/dataset/images/val/page.png"))

    assert found == Path("images/dataset/labels/val/page.txt")


def test_an_explicit_labels_root_skips_the_substitution():
    found = label_path_for_image(Path("anywhere/page.jpg"), labels_root=Path("out/labels"))

    assert found == Path("out/labels/page.txt")


def test_a_path_without_an_images_component_is_rejected():
    with pytest.raises(ValueError, match="no 'images' component"):
        label_path_for_image(Path("data/scans/page.png"))


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".tif"])
def test_the_label_suffix_replaces_any_image_suffix(suffix):
    found = label_path_for_image(Path(f"images/page{suffix}"))

    assert found.suffix == ".txt"


def test_the_recognised_image_suffixes_are_lowercase():
    assert all(s == s.lower() and s.startswith(".") for s in IMAGE_SUFFIXES)


def test_a_page_reports_its_stem():
    assert page("page_007").stem == "page_007"


def test_a_page_lists_its_symbols_in_order():
    subject = page("p", SymbolClass.ACCENT, SymbolClass.BEAM)

    assert subject.symbols() == (SymbolClass.ACCENT, SymbolClass.BEAM)


def test_a_page_may_carry_no_annotations():
    assert LabeledImage(image_path=Path("images/blank.png")).annotations == ()


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def test_the_split_holds_out_the_requested_fraction():
    result = split_dataset(pages(100), val_fraction=0.2)

    assert len(result.val) == 20
    assert len(result.train) == 80


def test_every_page_lands_in_exactly_one_half():
    source = pages(50)

    result = split_dataset(source, val_fraction=0.3)

    combined = [image.stem for image in (*result.train, *result.val)]
    assert sorted(combined) == sorted(image.stem for image in source)
    assert len(combined) == len(set(combined))


def test_the_same_seed_gives_the_same_split():
    """Two runs are only comparable if they evaluate on the same pages."""
    first = split_dataset(pages(40), seed=7)
    second = split_dataset(pages(40), seed=7)

    assert [i.stem for i in first.val] == [i.stem for i in second.val]


def test_a_different_seed_gives_a_different_split():
    first = split_dataset(pages(40), seed=1)
    second = split_dataset(pages(40), seed=2)

    assert [i.stem for i in first.val] != [i.stem for i in second.val]


def test_input_order_does_not_change_the_split():
    """Pages are sorted before shuffling, so filesystem ordering cannot leak
    into which pages are held out.
    """
    forward = pages(30)
    backward = list(reversed(forward))

    assert [i.stem for i in split_dataset(forward, seed=3).val] == [
        i.stem for i in split_dataset(backward, seed=3).val
    ]


def test_splitting_does_not_reorder_the_caller_list():
    source = pages(10)
    before = [image.stem for image in source]

    split_dataset(source, seed=1)

    assert [image.stem for image in source] == before


def test_a_positive_fraction_holds_out_at_least_one_page():
    """Rounding down to an empty validation set would evaluate on nothing."""
    result = split_dataset(pages(3), val_fraction=0.1)

    assert len(result.val) == 1
    assert len(result.train) == 2


def test_a_single_page_is_not_split_away_from_training():
    result = split_dataset(pages(1), val_fraction=0.5)

    assert result.total == 1


def test_a_zero_fraction_holds_out_nothing():
    result = split_dataset(pages(10), val_fraction=0.0)

    assert result.val == ()
    assert len(result.train) == 10


def test_an_empty_dataset_splits_into_nothing():
    result = split_dataset([], val_fraction=0.2)

    assert result.total == 0


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_an_out_of_range_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="val_fraction must be in"):
        split_dataset(pages(10), val_fraction=fraction)


def test_the_total_counts_both_halves():
    result = split_dataset(pages(25), val_fraction=0.2)

    assert result.total == 25


# --------------------------------------------------------------------------- #
# Class distribution
# --------------------------------------------------------------------------- #


def test_annotations_are_counted_per_class():
    counts = class_distribution(
        [
            page("a", SymbolClass.CROSS_NOTEHEAD, SymbolClass.CROSS_NOTEHEAD),
            page("b", SymbolClass.CROSS_NOTEHEAD, SymbolClass.ROUND_NOTEHEAD),
        ]
    )

    assert counts[SymbolClass.CROSS_NOTEHEAD] == 3
    assert counts[SymbolClass.ROUND_NOTEHEAD] == 1


def test_an_unseen_class_counts_zero():
    counts = class_distribution([page("a", SymbolClass.CROSS_NOTEHEAD)])

    assert counts[SymbolClass.TRIANGLE_NOTEHEAD] == 0


def test_counting_an_empty_dataset_yields_nothing():
    assert class_distribution([]) == {}


def test_pages_without_annotations_contribute_nothing():
    assert class_distribution([page("blank")]) == {}


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #


def test_data_yaml_declares_the_class_count(tmp_path):
    path = tmp_path / "data.yaml"

    write_data_yaml(path, tmp_path)

    assert f"nc: {NUM_CLASSES}" in path.read_text(encoding="utf-8")


def test_data_yaml_lists_every_class_by_id(tmp_path):
    path = tmp_path / "data.yaml"

    write_data_yaml(path, tmp_path)

    body = path.read_text(encoding="utf-8")
    for index, name in enumerate(class_names()):
        assert f"  {index}: {name}" in body


def test_data_yaml_orders_names_by_id_not_alphabetically(tmp_path):
    """Ultralytics reads the mapping positionally; sorting would remap."""
    path = tmp_path / "data.yaml"

    write_data_yaml(path, tmp_path)

    listed = [
        line.split(": ", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("  ") and ": " in line
    ]
    assert listed == class_names()


def test_data_yaml_records_the_split_directories(tmp_path):
    path = tmp_path / "data.yaml"

    write_data_yaml(path, tmp_path, train_dir="images/fit", val_dir="images/holdout")

    body = path.read_text(encoding="utf-8")
    assert "train: images/fit" in body
    assert "val: images/holdout" in body


def test_data_yaml_writes_an_absolute_root(tmp_path):
    """Ultralytics resolves a relative `path` against its own datasets
    directory, not the working directory, so a relative root silently sends
    training somewhere else.
    """
    path = tmp_path / "data.yaml"

    write_data_yaml(path, Path("datasets/relative"))

    root_line = next(
        line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("path: ")
    )
    assert Path(root_line.removeprefix("path: ")).is_absolute()


def test_data_yaml_uses_posix_paths(tmp_path):
    """A Windows backslash path breaks the YAML on a Linux training box."""
    path = tmp_path / "data.yaml"

    write_data_yaml(path, tmp_path)

    root_line = next(
        line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("path: ")
    )
    assert "\\" not in root_line


def test_data_yaml_warns_against_hand_editing(tmp_path):
    path = tmp_path / "data.yaml"

    write_data_yaml(path, tmp_path)

    assert path.read_text(encoding="utf-8").startswith("#")


def test_data_yaml_creates_missing_directories(tmp_path):
    path = tmp_path / "nested" / "config" / "data.yaml"

    write_data_yaml(path, tmp_path)

    assert path.exists()
