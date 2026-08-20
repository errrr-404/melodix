"""Stage 2: symbol recognition.

Finds and classifies the symbols on a page that Stage 1 has already measured.
The detector reports **shapes**, never voices: a cross notehead is a cross
notehead whatever line it sits on, and pairing that shape with a staff position
to get a drum voice is Stage 3's work.

The two halves of this package have very different dependencies. The schema and
the dataset tooling are plain data and importable anywhere::

    from melodix.vision import SymbolClass, BoundingBox, split_dataset

while the detector itself needs torch and ultralytics, installed via the
``vision`` extra::

    pip install -e ".[dev,vision]"

Keeping that import out of :mod:`~melodix.vision.labels` and
:mod:`~melodix.vision.dataset` means annotations can be written and validated
on a machine that will never run training.
"""

from melodix.vision.dataset import (
    IMAGE_SUFFIXES,
    Annotation,
    BoundingBox,
    DatasetSplit,
    LabeledImage,
    class_distribution,
    label_path_for_image,
    parse_label_file,
    split_dataset,
    write_data_yaml,
    write_label_file,
)
from melodix.vision.detector import (
    Detection,
    DetectorConfig,
    DetectorNotAvailableError,
    PageDetections,
    SymbolDetector,
)
from melodix.vision.labels import (
    LABELS,
    NUM_CLASSES,
    SymbolCategory,
    SymbolClass,
    SymbolLabel,
    class_names,
    label_for_id,
    label_for_name,
    labels_in_category,
)

__all__ = [
    # labels
    "LABELS",
    "NUM_CLASSES",
    "SymbolCategory",
    "SymbolClass",
    "SymbolLabel",
    "class_names",
    "label_for_id",
    "label_for_name",
    "labels_in_category",
    # dataset
    "IMAGE_SUFFIXES",
    "Annotation",
    "BoundingBox",
    "DatasetSplit",
    "LabeledImage",
    "class_distribution",
    "label_path_for_image",
    "parse_label_file",
    "split_dataset",
    "write_data_yaml",
    "write_label_file",
    # detector
    "Detection",
    "DetectorConfig",
    "DetectorNotAvailableError",
    "PageDetections",
    "SymbolDetector",
]
