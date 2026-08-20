"""Stage 1: spatial geometry and grid engine.

Pure classical computer vision. Establishes the coordinate system that every
later stage speaks in: staff line rows, measure boundaries, and system
groupings.

Order matters. :func:`deskew` runs first, because both the staff pass and the
vertical pass assume lines level to within roughly a degree::

    from melodix.geometry import deskew, detect_staff_grids, detect_vertical_segments

    page = deskew(raw_page).image
    grids = detect_staff_grids(page)
    strokes = detect_vertical_segments(page, staff_spacing=grids[0].line_spacing)

:func:`detect_vertical_segments` returns barlines, stems and brackets
undifferentiated; :func:`build_systems` separates them, groups staves into
ensemble systems and slices measures::

    systems = build_systems(grids, strokes)
    for system in systems:
        for measure in system.measures_at(0):   # every player's first bar
            ...
"""

from melodix.geometry.systems import (
    BarlineColumn,
    Measure,
    System,
    SystemGroupingConfig,
    assign_systems,
    build_columns,
    build_systems,
    classify_barlines,
    slice_measures,
)

from melodix.geometry.barlines import (
    BarlineDetectionConfig,
    VerticalSegment,
    detect_vertical_segments,
    extract_vertical_segments,
    isolate_vertical_runs,
    merge_collinear,
)

from melodix.geometry.deskew import (
    DeskewConfig,
    DeskewResult,
    SkewEstimate,
    SkewMethod,
    deskew,
    estimate_skew,
    estimate_skew_hough,
    estimate_skew_projection,
    rotate_image,
)
from melodix.geometry.staff import (
    LINES_PER_STAFF,
    STAFF_TOP_POSITION,
    LineBand,
    StaffDetectionConfig,
    StaffGrid,
    StaffLine,
    binarize,
    detect_staff_grids,
    extract_line_bands,
    group_bands_into_grids,
    isolate_horizontal_runs,
    to_grayscale,
)

__all__ = [
    # barlines
    "BarlineDetectionConfig",
    "VerticalSegment",
    "detect_vertical_segments",
    "extract_vertical_segments",
    "isolate_vertical_runs",
    "merge_collinear",
    # staff
    "LINES_PER_STAFF",
    "STAFF_TOP_POSITION",
    "LineBand",
    "StaffDetectionConfig",
    "StaffGrid",
    "StaffLine",
    "binarize",
    "detect_staff_grids",
    "extract_line_bands",
    "group_bands_into_grids",
    "isolate_horizontal_runs",
    "to_grayscale",
    # deskew
    "DeskewConfig",
    "DeskewResult",
    "SkewEstimate",
    "SkewMethod",
    "deskew",
    "estimate_skew",
    "estimate_skew_hough",
    "estimate_skew_projection",
    "rotate_image",
    # systems
    "BarlineColumn",
    "Measure",
    "System",
    "SystemGroupingConfig",
    "assign_systems",
    "build_columns",
    "build_systems",
    "classify_barlines",
    "slice_measures",
]