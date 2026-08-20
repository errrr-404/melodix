# Stage 1 — Spatial Geometry

**Status: complete.** 318 tests passing.

## What Stage 1 does

Stage 1 establishes the coordinate system every later stage speaks in. It does
not recognise a single musical symbol — that is Stage 2's job. It answers three
questions about a page of drum notation:

1. **Is the page straight?** If not, straighten it.
2. **Where are the staves?** Find every five-line staff and build a grid that
   converts between pixel rows and musical positions.
3. **How is the page structured?** Group staves into ensemble systems, and cut
   each staff into measures at its barlines.

The output is a list of `System` objects. Each carries its staves, its barline
columns, and its measures, where every measure is addressable as
`(system_index, staff_index, measure_index)`.

## Order of operations

Order matters, and getting it wrong fails silently rather than loudly.

```python
from melodix.geometry import (
    build_systems,
    deskew,
    detect_staff_grids,
    detect_vertical_segments,
)

page = deskew(raw_page).image                  # 1. straighten first
grids = detect_staff_grids(page)               # 2. then find staves
strokes = detect_vertical_segments(            # 3. then find vertical ink
    page, staff_spacing=grids[0].line_spacing
)
systems = build_systems(grids, strokes)        # 4. then work out structure

for measure in systems[0].measures_at(0):      # every player's first bar
    ...
```

Deskew runs first because both later passes use morphological kernels that
degrade once the page tilts beyond roughly one degree. Detection runs before
structure because a stroke cannot be classified as a barline without knowing
which staff it should span.

## Modules

### `deskew.py` — page straightening

Estimates skew by two independent methods and applies the correction. The
projection method rotates through candidate angles and scores each by how
sharply ink concentrates into rows; the Hough method measures the angle of long
detected lines directly. `method="auto"` prefers Hough when it finds enough
long strokes and falls back to projection otherwise.

The sign convention is fixed and asserted directly: lines rising to the right
give positive skew, and the correction is its negation. Deskew declines to
rotate when the estimate is unconfident or the tilt is negligible, but reports
the estimate either way.

### `staff.py` — staff detection and the coordinate grid

Grayscale, Otsu threshold, then a morphological opening with a long horizontal
kernel that erases everything except long level strokes — noteheads, stems,
flags and text all vanish. Surviving rows collapse into candidate bands, whose
ink-weighted centroids give sub-pixel line rows. Bands are grouped five at a
time into staves, accepting a window only when its four gaps agree and no line
is implausibly thick for its spacing.

The result is `StaffGrid`, the contract between Stage 1 and Stage 3.

### `barlines.py` — vertical stroke detection

The mirror of the staff pass: a tall, one-pixel-wide kernel keeps only vertical
ink. Connected components — deliberately not a column projection, which would
fuse two barlines at the same column on different staves into one impossible
stroke spanning the gap between systems.

This module returns barlines, stems, brackets and box edges
**undifferentiated**, and that restraint is the point. A detector with no access
to the staff grid genuinely cannot tell a barline from a note stem: both are
thin vertical strokes, and beamed sixteenth runs routinely produce stems longer
than a barline. Any height threshold that excluded stems would also discard
barlines on a smaller system. Classification needs the staff grid, so it lives
in `systems.py`.

### `systems.py` — structure

Three tiers, in order:

- **Tier 1, span check.** A stroke is a barline for a staff when it reaches
  both that staff's top and bottom line, within tolerance. Stems fail because
  they hang off a notehead and stop short of one end.
- **Tier 2, grouping and alignment.** Staves are grouped into systems, then
  their barlines are gathered into columns sharing an x position.
- **Tier 3, slicing.** Consecutive columns bound measures.

The ordering in Tier 2 is easy to get backwards and fails silently.
**Column alignment cannot define a system.** Barlines at the left and right
margins align vertically down the whole page, so grouping staves by shared
columns fuses every system on the page into one — yielding a plausible-looking
result with the measure count of one system and the staff count of the page.
Systems are therefore established first, from vertical spacing and from strokes
that physically span more than one staff, and only then are columns computed
within each system.

This tier is also where the two engraving styles converge. A continuous rule
drawn through an entire system and separate per-staff rules produce the same
measure grid, so nothing downstream needs to know which style the engraver
used.

## The staff position convention

Pixel space is y-down. Staff space counts integer half-space steps upward from
the bottom line:

```
position 8  ────────────  top line        Closed Hi-Hat [42]
position 7                space 4
position 6  ────────────  line 4
position 5                space 3         Acoustic Snare [38]
position 4  ────────────  line 3
position 3                space 2
position 2  ────────────  line 2
position 1                space 1
position 0  ────────────  bottom line
```

Even positions are lines, odd are spaces. Ledger territory is simply negative
or greater than 8, so notes off the staff need no special branch.

Conversion interpolates between the *measured* line rows rather than assuming
uniform spacing, so mild scanner warp does not accumulate error toward the top
of the staff.

`grid.snap(y)` returns `None` rather than guessing when a centroid falls
between two positions. A wrong drum voice is worse than a dropped note.

## Test suite

318 tests, all passing.

| File | Tests | Covers |
|---|---|---|
| `tests/test_staff.py` | 102 | Detection pipeline, grouping, the 0-8 mapping, snapping |
| `tests/test_systems.py` | 75 | All three tiers, grouping, column alignment, slicing |
| `tests/test_deskew.py` | 73 | Sign convention, both estimators, rotation, integration |
| `tests/test_barlines.py` | 68 | Isolation, extraction, height floors, fragment merging |

`tests/helpers.py` draws synthetic pages — blank pages, staves, noteheads and
stems — in place on a given image, falling back to NumPy rasterisation when
OpenCV is unavailable.

Each suite was mutation-checked rather than merely run: inverting the staff
position mapping breaks 15 staff tests, removing the aspect-ratio guard breaks
the barlines test written for it, and making spacing always group staves — the
exact silent failure described above — breaks 7 systems tests.

## Known gaps

- `ruff` and `mypy` are configured in `pyproject.toml` over both `src` and
  `tests`, but neither is installed, so neither has been run against this code.
- `pyproject.toml` declares the console script `melodix = "melodix.cli:main"`,
  but `cli.py` does not exist yet. The entry point is currently broken.
- `merge_collinear` in `barlines.py` sorts fragments by x before y, so two
  fragments of one stroke can fail to merge when another segment at a similar
  column sorts between them. Not currently reachable through
  `detect_vertical_segments` on clean input, but worth fixing.
- The `README.md` layout section is aspirational; only `geometry/` exists.

## Next: Stage 2

Symbol recognition in `melodix.vision`. YOLO classes represent **symbol shapes
only** — `round_notehead`, `cross_notehead`, `accent`, `rest_quarter` — never
vertical positions. Position comes from `grid.snap()`, and Stage 3 combines
shape and position into a General MIDI percussion voice.

Folding position into the class label would multiply the class count, discard
the sub-pixel precision computed here, and force a full retrain whenever a kit
is re-voiced.
