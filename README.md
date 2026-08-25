# Melodix

Optical music recognition for drum notation. Converts drum sheet music (PDF or
image, including multi-drummer ensemble scores) into multi-track MIDI,
synthesized MP3 audio, and interactive `sync_map.json` metadata for
karaoke-style sheet highlighting.

## Pipeline

| Stage | Package | Technique | Output |
|---|---|---|---|
| 1. Spatial geometry | `melodix.geometry` | OpenCV | Staff grids, barlines, systems |
| 2. Symbol recognition | `melodix.vision` | YOLOv8 / PyTorch | Classified bounding boxes |
| 3. Reconstruction | `melodix.mapping`, `melodix.midi` | mido | Multi-track MIDI + sync map |
| 4. Synthesis | `melodix.audio` | FluidSynth / FFmpeg | WAV → MP3 |

## Layout

```
melodix/
├── pyproject.toml
├── src/melodix/
│   ├── ingest/          PDF and image → numpy arrays
│   ├── geometry/        Stage 1 — staff, deskew, barlines, systems
│   ├── vision/          Stage 2 — detector, labels, dataset
│   ├── mapping/         Stage 3 — percussion_map, reconstruct
│   ├── midi/            Stage 3 — writer
│   ├── audio/           Stage 4 — synth, encode
│   ├── export/          sync_map.json serialization
│   ├── types.py         Shared primitives
│   ├── pipeline.py      Orchestration
│   └── cli.py           Entry point
├── tests/               Mirrors the src tree
├── assets/soundfonts/   .sf2 files (gitignored)
├── models/              YOLO checkpoints (gitignored)
└── scripts/             Training and evaluation utilities
```

## Install

```bash
pip install -e ".[dev]"          # Stages 1 and 3
pip install -e ".[dev,vision]"   # adds torch + ultralytics
pip install -e ".[dev,audio]"    # adds pyfluidsynth + pydub
```

Stage 4 also needs system binaries: `libfluidsynth` and `ffmpeg`.

## Staff position convention

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
or greater than 8, so no branch is needed for notes off the staff.

```python
from melodix.geometry import detect_staff_grids

grids = detect_staff_grids(page)          # top to bottom
position = grids[0].snap(notehead_y)      # int, or None if ambiguous
```

`snap()` returns `None` rather than guessing when a centroid falls between two
positions — a wrong drum voice is worse than a dropped note.

## Testing

```bash
pytest                                    # whole suite
pytest tests/test_staff.py -v             # one Stage 1 module
pytest --cov=melodix --cov-report=term-missing
```

## Roadmap

- [x] **Phase 1.1** Scaffolding, `pyproject.toml`, `geometry/staff.py`
- [x] **Phase 1.2** `geometry/deskew.py` — skew estimation before detection
- [x] **Phase 1.3** `geometry/barlines.py` — measure segmentation
- [x] **Phase 1.4** `geometry/systems.py` — ensemble system grouping
- [x] **Phase 2** Symbol recognition — `vision/labels.py`, `vision/dataset.py`, `vision/detector.py`
  - Pretraining checkpoint only: synthetic data, 2 epochs. Needs real scans before shipping.
- [ ] **Phase 3** Reconstruction and MIDI
- [ ] **Phase 4** Synthesis and encoding
