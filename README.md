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
python scripts/fix_opencv.py     # required after [vision]: see below
pip install -e ".[dev,audio]"    # adds pyfluidsynth + pydub
```

Stage 4 also needs system binaries: `libfluidsynth` and `ffmpeg`.

### Training

**Ad-hoc `model.train()` calls are not the supported path.** They inherit
ultralytics' COCO-tuned augmentation defaults, and the drift is invisible until
someone reads `train_args` out of the checkpoint — which is exactly how the
first checkpoint came to be trained with half its pages mirrored. Use
`scripts/train_yolo.py`, or `notebooks/train_colab.ipynb`, which calls it. Every
run writes `melodix_resolved_config.json` beside its weights.

```bash
python scripts/verify_labels.py --data datasets/melodix_synth   # boxes on glyphs?
python scripts/train_yolo.py --data datasets/melodix_synth/data.yaml --check-only
python scripts/train_yolo.py --data datasets/melodix_synth/data.yaml
```

### OpenCV

ultralytics hard-requires `opencv-python` (the GUI build, which pulls Qt and
breaks headless Docker/CI); this project requires `opencv-python-headless`. Both
ship the same `cv2` package and overlap on 52 files, so installing the `vision`
extra leaves both present and whichever unpacked last wins. pip has no
dependency-override mechanism, so `pyproject.toml` cannot prevent this. Run
`python scripts/fix_opencv.py` after installing; `--check` verifies the state.
Installing with `uv` needs no fix — the override is declared in `[tool.uv]`.

That script is a **mitigation, not a repair**. The deeper issue is that
importing ultralytics mutates global state in shared libraries — it rebinds
`cv2.imread`/`imwrite`/`imshow` on Windows, disables OpenCV threading
process-wide, and replaces `torch.save`. Two live bugs have come from it.
`docs/ultralytics-patches.md` is the audit; read it before adding a direct
`cv2.imread` or `cv2.imwrite` call anywhere. Use `melodix.ingest.read_grayscale`
and `melodix.ingest.write_image` instead.

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
  - Checkpoint: YOLOv8s, 30 epochs on Colab T4, synthetic data only. See `models/PROVENANCE.md`.
  - Pretraining only. No real page has been shown to this model, in training or evaluation.
- [x] **Phase 2.5** Reality-check tooling
  - `ingest/loader.py` — PDF and image to page arrays
  - `scripts/degrade.py` — scan-realistic degradation, boxes carried through geometry
  - `scripts/evaluate_real.py` — per-class + ensemble-slice metrics on real pages
  - `scripts/validate_yolo.py`, `scripts/fix_opencv.py`
- [x] **Phase 2.6** Training config correction
  - Augmentation stated explicitly in `scripts/train_yolo.py`; mirror augmentation refused
  - `notebooks/train_colab.ipynb` calls the script; resolved config written beside the weights
  - `scripts/verify_labels.py` — dataset label integrity; current dataset verified aligned
- [x] **Phase 2.7** Pre-retrain audit
  - `scale` derived from measured glyph size and localisation budget (0.5 → 0.20)
  - ultralytics patch surface enumerated in `docs/ultralytics-patches.md`; risky patches wrapped
  - **Retrain pending: needs GPU access. See `models/PROVENANCE.md`.**
- [x] **Phase 2.8** Look at a real page
  - `scripts/inspect_real.py` — annotated pages, Stage 1 overlay, no ground truth needed
  - `evaluate_real.py` reports small-class AP50 and centroid error against Stage 1's snap budget
  - **Finding: Stage 1 staff detection fails on speckled pages.** See `models/PROVENANCE.md`.
- [ ] **Phase 3** Reconstruction and MIDI
- [ ] **Phase 4** Synthesis and encoding
