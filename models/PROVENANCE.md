# Checkpoint provenance

Every fact below is read from **inside** the checkpoint file
(`torch.load(...)["train_args"]`, `["train_results"]`), not from a sibling
`runs/` directory. Reading run metadata from a neighbouring folder caused one
misdiagnosis already: an abandoned local CPU run was mistaken for the real
training run because its `args.yaml` sat next to a similarly named path.

Weights are gitignored. This file is the record of what they are.

---

## `models/stage2_synth/weights/best.pt`

**Status: pretraining checkpoint. Synthetic data only. Not shippable.**

| | |
|---|---|
| md5 | `fb59f31e541eb4a29f7c7c5f5b9e11a6` |
| size | 22,638,058 bytes (22.6 MB) |
| saved | 2026-08-25T13:05:22Z |
| ultralytics | 8.4.128 |
| base model | `yolov8s.pt` |
| epochs | **30 of 30 completed** |
| wall time | 6,236 s (103.9 min), 207.9 s/epoch |
| imgsz | 1280 |
| batch | 8 |
| workers | 8 |
| patience | 0 (no early stopping) |
| classes | 28, order verified identical to `melodix.vision.labels` |

### Where it ran

Google Colab, T4. The `device` field in `train_args` is the **empty string**,
which is ultralytics' "auto" and records nothing — it reads identically on CPU
and GPU, so it does *not* confirm the device. Three other pieces of evidence do:

- `data: /content/datasets/melodix_synth/data.yaml` — a Colab path
- `project: /content/drive/MyDrive/melodix_runs` — Drive mounted in Colab
- 207.9 s/epoch against 7,278 s/epoch for the same architecture and image size
  on this machine's CPU: **35× faster**, which no CPU explains

The 22.6 MB size versus 85.5 MB for the per-epoch checkpoints is expected —
ultralytics strips optimizer and EMA state from `best.pt` and `last.pt`, leaving
inference-only weights.

### Baseline status

**This checkpoint is the baseline and must not be overwritten.** The corrected
augmentation run (`stage2_corrected_aug`) exists to be compared against it, and
that comparison is impossible if the file is replaced. Train new runs under a
new `--name`.

### Augmentation actually used

Read from the checkpoint, these are the **ultralytics stock defaults**, not the
values in `scripts/train_yolo.py`:

| setting | in the checkpoint | `train_yolo.py` intends |
|---|---|---|
| `mosaic` | **1.0** | 0.0 |
| `fliplr` | **0.5** | 0.0 |
| `degrees` | **0.0** | 1.0 |
| `max_det` | 3000 | 3000 |
| `close_mosaic` | **10** | 0 |
| `scale` | 0.5 | 0.35 |
| `translate` | 0.1 | 0.05 |
| `hsv_s` | 0.7 | 0.0 |
| `hsv_v` | 0.4 | 0.25 |

So this checkpoint was **not** trained through `scripts/train_yolo.py`. It was
trained with mosaic stitching on, horizontal flips on half the pages, and no
rotation. The reasoning against the first two is in that script's docstring:
mosaic destroys the grid regularity of notation and its seams read as barlines,
and notation is not mirror-symmetric so a flipped page teaches a symbol that
cannot occur. `degrees=0.0` means the model has no training-time exposure to the
residual tilt Stage 1 deskew leaves behind (measured max 0.10 deg).

Ranked by expected harm:

**`fliplr=0.5` is the serious one.** Mirrored notation cannot occur. Time
signature digits mirror into non-glyphs, an accent (`>`) mirrors into `<` which
is not in the schema, and stems attach right for stem-up and left for stem-down,
so mirroring produces stem-up-on-the-left. Worse than wasted samples: it teaches
*invariance* to left-right asymmetry, which is the exact cue those glyphs are
distinguished by. A sufficient explanation for `time_signature` at 0.872.

**`mosaic=1.0` is second**, and `close_mosaic=10` did apply — confirmed from
`train_args`. So epochs 1-20 ran mosaicked and 21-30 ran clean.

**The training curve does not show the expected recovery when mosaic closed.**
Epoch 21, the first clean epoch, *dropped* 0.0032, and the mean gain per epoch
was higher with mosaic on (epochs 16-20: +0.00162) than off (21-25: +0.00076).
The curve simply flattens asymptotically throughout. This is confounded — the
learning rate decays over the same window, so a slowing gain rate is expected
regardless, and one run cannot separate the two causes. The honest reading is
that the data neither confirms nor refutes that mosaic hurt, but shows no
visible recovery at the transition, which is mild evidence against it having
been a large drag.

**`degrees=0.0` is third**: a gap to fill rather than damage to undo.

None of this is visible in the synthetic metrics, because validation applies no
augmentation and the val pages come from the same generator. Retraining through
`scripts/train_yolo.py` is the controlled comparison.

### Metrics

Two sets, measured on two different val splits. Both are synthetic.

**Recorded inside the checkpoint** (Colab val split, at the best epoch):

| | |
|---|---|
| mAP50 | 0.9888 |
| mAP50-95 | 0.8778 |
| precision | 0.99487 |
| recall | 0.98393 |

**Re-measured locally** (`scripts/validate_yolo.py`, local val snapshot of 503
pages, imgsz 1280, CPU, ~16 min):

| | |
|---|---|
| mAP50 | 0.98876 |
| mAP50-95 | **0.91182** |
| precision | 0.99468 |
| recall | 0.98419 |

The two disagree on mAP50-95 by 3.4 points while agreeing on mAP50, precision
and recall to within 0.0003. That pattern means **box localisation differs, not
detection** — the model finds the same symbols either way, but scores better
against local labels at strict IoU thresholds.

The likely cause is that the local dataset is not the snapshot Colab trained on.
The generator was fixed between the two: `augment()` used to rotate the page
without rotating its ground-truth boxes, so on any tilted page the labels sat
beside their symbols. That defect depresses strict-IoU scores and barely touches
mAP50 — exactly the observed signature. Unproven without the Colab snapshot, but
it is the reading that fits.

**Treat 0.8778 as this checkpoint's own number** and 0.91182 as a re-measurement
against different labels.

The direction is the one the hypothesis predicts, which strengthens it. If the
training labels carried box misalignment, the model still learns the true
average symbol position — noise around a correct mean does not shift the mean.
Evaluating that model against *noisy* labels then scores worse than evaluating
it against *clean* ones. Measured 0.9118 > recorded 0.8778 is exactly that, and
it implies this checkpoint is slightly better than its own record states.

**Falsifiable prediction.** Retraining on the corrected labels should improve
localisation further: mAP50-95 above 0.9118 on the current dataset, with mAP50
roughly unchanged near 0.9888. If mAP50-95 lands at or below 0.9118, the label
hypothesis is wrong and the gap has another cause. Recorded before the run so
it cannot be reinterpreted afterwards.

Per-class mAP50-95 from the local run is in `models/stage2_synth/validation.json`.
Weakest classes:

| mAP50-95 | class |
|---|---|
| 0.503 | `augmentation_dot` |
| 0.687 | `staccato` |
| 0.786 | `tie_slur` |
| 0.838 | `repeat_dots` |

`augmentation_dot` and `staccato` are the same glyph — a small filled circle —
separated only by position relative to a notehead. A detector that deliberately
does not encode position cannot fully separate them, so Stage 3 should expect to
disambiguate these two geometrically rather than trusting the class.

### Dataset

Local snapshot at time of measurement:

| | |
|---|---|
| train | 2,003 pages |
| val | 503 pages |
| annotations | 516,134 |
| size | 1.4 GB |
| generator | `scripts/generate_synthetic_dataset.py` |
| classes present | 28 of 28 |

Procedurally rendered, ground truth exact by construction. **Synthetic only** —
no real scan has ever been shown to this model, in training or in evaluation.
Every number on this page is a synthetic-to-synthetic measurement and none of
them predicts real-page performance.

---

## `runs_local_cpu_abandoned/` — not a model of record

An abandoned local CPU run from 2026-08-21. Reached epochs 1–3 of 30 and was
killed; its `results.csv` retains a single row. Unrelated to the checkpoint
above — different weights entirely (md5 `92877aac9f3e15ccbfc7e6b37028d68a`,
89.7 MB, unstripped).

Kept only because its `args.yaml` is the artifact that caused the original
misdiagnosis, and renaming it is cheaper than explaining it twice. Gitignored.

---

## Label integrity of the current dataset

Checked with `scripts/verify_labels.py` over 60 pages and 24,188 boxes.

**Verdict: aligned. No regeneration needed.**

The decisive measurement is centroid offset against distance from the page
centre. A label set left behind by a rotation is displaced in proportion to
radius; nothing else produces that pattern. Measured correlation **-0.058**,
effect size 0.005, and mean offset flat across bands (0.0253 at the centre to
0.0205 at the edge — if anything falling):

| radius band | mean offset | boxes |
|---|---|---|
| 0.00-0.25 | 0.0253 | 968 |
| 0.25-0.50 | 0.0246 | 3,035 |
| 0.50-0.75 | 0.0237 | 1,615 |
| 0.75-1.01 | 0.0205 | 589 |

5.1% of boxes flag on the per-box heuristic, concentrated in `open_modifier`,
`accent` and `marcato`. That is a measurement confound, not misalignment:
those are small glyphs that sit against beams and noteheads, and the
neighbour's ink inside the box drags the measured centre of mass. Confirmed by
cropping and looking — the boxes sit exactly on the glyphs.

Two false alarms were found and fixed in the checker itself along the way, both
recorded in its module docstring: it thresholded ink at a fixed 128, which reads
blurred one-pixel strokes as paper, and it judged on correlation without effect
size, which flags a clean dataset correlating at 0.79 across offsets that are
all under 0.01.

## Known gaps

- **No real-page evaluation exists.** Until `scripts/evaluate_real.py` is run
  against annotated real scans, nothing here says the model works.
- **Trained on stock augmentation**, not the project's documented settings.
- **The Colab dataset snapshot was not preserved**, so the checkpoint's own
  metrics cannot be reproduced exactly.
- `train_args.device` records nothing useful; treat wall time per epoch as the
  device evidence.
