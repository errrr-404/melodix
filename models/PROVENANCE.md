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

**No synthetic run can settle whether mosaic helped or hurt.** mAP50 sits at
0.9888 on synthetic val — there is no headroom in which a difference could show
itself. That is a stronger statement than the confound below and is the real
reason the question stays open: it needs real pages, not a better-controlled
synthetic run.

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

## Predictions recorded before the corrected run

Written down before `stage2_corrected_aug` starts, so neither can be
reinterpreted afterwards.

### What the run CAN test

**1. Label alignment.** If the Colab snapshot carried misaligned boxes and the
current dataset does not, the corrected run should reach **mAP50-95 above
0.9118** with **mAP50 roughly flat near 0.9888**. At or below 0.9118, the label
hypothesis is wrong and the gap has another cause.

**2. `scale` starving the small classes.** The stock `scale=0.5` took
`augmentation_dot` and `staccato` from 5.49 px at model input down to 2.75 px at
the minimum draw. The corrected `scale=0.20` floors them at 4.39 px. If that was
the constraint, **`augmentation_dot` (0.503) and `staccato` (0.687) should rise
materially while the saturated classes stay flat** — `cross_notehead` 0.995,
`percussion_clef` 0.995, `round_notehead` 0.981 have nowhere to go. If every
class moves together, the cause is elsewhere and `scale` was not the lever.

A caveat recorded in advance, because it bounds the expected size of the win.
The localisation budget — how far a predicted box may sit from truth and still
clear an IoU threshold, `d(1-t)/(1+t)` — is **sub-pixel for these classes even
at scale=0**:

| dot size at model input | IoU 0.50 budget | IoU 0.75 budget |
|---|---|---|
| 2.75 px (stock scale=0.5) | 0.92 px | 0.39 px |
| 4.39 px (corrected scale=0.20) | 1.46 px | 0.63 px |
| 5.49 px (no scale augmentation) | 1.83 px | 0.78 px |

YOLOv8's finest head is P3 at stride 8. So these classes are near the floor
*intrinsically*, and mAP50-95 — which averages IoU 0.50 through 0.95 — is
mechanically capped for them whatever `scale` does. Expect a real improvement,
not a fix. The durable answer is a larger render, or letting Stage 3
disambiguate dots geometrically rather than trusting the class.

### What the run CANNOT test

**Whether removing mirror augmentation helped.** Mirrored pages were never in
the synthetic val set either, so synthetic metrics are blind to it by
construction. `fliplr=0.0` is justified on domain grounds — mirrored notation
cannot occur — and stands regardless of what the number does. Answering it needs
real pages.

## mAP50-95 under-reports the small classes. Do not read it as model quality.

`augmentation_dot` at 0.503 and `staccato` at 0.687 are the two weakest classes
in the baseline. A large part of that number is the metric, not the model.

mAP50-95 averages ten IoU thresholds from 0.50 to 0.95. For two boxes of side
`d` offset by `delta`, IoU clears a threshold `t` only while
`delta < d(1-t)/(1+t)`. These glyphs reach the model at 5.49 px:

| IoU threshold | budget for a 5.49 px glyph |
|---|---|
| 0.50 | 1.83 px |
| 0.75 | 0.78 px |
| 0.90 | **0.29 px** |

YOLOv8's finest head is P3 at stride 8. Roughly half the thresholds mAP50-95
averages are therefore **unavailable at this glyph size regardless of detector
quality**, and a perfect detector would still score well under 1.0. The number
is partly measuring the metric's own ceiling.

**The metric is also misaligned with the product.** Stage 3 does two things with
a detection: takes its centroid into `StaffGrid.snap`, and associates it with a
nearby notehead. At 14 px staff spacing, snap's 0.4-position tolerance is
**+/-2.8 px** vertically; horizontal association across noteheads ~40 px apart is
looser still. So the application needs about 2.8 px where IoU 0.75 demands 0.63 px
— a target four times stricter than anything downstream cares about, applied to
the classes least able to meet it.

`scripts/evaluate_real.py` now reports both: AP50 for the small classes on its
own, and per-class centroid error in pixels against the snap budget, derived
from the geometry module rather than hard-coded. Read those for whether the
model is good enough. Read mAP for whether it is improving.

## Retrain status: pending, blocked on GPU access

`stage2_corrected_aug` has **not been run.** This machine's torch is a CPU-only
build (`2.13.0+cpu`, `cuda.is_available() == False`), and the measured local
cost is **7,278 s/epoch** — 60.6 hours for 30 epochs, against 104 minutes on the
T4 that produced the baseline. Run it from `notebooks/train_colab.ipynb`.

The training path itself is verified, so the Colab run is de-risked rather than
merely hoped for. A one-epoch smoke run at `imgsz=320` on 8 pages completed, and
every setting in `melodix_resolved_config.json` matches what ultralytics wrote
into the checkpoint's `train_args`:

| setting | resolved | in checkpoint |
|---|---|---|
| `fliplr` / `flipud` | 0.0 / 0.0 | 0.0 / 0.0 |
| `mosaic` / `close_mosaic` | 0.0 / 0 | 0.0 / 0 |
| `degrees` | 1.0 | 1.0 |
| `scale` / `translate` | 0.20 / 0.05 | 0.20 / 0.05 |
| `shear` / `perspective` | 0.0 / 0.0 | 0.0 / 0.0 |
| `hsv_h` / `hsv_s` / `hsv_v` | 0.0 / 0.0 / 0.25 | 0.0 / 0.0 / 0.25 |

That is the exact drift that produced the baseline's stock augmentation, now
closed and checked end to end rather than assumed.

## First look at a degraded page: Stage 1 is the fragile stage, not Stage 2

No real page exists in the repo yet, so this was measured on the closest
available proxy — a synthetic page put through `scripts/degrade.py` and
rendered to PDF, read back through `melodix.ingest`. It is not a real scan and
does not substitute for one. It is the first time anything has been looked at
rather than scored.

**Stage 2 transferred better than expected.** 504 detections, median confidence
0.55, with percussion clefs at 0.95, time signatures, and round and cross
noteheads at 0.84-0.93 sitting correctly on their staff positions.

**Stage 1 found zero staves on the same page.** Not degraded — zero. The staff
lines are plainly visible in the image, and the detector read symbols off them
successfully, but `detect_staff_grids` returned an empty list. Everything
downstream of Stage 1 depends on that grid, so on this page the pipeline has no
coordinate system at all.

Isolated by running each degradation effect alone against staff detection:

| effect alone | staves found (5 expected) |
|---|---|
| ink, rotate, perspective, blur, texture, brightness, gamma, jpeg | 5 |
| **speckle** | **1** |

Cumulatively in effect order, detection holds at 5 through blur, drops to 1 once
texture is added, and reaches **0** once brightness is applied.

The mechanism is that `isolate_horizontal_runs` opens with a kernel 35% of the
page width. A morphological opening needs an unbroken run, so a single
salt-noise pixel inside a staff line breaks it and the opening erases the whole
line.

Measured across twelve noise seeds, it degrades stochastically rather than
cleanly — it depends on whether a speck happens to land on a line:

| salt-and-pepper density | full 5/5 | partial | none |
|---|---|---|---|
| 0.0000 | 12/12 | 0 | 0 |
| 0.0010 | 11/12 | 1 | 0 |
| 0.0025 (degrade.py default) | 8/12 | **4** | 0 |
| 0.0050 | 4/12 | 8 | 0 |
| 0.0100 | 0/12 | 12 | 0 |

Losing one staff on an ensemble page means losing one player's part, silently.

A morphological closing before the opening recovers it in a bench probe — a 3x1
or 5x1 close restores 5/5 staves on a page where the opening alone found 5 but
the noisy variant found fewer. **Not implemented**: it is a change to Stage 1's
detection path, it needs its own tests and a check that it does not fuse
adjacent lines at small engraving sizes, and this phase was scoped to
diagnostics. Recorded as the most promising next fix.

This inverts the project's working assumption. Four phases of effort have gone
into Stage 2's augmentation. On the first page anyone looked at, Stage 2 worked
and Stage 1 did not.

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
