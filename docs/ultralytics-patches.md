# What importing ultralytics does to the process

`import ultralytics` mutates global state in libraries this pipeline also uses.
Nothing announces it. This is the audit, so the next person does not re-derive
it from a mystery.

Audited against **ultralytics 8.4.123 / 8.4.128**, OpenCV 5.0.0, torch 2.13.0.
Re-run the audit when ultralytics is upgraded — the surface is not stable and
is not part of any public contract.

The patch sites are `ultralytics/utils/__init__.py` (lines ~165 and ~1537) and
`ultralytics/utils/patches.py`.

---

## Applied on every platform

| what | effect | verdict |
|---|---|---|
| `torch.save = torch_save` | Adds retry-on-failure and a dill fallback | **Safe.** The pipeline never calls `torch.save`; only ultralytics writes checkpoints. |
| `cv2.setNumThreads(0)` | Disables OpenCV's internal threading **process-wide** | **Accepted, measured.** See below. |
| `np.set_printoptions(linewidth=320, ...)` | Changes how arrays repr | **Safe.** Nothing parses array reprs. Cosmetic only, but it does mean a debug print looks different depending on whether the detector has been imported. |
| `torch.set_printoptions(...)` | Same for tensors | **Safe**, same reasoning. |
| `os.environ["NUMEXPR_MAX_THREADS"]`, `TF_CPP_MIN_LOG_LEVEL`, `TORCH_CPP_LOG_LEVEL` | Set if unset | **Safe.** No component reads these. |

### `cv2.setNumThreads(0)` — accepted, with a number

Stage 1 is OpenCV-heavy: morphological opening with long kernels over full
pages. Losing OpenCV's threading is a real cost, so it was measured rather than
assumed, in separate processes to avoid warm-up noise:

| | threads | 10× morphological open, 3000×3000 |
|---|---|---|
| without ultralytics | 8 | 0.070 s |
| with ultralytics | 1 | 0.097 s |

**1.39× slower.** Not corrected, for two reasons. Ultralytics sets it because
OpenCV threading contends with the torch DataLoader, so re-enabling it inside a
process that is also running inference trades one problem for another. And a
39% slowdown on a pass that takes well under a second per page is not what
governs pipeline latency.

Worth revisiting only if Stage 1 becomes the bottleneck on large batches, and
then by isolating the two workloads rather than by fighting the setting.

---

## Applied on Windows only

Guarded by `if WINDOWS:` — which is what makes these dangerous. Code that
behaves one way in Linux CI behaves another way on a Windows workstation, and
the difference is invisible in both.

| what | difference | verdict |
|---|---|---|
| `cv2.imread` | Returns `(H, W, 1)` for `IMREAD_GRAYSCALE` where OpenCV returns `(H, W)`. Reads via `np.fromfile`+`imdecode` for non-ASCII paths. For `.tif`/`.tiff` it routes through `imdecodemulti` with `IMREAD_UNCHANGED`, **ignoring the flags it was given**. | **Wrapped.** Use `melodix.ingest.read_grayscale`. |
| `cv2.imwrite` | Writes via `imencode`+`tofile` inside a `try`, so a failure returns `False` instead of raising. | **Wrapped.** Use `melodix.ingest.write_image`. |
| `cv2.imshow` | Re-encodes the window title for non-ASCII | **Safe.** Headless build; the pipeline never opens a window. |

### `cv2.imread` — this one was a live bug

`scripts/evaluate_real.py` imports the detector (hence ultralytics) and then
read pages with `cv2.imread(..., IMREAD_GRAYSCALE)`. On Windows that returned a
`(H, W, 1)` array, so every downstream shape assumption was off by an axis.

It surfaced as an *order-dependent test failure* — twelve tests that passed
alone and failed in the full suite, because another test module had imported
ultralytics first. That is partly luck. Without that ordering it would have
appeared on the first real-page evaluation as inexplicable numbers, which is
precisely the moment when a wrong number is most likely to be believed.

`read_grayscale` normalises the result whether or not the patch is live, and
`tests/test_ingest.py` patches `cv2.imread` to prove immunity rather than
trusting that the real patch stays as it is.

---

## The broader point

ultralytics is not merely an awkward dependency with an OpenCV conflict. It
**mutates global state in a shared process**, on import, conditionally by
platform, without a documented contract.

`scripts/fix_opencv.py` addresses one symptom — two OpenCV distributions
fighting over the same files — and is a **mitigation, not a repair**. It does
nothing about the monkey-patching, which is the more consequential half.

The cleaner long-term answer is full isolation: ultralytics imported only inside
`melodix.vision.detector`, and inference run in a subprocess or a separate
service so its global mutations never reach Stage 1 or Stage 3. `detector.py`
already keeps the *import* lazy and confined, which is most of the way there.
The remaining gap is that a single process which has ever run a detection keeps
the patched globals for its lifetime.

Worth doing if a third patch-related bug appears. Two — the `imread` shape and
the silent `imwrite` — were absorbed by wrappers, which is proportionate for
now.
