# acentos-ROC — Project Analysis

**Date:** 2026-08-02
**Analyzed at commit:** `f5c0083` ("add example png")
**Scope:** structural review of the project layout, packaging, and OCR pipeline.

> **Status note:** Findings #1–#9 are preserved as originally written, describing the
> state at `f5c0083`. Findings #10–#13 were added later while working on the others —
> each is dated and marked.
>
> Current status as of 2026-08-08:
> - **#1 — fixed** (commit `4607546`).
> - **#3 — fixed**, but its stated diagnosis was **wrong**. See the correction below.
> - **#4 — fixed** (commit `82dc59b`): every pipeline stage is written to disk
>   under `--debug`.
> - **#5 — fixed** (2026-08-07): one Tesseract pass per image instead of two, ~45%
>   faster.
> - **#6 — fixed** (2026-08-08): 143 tests. Every filter has behavioural coverage,
>   plus contract tests parameterised over the registry.
> - **#7 — fixed** (2026-08-07): the three empty scaffolded modules are deleted.
> - **#9 — fixed**: `.cursorrules` names the `acentos_ocr` paths.
> - **#10 — fixed**: the sign inversion is corrected and covered by tests. Fixing it
>   exposed **#11**.
> - **#11 — fixed**: the estimator is replaced by a projection-profile search that
>   works on photographs. **Deskew became the default on 2026-08-08**, once a
>   15-image corpus replaced the two samples that had disagreed about it.
> - **#12 — fixed** (2026-08-08): geometric reading-order reconstruction in
>   `src/acentos_ocr/layout/`. Corpus CER 18.9% → 12.0%.
> - **#13 — investigated, closed as capture-limited** (2026-08-08): the two worst
>   images fail on perspective and small text, and neither upscaling nor confidence
>   filtering helps corpus-wide.
> - **#2 — fixed** (2026-08-08): a filter registry plus `--pipeline` on both CLIs.
>   Stacks are now strings, and the corpus harness can A/B several in one run.
> - **#8 — open.**
>
> Quality is now measured across the whole transcribed corpus by
> `scripts/evaluate_corpus.py`, not on single images, against 15 manual
> transcriptions.
>
> | Change | Corpus CER |
> | --- | --- |
> | original binarising stack | ~30% |
> | drop binarisation, `--psm 3` (#3) | 24.0% |
> | deskew on by default (#11) | 18.9% |
> | reading-order reconstruction (#12) | 14.8% |
> | residual ordering cases (#12) | **12.0%** |
>
> Nothing in the two remaining open findings is expected to move that number: #2 is
> a composability change and #8 is metadata. The next real accuracy gain would have
> to come from dewarping (#13) or from better captures.

> ### Correction to finding #3 (2026-08-02)
>
> #3 claimed the default stack was hostile to accents because blur and morphological
> close eroded diacritics. The per-step debug images from #4 show that is **not** the
> mechanism — accents survive thresholding intact. The real problem was that
> **binarising at all destroys quality**: adaptive threshold converted paper texture
> into speckle across the whole background, and Tesseract 5's LSTM engine binarises
> internally anyway, far better than a hand-tuned threshold can.
>
> Measured as character error rate against a manual transcription of a Cayman job
> listing: the old binarising stack scored **30.5%**, grayscale + light blur scores
> **8.9%**. The default pipeline is now grayscale + `GaussianBlur(3)`, and the
> default `--psm` moved from 6 to 3 (16.2% → 8.9% CER on the same image).
>
> The broader lesson: the architecture advice in `suggested_instructions.md`
> prescribed a preprocessing chain appropriate to pre-LSTM Tesseract. Most of it was
> actively harmful against Tesseract 5.

## Current focus (set 2026-08-02, until further notice)

Near-term work targets the **`Raw-Photos-Of-Cayman-Job-Listings`** corpus —
handheld phone photos of newspaper job listings, **all in English**. Spanish and
diacritic handling are deprioritised until this changes.

That focus drove the re-ordering of the work below, and in particular promoted
**#10** and **#11**: a handheld photo of a clipping is skewed by definition, so
deskew is the obvious filter to reach for, and it was both inverted and unusable
on photographs.

Two gaps none of the original nine findings covered:

- **The sample images are unrepresentative.** `fluoxetine.png` is a pill label and
  `document.png` is a Spanish book page. Neither is a newspaper clipping photo.
  **Still open** — a representative Cayman image should be committed as a third
  sample. Less pressing now that the corpus harness exists, but the two committed
  samples are still the only thing a fresh clone can measure against.
- ~~**There is no ground truth.**~~ **Closed 2026-08-08.** 15 manual
  transcriptions now cover the corpus (16 photos; `IMG_1600` is the gap), consumed
  by `scripts/evaluate_corpus.py`. Every quality claim in this document and the
  README is now anchored to character error rate rather than to Tesseract's
  self-reported confidence.

For English-only work the system models are both faster and more accurate than
`tessdata_best` (84.44% vs 82.74% on `fluoxetine.png`, at half the wall clock),
so `--tessdata-dir /usr/share/tesseract-ocr/5/tessdata` is the better setting for
this corpus even though the project default remains `tessdata_best`. Worth
re-testing against the corpus rather than the two samples — that comparison
predates the harness.

---

## Context

A pipe-and-filter OCR preprocessing pipeline in Python 3.13, wrapping OpenCV → Tesseract.
The architecture came from a GPT consult saved in-repo as `suggested_instructions.md`
(gitignored), which prescribed the directory layout, the Strategy pattern, the
`BaseFilter` ABC, and the `.cursorrules` file that enforces it on Cursor.

The name decodes the real goal: **`acentos`** + **ROC** (*Reconocimiento Óptico de
Caracteres*). The other in-repo transcript, `acentosROC.pl` (644 lines), is entirely
about OCR accuracy on accented characters — é, ü, ñ, ç — comparing Tesseract /
PaddleOCR / EasyOCR and prescribing binarization, skew correction, and morphological
filtering. That is the actual problem the project set out to solve.

**Size at time of analysis:** 365 lines of Python, 6 commits, ending on `f5c0083`,
preceded by the telling `2f47a41 CLI is working now, but quality is bad`.

## The architecture is sound

Credit where due — the Strategy pattern is implemented faithfully. `BaseFilter`
(`src/filters/base.py:8`) enforces `apply(np.ndarray) -> np.ndarray`;
`PreprocessingPipeline` (`src/core/processor.py:10`) holds an ordered list and returns
`self` from `add_filter` for chaining; `TesseractWrapper` returns a structured
`OCRResult` dataclass with text, confidence, and a per-word DataFrame rather than a
bare string. Every filter is a clean, small, single-purpose class. Nothing here is
spaghetti.

---

## Findings

### 1. Packaging split-brain — the most consequential structural issue

`pyproject.toml:27-28` declares `packages.find where = ["src"]`, which publishes
`core`, `filters`, `ocr`, `utils` as *top-level* packages. But every module imports
`src.core.*` / `src.filters.*`. Both paths currently resolve:

```
from src.core.processor import PreprocessingPipeline   → OK  (namespace pkg from repo root)
from core.processor      import PreprocessingPipeline   → OK  (installed copy)
```

Those are **two distinct copies of the same classes**, and the installed one is stale —
dated Mar 12, containing only `base, binarization, deskew, grayscale, noise_reduction,
threshold`. It is missing `clahe`, `gaussian_blur`, `morphology`, and `resize` entirely,
and it is a non-editable install so it will not pick up edits. It works today only
because nothing does an `isinstance` check. Also: `core`/`filters`/`ocr`/`utils` are
dangerously generic names to occupy in `site-packages`. There are no `__init__.py`
files anywhere.

**→ FIXED 2026-08-02.** Collapsed into a single `acentos_ocr` package under `src/`,
added the missing `__init__.py` files, made package discovery explicit, and switched
to an editable install so the tree and environment cannot drift again.

### 2. Four filters are built but unreachable — FIXED

`build_default_pipeline` (`main.py:15-21`) hardcodes
Grayscale → GaussianBlur(5) → AdaptiveThreshold(15,5) → Morphology(close,2).
`CLAHEFilter`, `DeskewFilter`, and `ResizeFilter` are written, tested by nobody, and
wired into nothing. There is no CLI flag to compose a stack — so the entire payoff of
the Strategy pattern (A/B testing preprocessing stacks, which
`suggested_instructions.md:52` explicitly calls out as the point) requires editing
source.

**→ FIXED 2026-08-08.** `src/acentos_ocr/filters/registry.py` maps every filter to a
short name and builds it from a `name:key=value` spec, coercing arguments with the
constructor's own type annotations so the filters stay the single source of truth for
what they accept. `main.py --pipeline` composes a stack from the command line;
`scripts/evaluate_corpus.py --pipeline` takes several and scores them all against the
corpus in one run. `build_default_pipeline` is now defined *as* a spec
(`DEFAULT_SPEC`), so the documented default and the running default cannot diverge.

A test asserts every filter module in the package appears in the registry, which is
the precise condition this finding described and would have caught it.

**What the capability immediately showed.** Four results that previously each needed
a source edit and a separate run, `--psm 3`, reading order on:

| Stack | CER | word miss |
| --- | --- | --- |
| **`grayscale deskew blur:ksize=3`** (default) | **12.0%** | 10.7% |
| `grayscale deskew blur:ksize=5` | 12.4% | 11.6% |
| `grayscale deskew` (no blur) | 12.4% | 11.0% |
| `grayscale deskew clahe blur:ksize=3` | 18.1% | 12.1% |
| `grayscale deskew blur:ksize=5 threshold morphology` (original) | 27.1% | 21.3% |

- The **blur is nearly irrelevant** — removing it costs 0.4 points, widening it to 5
  costs the same. The default stack's least consequential element.
- **CLAHE hurts**, +6.1 points, and catastrophically on `IMG_1648` (4.3% → 48.5%).
  Boosting local contrast on newsprint amplifies paper texture along with the ink.
- The **original binarising stack is confirmed catastrophic at corpus scale**, 27.1%
  against 12.0%, failing `IMG_1598` outright at 100% CER. Finding #3 measured that on
  one image; fifteen agree.

### 3. The default stack is actively hostile to accents

This is likely the "quality is bad" cause. A 5×5 Gaussian blur followed by adaptive
threshold and a morphological *close* is tuned for solid body text — and diacritics are
exactly the small, thin, isolated marks that those operations erode or merge into the
letter below. Compounding it, `--lang` defaults to `eng` (`main.py:53`), so Tesseract
is not even loading Spanish character models. For an accents-focused project the
defaults are working against the goal.

### 4. Debug mode cannot tell you where quality dies

`--debug` saves only the *final* image (`main.py:73`). `processor.py:33-34` prints each
filter's name but never saves the intermediate — the per-step `imwrite` that the
blueprint suggested (`suggested_instructions.md:203`) stayed commented out. To diagnose
a 4-stage pipeline you need all 4 images.

### 5. Tesseract runs twice per image — FIXED

`tesseract_wrapper.py:43` calls `image_to_data`, then `:52` calls `image_to_string` on
the same input — double the OCR cost per page. The full text can be reconstructed from
the DataFrame instead.

**→ FIXED 2026-08-07.** `_reconstruct_text` assembles the page from the word-level
dataframe the first pass already returned, grouping by block/paragraph/line. CER is
byte-identical on `IMG_1595` (41.0% at psm 3, 53.7% at psm 4); wall clock for that
two-PSM run fell from 34.5s to 19.0s. Confidence moved by <0.15 points because the
filter now also drops whitespace-only tokens that were previously averaged in.

### 6. Zero tests — FIXED

`tests/` is an empty directory, `pytest` is declared in
`[project.optional-dependencies].dev`, and the blueprint specifically called for
per-filter validation. Every filter is unverified.

**→ FIXED 2026-08-08.** 143 tests. The per-filter validation the finding named is
now in `tests/test_filters.py`: every filter has behavioural coverage, and two
contract tests parameterised over the registry apply to all of them at once, so a
newly added filter is covered the moment it is registered.

The contract tests are the more valuable half. One asserts uint8 output; the other
asserts **no filter mutates its input**, which the pipeline silently depends on —
an in-place edit would corrupt the debug images written for earlier stages and make
results differ depending on whether `--debug` was passed.

Behaviour is pinned rather than implementation: `open` must remove an isolated
speck and `close` must fill a pinhole, not "MORPH_OPEN was passed". Several
undocumented quirks are now nailed down — `GaussianBlurFilter` and
`AdaptiveThresholdFilter` silently round an even kernel up to odd,
`MorphologyFilter` clamps a zero kernel to 1, `ResizeFilter` returns the input
object unchanged rather than a copy when no resize is needed, and `GrayscaleFilter`
is idempotent on single-channel input.

**The tests were verified to fail.** A test suite that has never failed is not
evidence of anything, so three deliberate regressions were introduced and measured:
removing the even-kernel correction failed 3 tests, letting `ResizeFilter` shrink
failed 1, and swapping `open` with `close` failed 2.

That exercise turned up a trap worth recording. Swapping two OpenCV constants
leaves the file byte-length identical, and `git checkout` restored it within the
same second, so CPython reused the **stale `.pyc`** — timestamp invalidation
compares mtime and size, and neither had changed. The restored code kept behaving
like the mutated code until `__pycache__` was cleared. Anything that rewrites source
programmatically should clear the caches before trusting a test result.

### 7. Three empty files are committed — FIXED

`src/core/engine.py`, `src/filters/binarization.py`, `src/filters/noise_reduction.py` —
scaffolded from the blueprint's directory diagram, never implemented. `engine.py`'s
intended "orchestrator" role got absorbed into `main.py`.

**→ FIXED 2026-08-07.** Deleted. Nothing imported them, and the work they implied is
already done elsewhere — `threshold.py` and `morphology.py` cover the binarisation
placeholder, and `core/pipelines.py` now holds the orchestration role `engine.py` was
scaffolded for.

### 8. Licensing metadata is absent

AGPL-3.0 alongside a commercial agreement is a legitimate open-core dual-license, but
`pyproject.toml` has no `license` field (package metadata reads blank) and the README
never mentions licensing at all.

### 9. Minor: `.cursorrules` path inconsistency — FIXED

`.cursorrules` line 5 tells the AI filters inherit from `filters.base.BaseFilter` while
the code uses `src.filters.base` — a small inconsistency that feeds finding #1.

**→ FIXED** as a consequence of #1. `.cursorrules` now names the `acentos_ocr` paths
and explicitly forbids the `src.*` import prefix.

### 10. `DeskewFilter` rotates the wrong way and doubles the skew — FIXED

*Added 2026-08-02, found while smoke-testing the filters after the uv migration.
Not part of the original nine findings.*

`src/acentos_ocr/filters/deskew.py:27` builds the point set with:

```python
points = np.column_stack(np.where(binary > 0))
```

`np.where` returns `(rows, cols)`, so this yields `(y, x)` pairs — but
`cv2.minAreaRect` expects `(x, y)`. Transposing the coordinates mirrors the page
across its diagonal, which **inverts the sign of the measured angle**. The filter
then rotates by that inverted angle, adding the skew it was supposed to remove.

Measured on synthetic pages of horizontal text bars at known skew:

| True skew | Angle from current `(y, x)` | Angle from correct `(x, y)` | Residual after `DeskewFilter` |
| --- | --- | --- | --- |
| −6.0° | −6.00° | +6.00° | **+12.00°** |
| −3.0° | −3.00° | +3.00° | **+6.00°** |
| 0.0° | 0.00° | 0.00° | 0.00° |
| +3.0° | +3.00° | −3.00° | **−6.00°** |
| +6.0° | +6.00° | −6.00° | **−12.00°** |

The output skew is consistently *double* the input, in the opposite direction. Only
a perfectly upright page is unaffected, because there is no sign to invert.

The `if angle < -45: angle += 90` correction on line 33 does not compensate for this;
it handles the separate `minAreaRect` convention where the returned angle wraps near
±90°, and it masks the bug by keeping the result in a plausible-looking range. On
`document.png` the raw value is `-88.947°`, which becomes a believable `1.05°`.

**Why it has not caused visible damage:** `DeskewFilter` is not in
`build_default_pipeline`, so it has never run in the real pipeline — see finding #2.
It would begin corrupting output the moment it is wired in.

**Fix:** swap the coordinate order, e.g.

```python
ys, xs = np.where(binary > 0)
points = np.column_stack((xs, ys)).astype(np.float32)
```

This must land *before* deskew is added to any pipeline, and it needs a regression
test using known-skew synthetic pages (finding #6) — the defect is invisible by
inspection and only shows up under measurement.

**→ FIXED 2026-08-02.** Coordinates swapped, and the angle now folds into (−45, 45]
from either side. Residual skew on synthetic pages went from −2× the input to 0.00°
across ±1° to ±8°. Covered by `tests/test_deskew.py`, the repo's first tests.

A second defect surfaced while testing: `max_angle_deg` **clamped** rather than
declined, so a page detected at 40° was rotated by exactly 15° — baking in a
rotation with no basis. It now returns the image unchanged when the detected angle
exceeds the limit.

Fixing #10 did **not** make deskew usable on this corpus — see **#11**.

### 11. `DeskewFilter`'s estimator does not work on photographs — FIXED

*Added 2026-08-02, found while fixing #10. Not part of the original nine findings.*

Independent of the sign inversion in #10, the angle estimator itself is unsuitable
for the Cayman corpus. `DeskewFilter` runs Otsu over the whole image and takes
`cv2.minAreaRect` of every dark pixel. On a clean scan that isolates the text block.
On a photograph it also marks background, shadows and the clipping's own edges —
about **10% of the frame** on `IMG_1594` — so the minimum-area rectangle spans
essentially the entire image:

| Added rotation | Detected angle | Ink | minAreaRect |
| --- | --- | --- | --- |
| +0° | 0.51° | 10.1% | 1967×2908 of 3024×1984 |
| +2° | **0.00°** | 10.5% | 1983×2935 |
| +5° | **0.00°** | 10.8% | 1983×2992 |
| +8° | **0.00°** | 11.1% | 1983×3023 |

It reports 0.00° no matter how far the page is rotated. Measured end-to-end, adding
deskew to the pipeline changes character error rate not at all at +2°, +5° and +8°
of induced skew, and makes it *worse* on the unmodified image (8.9% → 9.9%), because
a detected 0.51° is just above the threshold and buys a full interpolation pass for
nothing.

Note that confidence *rose* to 88.97% on that worse-CER run — a clean example of why
confidence cannot be used to judge these changes.

**Fix:** the estimator needs replacing for photographs, not repairing. Candidates are
a horizontal projection profile over a range of trial angles (maximise row-variance),
or a Hough transform over text runs joined by a wide morphological close. Restricting
the measurement to a detected page region would help either approach.

Until then `DeskewFilter` must stay out of the default pipeline. It is correct for
clean scans and is covered by tests; it simply does not address the corpus in hand.

**→ FIXED 2026-08-02.** The `minAreaRect` estimator is replaced by a projection
profile search. A morphological gradient isolates character strokes (text is
high-local-contrast; smooth background is not), then trial rotations are scored by
the squared row-to-row differences of the horizontal ink profile — sharp when text
lines are horizontal, flat otherwise. Coarse 1° sweep, then 0.1° refinement, on a
900px-tall working copy: about 0.06s on a 6 MP photo.

It recovers induced skew exactly on the real photos, where the old estimator
reported 0.00° for everything, and it is exact on synthetic scans too, so it
replaces the old approach rather than sitting alongside it:

| Induced | `IMG_1594` estimate | `IMG_1595` estimate |
| --- | --- | --- |
| +0° | −2.00° | −1.70° |
| +2° | −4.00° | −3.70° |
| +5° | −7.00° | −6.70° |
| +8° | −10.00° | −9.70° |

The constant offset is each photo's own intrinsic tilt — `IMG_1594` sits at +2.00°,
`IMG_1595` at +1.70°.

Two guards prevent invented rotations: the search is bounded by `max_angle_deg`, and
the peak-to-median score ratio must exceed `min_peak_ratio` (default 10). Measured
separation is clean — 20–36 for the real samples, 1.2 for a 40° page where no trial
angle helps.

**Deskew is opt-in via `--deskew`, not default.** Measured CER:

| Image | Induced | No deskew | Deskew |
| --- | --- | --- | --- |
| `IMG_1594` | 0° | 8.9% | **7.3%** |
| `IMG_1594` | −4° | 45.3% | **7.3%** |
| `IMG_1595` | 0° | **41.0%** | 48.4% |
| `IMG_1595` | +3° | 57.7% | **46.7%** |

It rescues a genuinely skewed page decisively, but regresses `IMG_1595` at rest, so
enabling it by default is not justified on two samples. Revisit once more ground
truth exists.

**→ REVISITED 2026-08-08, and the conclusion reverses.** With 15 transcriptions
instead of 2, deskew is worth 5.1 points corpus-wide at psm 3 (24.0% → 18.9% CER) at
negligible time cost. The earlier hesitation was undersampling: `IMG_1595` is one of
only two images it hurts.

| Per-image CER delta at psm 3 | |
| --- | --- |
| `IMG_1648` | **−40.9%** |
| `IMG_1658` | −18.8% |
| `IMG_1601` | −9.5% |
| `IMG_1603` | −6.8% |
| `IMG_1594` | −1.6% |
| `IMG_1657` | +4.3% |
| `IMG_1595` | +7.5% |
| 8 others | no change — no confident angle found |

The `min_peak_ratio` guard is doing its job: on 8 of 15 images the filter declines to
act at all, which is why four large wins cost only two modest regressions.

**→ Default flipped 2026-08-08.** `--deskew` is now on; `--no-deskew` disables it.
The two committed sample images (`fluoxetine.png`, `document.png`) are flat scans
where the estimator finds no angle, so their baselines are unchanged at 83.69% and
95.17% — the flip is a no-op on clean scans and only acts on handheld photographs.

### 12. Tesseract's layout analysis shreds full-width text into column blocks — FIXED

Found 2026-08-08 while investigating `IMG_1595`'s 41% CER, which looked at first like
lines being truncated mid-word:

```
line 12:  We are seeking an experienced Senior Project Sys
line 65:  tems Specialist tc
line 76:  y lead the management and optimization
```

Nothing is truncated. One sentence is cut into three vertical pieces and emitted far
apart, because the advert has a two-column middle section between full-width
paragraphs, and psm 3's page segmentation extends the column split through the
full-width text, then reads each strip top-to-bottom.

CER cannot distinguish this from a failure to read the page at all — it is
order-sensitive, so correct text in the wrong sequence scores like missing text. That
is what `word_miss_rate` was added for. `IMG_1595` at psm 3: **40.5% CER against 7.9%
word miss.** The words are there; the order is not.

**How common it is — and a correction.** The first psm-3 sweep flagged 6 of 15 images
this way, and it looked like a corpus-wide segmentation problem. It largely is not.
With deskew enabled, five of the six drop off the list — `IMG_1648` goes from 45.7%
CER / 5.4% word miss to **4.8%**. So most of the shredding is *skew-induced*: a tilted
page makes the layout analyser hallucinate column boundaries. Only `IMG_1595` is
genuinely mis-segmented, and deskew makes it worse.

**→ FIXED 2026-08-08.** `src/acentos_ocr/layout/` re-segments the page geometrically
and re-emits the words in reading order. Corpus CER **18.9% → 14.8%**, at no extra
OCR cost, because every word already comes back with a bounding box.

Investigating it turned up a second failure mode, the mirror of the first. The two
depend on skew:

| | Tesseract's block widths | What it does wrong |
| --- | --- | --- |
| skewed page | narrow (x 340–1500 of 3016) | applies the column split to the **whole page** |
| straight page | full (x 343–2569) | reads the **two-column body as full width** |

So enabling deskew did not remove the problem, it inverted it — which is why
`IMG_1595` got *worse* with deskew (40.5% → 48.0%) while every other affected image
got better.

Three design points, each of which was load-bearing:

1. **Segment the word boxes, not the pixels.** A morphological-gradient mask of a
   photograph also picks up the page border, the shadow gradient, and the sliver of
   the adjacent article at the edge of the frame. The word boxes are clean by
   construction and free.
2. **Cut one gap at a time, widest first.** The first implementation split at every
   qualifying gap simultaneously and produced 45 regions on `IMG_1595`: the page
   broke into horizontal bands *before* anything noticed the gutter, and each band
   then split into left and right on its own. The output interleaved the columns a
   few lines at a time — the same bug, at finer granularity. Cutting only the widest
   gap lets the two-column body survive as one region long enough to be split down
   the gutter as a whole. 36 regions, CER 48.0% → 4.6%.
3. **Decline on single-column pages.** Reordering trades Tesseract's block structure
   for raw geometry, and with no columns to repair that is a pure loss — grouping
   words into lines by vertical position alone merges a heading with body text at the
   same height. Without the guard, `IMG_1594` went 7.3% → 18.8% and six images
   regressed. With it, 13 of 15 are untouched and none regresses.

**→ Residual cases resolved 2026-08-08.** `IMG_1596`, `IMG_1597` and `IMG_1646` were
initially left behind because the first implementation gated on "does this page have
columns", and none of them does. That gate was answering the wrong question.

`IMG_1646` is a centred, single-column advert, and Tesseract had shredded it anyway:

    ...in a fast-   |  duties include dishwashing, food preparation, kitche
    paced restaurant.  |  n cleaning,  |  fits required under  |  closing date: 6 au

So the fix had to work with no column structure at all. Three changes:

1. **Assign words to regions, group lines within them.** The two failure modes are
   mirror images — a line split across blocks must be rejoined, a line merged across
   the gutter must be torn apart — and each needs a different granularity.
2. **Rejoin fragments that overlap vertically and sit side by side.** That is what a
   shredded line looks like once the region is known.
3. **Stop re-deriving lines from word positions.** This was the real cause of the
   `IMG_1594` regression, and it was misattributed at first to "reordering
   single-column pages is inherently unsafe". It is not. Tesseract tracks a baseline
   per line and stays correct on a photographed page whose lines sag; clustering
   words by vertical centre drops the trailing words of a sloping line onto the line
   below. Preserving Tesseract's line grouping makes single-column reordering safe,
   so the `has_columns` guard was removed as unnecessary.

Corpus CER **14.8% → 12.0%**. Nine images improve — `IMG_1646` 23.2% → 7.3%,
`IMG_1596` 17.9% → 6.0%, `IMG_1597` 15.4% → 12.5%, `IMG_1604` 34.5% → 30.3%,
`IMG_1599` 34.2% → 32.2% — and the harness now reports no sample dominated by
ordering damage.

One regression accepted: `IMG_1658` 4.7% → 9.3%. Its final short line becomes a
region of its own and moves ahead of the line above it. Worth 4.6 points on one
image against 2.8 points across the corpus, and fixing it would mean another
threshold tuned on a single sample.

---

## Suggested order of work

Re-ordered 2026-08-02 for the English Cayman-listings focus described above.

1. ~~Packaging split-brain (#1)~~ — done 2026-08-02, commit `4607546`.
2. ~~OCR quality, language half (#3)~~ — done 2026-08-02, commit `09a087d`.
   `spa+eng` default, Tesseract 5, project-local `tessdata_best`.
   `document.png` 49.37% → 64.47%.
3. ~~Per-step debug output (#4)~~ — done 2026-08-02, commit `82dc59b`.
4. ~~OCR quality, filter half (#3)~~ — done 2026-08-02. Diagnosis corrected; the
   binarising stack was the problem. CER on the Cayman sample 30.5% → 8.9%.
5. ~~Deskew sign inversion (#10)~~ — done 2026-08-02. Fixed and covered by the
   repo's first tests; residual skew on synthetic pages went from −2× the input
   to 0.00°. Fixing it exposed **#11**, below.
6. ~~A skew estimator that works on photographs (#11)~~ — done 2026-08-02.
   Projection-profile search, exact on both real photos and synthetic scans.
7. ~~More ground truth~~ — done 2026-08-08. 15 transcriptions now exist against 16
   photos (`IMG_1600` is the gap). This settled the deskew question; see #11.
8. ~~Single Tesseract pass (#5)~~ — done 2026-08-07, ~45% faster.
9. ~~Corpus evaluation harness~~ — done 2026-08-07. `scripts/evaluate_corpus.py`,
   `src/acentos_ocr/eval/`. Sweeps psm × deskew across every transcribed image and
   reports CER beside an order-insensitive word miss rate. Baseline in
   `results/baseline-2026-08-08.json`.
10. ~~Empty committed files (#7)~~ — done 2026-08-07.

11. ~~Flip the `--deskew` default~~ — done 2026-08-08. Also fixed `main.py --help`,
    which had always crashed: an unescaped `%` in the `--psm` help text parsed as a
    `%c` conversion when argparse formatted it.

### 13. The floor cases are capture-limited, not pipeline-limited — INVESTIGATED

`IMG_1599` (32.2% CER, 28.8% word miss) and `IMG_1604` (30.3% / 27.0%) are the two
worst images in the corpus by a wide margin and do not respond to any configuration.
Their word miss rates track their CER, so unlike #12 this is recognition failing
rather than ordering.

**What is actually wrong with each**

`IMG_1599` loses inter-word spacing. Tesseract returns `experienceinfast-pacedfine`
as one token, and splits others across the break (`experi` / `perience,`, `tendi` /
`ending`). The advert is set in condensed type with tight tracking, and the photo is
slightly soft; between them the word gaps stop being resolvable. The remaining errors
are ordinary character confusions — `kirchsn` for "kitchen", `nimum` for "minimum",
`y5d4gy` for "Y5D4G7".

`IMG_1604` is a narrow column shot at an angle. Its median word height is **31px**,
the smallest in the corpus, and the worst-distorted corner is where the letterhead
sits: `rawlinso`, `unter`, `caymar`, `|slands`. It also picks up the newspaper's own
page furniture (`weekly, 17 23 july 2026`) and single-character noise from the
adjacent article down the left edge, giving it a 40% insertion rate.

Both are dominated by **perspective and curvature rather than rotation** — in
`IMG_1599` the top lines are level while the bottom ones fan upward. `DeskewFilter`
applies a single global rotation and is structurally unable to correct that.

**Two things that did not work**, both measured across the whole corpus so they are
not worth retrying:

| Attempt | Corpus CER |
| --- | --- |
| baseline | **12.0%** |
| upscale ×1.5 | 12.7% |
| upscale ×2 | 12.6% |
| drop words below conf 30 | 12.2% |
| drop words below conf 60 | 13.3% |

Upscaling looked promising on a single sample — `IMG_1599` alone goes 32.2% → 22.9%
at ×1.5 — but the corpus says otherwise, and the per-image pattern is scatter rather
than signal: `IMG_1646` improves to 0.8% at ×2 while `IMG_1595`, `IMG_1602`,
`IMG_1603` and `IMG_1649` all get worse. That is Tesseract's layout analysis jittering
under a different input size, not a resolution effect. Confidence filtering is
monotonically worse: it trims some of `IMG_1604`'s noise (30.3% → 27.4% at 30) and
badly damages `IMG_1599` (32.2% → 40.1% at 60), whose real words carry low confidence
precisely because it is soft.

The `--scale` sweep added to `scripts/evaluate_corpus.py` during this work is kept —
the dimension is worth having, and now has a recorded answer.

**Conclusion:** these two need better photographs, not better code. Shot square-on
and closer, both would likely fall in line with the rest of the corpus. Dewarping is
the only code-side fix that would help, and it is a much larger undertaking than
anything attempted so far — worth revisiting only if a meaningful share of future
captures turn out this distorted.

**One ground-truth defect found.** `text-of-IMG_1599.md` opens with
`# SILVERERSIDE Restaurant`; the logo in the photograph reads **SILVERSIDE**. A small
share of that image's measured error is therefore unearnable. Worth a scan of the
other transcriptions for similar slips before trusting any single image's number.

---

**Next:**

1. **Licensing metadata (#8)** — `pyproject.toml` still has no `license` field and
   the README says nothing about licensing.
2. **Corpus housekeeping** — `IMG_1600.JPEG` has no transcription; the
   `text-of-IMG_1599.md` heading reads `SILVERERSIDE` where the logo reads
   `SILVERSIDE`, and the other transcriptions are worth a scan for similar slips;
   and the two committed sample images are still a pill label and a Spanish book
   page rather than anything resembling the target corpus.
3. **Per-image configuration**, if it still looks worthwhile — the old 15.8% oracle
   predates deskew and reading order, so the headroom over a single fixed stack
   needs remeasuring before anyone invests in adaptive selection.
