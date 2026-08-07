# acentos-ROC — Project Analysis

**Date:** 2026-08-02
**Analyzed at commit:** `f5c0083` ("add example png")
**Scope:** structural review of the project layout, packaging, and OCR pipeline.

> **Status note:** Findings #1–#9 are preserved as originally written, describing the
> state at `f5c0083`. Findings #10 and #11 were added later, on the same date, and
> describe defects found while working on the others — both are dated and marked.
>
> Current status as of 2026-08-02:
> - **#1 — fixed** (commit `4607546`).
> - **#3 — fixed**, but its stated diagnosis was **wrong**. See the correction below.
> - **#4 — fixed** (commit `82dc59b`): every pipeline stage is written to disk
>   under `--debug`.
> - **#10 — fixed**: the sign inversion is corrected and covered by tests. Fixing it
>   exposed **#11**.
> - **#11 — fixed**: the estimator is replaced by a projection-profile search that
>   works on photographs. Deskew is opt-in via `--deskew`, not default.
> - **#2, #5–#9 — open.**

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

This re-orders the work below. Finding #3's remaining half was framed around
preserving accent marks; for English newsprint photos the same filter stack is
still wrong, but for different reasons — perspective, uneven lighting, and
newsprint texture rather than eroded diacritics. It also promotes **#10**: a
handheld photo of a clipping is skewed by definition, deskew is the obvious
filter to reach for, and it currently doubles the skew instead of removing it.

Two gaps none of the ten findings covers:

- **The sample images are unrepresentative.** `fluoxetine.png` is a pill label and
  `document.png` is a Spanish book page. Neither is a newspaper clipping photo. A
  representative Cayman image should be committed as a third sample.
- **There is no ground truth.** Every measurement so far is Tesseract's
  self-reported confidence, which is not accuracy — see the note in the README
  baselines section. Filter tuning needs ground-truth transcriptions and a
  character error rate to be meaningful.

For English-only work the system models are both faster and more accurate than
`tessdata_best` (84.44% vs 82.74% on `fluoxetine.png`, at half the wall clock),
so `--tessdata-dir /usr/share/tesseract-ocr/5/tessdata` is the better setting for
this corpus even though the project default remains `tessdata_best`.

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

### 2. Four filters are built but unreachable

`build_default_pipeline` (`main.py:15-21`) hardcodes
Grayscale → GaussianBlur(5) → AdaptiveThreshold(15,5) → Morphology(close,2).
`CLAHEFilter`, `DeskewFilter`, and `ResizeFilter` are written, tested by nobody, and
wired into nothing. There is no CLI flag to compose a stack — so the entire payoff of
the Strategy pattern (A/B testing preprocessing stacks, which
`suggested_instructions.md:52` explicitly calls out as the point) requires editing
source.

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

### 6. Zero tests

`tests/` is an empty directory, `pytest` is declared in
`[project.optional-dependencies].dev`, and the blueprint specifically called for
per-filter validation. Every filter is unverified.

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

### 9. Minor: `.cursorrules` path inconsistency

`.cursorrules` line 5 tells the AI filters inherit from `filters.base.BaseFilter` while
the code uses `src.filters.base` — a small inconsistency that feeds finding #1.

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

**→ REVISITED 2026-08-07, and the conclusion reverses.** With 15 transcriptions
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
act at all, which is why four large wins cost only two modest regressions. Flipping
the default is pending a decision.

### 12. Tesseract's layout analysis shreds full-width text into column blocks

Found 2026-08-07 while investigating `IMG_1595`'s 41% CER, which looked at first like
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

Remaining options for that residual case: run psm 4 on a detected column region, split
the page into column regions before OCR, or reorder Tesseract's blocks using the
bounding boxes already present in the dataframe.

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
7. ~~More ground truth~~ — done 2026-08-07. 15 transcriptions now exist against 16
   photos (`IMG_1600` is the gap). This settled the deskew question; see #11.
8. ~~Single Tesseract pass (#5)~~ — done 2026-08-07, ~45% faster.
9. ~~Corpus evaluation harness~~ — done 2026-08-07. `scripts/evaluate_corpus.py`,
   `src/acentos_ocr/eval/`. Sweeps psm × deskew across every transcribed image and
   reports CER beside an order-insensitive word miss rate. Baseline in
   `results/baseline-2026-08-07.json`.
10. ~~Empty committed files (#7)~~ — done 2026-08-07.

**Next:**

1. **Decide the `--deskew` default** — the corpus says yes (24.0% → 18.9%); the
   change itself is one line plus a docs pass.
2. **Reading order for `IMG_1595` (#12)** — the one genuinely mis-segmented sample.
3. **The floor cases** — `IMG_1599` (~34% CER in all six configurations) and
   `IMG_1604` (~30%) do not move for any setting, and their word miss rates are high
   too, so these are real recognition failures. Look at the photographs.
4. **Composable pipelines via CLI (#2)** — now more valuable, because the harness
   gives it something to measure against. Per-image oracle CER is 15.8% against
   18.9% for the best fixed configuration, so roughly 3 points sit in per-image
   adaptation.
5. **Remaining cleanup** — #8 (licensing), #9 (`.cursorrules`), and per-filter tests
   beyond deskew and the metrics (#6).
