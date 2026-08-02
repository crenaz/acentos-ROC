# acentos-ROC — Project Analysis

**Date:** 2026-08-02
**Analyzed at commit:** `f5c0083` ("add example png")
**Scope:** structural review of the project layout, packaging, and OCR pipeline.

> **Status note:** Findings #1–#9 are preserved as originally written, describing the
> state at `f5c0083`. Finding #10 was added later, on the same date, and describes a
> defect found while working on the others — it is dated and marked as such.
>
> Current status as of 2026-08-02:
> - **#1 — fixed** (commit `4607546`).
> - **#3 — fixed**, but its stated diagnosis was **wrong**. See the correction below.
> - **#4 — fixed** (commit `82dc59b`): every pipeline stage is written to disk
>   under `--debug`.
> - **#2, #5–#10 — open.**

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

### 5. Tesseract runs twice per image

`tesseract_wrapper.py:43` calls `image_to_data`, then `:52` calls `image_to_string` on
the same input — double the OCR cost per page. The full text can be reconstructed from
the DataFrame instead.

### 6. Zero tests

`tests/` is an empty directory, `pytest` is declared in
`[project.optional-dependencies].dev`, and the blueprint specifically called for
per-filter validation. Every filter is unverified.

### 7. Three empty files are committed

`src/core/engine.py`, `src/filters/binarization.py`, `src/filters/noise_reduction.py` —
scaffolded from the blueprint's directory diagram, never implemented. `engine.py`'s
intended "orchestrator" role got absorbed into `main.py`.

### 8. Licensing metadata is absent

AGPL-3.0 alongside a commercial agreement is a legitimate open-core dual-license, but
`pyproject.toml` has no `license` field (package metadata reads blank) and the README
never mentions licensing at all.

### 9. Minor: `.cursorrules` path inconsistency

`.cursorrules` line 5 tells the AI filters inherit from `filters.base.BaseFilter` while
the code uses `src.filters.base` — a small inconsistency that feeds finding #1.

### 10. `DeskewFilter` rotates the wrong way and doubles the skew

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
5. **Deskew sign inversion (#10)** — handheld clipping photos are skewed by
   definition, so deskew moves from unused to essential. It must be corrected before
   being wired in, and it needs a measurement-based test since the defect is
   invisible by inspection.
6. **More ground truth** — one transcription (`IMG_1594`) currently anchors every
   quality claim. More would make CER measurements trustworthy across the corpus.
   Convention: `<base>/text-of-<image stem>.md`, consumed by
   `scripts/evaluate_cer.py`.
7. **Composable pipelines via CLI (#2)** — unlocks A/B testing stacks against the
   real corpus without editing source.
8. **Tests (#6)**, then the cleanup items (#5, #7, #8, #9).
