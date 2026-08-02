# acentos-ROC — Project Analysis

**Date:** 2026-08-02
**Analyzed at commit:** `f5c0083` ("add example png")
**Scope:** structural review of the project layout, packaging, and OCR pipeline.

> **Status note:** Finding #1 was fixed on 2026-08-02 in the commit that immediately
> precedes this document. Findings #2 through #9 are still open. The findings below
> are preserved as originally written, describing the state at `f5c0083`.

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

---

## Suggested order of work

1. ~~Packaging split-brain (#1)~~ — done 2026-08-02.
2. **OCR quality (#3)** — Spanish language data, a diacritic-preserving filter stack.
   Highest impact on the actual goal.
3. **Per-step debug output (#4)** — needed to diagnose #3 empirically rather than by guess.
4. **Composable pipelines via CLI (#2)** — unlocks A/B testing the stacks from #3.
5. **Tests (#6)**, then the cleanup items (#5, #7, #8, #9).
