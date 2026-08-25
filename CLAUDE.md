# acentos-ROC

A modular OCR pipeline: OpenCV preprocessing filters (Strategy pattern) → Tesseract via
pytesseract → optional reading-order rebuild. Accuracy is measured, not asserted.

Long-form context lives in two documents — read them rather than re-deriving:

- `README.md` — what the pipeline does, current measured accuracy, how to run it.
- `PROJECT-ANALYSIS-2026-08-02.md` — decisions, negative results, and the live
  next-steps checklist. Update the checklist in the same commit as the work it
  describes; it has gone stale before.

## Commands

```bash
uv run pytest -q                       # 180 tests as of 2026-08-25
uv run python main.py INPUT [INPUT...] # images or PDFs; --save-text DIR writes .txt
uv run python scripts/evaluate_corpus.py   # the real accuracy measurement
uv run python scripts/evaluate_cer.py      # one image only; see caveat below
```

Both scripts have substantial `--help` worth reading before sweeping configurations.

## The corpus

**The corpus lives outside this repo and is not under version control. There is no
undo.** Back up transcriptions before editing them.

- Located via `$ACENTOS_CORPUS`, which is **not** exported by default — set it per
  shell. Hard-coded paths in older notes are stale; find the corpus by filename
  (`IMG_*.JPEG`, `IMG_*.png`, `text-of-*.md`) instead of trusting a remembered path.
- Pairing convention: image `IMG_1594.JPEG` ↔ transcription `text-of-IMG_1594.md`,
  matched on **stem** by `rglob` at any depth under the root (`eval/corpus.py`).
- **Stem collisions double-count.** Two images sharing a stem under the root
  (`IMG_1722.png` and `IMG_1722.JPEG`) both pair with the one truth file and score it
  twice. `.corpus-ignore` cannot fix this — it filters by stem and would drop both.
  Delete or move one. Likewise, never put a *copy* of a `text-of-*.md` under the root:
  the rglob finds it and it can shadow the live truth.
- `.corpus-ignore` lists stems that are deliberately untranscribed, one per line.
  Use it for reference shots and unusable frames — don't delete a photo to quiet a
  warning, and don't leave a known non-sample warning forever.

### Ground truth is off limits

`text-of-*.md` files are hand-made ground truth. **Nothing in this application may
write that namespace.** `utils/text_io.py` enforces it by raising on any `text-of-*`
filename, and `tests/test_text_io.py` pins that behaviour — OCR output uses `.txt`,
which `discover()` cannot see. Keep it that way; the guard is the reason a benchmark
run can't silently overwrite the thing it is being scored against.

### Measuring

- **Measure with `evaluate_corpus.py` across the whole corpus, never a single image.**
  Single samples disagree: the two transcriptions that existed early on pointed in
  opposite directions on whether `--deskew` helps.
- **Tesseract's confidence score is not accuracy.** Score against ground truth.
- CER is order-sensitive, word miss rate is not. A high CER with a low word miss rate
  means the words were read correctly and emitted in the wrong order — a segmentation
  failure needing a different fix from a blurry photo.
- A sample is **one advertisement, cropped** — not a full newspaper spread. Slicing
  neighbouring ads into the frame produces junk lines that measure the photograph
  rather than the pipeline.

## PDF input

`main.py` accepts `.pdf` alongside images (`utils/pdf_io.py`). Two rules drive it:

- **A text layer wins.** A digital-born PDF already holds exact characters; OCR could
  only introduce errors, so the embedded text is used as-is. `--force-ocr` overrides
  this for files where another tool left a bad text layer behind.
- **Render at native resolution, not at a DPI.** Phone scan-to-PDF sizes the page box
  to the pixels at 72 DPI, so an ordinary sheet can declare itself 28 inches wide —
  rendering *that* at 300 DPI turns a 2053-pixel scan into an 8559-pixel one made
  almost entirely of interpolation. The default follows the page's own largest image;
  `--pdf-dpi` overrides, and a 6000-pixel long-edge clamp catches the pathological
  cases.

A PDF expands to one work item per page, named `<stem>-p01`, `-p02`, … so pages flow
through `resolve_text_paths` with distinct stems like any other input.

Test fixtures build PDFs in memory (`tests/test_pdf_io.py`). **Do not commit sample
PDFs** — the documents this feature exists for are scans of personal paperwork.

## Adding a preprocessing filter

1. **Inherit** from `BaseFilter` (`src/acentos_ocr/filters/base.py`).
2. **Implement** `apply(self, image: np.ndarray) -> np.ndarray`.
3. **Register it** in `FILTERS` in `filters/registry.py` under a short CLI name.
   Unregistered filters are unreachable from `--pipeline`, which defeats the point of
   the Strategy pattern. Constructor type annotations are the single source of truth
   for parameter coercion — declare them and don't restate defaults in the registry.
4. **Place** it in `src/acentos_ocr/filters/` as its own file. Import siblings
   relatively (`from .base import BaseFilter`); never use `src.*` as an import prefix.
5. **Document** it with a docstring naming the OpenCV technique (Gaussian blur, Otsu
   thresholding, deskew).
6. Use `cv2` and `numpy`; assume BGR or grayscale `np.ndarray` input; type-hint
   everything.

```python
import cv2
import numpy as np
from .base import BaseFilter


class NewFilterName(BaseFilter):
    """Short description naming the OpenCV technique used."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        processed_image = image
        return processed_image
```

A new filter is not "done" until it has tests and a corpus measurement showing whether
it helps. Defaults are on-by-default only when the corpus says so.

## Environment

All tooling runs in **WSL2 Ubuntu** — never PowerShell or `cmd.exe`. Note that
`/mnt/c` is a 9p mount and slow for large files; the photo corpus lives there, so
image-heavy runs are I/O-bound.

Python ≥3.13, managed with `uv`. Tesseract 5 with project-local `tessdata_best`
(`scripts/fetch_tessdata.sh`). Licence is AGPL-3.0-only; keep the SPDX header on new
source files.
