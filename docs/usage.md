# Usage

Running the pipeline: single files, batches, PDFs, debugging, and composing a
preprocessing stack.

See also: [installation](installation.md) · [accuracy](accuracy.md) · [development](development.md)

## Running OCR

Input is an image or a PDF; several may be given at once, mixed freely. Assuming
`document.png` lives in the project root:

```bash
uv run python main.py document.png --debug --out-debug-dir debug --psm 6
```

`uv run` executes inside the locked environment without needing to activate it.

## Transcribing a set of pictures

Several images can be given at once, and `--save-text` writes each one's text to
its own file — `DIR/<stem>.txt`, UTF-8, directory created if needed:

```bash
uv run python main.py ~/clippings/July18/*.JPEG --save-text transcripts/
```

```
=== [1/12] IMG_1594.JPEG ===
=== OCR TEXT ===
...
=== SUMMARY ===
Average confidence: 82.41%
Config used: --oem 3 --psm 3 --tessdata-dir /…/tessdata
Wrote transcripts/IMG_1594.txt
...
=== BATCH ===
12 of 12 pages transcribed, mean confidence 80.31% over 12 OCR'd
```

Three things worth knowing about the batch behaviour:

- **`<stem>.txt` is not the corpus convention, deliberately.** A hand-made
  transcription of `IMG_1594.JPEG` is `text-of-IMG_1594.md`, and those are ground
  truth — the yardstick every CER figure is measured against. OCR output
  under that name would be the pipeline grading its own homework, so the writer
  refuses the `text-of-*` namespace outright. `.txt` is also invisible to
  `corpus.discover`, which globs only `text-of-*.md` and image suffixes, so
  pointing `--save-text` at the corpus tree cannot disturb it either.
- **Colliding output names are refused before any OCR runs.** `a/IMG_1.JPEG` and
  `b/IMG_1.png` both want `IMG_1.txt`; that is caught in the second it takes to
  compare names, rather than after eleven pages of work have been overwritten.
- **One bad file does not sink the batch.** An unreadable image or a Tesseract
  failure is reported, the remaining pictures still run, and the command exits
  non-zero so a partial run is not mistaken for a clean one.

A single image prints exactly what it always did — the per-picture headers and
the batch summary appear only when there is more than one.

## Reading PDFs

PDFs can be given anywhere an image can, mixed freely with them. Each page becomes
its own unit of work, written as `<stem>-p01.txt`, `-p02.txt`, and so on:

```bash
uv run python main.py scan.pdf ~/clippings/*.JPEG --save-text transcripts/
```

No extra system package is needed — `uv sync` pulls `pypdfium2`, which ships its own
PDF engine as a wheel.

Two behaviours are worth understanding, because both are decisions rather than
defaults inherited from a library:

- **A text layer is used as-is, and OCR is the fallback.** A digital-born PDF —
  exported from a word processor, a browser, a design tool — already contains the
  exact characters. Rasterising one so Tesseract can guess at the pixels can only
  introduce errors where there were none, so embedded text is extracted directly and
  the pipeline is never invoked. The check runs per page, since a scan appended to a
  digital document is a real thing, and a page needs 32 characters to count: a stray
  annotation, or a bad text layer left behind by some other OCR tool, is worse than
  nothing. `--force-ocr` ignores the text layer entirely, which is the escape hatch
  for exactly that case.

- **Pages render at the scan's own resolution, not at a chosen DPI.** This matters
  more than it sounds. Phone "scan to PDF" wraps a photo in a page box sized to the
  pixel count at 72 DPI, so an ordinary sheet of paper can declare itself 28 inches
  wide. Rendering one of those at a sensible-looking 300 DPI turned a 2053-pixel scan
  into an **8559-pixel** one — sixteen times the pixels, none of them carrying any
  more detail, and a correspondingly slower OCR pass. The default instead follows the
  native resolution of the page's own largest image, reproducing the scan exactly.
  `--pdf-dpi` overrides it when you do want a fixed resolution, and a 6000-pixel
  long-edge clamp catches pathological page boxes either way.

For a scanned page, the result is identical to extracting the image yourself with
`pdfimages -all` and running the pipeline over that — the same words, differing only
in blank-line placement. The point of the built-in path is that it picks the right
resolution for you, and knows when not to OCR at all.

## Diagnosing a bad result

`--debug` writes one image per pipeline stage, numbered in execution order:

```
Pipeline steps (images written to debug/):
  [00] source                       2016x1512x3      uint8
  [01] applied Grayscale            2016x1512        uint8
  [02] applied GaussianBlur         2016x1512        uint8
  [03] applied AdaptiveThreshold    2016x1512        uint8
  [04] applied Morphology           2016x1512        uint8
```

```
debug/document_00_source.png
debug/document_01_Grayscale.png
debug/document_02_GaussianBlur.png
debug/document_03_AdaptiveThreshold.png
debug/document_04_Morphology.png
```

Comparing consecutive images shows *which* stage degraded the page, which a single
final image cannot. The printed shape and dtype make channel drops and type changes
visible — a common cause of a filter silently doing nothing useful.

Note that these are full-resolution lossless PNGs; a 12 MP phone photo produces
several megabytes per stage.

## Flag reference

- `--save-text DIR`: write each picture's OCR text to `DIR/<stem>.txt` — see
  [Transcribing a set of pictures](#transcribing-a-set-of-pictures).
- `--debug`: prints each pipeline step with its shape and dtype, reports which tessdata directory was used, and writes **every intermediate image** to the `debug/` directory.
- `--out-debug-dir`: where `--debug` writes its intermediate images (default `debug/`).
- `--psm`: Tesseract Page Segmentation Mode (6 works well for blocks of text).
- `--lang`: Tesseract language code. Defaults to `spa+eng`.
- `--deskew` / `--no-deskew`: correct page skew before OCR. **On by default** — see
  [Deskew](accuracy.md#deskew---deskew).
- `--reading-order` / `--no-reading-order`: rebuild reading order from word geometry
  on multi-column pages. **On by default** — see
  [Reading order](accuracy.md#reading-order---reading-order).
- `--pipeline`: compose the preprocessing stack explicitly — see
  [Composing a pipeline](#composing-a-pipeline). Overrides `--deskew`.
- `--pdf-dpi`: resolution to render PDF pages at. Defaults to each page's native resolution — see [Reading PDFs](#reading-pdfs).
- `--force-ocr`: OCR PDF pages even when they already carry a text layer — see [Reading PDFs](#reading-pdfs).
- `--oem`: OCR Engine Mode — `1` (LSTM) or `3` (default). Legacy modes are gone in Tesseract 5.
- `--tessdata-dir`: override the language-model directory (defaults to project-local `tessdata/`).
- `--tesseract-cmd`: override path to Tesseract binary if needed (default is usually `/usr/bin/tesseract`).

## Composing a pipeline

Every filter is reachable by name, so a preprocessing stack is a command-line
argument rather than a source edit:

```bash
uv run python main.py photo.JPEG --pipeline grayscale deskew blur:ksize=3
uv run python main.py photo.JPEG --pipeline grayscale clahe:clip_limit=3 threshold
```

Each entry is `name` or `name:key=value,key=value`. Arguments are coerced using the
filter constructor's own type annotations, so the filters remain the single source
of truth for what they accept, and `main.py --help` lists every filter with its
defaults. An unknown name or parameter is a command-line error listing the valid
options, not a traceback.

The same flag on `scripts/evaluate_corpus.py` takes several stacks and scores them
against the whole corpus in one run:

```bash
uv run python scripts/evaluate_corpus.py \
    --pipeline 'grayscale deskew blur:ksize=3' \
    --pipeline 'grayscale deskew clahe blur:ksize=3'
```

Which stacks actually win is measured, not assumed — see
[Filter stacks compared](accuracy.md#filter-stacks-compared). Two of the filters
already in the tree make results *worse* on this corpus.
