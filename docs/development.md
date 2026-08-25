# Development

Tests, project layout, and the conventions worth keeping.

See also: [installation](installation.md) · [usage](usage.md) · [accuracy](accuracy.md)

For how to add a filter, and the licensing terms contributions are accepted under,
see [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Tests

```bash
uv run pytest -q          # 180 tests, about three seconds
```

| File | Covers |
| --- | --- |
| `test_filters.py` | each filter's behaviour, plus the contract they share |
| `test_registry.py` | the `--pipeline` spec language and the filter registry |
| `test_deskew.py` | skew estimation on synthetic scans and photo-like pages |
| `test_layout.py` | region detection and reading-order reconstruction |
| `test_metrics.py` | CER, word miss rate, transcription normalisation |
| `test_corpus.py` | image-to-transcription pairing |
| `test_text_io.py` | `--save-text` output naming, and that it cannot touch ground truth |
| `test_pdf_io.py` | text-layer detection, page rendering, and native-resolution choice |
| `test_main_inputs.py` | flattening images and PDF pages into per-page work items |
| `test_pipelines.py` | the default stack's composition |

No image, PDF or Tesseract install is needed — every fixture is synthesised, so the
suite runs anywhere `uv sync` succeeds. Accuracy claims are *not* tested here;
those come from `scripts/evaluate_corpus.py` against the real corpus, because a
synthetic page cannot tell you whether a change helps a photograph.

Three conventions worth keeping if you add tests:

- **Pin behaviour, not implementation.** The filter tests assert that `open`
  removes an isolated speck and `close` fills a pinhole, rather than that a
  particular OpenCV constant was passed.
- **Check the test can fail.** The filter tests were verified by deliberately
  breaking the code: removing the even-kernel correction failed 3 tests, letting
  `ResizeFilter` shrink failed 1, and swapping `open` with `close` failed 2.
- **Build fixtures, don't check them in.** The PDF tests construct both a
  digital-born and an image-only document in memory. The files this project is
  pointed at are photographs of newspapers and scans of personal paperwork, and
  neither belongs in a public repository.

## Project layout

All code lives in a single importable package, `acentos_ocr`, under a `src/` layout:

```
src/acentos_ocr/
├── core/      pipeline orchestration (PreprocessingPipeline, build_default_pipeline)
├── eval/      corpus discovery and accuracy metrics
├── filters/   individual preprocessing steps (BaseFilter subclasses)
├── layout/    geometric page segmentation and reading-order reconstruction
├── ocr/       Tesseract wrapper + OCRResult
└── utils/     image loading/saving, PDF reading, OCR-text output
```

Always import through the package — `from acentos_ocr.filters.grayscale import GrayscaleFilter`.
Never use `src.` as an import prefix.

Supporting files at the root:

| Path | Holds |
| --- | --- |
| `main.py` | the CLI: argument parsing, input expansion, the per-page loop |
| `scripts/` | `evaluate_corpus.py`, `evaluate_cer.py`, `fetch_tessdata.sh` |
| `results/` | committed JSON from past measurement runs |
| `docs/` | this documentation |
| `PROJECT-ANALYSIS-2026-08-02.md` | working decision log and the live next-steps checklist |
