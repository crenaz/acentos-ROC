## Acentos OCR Pipeline

Modular OCR for photographed and scanned documents. OpenCV preprocessing, NumPy, and
pytesseract, structured as a pipe-and-filter (Strategy) pipeline, with a geometric
reading-order rebuild for multi-column pages.

Every default in it was chosen by measurement against hand-typed ground truth, not by
plausibility — several reasonable-sounding ideas are recorded in the docs as failures
so nobody retries them.

**Character error rate: 11.9%** across 15 transcribed Cayman newspaper job listings
(`--psm 3`, deskew and reading order on). Full numbers and method in
[docs/accuracy.md](docs/accuracy.md).

### Quickstart

Needs Tesseract 5 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # Python 3.13, deps, and the project itself
./scripts/fetch_tessdata.sh  # language models (~39 MB, gitignored)

uv run python main.py photo.JPEG
uv run python main.py scan.pdf ~/clippings/*.JPEG --save-text transcripts/
```

Full setup, including the Tesseract 5 backport PPA for Ubuntu 20.04, is in
[docs/installation.md](docs/installation.md).

### What it does

- **Photographs of print**, not just flat scans — deskew and reading-order rebuild
  exist because phone photos of newspapers break Tesseract's own page segmentation.
- **PDFs**, scanned or digital-born. A PDF that already carries a text layer is
  extracted exactly rather than being OCR'd, and scanned pages render at the scan's
  own resolution instead of a guessed DPI.
- **Batches**, with per-file failures reported rather than sinking the run, and
  output names checked for collisions before any work starts.
- **Composable preprocessing** — every filter is reachable by name from the command
  line, so a stack is an argument rather than a source edit.
- **Its own benchmark.** `scripts/evaluate_corpus.py` scores a whole corpus of images
  against manual transcriptions and reports CER alongside word miss rate, because the
  gap between the two distinguishes a page read wrongly from a page read out of order.

### Documentation

| Document | Covers |
| --- | --- |
| [docs/installation.md](docs/installation.md) | Tesseract 5, language models, the uv environment |
| [docs/usage.md](docs/usage.md) | running OCR, batches, PDFs, debugging, flag reference |
| [docs/accuracy.md](docs/accuracy.md) | how quality is measured, and every result so far |
| [docs/development.md](docs/development.md) | tests, conventions, project layout |
| [docs/licence.md](docs/licence.md) | AGPL-3.0 in practice, and dependency terms |
| [CONTRIBUTING.md](CONTRIBUTING.md) | what helps most, and how to add a filter |

`PROJECT-ANALYSIS-2026-08-02.md` is the working decision log — narrative rather than
reference, and where the next-steps checklist lives.

### Status

Working and measured, but young. The corpus is English-language Cayman job listings;
accented Spanish is the longer-term target and is not yet represented in the ground
truth at all, which is the single most useful thing a contributor could change.

Known weak point: pages dominated by perspective or curvature rather than rotation,
which a single global deskew cannot correct.

### Licence

Copyright (c) 2026 crenaz. **AGPL-3.0-only** — run it on your own documents and you
owe nothing; distribute it, or offer a modified version as a network service, and you
owe your users the source. A commercial licence is available on request.

Details, including why contributions carry a relicensing grant, are in
[docs/licence.md](docs/licence.md).
