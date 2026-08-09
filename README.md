## Acentos OCR Pipeline

Modular OCR project using OpenCV preprocessing, NumPy, and pytesseract, structured as a pipe-and-filter (Strategy) pipeline.

### 1. System prerequisites (Ubuntu 20.04 on WSL2)

Tesseract 5 is required. Ubuntu 20.04 (`focal`) only carries 4.1.1 in its own
archive, so it comes from a backport PPA:

```bash
sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
sudo apt update
sudo apt install -y tesseract-ocr libtesseract-dev
```

Confirm with `tesseract --version` (expect 5.x).

Language data is **not** installed via apt — see the next section. The project
supplies its own models so results do not depend on which `tesseract-ocr-*`
packages happen to be present system-wide.

> **Note on Tesseract 5:** the legacy OCR engine was removed, so `--oem 0` and
> `--oem 2` no longer exist. The CLI accepts only `--oem 1` (LSTM) and `--oem 3`
> (default).

### 2. Language models

Fetch the high-accuracy `tessdata_best` models into a project-local `tessdata/`
directory:

```bash
./scripts/fetch_tessdata.sh
```

This downloads `eng`, `spa`, and `osd` (~39 MB total). The directory is
gitignored — run the script once after cloning. It is idempotent; pass `--force`
to re-download.

`tessdata_best` is slower than Ubuntu's standard models, and its advantage is
narrower than it first appears — it wins on Spanish and *loses* on English:

| Models | `fluoxetine.png` (English) | `document.png` (Spanish) | Speed |
| --- | --- | --- | --- |
| System (`tesseract-ocr-*` packages) | **84.44%** | 61.23% | ~1.8s |
| `tessdata_best` | 82.74% | **64.47%** | ~3.7s |

So for **English-only work, the system models are both faster and better** — use
`--tessdata-dir /usr/share/tesseract-ocr/5/tessdata`. `tessdata_best` earns its
keep on accented Spanish, which is the project's long-term target.

Requires the system language packs for the fallback path:
`sudo apt install -y tesseract-ocr-eng tesseract-ocr-spa`.

The pipeline auto-detects `tessdata/` when populated and falls back to
Tesseract's system-wide lookup when it is not. Override with `--tessdata-dir`.

### 3. Python environment (uv, Python 3.13)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
`uv.lock` is authoritative — it pins the exact version of every transitive
dependency, so the pipeline is reproducible across machines.

```bash
cd /home/crenaz/projects/ONLINE/GITHUB/acentos-ROC
uv sync
```

That single command provisions Python 3.13 (per `.python-version`), creates `.venv`,
installs every locked dependency plus the `dev` group, and installs the project
itself in editable mode pointing at `src/acentos_ocr/` — so edits take effect
immediately and the installed copy can never drift from the source tree.

Reproducing the exact locked environment (e.g. in CI or Docker):

```bash
uv sync --frozen        # fails rather than silently re-resolving
```

Adding or updating a dependency:

```bash
uv add <package>        # updates pyproject.toml and uv.lock together
uv lock --upgrade       # deliberately re-resolve everything
```

Commit `uv.lock` and `.python-version` with any such change. Because OCR output is
sensitive to the OpenCV version, re-locking should always be followed by re-running
the sample images below and confirming the confidence scores are unchanged.

### 4. Running OCR on an image

Assuming `document.png` lives in the project root:

```bash
uv run python main.py document.png --debug --out-debug-dir debug --psm 6
```

`uv run` executes inside the locked environment without needing to activate it.

#### Diagnosing a bad result

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

Flags:

- `--debug`: prints each pipeline step with its shape and dtype, reports which tessdata directory was used, and writes **every intermediate image** to the `debug/` directory.
- `--psm`: Tesseract Page Segmentation Mode (6 works well for blocks of text).
- `--lang`: Tesseract language code. Defaults to `spa+eng`.
- `--deskew` / `--no-deskew`: correct page skew before OCR. **On by default** — see below.
- `--reading-order` / `--no-reading-order`: rebuild reading order from word geometry on multi-column pages. **On by default** — see below.
- `--pipeline`: compose the preprocessing stack explicitly — see below. Overrides `--deskew`.
- `--oem`: OCR Engine Mode — `1` (LSTM) or `3` (default). Legacy modes are gone in Tesseract 5.
- `--tessdata-dir`: override the language-model directory (defaults to project-local `tessdata/`).
- `--tesseract-cmd`: override path to Tesseract binary if needed (default is usually `/usr/bin/tesseract`).

#### Composing a pipeline

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

Measured that way, `--lang eng`, `--psm 3`, reading order on
(`results/stacks-2026-08-08.json`):

| Stack | CER | word miss | confidence |
| --- | --- | --- | --- |
| **`grayscale deskew blur:ksize=3`** *(default)* | **12.0%** | 10.7% | 88.1% |
| `grayscale deskew blur:ksize=5` | 12.4% | 11.6% | 88.0% |
| `grayscale deskew` *(no blur)* | 12.4% | 11.0% | 87.4% |
| `grayscale deskew clahe blur:ksize=3` | 18.1% | 12.1% | 85.4% |
| `grayscale deskew blur:ksize=5 threshold morphology` *(the original stack)* | 27.1% | 21.3% | 69.6% |

Three things fall out of that:

- **The blur is nearly irrelevant.** Dropping it entirely costs 0.4 points, and
  widening it to 5 costs the same. It is the least consequential knob in the stack.
- **CLAHE hurts, badly and unevenly.** +6.1 points overall, and it destroys
  `IMG_1648` specifically (4.3% → 48.5%). Boosting local contrast on newsprint
  amplifies paper texture along with the ink.
- **The original binarising stack is confirmed catastrophic at corpus scale.**
  27.1% against 12.0%, and it fails one image completely (`IMG_1598`, 100% CER).
  That was previously measured on a single image; the corpus agrees.

#### Reference baselines

With the default filter stack at `--psm 6`, the committed sample images score:

| Configuration | `fluoxetine.png` | `document.png` |
| --- | --- | --- |
| Tesseract 4.1.1, system models, `--lang eng`, binarising stack | 75.30% | 49.37% |
| Tesseract 4.1.1, system models, `--lang spa+eng`, binarising stack | 81.73% | 56.21% |
| Tesseract 5.5.1, system models, `--lang spa+eng`, binarising stack | 84.44% | 61.23% |
| Tesseract 5.5.1, `tessdata_best`, `--lang spa+eng`, binarising stack | 82.74% | 64.47% |
| **Current default** — grayscale + deskew + blur3, `--psm 3` | **83.69%** | **95.17%** |

Both samples are flat-scanned, so the deskew estimator finds no confident angle and
declines to act; enabling it by default left these two figures untouched.

#### Ground truth (the metric that matters)

Measured as character error rate against a manual transcription of a Cayman job
listing, via `scripts/evaluate_cer.py`:

| Pipeline | `--psm 3` | `--psm 4` | `--psm 6` |
| --- | --- | --- | --- |
| Old binarising stack | 34.4% | 30.5% | 51.9% |
| Grayscale + blur3 | 8.9% | 8.9% | 16.2% |
| **Current: grayscale + deskew + blur3** | **7.3%** | 10.3% | 14.4% |

Three things worth knowing from those measurements:

- **Binarising was the single biggest quality problem.** Tesseract 5's LSTM engine
  binarises internally, and does it far better than a hand-tuned adaptive threshold.
  Removing `AdaptiveThresholdFilter` and `MorphologyFilter` from the default stack
  cut character error rate from 30.5% to 8.9%. Those filters remain available for
  explicit use, but the preprocessing they represent was written for the pre-LSTM era.
- `spa+eng` beats either language alone on *both* images, including the English
  one, which is why it is the default rather than plain `spa`.
- Cost of `tessdata_best` is roughly 2× wall clock — `document.png` takes ~3.7s
  versus ~1.8s with the system models on a 4-core i7-1165G7.

#### Corpus evaluation

A single image cannot tell you whether a change helped — the first two
transcriptions pointed opposite ways on `--deskew`. `scripts/evaluate_corpus.py`
runs the whole transcribed corpus through a sweep of configurations:

```bash
export ACENTOS_CORPUS="/path/to/Cayman Job Clippings"
uv run python scripts/evaluate_corpus.py --psm 3 4 6 --deskew both --per-image
```

Pairing is by filename: an image `IMG_1594.JPEG` anywhere under the root is matched
with `text-of-IMG_1594.md`, also anywhere under the root. Images with no
transcription are listed rather than silently skipped.

It reports two metrics side by side, and **the gap between them is the point**:

| Metric | Sensitive to order? | Answers |
| --- | --- | --- |
| CER | yes | what you actually get out of the pipeline |
| word miss rate | no | what the recogniser managed to read |

- Both low → the page was read correctly.
- Both high → a recognition failure; the words genuinely are not there.
- **CER high, word miss low** → the words were recognised and then emitted in the
  wrong order. That is a page-segmentation failure, and no amount of image
  preprocessing will fix it.

Rates are micro-averaged (errors and lengths summed separately) so a two-line
advert cannot outweigh a full page. `--json` writes the raw per-image rows for
comparison against a later run.

Baseline across 15 transcribed Cayman job listings, `--lang eng`, 2026-08-08
(`results/baseline-2026-08-08.json`):

| Configuration | CER | word miss | confidence |
| --- | --- | --- | --- |
| **psm 3 + deskew + reading order** *(current default)* | **12.0%** | 10.7% | 88.1% |
| psm 3 + deskew | 18.9% | 10.7% | 88.1% |
| psm 6 | 22.3% | 12.1% | 84.4% |
| psm 6 + deskew | 22.3% | 11.9% | 84.3% |
| psm 4 + deskew | 23.6% | 11.2% | 87.4% |
| psm 3 *(default before 2026-08-08)* | 24.0% | 11.6% | 87.1% |
| psm 4 | 27.6% | 12.9% | 86.3% |

No single mode wins everywhere. Choosing the best configuration per image would
reach 15.8% — the ~3-point gap under the best fixed setting is what per-image
adaptation is worth, and it is much smaller than the 5-point deskew effect below.

Two images (`IMG_1599`, `IMG_1604`) sit near 30% in *every* configuration with
correspondingly high word miss rates. Those are recognition failures, and they are
**capture-limited rather than pipeline-limited** — both are dominated by perspective
and curvature rather than rotation, which a single global deskew cannot correct, and
`IMG_1604`'s text is only 31px tall.

Two remedies were tried and measured across the corpus. Neither works, so they are
recorded here rather than retried:

| Attempt | Corpus CER |
| --- | --- |
| baseline | **12.0%** |
| `--scale 1.5` | 12.7% |
| `--scale 2` | 12.6% |
| drop words below confidence 30 | 12.2% |
| drop words below confidence 60 | 13.3% |

Upscaling looks convincing on one sample — `IMG_1599` alone goes 32.2% → 22.9% at
×1.5 — and the corpus flatly contradicts it. The per-image pattern is scatter, not
signal: `IMG_1646` reaches 0.8% at ×2 while four other images get worse. That is
Tesseract's layout analysis reacting to a different input size, not a resolution
effect, and it is a good illustration of why single-image results are not evidence.

#### Deskew (`--deskew`)

**On by default** as of 2026-08-08 — worth 5.1 points corpus-wide (24.0% → 18.9% CER
at psm 3) for negligible extra time. Pass `--no-deskew` to disable. Per-image delta:

| Image | CER change with `--deskew` |
| --- | --- |
| `IMG_1648` | **−40.9%** |
| `IMG_1658` | −18.8% |
| `IMG_1601` | −9.5% |
| `IMG_1603` | −6.8% |
| `IMG_1594` | −1.6% |
| `IMG_1657` | +4.3% |
| `IMG_1595` | +7.5% |
| 8 others | unchanged — no confident angle found |

Four large wins against two modest regressions, because on more than half the corpus
the filter finds no confident angle and declines to act rather than guessing.

Note what the largest win actually is. `IMG_1648` improves by 41 points not because
its characters were unreadable, but because its skew made Tesseract's layout analysis
invent column boundaries and shred the page into out-of-order blocks — 45.7% CER
against a 5.4% word miss rate. Straightening the page fixed the segmentation. Skew
damage and reading-order damage are frequently the same bug.

#### Reading order (`--reading-order`)

**On by default.** Corpus CER 18.9% → 12.0%, at no extra OCR cost.

Tesseract's page segmentation mishandles the layout these adverts almost always
use — a full-width headline and intro, a two-column body, then a full-width
footer — and it fails in *both* directions depending on skew:

| | Block widths | Failure |
| --- | --- | --- |
| skewed page | narrow | applies the column split to the **whole page**, shredding full-width paragraphs into three pieces emitted far apart |
| straight page | full | reads the **two-column body as full width**, interleaving left and right line by line |

Either way the characters are read correctly and put in the wrong order, which
CER punishes exactly as hard as not reading them at all. On `IMG_1595`: 40.5% CER
against a 7.9% word miss rate.

The fix re-segments the page geometrically and re-emits the words. Every word
already comes back from the OCR pass with a bounding box, so this needs no second
pass. The segmentation is an XY-cut over a map of those boxes — not over pixels,
which sidesteps the page border, uneven lighting and newsprint texture that make
pixel-based layout analysis fragile on a photograph.

Three details carry the benefit:

- **Cut one gap at a time, widest first.** Splitting at every qualifying gap at
  once breaks the page into horizontal bands *before* anything notices the gutter,
  and then splits each band into left and right independently — which interleaves
  the columns a few lines at a time, reproducing the bug being fixed.
- **Assign words to regions, but keep Tesseract's lines inside them.** The two
  failure modes are mirror images and both need handling: a line Tesseract split
  across blocks must be rejoined, and a line it merged across the gutter must be
  torn apart. Per-word assignment does the tearing; grouping by Tesseract's own
  line number inside each region does the rejoining.
- **Never re-derive lines from word positions.** Tesseract tracks a baseline per
  line, so it stays correct on a photographed page whose lines sag across the
  frame. Clustering words by vertical centre does not — it drops the last words of
  a sloping line onto the line below. An earlier version did exactly that and cost
  `IMG_1594` 7.3% → 18.8%.

Corpus result: nine images improve, most dramatically `IMG_1595` 48.0% → **4.7%**,
`IMG_1646` 23.2% → **7.3%** and `IMG_1596` 17.9% → **6.0%**. Four are untouched.
One regresses: `IMG_1658` 4.7% → 9.3%, where a short trailing line becomes its own
region and moves ahead of the line above it.

> **The confidence figures are Tesseract's self-assessment, not accuracy.** A model
> can be confidently wrong. Treat the confidence table as a regression check — "did
> this change break something?" — and use `scripts/evaluate_cer.py` against a manual
> transcription whenever a decision actually depends on which configuration is better.

Recorded against Tesseract 5.5.1 and the current `uv.lock`.

### 5. Tests

```bash
uv run pytest -q          # 143 tests, about two seconds
```

| File | Covers |
| --- | --- |
| `test_filters.py` | each filter's behaviour, plus the contract they share |
| `test_registry.py` | the `--pipeline` spec language and the filter registry |
| `test_deskew.py` | skew estimation on synthetic scans and photo-like pages |
| `test_layout.py` | region detection and reading-order reconstruction |
| `test_metrics.py` | CER, word miss rate, transcription normalisation |
| `test_corpus.py` | image-to-transcription pairing |
| `test_pipelines.py` | the default stack's composition |

No image or Tesseract install is needed — every fixture is synthesised, so the
suite runs anywhere `uv sync` succeeds. Accuracy claims are *not* tested here;
those come from `scripts/evaluate_corpus.py` against the real corpus, because a
synthetic page cannot tell you whether a change helps a photograph.

Two conventions worth keeping if you add tests:

- **Pin behaviour, not implementation.** The filter tests assert that `open`
  removes an isolated speck and `close` fills a pinhole, rather than that a
  particular OpenCV constant was passed.
- **Check the test can fail.** The filter tests were verified by deliberately
  breaking the code: removing the even-kernel correction failed 3 tests, letting
  `ResizeFilter` shrink failed 1, and swapping `open` with `close` failed 2.

### 6. Project layout

All code lives in a single importable package, `acentos_ocr`, under a `src/` layout:

```
src/acentos_ocr/
├── core/      pipeline orchestration (PreprocessingPipeline, build_default_pipeline)
├── eval/      corpus discovery and accuracy metrics
├── filters/   individual preprocessing steps (BaseFilter subclasses)
├── layout/    geometric page segmentation and reading-order reconstruction
├── ocr/       Tesseract wrapper + OCRResult
└── utils/     image loading/saving
```

Always import through the package — `from acentos_ocr.filters.grayscale import GrayscaleFilter`.
Never use `src.` as an import prefix.

### 7. Licence

Copyright (c) 2026 crenaz. Licensed under **AGPL-3.0-only**.

The full text is in [`LICENSE`](LICENSE), `pyproject.toml` declares it as an SPDX
expression — so the built wheel carries `License-Expression: AGPL-3.0-only` and
bundles the text — and every source file opens with an SPDX header.

The AGPL is the GPL plus one extra obligation, [section
13](LICENSE): if you run a modified version and let other people use it **over a
network**, those users are entitled to your modified source. Merely running it
privately, or on your own documents, triggers nothing.

In practice:

| What you do | What you owe |
| --- | --- |
| Run it on your own images | nothing |
| Modify it and keep the changes to yourself | nothing |
| Distribute it, modified or not | source, under AGPL-3.0 |
| Offer it as a hosted service, modified | source, under AGPL-3.0, to your users |

A commercial licence is available on request for anyone who needs to build on this
without those obligations.

Contributions are accepted under AGPL-3.0 **plus** a grant allowing the copyright
holder to relicense them, which is what keeps that commercial option open. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

> Dependencies keep their own terms and none of them is AGPL: OpenCV is Apache-2.0,
> NumPy, pandas and Pillow are BSD-style, pytesseract is Apache-2.0. Tesseract
> itself is Apache-2.0 and is invoked as a separate process, not linked. The
> `tessdata_best` models fetched by `scripts/fetch_tessdata.sh` are Apache-2.0 and
> are gitignored rather than redistributed here.

### 8. Extending the pipeline

- Add new filters under `src/acentos_ocr/filters/`, each inheriting from `BaseFilter` and implementing `apply(self, image: np.ndarray) -> np.ndarray`.
- **Register it** in `src/acentos_ocr/filters/registry.py`. That one line is what makes it reachable as `--pipeline yourfilter:option=value`, with its arguments coerced from the constructor's type annotations. A test fails if a filter module is missing from the registry.
- Give it a behavioural test in `tests/test_filters.py`; the shared contract tests (uint8 output, no mutation of the input) pick it up automatically from the registry.
- Measure it before adopting it: `scripts/evaluate_corpus.py --pipeline '...' --pipeline '...'`. Two of the filters already in the tree make results *worse* on this corpus, which is only knowable by measuring.

Acentos-ROC
