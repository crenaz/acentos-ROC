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
- `--oem`: OCR Engine Mode — `1` (LSTM) or `3` (default). Legacy modes are gone in Tesseract 5.
- `--tessdata-dir`: override the language-model directory (defaults to project-local `tessdata/`).
- `--tesseract-cmd`: override path to Tesseract binary if needed (default is usually `/usr/bin/tesseract`).

#### Reference baselines

With the default filter stack at `--psm 6`, the committed sample images score:

| Configuration | `fluoxetine.png` | `document.png` |
| --- | --- | --- |
| Tesseract 4.1.1, system models, `--lang eng` | 75.30% | 49.37% |
| Tesseract 4.1.1, system models, `--lang spa+eng` | 81.73% | 56.21% |
| Tesseract 5.5.1, system models, `--lang spa+eng` | **84.44%** | 61.23% |
| Tesseract 5.5.1, `tessdata_best`, `--lang spa+eng` *(current default)* | 82.74% | **64.47%** |

Three things worth knowing from those measurements:

- `spa+eng` beats either language alone on *both* images, including the English
  one, which is why it is the default rather than plain `spa`.
- The **engine** mattered more than the **models**. On `document.png`, upgrading
  Tesseract 4.1.1 → 5.5.1 gained +5.02 points; `tessdata_best` on top of that
  gained a further +3.24. Neither configuration wins on both images.
- Cost of `tessdata_best` is roughly 2× wall clock — `document.png` takes ~3.7s
  versus ~1.8s with the system models on a 4-core i7-1165G7.

> **These numbers are Tesseract's self-reported confidence, not accuracy.** A model
> can be confidently wrong, and at these margins the metric cannot separate the two
> configurations: inspecting the text, system models and `tessdata_best` make
> *different* errors on `document.png` rather than one making fewer. Treat the table
> as a regression check — "did this change break something?" — not as a quality
> ranking. Deciding which configuration is genuinely better requires ground-truth
> transcriptions and a character error rate, which the project does not yet have.

Recorded against Tesseract 5.5.1 and the current `uv.lock`.

### 5. Project layout

All code lives in a single importable package, `acentos_ocr`, under a `src/` layout:

```
src/acentos_ocr/
├── core/      pipeline orchestration (PreprocessingPipeline)
├── filters/   individual preprocessing steps (BaseFilter subclasses)
├── ocr/       Tesseract wrapper + OCRResult
└── utils/     image loading/saving
```

Always import through the package — `from acentos_ocr.filters.grayscale import GrayscaleFilter`.
Never use `src.` as an import prefix.

### 6. Extending the pipeline

- Add new filters under `src/acentos_ocr/filters/`, each inheriting from `BaseFilter` and implementing `apply(self, image: np.ndarray) -> np.ndarray`.
- Wire new filters into the pipeline in `build_default_pipeline` inside `main.py` or create alternative pipelines in separate modules under `src/acentos_ocr/core/`.

Acentos-ROC
