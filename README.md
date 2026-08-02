## Acentos OCR Pipeline

Modular OCR project using OpenCV preprocessing, NumPy, and pytesseract, structured as a pipe-and-filter (Strategy) pipeline.

### 1. System prerequisites (Ubuntu 20.04 on WSL2)

Install Tesseract and its dev libraries:

```bash
sudo apt update
sudo apt install -y tesseract-ocr libtesseract-dev
```

### 2. Python environment (uv, Python 3.13)

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

### 3. Running OCR on an image

Assuming `document.png` lives in the project root:

```bash
uv run python main.py document.png --debug --out-debug-dir debug --psm 6
```

`uv run` executes inside the locked environment without needing to activate it.

Flags:

- `--debug`: prints filter steps and saves a processed image into the `debug/` directory.
- `--psm`: Tesseract Page Segmentation Mode (6 works well for blocks of text).
- `--tesseract-cmd`: override path to Tesseract binary if needed (default is usually `/usr/bin/tesseract`).

#### Reference baselines

With the default filter stack at `--psm 6`, the committed sample images score:

| Image | Average confidence | Notes |
| --- | --- | --- |
| `fluoxetine.png` | 75.30% | English label photo |
| `document.png` | 49.37% | Spanish prose, heavy diacritics |

Use these as a regression check after changing the filter stack or re-locking
dependencies. Recorded against Tesseract 4.1.1 and the current `uv.lock`.

### 4. Project layout

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

### 5. Extending the pipeline

- Add new filters under `src/acentos_ocr/filters/`, each inheriting from `BaseFilter` and implementing `apply(self, image: np.ndarray) -> np.ndarray`.
- Wire new filters into the pipeline in `build_default_pipeline` inside `main.py` or create alternative pipelines in separate modules under `src/acentos_ocr/core/`.

Acentos-ROC
