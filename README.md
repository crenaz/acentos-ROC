## Acentos OCR Pipeline

Modular OCR project using OpenCV preprocessing, NumPy, and pytesseract, structured as a pipe-and-filter (Strategy) pipeline.

### 1. System prerequisites (Ubuntu 20.04 on WSL2)

Install Tesseract and its dev libraries:

```bash
sudo apt update
sudo apt install -y tesseract-ocr libtesseract-dev
```

### 2. Python environment (Python 3.13.3)

Create and activate a virtual environment, then install the project in editable mode
via `pyproject.toml`:

```bash
cd /home/crenaz/projects/ONLINE/GITHUB/acentos-ROC
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For development extras (pytest, etc.):

```bash
pip install -e ".[dev]"
```

The editable install is deliberate: it points the environment at `src/acentos_ocr/`
so edits take effect immediately and the installed copy can never drift out of sync
with the source tree.

### 3. Running OCR on an image

Assuming `document.png` lives in the project root:

```bash
source .venv/bin/activate
python main.py document.png --debug --out-debug-dir debug --psm 6
```

Flags:

- `--debug`: prints filter steps and saves a processed image into the `debug/` directory.
- `--psm`: Tesseract Page Segmentation Mode (6 works well for blocks of text).
- `--tesseract-cmd`: override path to Tesseract binary if needed (default is usually `/usr/bin/tesseract`).

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
