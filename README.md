## Acentos OCR Pipeline

Modular OCR project using OpenCV preprocessing, NumPy, and pytesseract, structured as a pipe-and-filter (Strategy) pipeline.

### 1. System prerequisites (Ubuntu 20.04 on WSL2)

Install Tesseract and its dev libraries:

```bash
sudo apt update
sudo apt install -y tesseract-ocr libtesseract-dev
```

### 2. Python environment (Python 3.13.3)

Create and activate a virtual environment, then install dependencies via `pyproject.toml`:

```bash
cd /home/crenaz/projects/ONLINE/GITHUB/acentos-ROC
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

For development extras (pytest, etc.):

```bash
pip install ".[dev]"
```

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

### 4. Extending the pipeline

- Add new filters under `src/filters/`, each inheriting from `BaseFilter` and implementing `apply(self, image: np.ndarray) -> np.ndarray`.
- Wire new filters into the pipeline in `build_default_pipeline` inside `main.py` or create alternative pipelines in separate modules under `src/core/`.

Acentos-ROC
