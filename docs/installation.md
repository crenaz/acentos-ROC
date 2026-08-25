# Installation

Setting up Tesseract, the language models, and the Python environment.

See also: [usage](usage.md) · [accuracy](accuracy.md) · [development](development.md)

## 1. System prerequisites (Ubuntu 20.04 on WSL2)

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

Nothing extra is needed for PDF input: `pypdfium2` ships its own PDF engine as a
wheel and is installed by `uv sync`.

> **Note on Tesseract 5:** the legacy OCR engine was removed, so `--oem 0` and
> `--oem 2` no longer exist. The CLI accepts only `--oem 1` (LSTM) and `--oem 3`
> (default).

## 2. Language models

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

## 3. Python environment (uv, Python 3.13)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
`uv.lock` is authoritative — it pins the exact version of every transitive
dependency, so the pipeline is reproducible across machines.

```bash
cd /path/to/acentos-ROC
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
the [reference baselines](accuracy.md#reference-baselines) and confirming the
figures are unchanged.
