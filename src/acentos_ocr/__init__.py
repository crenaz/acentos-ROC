"""
Acentos OCR: a pipe-and-filter OCR pipeline built on OpenCV and Tesseract,
aimed at accurate recognition of accented characters.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("acentos-ocr")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
