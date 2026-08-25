# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

"""
Reading PDFs as pipeline input.

There are two kinds of PDF and only one of them wants OCR.

A digital-born PDF already carries a text layer: the exact characters are sitting
in the file, and extracting them is both instant and error-free. Rasterising one
and running it through Tesseract can only make it worse -- it throws away perfect
text and then guesses at the pixels. So the text layer is checked first, and OCR
is the fallback rather than the default.

A scanned PDF is an image in a container. Those pages are rendered to arrays and
handed to the ordinary preprocessing pipeline, which neither knows nor cares that
the pixels arrived in a PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pypdfium2 as pdfium

#: A page yielding fewer than this many non-whitespace characters is treated as
#: having no usable text layer. A scan is not always cleanly empty: a stray
#: annotation, or a bad OCR layer left behind by another tool, can leave a handful
#: of characters that are worse than nothing.
MIN_TEXT_LAYER_CHARS = 32

#: Used only when a page's native resolution cannot be determined. Tesseract is
#: tuned for roughly this and degrades noticeably below it.
DEFAULT_DPI = 300

#: PDF user-space units per inch, fixed by the format.
POINTS_PER_INCH = 72

#: Refuse to render beyond this on the long edge. Guards against pathological page
#: boxes -- a page declaring itself 28 inches wide at 300 DPI is 8559 pixels of
#: mostly interpolation, which is slow and no more legible.
MAX_EDGE_PIXELS = 6000

#: Below this, a "native" resolution is more likely a thumbnail or a logo than the
#: scan itself, so the DPI default is used instead.
MIN_NATIVE_EDGE_PIXELS = 1000


def is_pdf(path: str | Path) -> bool:
    """Whether this path should be read as a PDF rather than an image."""
    return Path(path).suffix.lower() == ".pdf"


def page_count(path: str | Path) -> int:
    with pdfium.PdfDocument(str(path)) as pdf:
        return len(pdf)


def extract_text_layer(path: str | Path) -> list[str]:
    """
    The embedded text of each page, in order, `""` where a page has none.

    The list is always as long as the document, so a caller can decide per page
    rather than for the whole file -- a scan appended to a digital document is a
    real thing, and it should not force the whole file down one route.
    """
    with pdfium.PdfDocument(str(path)) as pdf:
        pages = []
        for page in pdf:
            pages.append(page.get_textpage().get_text_range().strip())
        return pages


def has_text_layer(
    path: str | Path, min_chars: int = MIN_TEXT_LAYER_CHARS
) -> bool:
    """
    Whether any page carries enough embedded text to be worth extracting.

    Deliberately `any` and not `all`: one page of real text means the file is not
    a pure scan, and the caller should look page by page.
    """
    return any(len(text) >= min_chars for text in extract_text_layer(path))


def _native_scale(page: pdfium.PdfPage) -> float | None:
    """
    The render scale that reproduces the page's own image at its stored resolution.

    A scanned page is one big embedded image, and that image has a fixed pixel
    count. Rendering above it interpolates -- more pixels, no more detail, and a
    slower OCR pass. This finds the largest image on the page and returns the scale
    that lands on its native size.

    Returns None when the page has no image big enough to be the scan itself, which
    is the signal to fall back to a DPI.
    """
    width_pt, _ = page.get_size()
    if width_pt <= 0:
        return None

    widest = 0
    for obj in page.get_objects():
        if isinstance(obj, pdfium.PdfImage):
            metadata = obj.get_metadata()
            widest = max(widest, metadata.width)

    if widest < MIN_NATIVE_EDGE_PIXELS:
        return None
    return widest / width_pt


def _to_bgr(bitmap: pdfium.PdfBitmap) -> np.ndarray:
    """
    A pdfium bitmap as a contiguous 3-channel BGR array.

    Two hazards handled here. The buffer `to_numpy` returns is a view onto memory
    the bitmap owns, so it has to be copied before the bitmap is released or the
    array quietly becomes garbage. And the channel order depends on how pdfium was
    asked to render, so it is normalised rather than assumed.
    """
    array = bitmap.to_numpy()
    mode = bitmap.mode

    if mode == "BGRA":
        array = array[..., :3]
    elif mode == "RGB":
        array = array[..., ::-1]
    elif mode == "RGBA":
        array = array[..., 2::-1]
    elif mode in ("L", "GRAY"):
        array = np.repeat(array[..., np.newaxis], 3, axis=2)
    elif mode != "BGR":
        raise ValueError(f"Unsupported pdfium bitmap mode: {mode!r}")

    return np.ascontiguousarray(array)


def render_pages(
    path: str | Path, dpi: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield `(page_number, image)` for each page, page numbers starting at 1.

    With `dpi` given, every page is rendered at that resolution. With `dpi` None --
    the default -- each page is rendered at the native resolution of its own largest
    image, which is what a scan actually contains, and falls back to DEFAULT_DPI
    only for pages that have no such image.

    That default matters more than it sounds. Phone "scan to PDF" wraps a photo in a
    page box sized to the pixels at 72 DPI, so an ordinary sheet can declare itself
    28 inches wide; rendering *that* at 300 DPI turns a 2053-pixel scan into an 8559
    pixel one made almost entirely of interpolation.
    """
    with pdfium.PdfDocument(str(path)) as pdf:
        for index, page in enumerate(pdf, start=1):
            scale = _render_scale(page, dpi)
            bitmap = page.render(scale=scale)
            try:
                yield index, _to_bgr(bitmap)
            finally:
                bitmap.close()


def render_page(
    path: str | Path, number: int, dpi: int | None = None
) -> np.ndarray:
    """
    One page as a BGR array, `number` starting at 1.

    Exists so a caller can render a page at the moment it is needed rather than
    holding a whole document in memory: a fifty-page scan at native resolution is
    comfortably a gigabyte of arrays, and the pipeline only ever looks at one.
    """
    with pdfium.PdfDocument(str(path)) as pdf:
        if not 1 <= number <= len(pdf):
            raise IndexError(
                f"{Path(path).name} has {len(pdf)} page(s); asked for page {number}"
            )
        page = pdf[number - 1]
        bitmap = page.render(scale=_render_scale(page, dpi))
        try:
            return _to_bgr(bitmap)
        finally:
            bitmap.close()


def _render_scale(page: pdfium.PdfPage, dpi: int | None) -> float:
    """The scale factor for one page, clamped so the long edge stays sane."""
    if dpi is not None:
        scale = dpi / POINTS_PER_INCH
    else:
        scale = _native_scale(page) or (DEFAULT_DPI / POINTS_PER_INCH)

    width_pt, height_pt = page.get_size()
    longest_pt = max(width_pt, height_pt)
    if longest_pt > 0 and longest_pt * scale > MAX_EDGE_PIXELS:
        scale = MAX_EDGE_PIXELS / longest_pt

    return scale
