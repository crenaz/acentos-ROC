from __future__ import annotations

import numpy as np
import pandas as pd

from .regions import Rect, find_regions, has_columns

#: Occupancy is rasterised at roughly this many cells per median word height. Fine
#: enough to keep a column gutter distinct, coarse enough that a 12 MP page becomes
#: an array of a few hundred rows.
CELLS_PER_UNIT = 8

#: Two words belong to the same line when their vertical centres sit within this
#: many median word heights of each other.
LINE_TOLERANCE_SCALE = 0.6


def _occupancy(data: pd.DataFrame, scale: float) -> tuple[np.ndarray, int, int]:
    """Rasterise word bounding boxes into a coarse boolean page map.

    Built from Tesseract's word boxes rather than from the image, which sidesteps
    everything that makes pixel-based layout analysis fragile on a photograph: the
    dark page border, uneven lighting, newsprint texture, and the sliver of the
    adjacent article usually caught at the edge of the frame. It also costs nothing
    extra -- the boxes come back from the OCR pass that already ran.
    """
    left = np.floor(data["left"] / scale).astype(int)
    top = np.floor(data["top"] / scale).astype(int)
    right = np.ceil((data["left"] + data["width"]) / scale).astype(int)
    bottom = np.ceil((data["top"] + data["height"]) / scale).astype(int)

    grid = np.zeros((int(bottom.max()) + 1, int(right.max()) + 1), dtype=bool)
    for x0, y0, x1, y1 in zip(left, top, right, bottom):
        grid[y0:y1, x0:x1] = True
    return grid, grid.shape[1], grid.shape[0]


def _assign(data: pd.DataFrame, regions: list[Rect], scale: float) -> dict[int, list[int]]:
    """Map each region index to the row labels of the words inside it."""
    centres_x = (data["left"] + data["width"] / 2) / scale
    centres_y = (data["top"] + data["height"] / 2) / scale

    buckets: dict[int, list[int]] = {index: [] for index in range(len(regions))}
    for label, x, y in zip(data.index, centres_x, centres_y):
        for index, region in enumerate(regions):
            if region.contains(x, y):
                buckets[index].append(label)
                break
        else:
            # Regions tile the content, not the margins, so a word can sit just
            # outside one. Attach it to the nearest region rather than dropping it.
            nearest = min(
                range(len(regions)),
                key=lambda i: (
                    max(regions[i].x0 - x, 0, x - regions[i].x1) ** 2
                    + max(regions[i].y0 - y, 0, y - regions[i].y1) ** 2
                ),
            )
            buckets[nearest].append(label)
    return buckets


def _lines(block: pd.DataFrame, unit: float) -> list[str]:
    """Group a region's words into lines by vertical position, then read left to right."""
    tolerance = LINE_TOLERANCE_SCALE * unit
    block = block.assign(_centre=block["top"] + block["height"] / 2).sort_values("_centre")

    lines: list[list[int]] = []
    centre = None
    for label, row_centre in zip(block.index, block["_centre"]):
        if centre is None or row_centre - centre > tolerance:
            lines.append([label])
            centre = row_centre
        else:
            lines[-1].append(label)
            centre = block.loc[lines[-1], "_centre"].mean()

    return [
        " ".join(block.loc[line].sort_values("left")["text"])
        for line in lines
    ]


def reorder(data: pd.DataFrame) -> str | None:
    """
    Rebuild page text in true reading order from Tesseract's word boxes.

    Tesseract emits words grouped by the blocks its own page segmentation found,
    and on a mixed one-and-two-column advert those blocks are wrong -- see
    `regions.find_regions`. Every word is nevertheless returned with a correct
    bounding box, so the page can be re-segmented geometrically and the words
    re-emitted in the right order without running OCR a second time.

    Returns None when the page has no column structure, meaning the caller should
    keep Tesseract's own ordering. That case is not a failure: with nothing to
    repair, re-deriving the order from geometry only discards the block structure
    Tesseract got right.

    This corrects ordering only. A word Tesseract misread stays misread.
    """
    if data.empty:
        return None

    unit = float(data["height"].median())
    if not np.isfinite(unit) or unit <= 0:
        return None

    scale = max(1.0, unit / CELLS_PER_UNIT)
    grid, _, _ = _occupancy(data, scale)
    regions = find_regions(grid, unit / scale)
    if not regions or not has_columns(regions):
        return None

    chunks = []
    for index, labels in _assign(data, regions, scale).items():
        if not labels:
            continue
        chunks.append("\n".join(_lines(data.loc[labels], unit)))
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()
