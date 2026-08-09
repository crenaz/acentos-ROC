# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

import numpy as np
import pandas as pd

from .regions import Rect, find_regions

#: Occupancy is rasterised at roughly this many cells per median word height. Fine
#: enough to keep a column gutter distinct, coarse enough that a 12 MP page becomes
#: an array of a few hundred rows.
CELLS_PER_UNIT = 8

#: Two line fragments belong to the same visual line when their vertical extents
#: overlap by at least this fraction of the shorter one.
LINE_OVERLAP = 0.5


def _occupancy(data: pd.DataFrame, scale: float) -> np.ndarray:
    """
    Rasterise word bounding boxes into a coarse boolean page map.

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
    return grid


def _fragments(data: pd.DataFrame) -> list[dict]:
    """
    Tesseract's own line groupings, each with its bounding box.

    Deliberately keeps Tesseract's word-to-line grouping rather than re-deriving
    lines from word positions. Tesseract tracks a baseline per line, so it stays
    correct on a photographed page whose lines sag or tilt across the frame;
    clustering words by vertical centre does not, and drops the last words of a
    sloping line onto the line below. Measured on `IMG_1594`, re-deriving lines
    cost 7.3% -> 18.8% CER.

    What Tesseract gets wrong is coarser: which lines belong to which block, and
    what order the blocks come in. That is what this module repairs, so a *line* is
    the right atom to move around.
    """
    fragments = []
    for _, group in data.groupby(["block_num", "par_num", "line_num"], sort=False):
        fragments.append(
            {
                "text": " ".join(group.sort_values("left")["text"]),
                "x0": float(group["left"].min()),
                "x1": float((group["left"] + group["width"]).max()),
                "y0": float(group["top"].min()),
                "y1": float((group["top"] + group["height"]).max()),
            }
        )
    return fragments


def _same_visual_line(group: list[dict], candidate: dict) -> bool:
    """
    Whether a fragment continues the line an existing group represents.

    Requires vertical overlap *and* horizontal separation: two pieces of one
    sentence sit at the same height and side by side. Fragments that overlap
    horizontally are stacked lines of a paragraph, not one line in pieces.
    """
    top = min(f["y0"] for f in group)
    bottom = max(f["y1"] for f in group)
    overlap = min(bottom, candidate["y1"]) - max(top, candidate["y0"])
    shorter = min(bottom - top, candidate["y1"] - candidate["y0"])
    if shorter <= 0 or overlap < LINE_OVERLAP * shorter:
        return False
    return all(
        f["x1"] <= candidate["x0"] or candidate["x1"] <= f["x0"] for f in group
    )


def _assemble(fragments: list[dict]) -> list[str]:
    """Sort a region's fragments into lines, rejoining any that were split apart."""
    groups: list[list[dict]] = []
    for fragment in sorted(fragments, key=lambda f: (f["y0"] + f["y1"]) / 2):
        if groups and _same_visual_line(groups[-1], fragment):
            groups[-1].append(fragment)
        else:
            groups.append([fragment])

    return [
        " ".join(f["text"] for f in sorted(group, key=lambda f: f["x0"]))
        for group in groups
    ]


def reorder(data: pd.DataFrame) -> str | None:
    """
    Rebuild page text in true reading order from Tesseract's line geometry.

    Tesseract emits text grouped by the blocks its own page segmentation found, and
    on these newspaper adverts those blocks are wrong in one of two ways depending
    on skew -- see `regions.find_regions`. Either way the words are recognised
    correctly and delivered in the wrong sequence, which character error rate
    punishes exactly as hard as not reading them at all.

    Every line comes back with a bounding box, so the page can be re-segmented
    geometrically and the lines re-emitted in the right order with no second OCR
    pass. Two repairs happen here:

    * **Regions are emitted in reading order** -- a whole column at a time, rather
      than Tesseract's block order.
    * **Split lines are rejoined.** Where Tesseract cut one sentence into vertical
      strips, the pieces come back as separate lines that overlap vertically and sit
      side by side. Within a region those are merged back into one line.

    Returns None when the page yields no usable geometry. Ordering is all this
    corrects: a word Tesseract misread stays misread.
    """
    if data.empty:
        return None

    unit = float(data["height"].median())
    if not np.isfinite(unit) or unit <= 0:
        return None

    scale = max(1.0, unit / CELLS_PER_UNIT)
    regions = find_regions(_occupancy(data, scale), unit / scale)
    if not regions:
        return None

    # Words are assigned to regions, but lines are grouped within a region. Both
    # halves matter, because Tesseract's two failure modes are mirror images:
    # when it splits one line across blocks the pieces must be rejoined, and when
    # it merges two columns into a single line that line must be torn apart at the
    # gutter. Assigning per word handles the second; grouping by Tesseract's line
    # number inside the region preserves the baseline tracking that handles the
    # first.
    located = data.assign(_region=_assign_regions(data, regions, scale))

    chunks = []
    for index in range(len(regions)):
        part = located[located["_region"] == index]
        if part.empty:
            continue
        chunks.append("\n".join(_assemble(_fragments(part))))
    return "\n\n".join(chunk for chunk in chunks if chunk).strip() or None


def _assign_regions(data: pd.DataFrame, regions: list[Rect], scale: float) -> list[int]:
    centres_x = (data["left"] + data["width"] / 2) / scale
    centres_y = (data["top"] + data["height"] / 2) / scale

    assigned = []
    for x, y in zip(centres_x, centres_y):
        index = next((i for i, r in enumerate(regions) if r.contains(x, y)), None)
        if index is None:
            # Regions tile the content, not the margins, so a word can sit just
            # outside one. Attach it to the nearest rather than dropping it.
            index = min(
                range(len(regions)),
                key=lambda i: (
                    max(regions[i].x0 - x, 0, x - regions[i].x1) ** 2
                    + max(regions[i].y0 - y, 0, y - regions[i].y1) ** 2
                ),
            )
        assigned.append(index)
    return assigned
