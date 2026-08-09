# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A gutter must be at least this many median-word-heights wide to count as a
#: column separator rather than a wide inter-word space.
MIN_GUTTER_SCALE = 2.5

#: A horizontal gap must be at least this many median-word-heights tall to separate
#: one band from the next. Gaps between lines of a paragraph are smaller than this;
#: gaps between sections are larger.
MIN_BAND_GAP_SCALE = 0.7

#: Both sides of a vertical cut must be at least this wide, which is what stops a
#: bullet glyph from being severed from the text it introduces.
MIN_COLUMN_SCALE = 4.0

#: Depth is consumed one cut at a time, so a column of twenty paragraphs needs
#: twenty levels. Generous, and only a guard against a pathological page.
MAX_DEPTH = 64


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


def _widest_interior_gap(occupied: np.ndarray, minimum: int) -> tuple[int, int] | None:
    """
    The widest run of unoccupied positions with content on both sides.

    Leading and trailing runs are ignored: those are margins, and trimming them is
    the caller's job, not a cut.
    """
    if occupied.size == 0:
        return None
    padded = np.concatenate(([True], occupied, [True]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])

    best = None
    for start, end in zip(edges[::2], edges[1::2]):
        if start == 0 or end == len(occupied):
            continue
        if end - start >= minimum and (best is None or end - start > best[1] - best[0]):
            best = (int(start), int(end))
    return best


def _trim(occupancy: np.ndarray, rect: Rect) -> Rect | None:
    """Shrink a rectangle to the content inside it, discarding blank margin."""
    window = occupancy[rect.y0 : rect.y1, rect.x0 : rect.x1]
    if not window.any():
        return None
    rows = np.flatnonzero(window.any(axis=1))
    cols = np.flatnonzero(window.any(axis=0))
    return Rect(
        rect.x0 + int(cols[0]),
        rect.y0 + int(rows[0]),
        rect.x0 + int(cols[-1]) + 1,
        rect.y0 + int(rows[-1]) + 1,
    )


def find_regions(occupancy: np.ndarray, unit: float) -> list[Rect]:
    """
    Recursively cut a page into regions and return them in reading order.

    An XY-cut with two deliberate choices.

    **One cut at a time, at the most significant gap.** Splitting at every
    qualifying gap at once looks equivalent and is not: on a mixed layout the page
    would break into horizontal bands *before* anything noticed the column gutter,
    and each band would then be split into left and right independently. The output
    interleaves the columns a few lines at a time -- which is the very failure this
    module exists to repair. Taking the widest gap first lets the two-column body
    survive as one region long enough to be cut down the gutter as a whole.

    **Vertical cuts win ties.** Where a gutter and a band gap are equally wide, the
    gutter is the more meaningful boundary.

    On the newspaper adverts in this corpus -- a full-width headline and intro, a
    two-column body, then a full-width footer -- this produces: no full-height
    gutter at page level, so the widest section gap splits the page horizontally;
    recursion isolates the body; the body has a full-height gutter and splits into
    two columns; each column then splits into paragraphs.

    Tesseract's own segmentation gets this wrong in both directions -- without
    deskew it applies the column split to the entire page and shreds the full-width
    paragraphs, with deskew it reads the two-column body as full width and
    interleaves the columns line by line.

    `unit` is the median word height, in the same cell units as `occupancy`, and
    every threshold is expressed as a multiple of it so the constants hold at any
    resolution.
    """
    page = _trim(occupancy, Rect(0, 0, occupancy.shape[1], occupancy.shape[0]))
    return [] if page is None else _cut(occupancy, page, unit, 0)


def _cut(occupancy: np.ndarray, rect: Rect, unit: float, depth: int) -> list[Rect]:
    if depth >= MAX_DEPTH:
        return [rect]

    window = occupancy[rect.y0 : rect.y1, rect.x0 : rect.x1]
    gutter = _widest_interior_gap(
        window.any(axis=0), max(1, int(MIN_GUTTER_SCALE * unit))
    )
    if gutter is not None:
        left = gutter[0]
        right = rect.width - gutter[1]
        if min(left, right) < MIN_COLUMN_SCALE * unit:
            gutter = None  # a bullet or a marginal mark, not a column boundary

    band = _widest_interior_gap(
        window.any(axis=1), max(1, int(MIN_BAND_GAP_SCALE * unit))
    )

    if gutter is not None and (band is None or gutter[1] - gutter[0] >= band[1] - band[0]):
        halves = [
            Rect(rect.x0, rect.y0, rect.x0 + gutter[0], rect.y1),
            Rect(rect.x0 + gutter[1], rect.y0, rect.x1, rect.y1),
        ]
    elif band is not None:
        halves = [
            Rect(rect.x0, rect.y0, rect.x1, rect.y0 + band[0]),
            Rect(rect.x0, rect.y0 + band[1], rect.x1, rect.y1),
        ]
    else:
        return [rect]

    regions = []
    for half in halves:
        trimmed = _trim(occupancy, half)
        if trimmed is not None:
            regions.extend(_cut(occupancy, trimmed, unit, depth + 1))
    return regions
