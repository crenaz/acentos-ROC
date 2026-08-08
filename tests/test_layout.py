import numpy as np
import pandas as pd
import pytest

from acentos_ocr.layout.reading_order import reorder
from acentos_ocr.layout.regions import Rect, find_regions, has_columns

#: Occupancy grids below are expressed in cells, and `reading_order` rasterises at
#: 8 cells per median word height, so a "word height" here is 8 cells.
UNIT = 8.0


def grid(height=300, width=200):
    return np.zeros((height, width), dtype=bool)


def fill(g, x0, y0, x1, y1):
    g[y0:y1, x0:x1] = True
    return g


# --------------------------------------------------------------------------
# region detection
# --------------------------------------------------------------------------

def test_a_single_block_of_text_is_one_region():
    g = fill(grid(), 10, 10, 190, 290)
    assert find_regions(g, UNIT) == [Rect(10, 10, 190, 290)]


def test_margins_are_trimmed_not_treated_as_a_cut():
    g = fill(grid(), 30, 40, 160, 270)
    assert find_regions(g, UNIT) == [Rect(30, 40, 160, 270)]


def test_two_columns_are_returned_left_to_right():
    g = grid()
    fill(g, 0, 0, 70, 300)
    fill(g, 130, 0, 200, 300)  # 60-cell gutter, columns well over the minimum width
    regions = find_regions(g, UNIT)

    assert len(regions) == 2
    assert regions[0].x1 <= regions[1].x0


def test_stacked_bands_are_returned_top_to_bottom():
    g = grid()
    fill(g, 0, 0, 200, 60)
    fill(g, 0, 160, 200, 300)
    regions = find_regions(g, UNIT)

    assert len(regions) == 2
    assert regions[0].y1 <= regions[1].y0


def test_a_narrow_gap_is_not_a_column_boundary():
    """A bullet glyph sits close to its text; that spacing is not a gutter."""
    g = grid()
    fill(g, 10, 10, 16, 290)   # bullet strip
    fill(g, 26, 10, 190, 290)  # the text it introduces, 10 cells away
    assert len(find_regions(g, UNIT)) == 1


def test_a_wide_gap_beside_a_narrow_strip_is_not_a_column_boundary():
    """Both sides of a vertical cut must be wide enough to be real columns."""
    g = grid()
    fill(g, 0, 10, 6, 290)     # a stray mark in the margin
    fill(g, 80, 10, 200, 290)  # the actual text
    assert len(find_regions(g, UNIT)) == 1


def test_mixed_layout_keeps_full_width_text_whole_and_splits_only_the_body():
    """
    The layout this module exists for: full-width header, two-column body,
    full-width footer. Header and footer must survive intact, and the body must
    come out as all-of-left then all-of-right, never interleaved.
    """
    g = grid()
    fill(g, 0, 0, 200, 30)      # header, full width
    fill(g, 0, 60, 70, 200)     # body, left column
    fill(g, 130, 60, 200, 200)  # body, right column
    fill(g, 0, 240, 200, 290)   # footer, full width

    header, left, right, footer = find_regions(g, UNIT)
    assert header.width > 190 and header.y1 <= 60
    assert left.x1 <= right.x0      # left column precedes right
    assert left.y0 == right.y0      # and they sit side by side
    assert footer.width > 190 and footer.y0 >= 200


def test_an_empty_page_yields_no_regions():
    assert find_regions(grid(), UNIT) == []


# --------------------------------------------------------------------------
# the single-column guard
# --------------------------------------------------------------------------

def test_has_columns_detects_side_by_side_regions():
    assert has_columns([Rect(0, 0, 40, 100), Rect(60, 0, 100, 100)])


def test_has_columns_rejects_stacked_regions():
    assert not has_columns([Rect(0, 0, 100, 40), Rect(0, 60, 100, 100)])


def test_has_columns_rejects_a_barely_touching_overlap():
    """A footer brushing the bottom of a column is not a second column."""
    assert not has_columns([Rect(0, 0, 40, 100), Rect(60, 95, 100, 200)])


# --------------------------------------------------------------------------
# reordering
# --------------------------------------------------------------------------

HEIGHT = 20
CHAR = 11    # approximate glyph advance at that height
SPACE = 7    # inter-word gap: far narrower than the 50px gutter threshold

HEADER = "we are seeking an experienced specialist to lead the systems team"
LEFT = ["a bachelors degree in a related area", "five years of relevant experience"]
RIGHT = ["manage the technical support network", "provide training to existing users"]


def _line(text, x, y, jitter=0):
    words, cursor = [], x
    for index, word in enumerate(text.split()):
        width = CHAR * len(word)
        words.append({
            "text": word,
            "left": cursor,
            "top": y + (jitter if index % 2 else 0),
            "width": width,
            "height": HEIGHT,
        })
        cursor += width + SPACE
    return words


def page(*lines, shuffle=False):
    """Build a Tesseract-shaped word dataframe from (text, x, y) line specs."""
    rows = [w for text, x, y in lines for w in _line(text, x, y)]
    frame = pd.DataFrame(rows)
    # Row order must not matter -- geometry is the only input that should count.
    return frame.sample(frac=1, random_state=0) if shuffle else frame


def two_column_page(**kwargs):
    lines = [(HEADER, 0, 0)]
    lines += [(text, 0, 120 + 40 * i) for i, text in enumerate(LEFT)]
    lines += [(text, 700, 120 + 40 * i) for i, text in enumerate(RIGHT)]
    return page(*lines, **kwargs)


def test_reorder_declines_on_a_single_column_page():
    """
    None tells the caller to keep Tesseract's ordering. With no columns to repair,
    re-deriving order from geometry is a pure loss -- measured at 7.3% -> 18.8%
    CER on IMG_1594.
    """
    assert reorder(page(("alpha beta gamma", 0, 0), ("delta epsilon", 0, 60))) is None


def test_reorder_declines_on_an_empty_page():
    empty = pd.DataFrame(columns=["text", "left", "top", "width", "height"])
    assert reorder(empty) is None


def test_reorder_keeps_a_full_width_line_whole():
    """
    The IMG_1595 failure: a full-width sentence cut across column blocks and
    emitted in three distant pieces. It must come back as one line.
    """
    assert reorder(two_column_page()).splitlines()[0] == HEADER


def test_reorder_reads_each_column_fully_before_the_next():
    lines = reorder(two_column_page()).splitlines()
    assert [line for line in lines if line] == [HEADER, *LEFT, *RIGHT]


def test_reorder_depends_on_geometry_not_dataframe_row_order():
    assert reorder(two_column_page(shuffle=True)) == reorder(two_column_page())


@pytest.mark.parametrize("jitter", [0, 3, 6])
def test_reorder_tolerates_small_baseline_jitter_within_a_line(jitter):
    """Word boxes on one line vary by a few pixels; that must not split the line."""
    lines = [(HEADER, 0, 0)]
    lines += [(text, 0, 120 + 40 * i) for i, text in enumerate(LEFT)]
    lines += [(text, 700, 120 + 40 * i) for i, text in enumerate(RIGHT)]
    rows = [w for text, x, y in lines for w in _line(text, x, y, jitter=jitter)]

    assert reorder(pd.DataFrame(rows)).splitlines()[0] == HEADER
