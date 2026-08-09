# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

"""
Tests for DeskewFilter.

The original implementation passed (y, x) points to cv2.minAreaRect, which expects
(x, y). That mirrors the page across its diagonal and inverts the sign of the
measured angle, so the filter rotated the wrong way and doubled the skew. The bug
was invisible by inspection -- the output looked plausible and the reported angle
was in a believable range -- so these tests measure the residual skew of the
*result* rather than asserting on any intermediate value.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from acentos_ocr.filters.deskew import DeskewFilter


def make_text_page(skew_deg: float, width: int = 1200, height: int = 800) -> np.ndarray:
    """A white page of black horizontal bars standing in for lines of text."""
    page = np.full((height, width), 255, np.uint8)
    for y in range(100, height - 100, 45):
        cv2.rectangle(page, (150, y), (width - 150, y + 18), 0, -1)

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), skew_deg, 1.0)
    return cv2.warpAffine(
        page, matrix, (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def measure_skew(gray: np.ndarray) -> float:
    """Dominant text angle, folded into (-45, 45]. Independent of the filter."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows, cols = np.where(binary > 0)
    angle = float(cv2.minAreaRect(np.column_stack((cols, rows)).astype(np.float32))[-1])
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return angle


@pytest.mark.parametrize("skew", [-8.0, -6.0, -3.0, -1.0, 1.0, 3.0, 6.0, 8.0])
def test_deskew_straightens_a_skewed_page(skew: float) -> None:
    """The result should be close to upright, whichever way the page was tilted."""
    result = DeskewFilter().apply(make_text_page(skew))
    assert abs(measure_skew(result)) < 1.0


@pytest.mark.parametrize("skew", [-6.0, -3.0, 3.0, 6.0])
def test_deskew_reduces_skew_rather_than_amplifying_it(skew: float) -> None:
    """
    Regression test for the transposed-coordinates bug.

    Before the fix the residual was consistently -2x the input: a +6 deg page came
    out at -12 deg. Asserting the residual is smaller than the input catches any
    return to that behaviour, even if the exact numbers change.
    """
    page = make_text_page(skew)
    result = DeskewFilter().apply(page)
    assert abs(measure_skew(result)) < abs(measure_skew(page))


def test_deskew_leaves_an_upright_page_alone() -> None:
    """Below min_angle_deg the filter should return the image untouched."""
    page = make_text_page(0.0)
    assert np.array_equal(DeskewFilter().apply(page), page)


def test_deskew_ignores_skew_beyond_max_angle() -> None:
    """
    A rotation larger than max_angle_deg is not skew, it is a landscape page or a
    detection failure. Clamping to the limit would bake in a wrong rotation, so the
    filter should decline rather than guess.
    """
    page = make_text_page(40.0)
    result = DeskewFilter(max_angle_deg=15.0).apply(page)
    assert np.array_equal(result, page)


def test_deskew_returns_image_when_too_few_ink_pixels() -> None:
    """A blank page has nothing to measure; it should pass through unchanged."""
    blank = np.full((400, 400), 255, np.uint8)
    assert np.array_equal(DeskewFilter().apply(blank), blank)


def test_deskew_rejects_colour_input() -> None:
    with pytest.raises(ValueError):
        DeskewFilter().apply(np.zeros((100, 100, 3), np.uint8))


def make_photo_like_page(skew_deg: float) -> np.ndarray:
    """
    A skewed page over a noisy, unevenly lit background with a hard border.

    This is what defeated the previous minAreaRect estimator: Otsu marks the
    background and the page edge as ink, so the minimum-area rectangle spans the
    whole frame. The projection profile is driven by text-line periodicity and is
    unaffected by it.
    """
    rng = np.random.default_rng(0)
    frame = rng.integers(90, 150, (900, 1300), dtype=np.uint8)          # noisy background
    frame[:, :] = cv2.GaussianBlur(frame, (31, 31), 0)                   # uneven lighting
    page = make_text_page(skew_deg, width=1000, height=700)
    frame[100:800, 150:1150] = page
    cv2.rectangle(frame, (150, 100), (1150, 800), 0, 6)                  # hard page edge
    return frame


@pytest.mark.parametrize("skew", [-5.0, -2.0, 2.0, 5.0])
def test_deskew_works_over_a_noisy_background(skew: float) -> None:
    """Finding #11: the estimator must survive a photographic background."""
    estimated = DeskewFilter().estimate_angle(make_photo_like_page(skew))
    assert estimated is not None
    assert abs(estimated + skew) < 1.0


def test_estimate_angle_declines_when_rotation_exceeds_search_range() -> None:
    """No trial angle straightens a 40 deg page, so the peak never stands out."""
    assert DeskewFilter(max_angle_deg=15.0).estimate_angle(make_text_page(40.0)) is None


def test_estimate_angle_declines_on_a_blank_page() -> None:
    assert DeskewFilter().estimate_angle(np.full((400, 400), 255, np.uint8)) is None
