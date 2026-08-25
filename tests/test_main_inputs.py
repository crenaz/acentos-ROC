# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

"""
Tests for flattening the command line into per-page work items.

This is the seam that keeps PDFs out of the rest of main(): once inputs are expanded,
output naming, clash detection and the batch summary all operate on items and never
ask where the pixels came from.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import expand_inputs  # noqa: E402

from acentos_ocr.utils.text_io import resolve_text_paths  # noqa: E402

from test_pdf_io import make_scan_pdf, make_text_pdf  # noqa: E402


def test_an_image_expands_to_a_single_item():
    (item,) = expand_inputs([Path("IMG_1594.JPEG")], pdf_dpi=None, force_ocr=False)

    assert item.label == "IMG_1594.JPEG"
    assert item.name == Path("IMG_1594.JPEG")
    assert item.text is None
    assert item.load is not None


def test_a_scanned_pdf_expands_to_one_item_per_page(tmp_path):
    pdf = make_scan_pdf(tmp_path / "Police_Clearance.pdf", pages=3)

    items = expand_inputs([pdf], pdf_dpi=None, force_ocr=False)

    assert [item.label for item in items] == [
        "Police_Clearance.pdf page 1",
        "Police_Clearance.pdf page 2",
        "Police_Clearance.pdf page 3",
    ]
    assert all(item.text is None and item.load is not None for item in items)


def test_pdf_pages_get_distinct_zero_padded_stems(tmp_path):
    """
    They flow into resolve_text_paths like any other input, so they must not collide
    -- and they sort correctly in a directory listing.
    """
    pdf = make_scan_pdf(tmp_path / "scan.pdf", pages=3)

    items = expand_inputs([pdf], pdf_dpi=None, force_ocr=False)
    written = resolve_text_paths([item.name for item in items], tmp_path / "out")

    assert [path.name for path in written] == [
        "scan-p01.txt",
        "scan-p02.txt",
        "scan-p03.txt",
    ]


def test_a_digital_born_pdf_carries_its_text_instead_of_an_image(tmp_path):
    """OCR on text that is already exact could only introduce errors."""
    pdf = make_text_pdf(tmp_path / "letter.pdf", "Se solicita mecanico con experiencia")

    (item,) = expand_inputs([pdf], pdf_dpi=None, force_ocr=False)

    assert item.text == "Se solicita mecanico con experiencia"
    assert item.load is None


def test_force_ocr_ignores_a_text_layer(tmp_path):
    """For files where another tool left a bad text layer behind."""
    pdf = make_text_pdf(tmp_path / "letter.pdf", "Se solicita mecanico con experiencia")

    (item,) = expand_inputs([pdf], pdf_dpi=None, force_ocr=True)

    assert item.text is None
    assert item.load is not None


def test_mixed_inputs_keep_their_command_line_order(tmp_path):
    pdf = make_scan_pdf(tmp_path / "scan.pdf", pages=2)

    items = expand_inputs(
        [Path("a.png"), pdf, Path("b.png")], pdf_dpi=None, force_ocr=False
    )

    assert [item.label for item in items] == [
        "a.png",
        "scan.pdf page 1",
        "scan.pdf page 2",
        "b.png",
    ]


def test_the_loader_is_lazy(tmp_path):
    """
    Nothing is rendered during expansion. A fifty-page scan at native resolution is
    comfortably a gigabyte of arrays, and the pipeline only ever looks at one.
    """
    pdf = make_scan_pdf(tmp_path / "scan.pdf", pages=2, native=(1600, 800))

    items = expand_inputs([pdf], pdf_dpi=None, force_ocr=False)
    image = items[0].load()

    assert image.shape[1] == pytest.approx(1600, abs=2)
