# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

"""
Tests for PDF input.

The fixtures build real PDFs in memory rather than checking sample files into the
repository. The documents this feature exists for are scans of personal paperwork --
police clearances, identity records -- and those must not be committed anywhere.
"""

import pytest
import pypdfium2 as pdfium

from acentos_ocr.utils.pdf_io import (
    DEFAULT_DPI,
    MAX_EDGE_PIXELS,
    MIN_TEXT_LAYER_CHARS,
    extract_text_layer,
    has_text_layer,
    is_pdf,
    page_count,
    render_page,
    render_pages,
)


def make_text_pdf(path, text="Se solicita mecanico con experiencia", width=400, height=200):
    """A digital-born PDF: real text, no images."""
    content = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
    path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 "
        + f"{width} {height}".encode()
        + b"]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length " + str(len(content)).encode() + b">>stream\n"
        + content
        + b"\nendstream endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"trailer<</Root 1 0 R/Size 6>>\n%%EOF\n"
    )
    return path


def make_scan_pdf(path, pages=1, page_pt=(400, 200), native=(1600, 800)):
    """
    An image-only PDF, the shape a scanner or a phone produces.

    `native` is deliberately larger than `page_pt`, mirroring scan-to-PDF: the page
    box is small in points while the image inside it holds the real pixels.
    """
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        page = pdf.new_page(*page_pt)
        image = pdfium.PdfImage.new(pdf)
        bitmap = pdfium.PdfBitmap.new_native(*native, pdfium.raw.FPDFBitmap_BGR)
        bitmap.fill_rect((255, 255, 255, 255), 0, 0, *native)
        bitmap.fill_rect((0, 0, 0, 255), 40, 40, native[0] // 4, native[1] // 4)
        image.set_bitmap(bitmap)
        image.set_matrix(pdfium.PdfMatrix().scale(*page_pt))
        page.insert_obj(image)
        page.gen_content()
    pdf.save(str(path))
    return path


# --- is_pdf -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("scan.pdf", True),
        ("SCAN.PDF", True),
        ("IMG_1594.JPEG", False),
        ("IMG_1594.png", False),
        ("notes.md", False),
    ],
)
def test_is_pdf_recognises_the_suffix_case_insensitively(name, expected):
    assert is_pdf(name) is expected


# --- text layer -------------------------------------------------------------


def test_a_digital_born_pdf_reports_its_text_layer(tmp_path):
    pdf = make_text_pdf(tmp_path / "letter.pdf", "Se solicita mecanico con experiencia")

    assert has_text_layer(pdf) is True
    assert extract_text_layer(pdf) == ["Se solicita mecanico con experiencia"]


def test_a_scan_has_no_text_layer(tmp_path):
    """The whole point of the check: this is the file that needs OCR."""
    pdf = make_scan_pdf(tmp_path / "scan.pdf")

    assert has_text_layer(pdf) is False
    assert extract_text_layer(pdf) == [""]


def test_a_trivial_amount_of_text_does_not_count_as_a_text_layer(tmp_path):
    """
    A stray annotation, or a bad OCR layer some other tool left behind, can leave a
    few characters on a page that is really a scan. Taking those as the page's text
    would return a handful of junk instead of transcribing the image.
    """
    pdf = make_text_pdf(tmp_path / "stray.pdf", "x")

    assert len("x") < MIN_TEXT_LAYER_CHARS
    assert has_text_layer(pdf) is False


def test_extract_text_layer_returns_one_entry_per_page(tmp_path):
    pdf = make_scan_pdf(tmp_path / "three.pdf", pages=3)

    assert page_count(pdf) == 3
    assert extract_text_layer(pdf) == ["", "", ""]


# --- rendering --------------------------------------------------------------


def test_pages_render_as_contiguous_three_channel_bgr(tmp_path):
    """The pipeline's filters assume what load_image gives them."""
    pdf = make_scan_pdf(tmp_path / "scan.pdf")

    (number, image), = render_pages(pdf)

    assert number == 1
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype.name == "uint8"
    assert image.flags["C_CONTIGUOUS"]


def test_the_default_render_matches_the_scans_own_resolution(tmp_path):
    """
    Rendering above native interpolates: more pixels, no more detail, slower OCR.
    A 400x200pt page holding a 1600x800 image should come back 1600 wide, not
    400/72*300 = 1666.
    """
    pdf = make_scan_pdf(tmp_path / "scan.pdf", page_pt=(400, 200), native=(1600, 800))

    image = render_page(pdf, 1)

    assert image.shape[1] == pytest.approx(1600, abs=2)


def test_an_explicit_dpi_overrides_the_native_resolution(tmp_path):
    pdf = make_scan_pdf(tmp_path / "scan.pdf", page_pt=(400, 200), native=(1600, 800))

    image = render_page(pdf, 1, dpi=72)

    assert image.shape[1] == pytest.approx(400, abs=2)


def test_a_page_with_no_usable_image_falls_back_to_the_default_dpi(tmp_path):
    """A text-only page has no native resolution to honour."""
    pdf = make_text_pdf(tmp_path / "letter.pdf", width=400, height=200)

    image = render_page(pdf, 1)

    assert image.shape[1] == pytest.approx(400 / 72 * DEFAULT_DPI, abs=2)


def test_an_absurd_page_box_is_clamped(tmp_path):
    """
    Phone scan-to-PDF sizes the page to the pixels at 72 DPI, so a sheet can declare
    itself feet wide. Asking for 300 DPI on one of those would otherwise render tens
    of megapixels of interpolation.
    """
    pdf = make_scan_pdf(tmp_path / "huge.pdf", page_pt=(2054, 2787), native=(2053, 2785))

    image = render_page(pdf, 1, dpi=300)

    assert max(image.shape[:2]) == pytest.approx(MAX_EDGE_PIXELS, abs=2)


def test_render_pages_yields_every_page_in_order(tmp_path):
    pdf = make_scan_pdf(tmp_path / "three.pdf", pages=3)

    assert [number for number, _ in render_pages(pdf)] == [1, 2, 3]


def test_asking_for_a_page_that_does_not_exist_is_an_error(tmp_path):
    pdf = make_scan_pdf(tmp_path / "scan.pdf", pages=2)

    with pytest.raises(IndexError, match="has 2 page"):
        render_page(pdf, 3)

    with pytest.raises(IndexError):
        render_page(pdf, 0)
