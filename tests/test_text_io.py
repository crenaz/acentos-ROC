# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

import pytest
from acentos_ocr.eval.corpus import discover
from acentos_ocr.utils.text_io import resolve_text_paths, save_text

from test_corpus import make_corpus


def test_save_text_writes_utf8_and_creates_the_directory(tmp_path):
    target = tmp_path / "out" / "IMG_1594.txt"
    save_text(target, "Se solicita mecánico\nBuen sueldo")

    assert target.read_text(encoding="utf-8") == "Se solicita mecánico\nBuen sueldo\n"


def test_a_page_that_ocrd_to_nothing_is_an_empty_file(tmp_path):
    """Visible in `wc -c` when scanning a batch for failures."""
    target = tmp_path / "IMG_1594.txt"
    save_text(target, "")

    assert target.read_bytes() == b""


def test_save_text_refuses_the_ground_truth_namespace(tmp_path):
    """Machine output must never be able to overwrite a hand-made transcription."""
    truth = tmp_path / "text-of-IMG_1594.md"
    truth.write_text("typed out by hand", encoding="utf-8")

    with pytest.raises(ValueError, match="ground truth"):
        save_text(truth, "whatever the OCR produced")

    assert truth.read_text(encoding="utf-8") == "typed out by hand"


def test_resolve_text_paths_names_each_output_after_its_image(tmp_path):
    paths = resolve_text_paths(
        [tmp_path / "July18" / "IMG_1594.JPEG", tmp_path / "IMG_1595.png"],
        tmp_path / "out",
    )

    assert [p.name for p in paths] == ["IMG_1594.txt", "IMG_1595.txt"]
    assert all(p.parent == tmp_path / "out" for p in paths)


def test_resolve_text_paths_rejects_images_that_would_share_an_output(tmp_path):
    """Caught before any OCR runs, not after twelve pages of work."""
    with pytest.raises(ValueError, match="same file"):
        resolve_text_paths(
            [tmp_path / "a" / "IMG_1.JPEG", tmp_path / "b" / "IMG_1.png"],
            tmp_path / "out",
        )


def test_the_same_image_listed_twice_is_not_a_clash(tmp_path):
    repeated = tmp_path / "IMG_1.JPEG"
    assert resolve_text_paths([repeated, repeated], tmp_path / "out") == [
        tmp_path / "out" / "IMG_1.txt",
        tmp_path / "out" / "IMG_1.txt",
    ]


def test_saved_text_is_invisible_to_corpus_discovery(tmp_path):
    """
    Pointing --save-text at the corpus tree must not disturb the benchmark:
    `.txt` is neither an image suffix nor a `text-of-*.md` transcription.
    """
    make_corpus(tmp_path, ["IMG_1594"])
    before = discover(tmp_path)

    for path in resolve_text_paths([tmp_path / "IMG_1594.JPEG"], tmp_path):
        save_text(path, "machine output")

    samples, unmatched = discover(tmp_path)
    assert [s.stem for s in samples] == [s.stem for s in before[0]] == ["IMG_1594"]
    assert unmatched == before[1] == []
    assert samples[0].read_truth().strip() == "# IMG_1594"
