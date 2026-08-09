# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

import numpy as np
import pytest
from acentos_ocr.eval.corpus import IGNORE_FILE, Sample, discover, read_ignored
from acentos_ocr.utils.image_io import save_image


def make_corpus(root, stems_with_truth, stems_without_truth=()):
    """Build a corpus tree matching the real one: photos nested, transcriptions flat."""
    photos = root / "Raw-Photos-Of-Cayman-Job-Listings" / "July18"
    photos.mkdir(parents=True)
    blank = np.full((8, 8, 3), 255, dtype=np.uint8)

    for stem in list(stems_with_truth) + list(stems_without_truth):
        save_image(photos / f"{stem}.JPEG", blank)
    for stem in stems_with_truth:
        (root / f"text-of-{stem}.md").write_text(f"# {stem}\n", encoding="utf-8")
    return root


def test_discover_pairs_images_with_transcriptions(tmp_path):
    make_corpus(tmp_path, ["IMG_1594", "IMG_1595"])
    samples, unmatched = discover(tmp_path)

    assert [s.stem for s in samples] == ["IMG_1594", "IMG_1595"]
    assert unmatched == []
    assert samples[0].read_truth().strip() == "# IMG_1594"


def test_discover_reports_untranscribed_images_rather_than_hiding_them(tmp_path):
    """A benchmark that silently shrinks stops meaning anything."""
    make_corpus(tmp_path, ["IMG_1594"], stems_without_truth=["IMG_1600"])
    samples, unmatched = discover(tmp_path)

    assert [s.stem for s in samples] == ["IMG_1594"]
    assert [p.name for p in unmatched] == ["IMG_1600.JPEG"]


def test_discover_matches_on_stem_not_directory_depth(tmp_path):
    """Photos can be reorganised without breaking the pairing."""
    make_corpus(tmp_path, ["IMG_1594"])
    deeper = tmp_path / "Raw-Photos-Of-Cayman-Job-Listings" / "July26" / "extra"
    deeper.mkdir(parents=True)
    save_image(deeper / "IMG_1595.JPEG", np.full((8, 8, 3), 255, dtype=np.uint8))
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "text-of-IMG_1595.md").write_text("hi", encoding="utf-8")

    samples, unmatched = discover(tmp_path)
    assert sorted(s.stem for s in samples) == ["IMG_1594", "IMG_1595"]
    assert unmatched == []


def test_discover_ignores_non_image_files(tmp_path):
    make_corpus(tmp_path, ["IMG_1594"])
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    samples, unmatched = discover(tmp_path)
    assert len(samples) == 1
    assert unmatched == []


def test_discover_rejects_a_missing_root(tmp_path):
    with pytest.raises(NotADirectoryError):
        discover(tmp_path / "nope")


def test_sample_stem_comes_from_the_image(tmp_path):
    sample = Sample(image=tmp_path / "IMG_1.JPEG", truth=tmp_path / "text-of-IMG_1.md")
    assert sample.stem == "IMG_1"


def test_ignored_images_are_neither_samples_nor_reported_as_missing(tmp_path):
    """
    A photograph that is not a sample -- a masthead kept for provenance -- should
    not have to be deleted to quiet the warning, nor keep raising one.
    """
    make_corpus(tmp_path, ["IMG_1594"], stems_without_truth=["IMG_1600", "IMG_1601"])
    (tmp_path / IGNORE_FILE).write_text("IMG_1600\n", encoding="utf-8")

    samples, unmatched = discover(tmp_path)
    assert [s.stem for s in samples] == ["IMG_1594"]
    assert [p.stem for p in unmatched] == ["IMG_1601"]


def test_the_ignore_file_supports_comments_and_blank_lines(tmp_path):
    make_corpus(tmp_path, ["IMG_1594"], stems_without_truth=["IMG_1600"])
    (tmp_path / IGNORE_FILE).write_text(
        "# not samples\n\nIMG_1600   # masthead\n", encoding="utf-8")

    assert read_ignored(tmp_path) == {"IMG_1600"}
    assert discover(tmp_path)[1] == []


def test_a_missing_ignore_file_is_fine(tmp_path):
    make_corpus(tmp_path, ["IMG_1594"])
    assert read_ignored(tmp_path) == set()
