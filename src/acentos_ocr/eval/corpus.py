# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}

#: Environment variable pointing at the corpus root, so the scripts carry no
#: machine-specific absolute path.
CORPUS_ENV_VAR = "ACENTOS_CORPUS"

#: Optional file in the corpus root listing image stems that are deliberately not
#: transcribed, one per line, `#` for comments. Not every photograph in a corpus
#: directory is a sample -- there are reference shots, mastheads, duplicates -- and
#: those should be distinguishable from a transcription somebody still owes.
IGNORE_FILE = ".corpus-ignore"


@dataclass(frozen=True)
class Sample:
    """An image paired with its manual transcription."""

    image: Path
    truth: Path

    @property
    def stem(self) -> str:
        return self.image.stem

    def read_truth(self) -> str:
        return self.truth.read_text(encoding="utf-8")


def default_root() -> Path | None:
    root = os.environ.get(CORPUS_ENV_VAR)
    return Path(root) if root else None


def read_ignored(root: Path) -> set[str]:
    """Image stems the corpus deliberately excludes, from the optional ignore file."""
    path = Path(root) / IGNORE_FILE
    if not path.is_file():
        return set()
    return {
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def discover(root: Path) -> tuple[list[Sample], list[Path]]:
    """
    Pair every image under `root` with its transcription.

    Convention: an image `IMG_1594.JPEG` is transcribed in `text-of-IMG_1594.md`.
    Rather than assuming a fixed directory depth, this indexes every
    `text-of-*.md` beneath the root and matches on stem, so the photo tree can be
    reorganised without breaking the pairing.

    Returns the matched samples and any images that have no transcription yet --
    the latter are reported rather than silently skipped, because a corpus quietly
    shrinking is how a benchmark stops meaning anything.

    Images listed in `.corpus-ignore` are left out of both. A photograph that is
    not a sample -- a masthead shot kept for provenance, say -- should not be
    deleted to quiet the warning, and should not keep raising one either.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus root does not exist: {root}")

    ignored = read_ignored(root)
    truths = {
        path.stem.removeprefix("text-of-"): path
        for path in root.rglob("text-of-*.md")
    }

    samples: list[Sample] = []
    unmatched: list[Path] = []
    for image in sorted(root.rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES or image.stem in ignored:
            continue
        truth = truths.get(image.stem)
        if truth is None:
            unmatched.append(image)
        else:
            samples.append(Sample(image=image, truth=truth))

    return samples, unmatched
