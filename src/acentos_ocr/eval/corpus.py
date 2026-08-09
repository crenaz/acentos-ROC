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


def discover(root: Path) -> tuple[list[Sample], list[Path]]:
    """
    Pair every image under `root` with its transcription.

    Convention: an image `IMG_1594.JPEG` is transcribed in `text-of-IMG_1594.md`.
    Rather than assuming a fixed directory depth, this indexes every
    `text-of-*.md` beneath the root and matches on stem, so the photo tree can be
    reorganised without breaking the pairing.

    Returns the matched samples and any images that have no transcription yet --
    the latter are reported rather than silently skipped, because a corpus
    quietly shrinking is how a benchmark stops meaning anything.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus root does not exist: {root}")

    truths = {
        path.stem.removeprefix("text-of-"): path
        for path in root.rglob("text-of-*.md")
    }

    samples: list[Sample] = []
    unmatched: list[Path] = []
    for image in sorted(root.rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        truth = truths.get(image.stem)
        if truth is None:
            unmatched.append(image)
        else:
            samples.append(Sample(image=image, truth=truth))

    return samples, unmatched
