# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

#: Suffix for machine-produced transcriptions.
#:
#: Deliberately not the corpus convention. A hand-made transcription of
#: `IMG_1594.JPEG` is `text-of-IMG_1594.md` (see `eval/corpus.py`), and those are
#: ground truth: the yardstick every CER number in the README is measured against.
#: OCR output written under that name would be the pipeline grading its own
#: homework, so this writes `IMG_1594.txt` instead. `.txt` is also invisible to
#: `corpus.discover`, which globs only `text-of-*.md` and image suffixes -- so
#: pointing `--save-text` at the corpus tree cannot disturb it either.
TEXT_SUFFIX = ".txt"

#: Prefix owned by hand-made transcriptions, which this module refuses to write.
GROUND_TRUTH_PREFIX = "text-of-"


def save_text(path: str | Path, text: str) -> None:
    """
    Write OCR text to `path` as UTF-8, creating the parent directory.

    Refuses any filename in the ground-truth namespace. Nothing in the CLI can
    ask for one -- output names are always `<stem>.txt` -- but the guard makes
    the rule an invariant of the writer rather than a property of its callers,
    so a future caller cannot quietly overwrite a transcription somebody typed
    out by hand.
    """
    path = Path(path)
    if path.name.startswith(GROUND_TRUTH_PREFIX):
        raise ValueError(
            f"Refusing to write {path.name}: the {GROUND_TRUTH_PREFIX}* namespace "
            "belongs to hand-made transcriptions used as ground truth."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    # A page that OCR'd to nothing is an empty file, not a lone newline: the
    # difference is visible in `wc -c` when scanning a batch for failures.
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")


def resolve_text_paths(images: Sequence[str | Path], out_dir: str | Path) -> list[Path]:
    """
    Map each image to `out_dir/<stem>.txt`, returning paths parallel to `images`.

    Raises ValueError if two *different* images share a stem -- `a/IMG_1.JPEG`
    and `b/IMG_1.png` both want `IMG_1.txt`, and the second run would silently
    replace the first page's text with the other's. Checked up front, before any
    OCR runs, so the batch fails in the second it takes to compare names rather
    than after twelve pages of work.
    """
    out_dir = Path(out_dir)
    paths = [out_dir / f"{Path(image).stem}{TEXT_SUFFIX}" for image in images]

    sources: dict[str, set[Path]] = defaultdict(set)
    for image, path in zip(images, paths):
        sources[path.name].add(Path(image).resolve())

    clashes = {name: found for name, found in sources.items() if len(found) > 1}
    if clashes:
        detail = "; ".join(
            f"{name} <- " + ", ".join(str(source) for source in sorted(found))
            for name, found in sorted(clashes.items())
        )
        raise ValueError(
            f"Different images would be written to the same file: {detail}. "
            "Rename them, or run them into separate directories."
        )

    return paths
