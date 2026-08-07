from __future__ import annotations

from pathlib import Path

# Project-local high-accuracy models, populated by ./scripts/fetch_tessdata.sh.
# Preferred over the system tessdata when present, so results do not depend on
# which language packs happen to be installed system-wide.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESSDATA_DIR = PROJECT_ROOT / "tessdata"


def resolve_tessdata_dir(explicit: str | Path | None = None) -> Path | None:
    """
    Decide which tessdata directory to use.

    An explicit choice always wins. Otherwise use the project-local directory if
    it has been populated, and fall back to Tesseract's own system-wide lookup
    if it has not.
    """
    if explicit:
        return Path(explicit)
    if any(DEFAULT_TESSDATA_DIR.glob("*.traineddata")):
        return DEFAULT_TESSDATA_DIR
    return None
