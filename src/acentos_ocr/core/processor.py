from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..filters.base import BaseFilter
from ..utils.image_io import save_image


class PreprocessingPipeline:
    """
    Executes a sequence of filters against an image (pipe-and-filter pattern).
    """

    def __init__(
        self,
        debug: bool = False,
        debug_dir: str | Path | None = None,
    ) -> None:
        self._filters: List[BaseFilter] = []
        self.debug = debug
        self.debug_dir = Path(debug_dir) if debug_dir else None

    def add_filter(self, img_filter: BaseFilter) -> "PreprocessingPipeline":
        self._filters.append(img_filter)
        return self

    @property
    def filters(self) -> List[BaseFilter]:
        return list(self._filters)

    def run(self, image: np.ndarray, name: str = "image") -> np.ndarray:
        """
        Apply every filter in order and return the result.

        When debug_dir is set, each intermediate image is written out as
        <name>_<NN>_<FilterName>.png, starting with _00_source. Diagnosing a bad
        result means seeing which stage degraded it, which a single final image
        cannot show.
        """
        processed = image.copy()

        self._log_step(0, "source", processed)
        self._save_step(name, 0, "source", processed)

        for index, img_filter in enumerate(self._filters, start=1):
            processed = img_filter.apply(processed)
            self._log_step(index, img_filter.name, processed)
            self._save_step(name, index, img_filter.name, processed)

        return processed

    def _log_step(self, index: int, step: str, image: np.ndarray) -> None:
        if not self.debug:
            return
        # Shape and dtype make channel drops and type changes visible, which are a
        # common cause of a filter silently doing nothing useful.
        shape = "x".join(str(d) for d in image.shape)
        label = "source" if index == 0 else f"applied {step}"
        print(f"  [{index:02d}] {label:<28} {shape:<16} {image.dtype}")

    def _save_step(self, name: str, index: int, step: str, image: np.ndarray) -> None:
        if self.debug_dir is None:
            return
        save_image(self.debug_dir / f"{name}_{index:02d}_{step}.png", image)
