from __future__ import annotations

from typing import List

import numpy as np

from src.filters.base import BaseFilter


class PreprocessingPipeline:
    """
    Executes a sequence of filters against an image (pipe-and-filter pattern).
    """

    def __init__(self, debug: bool = False) -> None:
        self._filters: List[BaseFilter] = []
        self.debug = debug

    def add_filter(self, img_filter: BaseFilter) -> "PreprocessingPipeline":
        self._filters.append(img_filter)
        return self

    @property
    def filters(self) -> List[BaseFilter]:
        return list(self._filters)

    def run(self, image: np.ndarray) -> np.ndarray:
        processed = image.copy()

        for f in self._filters:
            processed = f.apply(processed)

            if self.debug:
                print(f"Applied filter: {f.name}")

        return processed

