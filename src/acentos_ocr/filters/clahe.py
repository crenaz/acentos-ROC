# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFilter


class CLAHEFilter(BaseFilter):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Improves local contrast and often helps Tesseract on photos with uneven lighting.
    """

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8) -> None:
        super().__init__(name="CLAHE")
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size),
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            raise ValueError("CLAHEFilter expects a grayscale image.")
        return self._clahe.apply(image)
