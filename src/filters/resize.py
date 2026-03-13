from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFilter


class ResizeFilter(BaseFilter):
    """
    Upscale image so the shorter side is at least min_side pixels.
    Tesseract tends to perform better with ~300 DPI; upscaling small photos can help.
    Only enlarges; never shrinks.
    """

    def __init__(self, min_side: int = 1500) -> None:
        super().__init__(name="Resize")
        self.min_side = min_side

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        short = min(h, w)
        if short >= self.min_side:
            return image
        scale = self.min_side / short
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(
            image, (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )
