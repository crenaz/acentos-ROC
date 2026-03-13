from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFilter


class MorphologyFilter(BaseFilter):
    """
    Morphological operation on a binary image (after threshold).
    Closing (dilation then erosion) fills small gaps in text strokes.
    Opening (erosion then dilation) removes small noise/speckle.
    """

    def __init__(
        self,
        op: str = "close",
        kernel_size: int = 2,
    ) -> None:
        super().__init__(name="Morphology")
        self.op = op.strip().lower()
        if self.op not in ("open", "close", "dilate", "erode"):
            raise ValueError("op must be one of: open, close, dilate, erode")
        self.kernel_size = max(1, kernel_size)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_size, self.kernel_size),
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            raise ValueError("MorphologyFilter expects a grayscale/binary image.")
        if self.op == "open":
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, self._kernel)
        if self.op == "close":
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, self._kernel)
        if self.op == "dilate":
            return cv2.dilate(image, self._kernel)
        return cv2.erode(image, self._kernel)
