from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFilter


class GaussianBlurFilter(BaseFilter):
    """
    Gaussian blur for noise reduction before thresholding.
    Smooths the image so adaptive threshold produces cleaner text edges.
    """

    def __init__(self, ksize: int = 5, sigma_x: float = 0) -> None:
        super().__init__(name="GaussianBlur")
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1
        self.sigma_x = sigma_x

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (self.ksize, self.ksize), self.sigma_x)
