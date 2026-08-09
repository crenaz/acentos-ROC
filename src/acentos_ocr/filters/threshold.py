# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

import cv2
import numpy as np

from .base import BaseFilter


class AdaptiveThresholdFilter(BaseFilter):
    """
    Adaptive thresholding to produce a high-contrast binary image
    that tends to work well for Tesseract.
    """

    def __init__(self, block_size: int = 15, constant: int = 5) -> None:
        super().__init__(name="AdaptiveThreshold")
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.constant = constant

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            raise ValueError("AdaptiveThresholdFilter expects a grayscale image.")

        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.constant,
        )

