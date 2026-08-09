# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

import cv2
import numpy as np

from .base import BaseFilter


class GrayscaleFilter(BaseFilter):
    """Convert BGR images to single-channel grayscale."""

    def __init__(self) -> None:
        super().__init__(name="Grayscale")

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

