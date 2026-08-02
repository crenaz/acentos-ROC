from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFilter


class DeskewFilter(BaseFilter):
    """
    Correct skew (tilt) in document photos using the dominant text angle.
    Uses Otsu threshold + minAreaRect to estimate angle, then rotates the image.
    """

    def __init__(self, max_angle_deg: float = 15.0, min_angle_deg: float = 0.5) -> None:
        super().__init__(name="Deskew")
        self.max_angle_deg = max_angle_deg
        self.min_angle_deg = min_angle_deg

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            raise ValueError("DeskewFilter expects a grayscale image.")

        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        points = np.column_stack(np.where(binary > 0))
        if points.shape[0] < 100:
            return image

        rect = cv2.minAreaRect(points)
        angle = float(rect[-1])
        if angle < -45:
            angle += 90
        angle = max(-self.max_angle_deg, min(self.max_angle_deg, angle))
        if abs(angle) < self.min_angle_deg:
            return image

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
