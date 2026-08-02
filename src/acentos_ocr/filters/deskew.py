from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFilter


class DeskewFilter(BaseFilter):
    """
    Correct skew (tilt) using the dominant text angle.
    Uses Otsu threshold + minAreaRect to estimate the angle, then rotates.

    Works on clean scans, where Otsu isolates text against a uniform background.

    Does NOT work on photographs. Otsu over a photo marks background, shadows and
    the clipping's own edges as ink -- around 10% of the frame on the Cayman
    samples -- so the minimum-area rectangle spans the whole image and its angle is
    meaningless. On a 6 MP clipping photo it reports 0.00 degrees regardless of how
    far the page is actually rotated, making the filter a no-op at best. Photographs
    need a different estimator (text-line projection profiles, or a Hough transform
    over morphologically joined text runs). See finding #11 in
    PROJECT-ANALYSIS-2026-08-02.md.
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
        # np.where returns (rows, cols) i.e. (y, x), but cv2.minAreaRect expects
        # (x, y). Passing them in row-major order mirrors the page across its
        # diagonal, which inverts the sign of the measured angle -- the filter then
        # rotates the wrong way and doubles the skew instead of removing it.
        rows, cols = np.where(binary > 0)
        if rows.size < 100:
            return image
        points = np.column_stack((cols, rows)).astype(np.float32)

        rect = cv2.minAreaRect(points)
        angle = float(rect[-1])
        # minAreaRect reports the angle of one arbitrary edge, so a near-upright page
        # can come back close to either 0 or 90. Fold it into (-45, 45].
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        if abs(angle) < self.min_angle_deg:
            # Too small to be worth an interpolation pass.
            return image
        if abs(angle) > self.max_angle_deg:
            # Beyond this it is not page skew -- more likely a landscape page or a
            # failed measurement. Clamping to the limit would bake in a rotation we
            # have no reason to believe in, so decline instead.
            return image

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
