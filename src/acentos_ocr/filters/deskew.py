# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseFilter


class DeskewFilter(BaseFilter):
    """
    Correct page skew by maximising the sharpness of the horizontal projection
    profile over a range of trial rotations.

    When text lines are horizontal, the per-row ink counts alternate strongly
    between line and gap, so squared row-to-row differences peak. Rotating away
    from true smears the lines together and flattens the profile. Searching for
    that peak recovers the skew angle.

    This replaces an earlier estimator that took `cv2.minAreaRect` over every dark
    pixel. That approach only worked on clean scans: on a photograph, Otsu also
    marks background, shadows and the page's own edges as ink -- roughly 10% of the
    frame on the Cayman samples -- so the rectangle spanned the whole image and
    reported 0 degrees regardless of the true rotation. The projection profile is
    driven by text-line periodicity instead, which survives a noisy background.

    Two guards keep it from inventing a rotation:

    * The trial range is bounded by `max_angle_deg`. If the page is rotated further
      than that, no trial angle straightens it and the score surface stays flat, so
      the peak-to-median ratio collapses -- measured at 1.2 for a 40-degree page
      versus 20-36 for the real samples. Below `min_peak_ratio` the filter declines.
    * Rotations smaller than `min_angle_deg` are not worth an interpolation pass.
    """

    def __init__(
        self,
        max_angle_deg: float = 15.0,
        min_angle_deg: float = 0.5,
        coarse_step_deg: float = 1.0,
        fine_step_deg: float = 0.1,
        work_height: int = 900,
        min_peak_ratio: float = 10.0,
    ) -> None:
        super().__init__(name="Deskew")
        self.max_angle_deg = max_angle_deg
        self.min_angle_deg = min_angle_deg
        self.coarse_step_deg = coarse_step_deg
        self.fine_step_deg = fine_step_deg
        self.work_height = work_height
        self.min_peak_ratio = min_peak_ratio

    def _text_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Isolate text-like detail, downscaled for speed.

        A morphological gradient responds to local contrast, so character strokes
        survive and smooth background does not. This is the step that makes the
        estimator work on photographs rather than only on scans.
        """
        height, width = image.shape[:2]
        if height > self.work_height:
            scale = self.work_height / height
            image = cv2.resize(
                image, (max(1, int(width * scale)), self.work_height),
                interpolation=cv2.INTER_AREA,
            )
        gradient = cv2.morphologyEx(
            image, cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        _, mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    @staticmethod
    def _profile_score(mask: np.ndarray) -> float:
        profile = mask.sum(axis=1, dtype=np.float64)
        return float(np.square(np.diff(profile)).sum())

    def _score_at(self, mask: np.ndarray, angle: float) -> float:
        height, width = mask.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            mask, matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderValue=0,
        )
        return self._profile_score(rotated)

    def estimate_angle(self, image: np.ndarray) -> float | None:
        """
        Rotation in degrees that would straighten the page, or None if no
        confident estimate exists. Exposed separately so it can be measured
        directly without paying for the warp.
        """
        mask = self._text_mask(image)
        if int((mask > 0).sum()) < 100:
            return None

        coarse = np.arange(
            -self.max_angle_deg,
            self.max_angle_deg + self.coarse_step_deg / 2,
            self.coarse_step_deg,
        )
        scores = np.array([self._score_at(mask, a) for a in coarse])

        median = float(np.median(scores))
        if median <= 0 or float(scores.max()) / median < self.min_peak_ratio:
            # No trial angle stands out: either there is no line structure, or the
            # true angle lies outside the search range. Either way, do not guess.
            return None

        best = float(coarse[int(scores.argmax())])
        fine = np.arange(
            best - self.coarse_step_deg,
            best + self.coarse_step_deg + self.fine_step_deg / 2,
            self.fine_step_deg,
        )
        return float(max(fine, key=lambda a: self._score_at(mask, a)))

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            raise ValueError("DeskewFilter expects a grayscale image.")

        angle = self.estimate_angle(image)
        if angle is None or abs(angle) < self.min_angle_deg:
            return image

        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(
            image, matrix, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
