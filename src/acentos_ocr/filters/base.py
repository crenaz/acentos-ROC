# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseFilter(ABC):
    """
    Abstract base class for all image preprocessing filters.

    Every filter takes a numpy.ndarray (BGR or grayscale) and must
    return a numpy.ndarray of the same type.
    """

    def __init__(self, name: str = "BaseFilter") -> None:
        self.name = name

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image and return the transformed version.
        """

    def __repr__(self) -> str:
        return f"<Filter: {self.name}>"

