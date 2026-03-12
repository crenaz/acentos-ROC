from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import cv2


@dataclass
class OCRResult:
    text: str
    confidence: float
    metadata: pd.DataFrame
    config_used: str

    def __repr__(self) -> str:
        return f"<OCRResult(conf={self.confidence:.2f}%, text_len={len(self.text)})>"


class TesseractWrapper:
    """
    Thin wrapper around pytesseract that returns structured results.
    """

    def __init__(self, tesseract_cmd: str | None = None, lang: str = "eng") -> None:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.lang = lang
        self.default_config = "--oem 3 --psm 3"

    def process_image(self, image: np.ndarray, custom_config: str | None = None) -> OCRResult:
        config = custom_config or self.default_config

        if image.ndim == 3:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = Image.fromarray(image)

        raw_data = pytesseract.image_to_data(
            image_pil,
            lang=self.lang,
            config=config,
            output_type=pytesseract.Output.DATAFRAME,
        )

        clean_data = raw_data[raw_data["text"].notna() & (raw_data["text"] != "")]

        full_text = pytesseract.image_to_string(
            image_pil,
            lang=self.lang,
            config=config,
        ).strip()

        avg_conf = float(clean_data["conf"].mean()) if not clean_data.empty else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_conf,
            metadata=clean_data,
            config_used=config,
        )

