import numpy as np

from acentos_ocr.core.pipelines import build_default_pipeline


def filter_names(pipeline):
    return [f.name for f in pipeline.filters]


def test_deskew_is_on_by_default():
    """Corpus-measured: 24.0% -> 18.9% CER at psm 3. See PROJECT-ANALYSIS #11."""
    assert filter_names(build_default_pipeline()) == ["Grayscale", "Deskew", "GaussianBlur"]


def test_deskew_can_be_turned_off():
    assert filter_names(build_default_pipeline(deskew=False)) == ["Grayscale", "GaussianBlur"]


def test_deskew_runs_after_grayscale():
    """DeskewFilter rejects colour input, so ordering here is load-bearing."""
    names = filter_names(build_default_pipeline())
    assert names.index("Grayscale") < names.index("Deskew")


def test_the_default_pipeline_does_not_binarise():
    """
    Tesseract 5's LSTM engine binarises internally and better than a hand-tuned
    threshold; adding our own cost 30.5% -> 8.9% CER to remove. See finding #3.
    """
    names = filter_names(build_default_pipeline())
    assert "AdaptiveThreshold" not in names
    assert "Morphology" not in names


def test_the_default_pipeline_runs_end_to_end_on_a_blank_page():
    page = np.full((64, 48, 3), 255, dtype=np.uint8)
    result = build_default_pipeline().run(page)
    # Grayscale drops the channel axis; a featureless page gives deskew no angle
    # to find, so it must pass the image through rather than invent a rotation.
    assert result.shape == (64, 48)
