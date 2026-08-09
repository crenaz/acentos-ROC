"""Behavioural tests for the individual filters.

Every filter became reachable by name when `--pipeline` landed (finding #2), so a
user can now compose a stack out of components that nothing verified. These pin
what each one actually does, including the quirks worth knowing about.
"""
import numpy as np
import pytest

from acentos_ocr.filters.registry import FILTERS, build_filter

RNG = np.random.default_rng(0)


def noise(height=40, width=60):
    return RNG.integers(0, 256, (height, width), dtype=np.uint8)


def colour(height=40, width=60):
    return RNG.integers(0, 256, (height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# the contract every filter shares
# --------------------------------------------------------------------------

#: Filters that operate on a single channel and reject colour outright.
GRAYSCALE_ONLY = ["clahe", "threshold", "morphology", "deskew"]


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_apply_returns_a_uint8_array(name):
    image = noise()
    result = build_filter(name).apply(image)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_apply_does_not_mutate_its_input(name):
    """
    The pipeline threads one array through every filter in turn. A filter that
    edited in place would corrupt the debug images written for the stages before
    it, and make results depend on whether --debug was passed.
    """
    image = noise()
    original = image.copy()
    build_filter(name).apply(image)
    assert np.array_equal(image, original)


@pytest.mark.parametrize("name", GRAYSCALE_ONLY)
def test_grayscale_only_filters_reject_colour(name):
    with pytest.raises(ValueError, match="grayscale"):
        build_filter(name).apply(colour())


# --------------------------------------------------------------------------
# GrayscaleFilter
# --------------------------------------------------------------------------

def test_grayscale_drops_the_channel_axis():
    assert build_filter("grayscale").apply(colour(30, 50)).shape == (30, 50)


def test_grayscale_passes_a_single_channel_image_through():
    """Idempotent, so it is safe to have in a stack twice or after another filter."""
    image = noise()
    assert np.array_equal(build_filter("grayscale").apply(image), image)


def test_grayscale_weights_channels_perceptually():
    """
    Not a flat mean: green carries most of the luminance and blue least. Worth
    pinning because ink on newsprint is rarely neutral, and a naive mean would
    render coloured headings at a different density.
    """
    filter_ = build_filter("grayscale")
    green = filter_.apply(np.full((4, 4, 3), (0, 255, 0), dtype=np.uint8))[0, 0]
    blue = filter_.apply(np.full((4, 4, 3), (255, 0, 0), dtype=np.uint8))[0, 0]
    assert green > blue


# --------------------------------------------------------------------------
# GaussianBlurFilter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("given, expected", [(3, 3), (4, 5), (5, 5), (0, 1), (8, 9)])
def test_blur_rounds_an_even_kernel_up_to_odd(given, expected):
    """OpenCV requires an odd kernel; an even one is silently corrected, not rejected."""
    assert build_filter(f"blur:ksize={given}").ksize == expected


def test_blur_with_a_kernel_of_one_is_a_no_op():
    image = noise()
    assert np.array_equal(build_filter("blur:ksize=1").apply(image), image)


def test_blur_reduces_local_variation():
    image = noise()
    blurred = build_filter("blur:ksize=5").apply(image)
    assert blurred.var() < image.var()


def test_blur_preserves_shape_and_accepts_colour():
    """Unlike the single-channel filters, blur has no guard and works on either."""
    assert build_filter("blur").apply(colour(20, 30)).shape == (20, 30, 3)
    assert build_filter("blur").apply(noise(20, 30)).shape == (20, 30)


# --------------------------------------------------------------------------
# CLAHEFilter
# --------------------------------------------------------------------------

def test_clahe_expands_the_range_of_a_flat_image():
    low_contrast = np.linspace(100, 140, 64 * 64, dtype=np.uint8).reshape(64, 64)
    result = build_filter("clahe").apply(low_contrast)
    # np.ptp as a function, not a method: ndarray.ptp() was removed in NumPy 2.
    assert np.ptp(result) > np.ptp(low_contrast)


def test_clahe_preserves_shape():
    assert build_filter("clahe").apply(noise(30, 40)).shape == (30, 40)


def test_clahe_clip_limit_controls_how_aggressive_it_is():
    """
    Measured on the corpus, CLAHE costs 6.1 points of CER -- it amplifies newsprint
    texture along with the ink. The clip limit is the knob that governs that, so it
    has to actually do something.
    """
    image = noise()
    gentle = build_filter("clahe:clip_limit=1").apply(image)
    aggressive = build_filter("clahe:clip_limit=40").apply(image)
    assert not np.array_equal(gentle, aggressive)


# --------------------------------------------------------------------------
# AdaptiveThresholdFilter
# --------------------------------------------------------------------------

def test_threshold_output_is_strictly_binary():
    result = build_filter("threshold").apply(noise())
    assert set(np.unique(result).tolist()) <= {0, 255}


@pytest.mark.parametrize("given, expected", [(15, 15), (16, 17), (3, 3)])
def test_threshold_rounds_an_even_block_size_up_to_odd(given, expected):
    assert build_filter(f"threshold:block_size={given}").block_size == expected


def test_threshold_preserves_shape():
    assert build_filter("threshold").apply(noise(30, 40)).shape == (30, 40)


# --------------------------------------------------------------------------
# MorphologyFilter
# --------------------------------------------------------------------------

def speckled_page():
    """A solid block of ink, one isolated speck, and one pinhole in the block."""
    page = np.zeros((40, 40), dtype=np.uint8)
    page[10:30, 10:30] = 255      # the block
    page[35, 35] = 255            # isolated speck
    page[20, 20] = 0              # pinhole
    return page


def test_morphology_rejects_an_unknown_operation_at_construction():
    """Fail when the stack is built, not part-way through a corpus run."""
    with pytest.raises(ValueError, match="open, close, dilate, erode"):
        build_filter("morphology:op=sideways")


@pytest.mark.parametrize("op", ["open", "close", "dilate", "erode"])
def test_every_documented_operation_runs(op):
    assert build_filter(f"morphology:op={op}").apply(speckled_page()).shape == (40, 40)


def test_morphology_clamps_a_zero_kernel_to_one():
    assert build_filter("morphology:kernel_size=0").kernel_size == 1


def test_dilate_grows_ink_and_erode_shrinks_it():
    page = speckled_page()
    ink = int((page > 0).sum())
    assert int((build_filter("morphology:op=dilate,kernel_size=3").apply(page) > 0).sum()) > ink
    assert int((build_filter("morphology:op=erode,kernel_size=3").apply(page) > 0).sum()) < ink


def test_open_removes_an_isolated_speck():
    result = build_filter("morphology:op=open,kernel_size=3").apply(speckled_page())
    assert result[35, 35] == 0      # speck gone
    assert result[15, 15] == 255    # block survives


def test_close_fills_a_pinhole():
    result = build_filter("morphology:op=close,kernel_size=3").apply(speckled_page())
    assert result[20, 20] == 255


# --------------------------------------------------------------------------
# ResizeFilter
# --------------------------------------------------------------------------

def test_resize_enlarges_until_the_short_side_meets_the_minimum():
    result = build_filter("resize:min_side=200").apply(noise(50, 80))
    assert min(result.shape[:2]) == 200


def test_resize_preserves_aspect_ratio():
    result = build_filter("resize:min_side=300").apply(noise(100, 250))
    assert result.shape[:2] == (300, 750)


def test_resize_never_shrinks_a_large_image():
    """Documented as enlarge-only; downscaling would silently discard detail."""
    image = noise(400, 900)
    result = build_filter("resize:min_side=100").apply(image)
    assert result.shape == image.shape
    assert result is image      # and does so without paying for a copy


def test_resize_handles_colour_images():
    assert build_filter("resize:min_side=120").apply(colour(30, 60)).shape == (120, 240, 3)
