import pytest

from acentos_ocr.core.pipelines import (
    DEFAULT_SPEC,
    NO_DESKEW_SPEC,
    build_default_pipeline,
    build_pipeline,
)
from acentos_ocr.filters.base import BaseFilter
from acentos_ocr.filters.registry import FILTERS, build_filter, describe_filters


def test_every_registered_name_builds_with_no_arguments():
    """A filter in the registry must be usable by name alone."""
    for name in FILTERS:
        assert isinstance(build_filter(name), BaseFilter)


def test_every_filter_in_the_package_is_registered():
    """
    A filter that exists but is not registered is unreachable from the CLI, which
    is the situation finding #2 described: four filters were built and wired into
    nothing.
    """
    import pkgutil

    import acentos_ocr.filters as package

    modules = {
        name for _, name, _ in pkgutil.iter_modules(package.__path__)
        if name not in ("base", "registry")
    }
    registered = {cls.__module__.rsplit(".", 1)[-1] for cls in FILTERS.values()}
    assert modules == registered


@pytest.mark.parametrize(
    "spec, attribute, expected",
    [
        ("blur:ksize=7", "ksize", 7),
        ("clahe:clip_limit=3.5", "clip_limit", 3.5),
        ("clahe:tile_size=16", "tile_size", 16),
        ("morphology:op=open", "op", "open"),
        ("threshold:block_size=21", "block_size", 21),
        ("resize:min_side=800", "min_side", 800),
    ],
)
def test_arguments_are_parsed_and_coerced(spec, attribute, expected):
    """
    Values arrive from the CLI as strings and must reach the constructor as the
    type it declares. The filter modules use postponed annotations, so this only
    works because the registry resolves them.
    """
    value = getattr(build_filter(spec), attribute)
    assert value == expected
    assert type(value) is type(expected)


def test_several_arguments_can_be_given_at_once():
    morphology = build_filter("morphology:op=dilate,kernel_size=4")
    assert (morphology.op, morphology.kernel_size) == ("dilate", 4)


def test_whitespace_around_arguments_is_tolerated():
    assert build_filter("blur: ksize = 9 ").ksize == 9


@pytest.mark.parametrize(
    "spec, message",
    [
        ("nope", "unknown filter"),
        ("blur:radius=3", "has no parameter"),
        ("blur:ksize", "expected key=value"),
        ("blur:ksize=wide", "invalid literal"),
    ],
)
def test_bad_specs_raise_a_useful_message(spec, message):
    with pytest.raises(ValueError, match=message):
        build_filter(spec)


def test_an_unknown_filter_lists_the_available_ones():
    """The error has to be actionable -- it is the only discovery mechanism."""
    with pytest.raises(ValueError) as error:
        build_filter("sharpen")
    for name in FILTERS:
        assert name in str(error.value)


def test_build_pipeline_preserves_the_given_order():
    pipeline = build_pipeline(["grayscale", "clahe", "blur:ksize=3", "threshold"])
    assert [f.name for f in pipeline.filters] == [
        "Grayscale", "CLAHE", "GaussianBlur", "AdaptiveThreshold",
    ]


def test_build_pipeline_accepts_an_empty_stack():
    assert build_pipeline([]).filters == []


def test_the_default_pipeline_is_its_spec():
    """
    The documented default and the running default are the same string, so they
    cannot drift apart.
    """
    assert [f.name for f in build_default_pipeline().filters] == [
        f.name for f in build_pipeline(DEFAULT_SPEC).filters
    ]
    assert [f.name for f in build_default_pipeline(deskew=False).filters] == [
        f.name for f in build_pipeline(NO_DESKEW_SPEC).filters
    ]


def test_describe_filters_lists_every_filter_with_its_defaults():
    described = describe_filters()
    for name in FILTERS:
        assert name in described
    assert "ksize=5" in described        # GaussianBlurFilter's default
    assert "(no parameters)" in described  # GrayscaleFilter takes none
