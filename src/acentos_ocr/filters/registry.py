# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

import inspect
from typing import Any

from .base import BaseFilter
from .clahe import CLAHEFilter
from .deskew import DeskewFilter
from .gaussian_blur import GaussianBlurFilter
from .grayscale import GrayscaleFilter
from .morphology import MorphologyFilter
from .resize import ResizeFilter
from .threshold import AdaptiveThresholdFilter

#: Every filter, addressable by a short CLI name. Adding a filter here is what
#: makes it reachable without editing source -- which was the whole point of the
#: Strategy pattern the project is built on, and was not true until now.
FILTERS: dict[str, type[BaseFilter]] = {
    "grayscale": GrayscaleFilter,
    "deskew": DeskewFilter,
    "blur": GaussianBlurFilter,
    "clahe": CLAHEFilter,
    "morphology": MorphologyFilter,
    "resize": ResizeFilter,
    "threshold": AdaptiveThresholdFilter,
}


def _signature(cls: type[BaseFilter]) -> inspect.Signature:
    """
    The constructor signature with annotations resolved to real types.

    The filter modules use `from __future__ import annotations`, so without
    eval_str every annotation arrives as the *string* "int" and every parameter
    would be passed through as text.
    """
    return inspect.signature(cls.__init__, eval_str=True)


def _coerce(value: str, annotation: Any) -> Any:
    """Convert a CLI string to the type the constructor declares."""
    if annotation is bool:
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "on"):
            return True
        if lowered in ("false", "no", "0", "off"):
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    return value


def build_filter(spec: str) -> BaseFilter:
    """
    Construct one filter from a `name` or `name:key=value,key=value` spec.

    Parameters are coerced using the constructor's own type annotations, so the
    filters stay the single source of truth for their arguments and nothing has to
    be restated here.

        blur                    -> GaussianBlurFilter()
        blur:ksize=3            -> GaussianBlurFilter(ksize=3)
        morphology:op=open,kernel_size=3
    """
    name, _, argument_text = spec.partition(":")
    name = name.strip().lower()
    if name not in FILTERS:
        raise ValueError(
            f"unknown filter {name!r}. Available: {', '.join(sorted(FILTERS))}"
        )

    cls = FILTERS[name]
    signature = _signature(cls)
    kwargs: dict[str, Any] = {}

    for pair in (p for p in argument_text.split(",") if p.strip()):
        key, separator, value = pair.partition("=")
        key = key.strip()
        if not separator:
            raise ValueError(
                f"filter {name!r}: expected key=value, got {pair.strip()!r}"
            )
        if key not in signature.parameters:
            accepted = [p for p in signature.parameters if p != "self"]
            raise ValueError(
                f"filter {name!r} has no parameter {key!r}. "
                f"Accepts: {', '.join(accepted) or 'no parameters'}"
            )
        annotation = signature.parameters[key].annotation
        try:
            kwargs[key] = _coerce(value.strip(), annotation)
        except ValueError as error:
            raise ValueError(f"filter {name!r}, parameter {key!r}: {error}") from error

    return cls(**kwargs)


def describe_filters() -> str:
    """One line per filter, with its parameters and defaults, for CLI help."""
    lines = []
    for name in sorted(FILTERS):
        signature = _signature(FILTERS[name])
        params = [
            f"{key}={parameter.default!r}"
            for key, parameter in signature.parameters.items()
            if key != "self"
        ]
        lines.append(f"  {name:<12}{', '.join(params) or '(no parameters)'}")
    return "\n".join(lines)
