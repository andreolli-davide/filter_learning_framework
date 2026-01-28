from abc import ABC, ABCMeta, abstractmethod
from enum import IntEnum, auto
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel, Field

# Type variable for filter hyperparameters, constrained to Pydantic BaseModel
FilterHyperparameters = TypeVar("FilterHyperparameters", bound=BaseModel)


class FilterType(IntEnum):
    NO_OP = 0
    NOISE_REDUCTION = auto()
    COLOR_CORRECTION = auto()
    CONTRAST_ENHANCEMENT = auto()
    EDGE_SHARPNING = auto()


class FilterAdapterMeta(ABCMeta):
    """
    Metaclass that enforces:
    1. A 'name' string attribute.
    2. An 'initial_hyperparameters' attribute matching the Generic TypeVar.
    """

    def __new__(mcls, name, bases, dct):
        # Skip validation for the base class itself
        if name == "FilterAdapter":
            return super().__new__(mcls, name, bases, dct)

        if "name" not in dct:
            raise TypeError(f"Class '{name}' must define the class attribute 'name'.")

        if "filter_type" not in dct:
            raise TypeError(
                f"Class '{name}' must define the class attribute 'filter_type'."
            )

        if "initial_hyperparameters" not in dct:
            raise TypeError(f"Class '{name}' must define 'initial_hyperparameters'.")

        return super().__new__(mcls, name, bases, dct)


class FilterAdapter(ABC, Generic[FilterHyperparameters], metaclass=FilterAdapterMeta):
    """
    Abstract Base Class for image filter adapters.
    All subclasses must define 'name' and 'initial_hyperparameters'.
    """

    @classmethod
    @abstractmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: FilterHyperparameters
    ) -> np.ndarray:
        """Applies the filter to the provided image."""
        pass


class NoOpFilterParameters(BaseModel):
    """Parameters for No-Op filter (empty)."""

    pass


class NoOpFilterAdapter(FilterAdapter[NoOpFilterParameters]):
    """
    A concrete implementation that fulfills all metaclass requirements.
    """

    name = "NoOpFilter"
    filter_type = FilterType.NO_OP
    initial_hyperparameters = NoOpFilterParameters()

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: NoOpFilterParameters
    ) -> np.ndarray:
        return image


class MedianBlurParameters(BaseModel):
    """Parameters for Median Blur filter."""

    kernel_size: int = Field(
        default=3, ge=1, description="Size of the median filter kernel."
    )


class MedianBlurFilterAdapter(FilterAdapter[MedianBlurParameters]):
    """
    Median Blur filter adapter implementation.
    """

    name = "MedianBlur"
    filter_type = FilterType.NOISE_REDUCTION
    initial_hyperparameters = MedianBlurParameters(kernel_size=3)

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: MedianBlurParameters
    ) -> np.ndarray:
        import cv2

        return cv2.medianBlur(image, parameters.kernel_size)

