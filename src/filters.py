from abc import ABC, ABCMeta, abstractmethod
from enum import IntEnum, auto
from inspect import Parameter
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
    Metaclass that enforces required class attributes for FilterAdapter subclasses.

    All concrete FilterAdapter subclasses must define:
    1. 'name' (str): A unique identifier for the filter.
    2. 'filter_type' (FilterType): The category/type of the filter.
    3. 'initial_hyperparameters' (FilterHyperparameters): Default parameter values.
    4. 'parameters_class' (Type[FilterHyperparameters]): The class defining filter parameters.

    This metaclass validates these attributes at class definition time, ensuring
    consistent interface across all filter implementations.
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

        if "parameters_class" not in dct:
            raise TypeError(f"Class '{name}' must define 'parameters_class'.")

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
    parameters_class = NoOpFilterParameters
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
    parameter_class = MedianBlurParameters
    initial_hyperparameters = MedianBlurParameters(kernel_size=3)

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: MedianBlurParameters
    ) -> np.ndarray:
        import cv2

        return cv2.medianBlur(image, parameters.kernel_size)
