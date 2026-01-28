"""
filters.py

This module defines a flexible and extensible framework for image filtering operations,
using an adapter pattern and Pydantic for parameter validation. Each filter is represented
by a class that specifies its type, parameters, and application logic. The framework
supports easy extension for new filters and provides detailed documentation for each filter
and its parameters.

Filter Types Implemented:
- No-Op (does nothing)
- Median Blur (noise reduction)
- Bilateral Filter (edge-preserving smoothing)
- CLAHE (contrast enhancement)
- Gamma Correction (color correction)
- Unsharp Mask (edge sharpening)
- Laplacian Sharpen (edge sharpening)

Each filter exposes its hyperparameters, with descriptions of their effects and how
changing their values influences the filter's behavior.
"""

from abc import ABC, ABCMeta, abstractmethod
from enum import IntEnum, auto
from typing import Generic, TypeVar

import numpy as np
from pydantic import BaseModel, Field

FilterHyperparameters = TypeVar("FilterHyperparameters", bound=BaseModel)


class FilterType(IntEnum):
    """
    Enum representing the main categories of image filters.

    Attributes:
        NO_OP: No operation (identity filter).
        NOISE_REDUCTION: Filters that reduce noise.
        COLOR_CORRECTION: Filters that adjust color or intensity.
        CONTRAST_ENHANCEMENT: Filters that enhance image contrast.
        EDGE_SHARPNING: Filters that enhance edges and details.
    """

    NO_OP = 0
    NOISE_REDUCTION = auto()
    COLOR_CORRECTION = auto()
    CONTRAST_ENHANCEMENT = auto()
    EDGE_SHARPNING = auto()


class FilterAdapterMeta(ABCMeta):
    """
    Metaclass that enforces required class attributes for FilterAdapter subclasses.

    All concrete FilterAdapter subclasses must define:
        - 'name' (str): A unique identifier for the filter.
        - 'filter_type' (FilterType): The category/type of the filter.
        - 'initial_hyperparameters' (FilterHyperparameters): Default parameter values.
        - 'parameters_class' (Type[FilterHyperparameters]): The class defining filter parameters.

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

    Subclasses must define:
        - name: Unique string identifier for the filter.
        - filter_type: The category/type of the filter.
        - parameters_class: The Pydantic model for filter parameters.
        - initial_hyperparameters: Default parameter values.

    Each subclass must implement the apply_filter method, which applies the filter
    to an image using the provided parameters.
    """

    @classmethod
    @abstractmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: FilterHyperparameters
    ) -> np.ndarray:
        """
        Apply the filter to the provided image using the given parameters.

        Args:
            image: Input image as a NumPy array.
            parameters: Filter-specific hyperparameters.

        Returns:
            The filtered image as a NumPy array.
        """
        pass


class NoOpFilterParameters(BaseModel):
    """
    Parameters for No-Op filter.

    This filter does not modify the image and has no parameters.
    """

    pass


class NoOpFilterAdapter(FilterAdapter[NoOpFilterParameters]):
    """
    No-Op Filter Adapter.

    This filter returns the input image unchanged. Useful as a placeholder or for testing.
    """

    name = "NoOpFilter"
    filter_type = FilterType.NO_OP
    parameters_class = NoOpFilterParameters
    initial_hyperparameters = NoOpFilterParameters()

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: NoOpFilterParameters
    ) -> np.ndarray:
        """
        Return the image unchanged.
        """
        return image


class MedianBlurParameters(BaseModel):
    """
    Parameters for Median Blur filter.

    Attributes:
        kernel_size: Size of the median filter kernel (must be odd and >= 1).
            - Higher values increase smoothing and noise reduction, but may blur details.
            - Lower values preserve more detail but reduce noise less.
    """

    kernel_size: int = Field(
        default=3, ge=1, description="Size of the median filter kernel."
    )


class MedianBlurFilterAdapter(FilterAdapter[MedianBlurParameters]):
    """
    Median Blur Filter Adapter.

    Applies a median blur to the image, which is effective for removing salt-and-pepper noise
    while preserving edges better than a simple mean blur.

    Parameters:
        kernel_size: Size of the median filter kernel.
    """

    name = "MedianBlur"
    filter_type = FilterType.NOISE_REDUCTION
    parameters_class = MedianBlurParameters
    initial_hyperparameters = MedianBlurParameters(kernel_size=3)

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: MedianBlurParameters
    ) -> np.ndarray:
        """
        Apply median blur to the image.

        Args:
            image: Input image.
            parameters: MedianBlurParameters.

        Returns:
            Blurred image.
        """
        import cv2

        return cv2.medianBlur(image, parameters.kernel_size)


class BilateralFilterParameters(BaseModel):
    diameter: int = Field(
        default=5, ge=1, description="Diameter of each pixel neighborhood."
    )
    sigmaColor: float = Field(
        default=50.0, ge=0.0, description="Filter sigma in color space."
    )
    sigmaSpace: float = Field(
        default=50.0, ge=0.0, description="Filter sigma in coordinate space."
    )


class BilateralFilterAdapter(FilterAdapter[BilateralFilterParameters]):
    """
    Bilateral Filter Adapter.

    Applies bilateral filtering to the image, which smooths images while preserving edges.
    Useful for noise reduction without blurring edges.

    Parameters:
        diameter: Diameter of pixel neighborhood.
        sigmaColor: Filter sigma in color space.
        sigmaSpace: Filter sigma in coordinate space.
    """

    name = "BilateralFilter"
    filter_type = FilterType.NOISE_REDUCTION
    parameters_class = BilateralFilterParameters
    initial_hyperparameters = BilateralFilterParameters(
        diameter=5, sigmaColor=50.0, sigmaSpace=50.0
    )

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: BilateralFilterParameters
    ) -> np.ndarray:
        """
        Apply bilateral filter to the image.

        Args:
            image: Input image.
            parameters: BilateralFilterParameters.

        Returns:
            Filtered image.
        """
        import cv2

        return cv2.bilateralFilter(
            src=image,
            d=parameters.diameter,
            sigmaColor=parameters.sigmaColor,
            sigmaSpace=parameters.sigmaSpace,
        )


class SaturationBoostParameters(BaseModel):
    """
    Parameters for Saturation Boost filter.

    Attributes:
        boost_factor: Factor to boost saturation.
            - Values > 1 increase color vividness.
            - Values < 1 decrease saturation, making colors more muted.
    """

    boost_factor: float = Field(
        default=1.3, ge=0.0, description="Factor to boost saturation."
    )


class SaturationBoostFilterAdapter(FilterAdapter[SaturationBoostParameters]):
    """
    Saturation Boost Filter Adapter.

    Increases or decreases the saturation of an image by multiplying the saturation channel
    in HSV color space by a given factor. Useful for making colors more vivid or more muted.

    Parameters:
        boost_factor: Factor to multiply the saturation channel.
            - Values > 1 increase color vividness.
            - Values < 1 decrease saturation, making colors more muted.
    """

    name = "SaturationBoost"
    filter_type = FilterType.COLOR_CORRECTION
    parameters_class = SaturationBoostParameters
    initial_hyperparameters = SaturationBoostParameters(boost_factor=1.3)

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: SaturationBoostParameters
    ) -> np.ndarray:
        """
        Apply saturation boost to the image.

        Args:
            image: Input image.
            parameters: SaturationBoostParameters.

        Returns:
            Image with adjusted saturation.
        """
        import cv2

        image = image.astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        s = s * parameters.boost_factor
        s = np.clip(s, 0, 255).astype(np.uint8)

        hsv_boosted = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)


class ClaheParameters(BaseModel):
    """
    Parameters for CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Attributes:
        clipLimit: Threshold for contrast limiting.
            - Higher values give more contrast enhancement but may amplify noise.
        tileGridSize: Size of grid for histogram equalization.
            - Larger grids give coarser equalization; smaller grids give finer local contrast.
    """

    clipLimit: float = Field(
        ..., ge=0.0, description="Threshold for contrast limiting."
    )
    tileGridSize: tuple[int, int] = Field(
        ..., description="Size of grid for histogram equalization."
    )


class SoftClaheFilterAdapter(FilterAdapter[ClaheParameters]):
    """
    CLAHE Filter Adapter (Soft).

    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast.
    Useful for improving the visibility of features in both grayscale and color images.

    Parameters:
        clipLimit: Lower value (e.g., 2.0) for moderate contrast enhancement.
        tileGridSize: Grid size for local histogram equalization.
    """

    name = "CLAHE (with starting 'clip limit' set to 2.0)"
    filter_type = FilterType.CONTRAST_ENHANCEMENT
    parameters_class = ClaheParameters
    initial_hyperparameters = ClaheParameters(clipLimit=2.0, tileGridSize=(8, 8))

    @classmethod
    def apply_filter(cls, image: np.ndarray, parameters: ClaheParameters) -> np.ndarray:
        """
        Apply CLAHE to the image.

        Args:
            image: Input image.
            parameters: ClaheParameters.

        Returns:
            Contrast-enhanced image.
        """
        import cv2

        image = image.astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=parameters.clipLimit, tileGridSize=parameters.tileGridSize
        )
        if len(image.shape) == 2:  # Grayscale image
            return clahe.apply(image)
        elif len(image.shape) == 3:  # Color image
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            luminosity, a, b = cv2.split(lab)
            l_eq = clahe.apply(luminosity)
            lab_eq = cv2.merge((l_eq, a, b))
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        else:
            raise ValueError("Unsupported image shape for CLAHE filter.")


class HardClaheFilterAdapter(FilterAdapter[ClaheParameters]):
    """
    CLAHE Filter Adapter (Hard).

    Like Soft CLAHE, but with a higher clip limit for stronger contrast enhancement.

    Parameters:
        clipLimit: Higher value (e.g., 4.0) for stronger contrast enhancement.
        tileGridSize: Grid size for local histogram equalization.
    """

    name = "CLAHE (with starting 'clip limit' set to 4.0)"
    filter_type = FilterType.CONTRAST_ENHANCEMENT
    parameters_class = ClaheParameters
    initial_hyperparameters = ClaheParameters(clipLimit=4.0, tileGridSize=(8, 8))

    @classmethod
    def apply_filter(cls, image: np.ndarray, parameters: ClaheParameters) -> np.ndarray:
        """
        Apply CLAHE to the image.

        Args:
            image: Input image.
            parameters: ClaheParameters.

        Returns:
            Contrast-enhanced image.
        """
        import cv2

        image = image.astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=parameters.clipLimit, tileGridSize=parameters.tileGridSize
        )
        if len(image.shape) == 2:  # Grayscale image
            return clahe.apply(image)
        elif len(image.shape) == 3:  # Color image
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            luminosity, a, b = cv2.split(lab)
            l_eq = clahe.apply(luminosity)
            lab_eq = cv2.merge((l_eq, a, b))
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        else:
            raise ValueError("Unsupported image shape for CLAHE filter.")


class GammaCorrectionParameters(BaseModel):
    """
    Parameters for Gamma Correction filter.

    Attributes:
        gamma: Gamma value for correction.
            - Values > 1 darken the image.
            - Values < 1 lighten the image.
    """

    gamma: float = Field(default=1.2, gt=0.0, description="Gamma value for correction.")


class GammaCorrectionFilterAdapter(FilterAdapter[GammaCorrectionParameters]):
    """
    Gamma Correction Filter Adapter.

    Adjusts the brightness and contrast of the image using a gamma curve.

    Parameters:
        gamma: Gamma value for correction.
    """

    name = "GammaCorrection"
    filter_type = FilterType.COLOR_CORRECTION
    parameters_class = GammaCorrectionParameters
    initial_hyperparameters = GammaCorrectionParameters(gamma=1.2)

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: GammaCorrectionParameters
    ) -> np.ndarray:
        """
        Apply gamma correction to the image.

        Args:
            image: Input image.
            parameters: GammaCorrectionParameters.

        Returns:
            Gamma-corrected image.
        """
        import cv2

        inv_gamma = 1.0 / parameters.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )

        return cv2.LUT(image, table)


class UnsharpMaskParameters(BaseModel):
    """
    Parameters for Unsharp Mask filter.

    Attributes:
        sigma: Gaussian blur sigma for unsharp masking.
            - Higher values blur more, affecting the scale of sharpening.
        strength: Strength of the sharpening effect.
            - Higher values increase the sharpening effect.
    """

    sigma: float = Field(
        default=1.2, gt=0.0, description="Gaussian blur sigma for unsharp masking."
    )
    strength: float = Field(
        default=1.5, gt=0.0, description="Strength of the sharpening effect."
    )


class UnsharpMaskFilterAdapter(FilterAdapter[UnsharpMaskParameters]):
    """
    Unsharp Mask Filter Adapter.

    Enhances edges and details by subtracting a blurred version of the image from the original,
    then adding the result back to the original image.

    Parameters:
        sigma: Standard deviation for Gaussian blur (controls the scale of details to sharpen).
        strength: Multiplier for the sharpening effect.
    """

    name = "UnsharpMask"
    filter_type = FilterType.EDGE_SHARPNING
    parameters_class = UnsharpMaskParameters
    initial_hyperparameters = UnsharpMaskParameters(sigma=1.2, strength=1.5)

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: UnsharpMaskParameters
    ) -> np.ndarray:
        """
        Apply unsharp mask to the image.

        Args:
            image: Input image.
            parameters: UnsharpMaskParameters.

        Returns:
            Sharpened image.
        """
        import cv2

        blurred = cv2.GaussianBlur(image, (0, 0), parameters.sigma)

        # Unsharp Mask formula:
        # Sharpened = Original + (Original - Blurred) * Strength
        # Reformulated for cv2.addWeighted:
        # Sharpened = Original * (1 + Strength) + Blurred * (-Strength)

        return cv2.addWeighted(
            image, 1.0 + parameters.strength, blurred, -parameters.strength, 0
        )


class LaplacianSharpenParameters(BaseModel):
    """
    Parameters for Laplacian Sharpen filter.

    Attributes:
        ksize: Kernel size for the Laplacian operator (must be odd and positive).
            - Higher values detect broader edges and may increase the sharpening effect.
            - Lower values focus on finer details.
        scale: Optional scale factor for the computed Laplacian.
            - Higher values amplify the Laplacian response, increasing sharpening.
        delta: Optional delta value added to the results.
            - Can be used to shift the output intensity.
    """

    ksize: int = Field(
        default=3, ge=1, description="Kernel size (must be odd and positive)."
    )
    scale: float = Field(
        default=1.0, description="Optional scale factor for the computed Laplacian."
    )
    delta: float = Field(
        default=0.0, description="Optional delta value added to the results."
    )


class LaplacianSharpenFilterAdapter(FilterAdapter[LaplacianSharpenParameters]):
    """
    Laplacian Sharpen Filter Adapter.

    Enhances edges by subtracting the Laplacian (second derivative) of the image from the original.
    This highlights regions of rapid intensity change, making edges more pronounced.

    Parameters:
        ksize: Kernel size for the Laplacian operator.
        scale: Scale factor for the Laplacian.
        delta: Value added to the Laplacian result.
    """

    name = "LaplacianSharpen"
    filter_type = FilterType.EDGE_SHARPNING
    parameters_class = LaplacianSharpenParameters
    initial_hyperparameters = LaplacianSharpenParameters()

    @classmethod
    def apply_filter(
        cls, image: np.ndarray, parameters: LaplacianSharpenParameters
    ) -> np.ndarray:
        """
        Apply Laplacian sharpening to the image.

        Args:
            image: Input image.
            parameters: LaplacianSharpenParameters.

        Returns:
            Sharpened image.
        """
        import cv2

        laplacian = cv2.Laplacian(
            image,
            cv2.CV_64F,
            ksize=parameters.ksize,
            scale=parameters.scale,
            delta=parameters.delta,
        )
        sharpened = cv2.subtract(image.astype(np.float64), laplacian)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened
