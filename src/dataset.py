"""
dataset.py
This module defines a structured framework for handling datasets of images and their associated labels,
using Pydantic for parameter validation and type safety. It provides classes for representing image labels,
individual samples, and the dataset as a whole, including utilities for loading from disk or Kaggle,
caching images, and random sampling. The design supports extensibility and clear documentation for each
component and its parameters.

Main Components:
- Label: Represents the coordinates for the four corners of an image label.
- Sample: Encapsulates an image, its labels, and its magnitude (HCM or LCM).
- Dataset: Manages a collection of samples, supports loading, copying, and random sampling.

Each class exposes its fields with descriptions, and methods are documented for clarity and maintainability.
"""

import tempfile
from abc import ABC
from enum import StrEnum
from pathlib import Path
from typing import List, Optional
from warnings import deprecated

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr


class Label(BaseModel):
    """
    Represents a label for an image, containing the coordinates for the four corners.
    """

    top_left: float = Field(..., description="Top-left coordinate value.")
    top_right: float = Field(..., description="Top-right coordinate value.")
    bottom_left: float = Field(..., description="Bottom-left coordinate value.")
    bottom_right: float = Field(..., description="Bottom-right coordinate value.")


class SampleMagnitude(StrEnum):
    """
    Enum for sample magnitudes.
    """

    HCM = "hcm"
    LCM = "lcm"


class Sample(BaseModel):
    """
    Represents a single dataset sample, including image path, labels, and magnitude.
    """

    image_path: Path = Field(..., description="Path to the image file.")
    labels_path: Optional[Path] = Field(
        default=None, description="Path to the labels file."
    )
    labels: List[Label] = Field(description="List of labels for the image.")
    magnitude: SampleMagnitude = Field(
        ..., description="Sample magnitude (HCM or LCM)."
    )
    _numpy_image: Optional[np.ndarray] = PrivateAttr(default=None)

    def load_image(self) -> np.ndarray:
        """
        Loads the image from disk as a numpy array and caches it.
        Returns:
            np.ndarray: The loaded image.
        Raises:
            Exception: If the image cannot be loaded.
        """
        if self._numpy_image is None:
            import cv2

            image = cv2.imread(self.image_path.as_posix())
            if image is None:
                raise Exception(f"Failed to load image '{self.image_path.as_posix()}'.")
            self._numpy_image = image

        return self._numpy_image

    def unload_image(self):
        """
        Unloads the cached image from memory.
        """
        self._numpy_image = None


class Dataset(ABC):
    """
    Represents a dataset containing multiple samples.
    """

    base_path: Optional[Path] = None  # Base directory of the dataset
    samples: List[Sample] = []  # List of loaded samples

    @classmethod
    def load_from_kaggle(
        cls, destination_path: Optional[Path] = None
    ) -> FileSystemDataset:
        """
        Downloads and loads the dataset from Kaggle.
        Args:
            destination_path (Optional[Path]): Where to download the dataset.
        """
        import kagglehub

        download_path = kagglehub.dataset_download(
            "shahidzikria/malariahcm1000",
            path=destination_path.as_posix() if destination_path else None,
        )

        instance = object.__new__(FileSystemDataset)
        instance.base_path = Path(download_path)
        instance._load_samples()
        return instance

    @classmethod
    def load_from_path(cls, path: Path) -> FileSystemDataset:
        """
        Loads the dataset from a local directory.
        Args:
            path (Path): Path to the dataset directory.
        Raises:
            Exception: If the path does not exist or is not a directory.
        """
        if not path.exists():
            raise Exception("Given path doesn't exist.")

        if not path.is_dir():
            raise Exception("Given path is not a directory.")

        instance = object.__new__(FileSystemDataset)
        instance.base_path = path
        instance._load_samples()
        return instance

    @classmethod
    def create_temporary(cls, samples: List[Sample]) -> TemporaryDataset:
        from tqdm import tqdm

        temporary_directory = tempfile.TemporaryDirectory()
        temporary_directory_path = Path(temporary_directory.name)

        images_path = temporary_directory_path / "images"
        labels_path = temporary_directory_path / "labels"

        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)

        temporary_samples: List[Sample] = []

        for sample in tqdm(
            samples, desc="Copying samples in a temporary directory.", unit="img"
        ):
            if sample.labels_path is None:
                raise Exception(
                    "A sample created without a path cannot be copied to a temporary dataset."
                )

            temporary_image_path = images_path / sample.image_path.name
            temporary_labels_path = labels_path / sample.labels_path.name

            sample.image_path.copy(
                temporary_image_path.as_posix(),
            )
            sample.labels_path.copy(
                temporary_labels_path.as_posix(),
            )

            temporary_samples.append(
                Sample(
                    image_path=temporary_image_path,
                    labels_path=temporary_labels_path,
                    labels=sample.labels,
                    magnitude=sample.magnitude,
                )
            )

        instance = object.__new__(TemporaryDataset)
        instance.base_path = temporary_directory_path
        instance.samples = temporary_samples
        return instance

    def unload_cached_images(self):
        """
        Unloads all cached images from memory for all samples.
        """
        from tqdm import tqdm

        for sample in tqdm(self.samples, desc="Unloading images", unit="img"):
            sample.unload_image()

    def pick_random_samples(self, k: int) -> List[Sample]:
        """
        Picks k random samples from the dataset.
        Args:
            k (int): Number of samples to pick.
            type (Literal["hcm", "lcm"]): Type of samples to pick (currently unused).
        Returns:
            List[Sample]: List of randomly picked samples.
        Raises:
            Exception: If base_path is not set.
        """
        if self.base_path is None:
            raise Exception(
                "Use 'load_from_kaggle' or 'load_from_path' before trying to pick a sample."
            )

        import random

        return random.choices(self.samples, k=k)


class FileSystemDataset(Dataset):
    """
    Dataset implementation for datasets stored on the filesystem.
    Provides methods for loading, copying, and cleaning up datasets.
    """

    @deprecated(
        "Use 'Template.load_from_kaggle' or 'Template.load_from_path' to instantiate FileSystemDataset."
    )
    def __init__(self, *args, **kwargs):
        # Prevent direct instantiation; use class methods instead.
        raise RuntimeError(
            "Use 'Template.load_from_kaggle' or 'Template.load_from_path' to instantiate FileSystemDataset."
        )

    def _load_samples(self):
        """
        Loads all samples from the dataset directory into memory.
        Raises:
            Exception: If base_path is not set or dataset is corrupted.
        """
        from tqdm import tqdm

        if self.base_path is None:
            raise Exception("External use of '_load_sample' is prohibited.")
        if len(self.samples) > 0:
            print("Samples are already loaded.")

        image_paths_generator = self.base_path.rglob("*.png")

        for image_path_absolute in tqdm(
            image_paths_generator, desc="Loading samples", unit="img"
        ):
            image_path = image_path_absolute.relative_to(self.base_path)

            magnitude: SampleMagnitude
            match image_path.parts[0]:
                case "source":
                    magnitude = SampleMagnitude.HCM
                case "target":
                    magnitude = SampleMagnitude.LCM
                case _:
                    raise Exception("Dataset seems to be corrupted.")

            labels_path = (
                self.base_path
                / image_path.parts[0]
                / "labels"
                / Path(*image_path.parts[2:])
            )

            labels_path = labels_path.with_suffix(".txt")
            labels_raw = labels_path.read_text()
            labels: List[Label] = []

            for label_raw in labels_raw.split("\n"):
                label_raw_parts = label_raw.split(" ")
                if len(label_raw_parts) != 5:
                    continue
                labels.append(
                    Label(
                        top_left=float(label_raw_parts[1]),
                        top_right=float(label_raw_parts[2]),
                        bottom_left=float(label_raw_parts[3]),
                        bottom_right=float(label_raw_parts[4]),
                    )
                )

            self.samples.append(
                Sample(
                    image_path=image_path_absolute,
                    labels_path=labels_path,
                    labels=labels,
                    magnitude=magnitude,
                )
            )

    def copy_to(self, destination_path: Path):
        """
        Copies the dataset to a new destination.
        Args:
            destination_path (Path): The path to copy the dataset to.
        Raises:
            Exception: If base_path is not set or destination already exists.
        """
        if self.base_path is None:
            raise Exception(
                "Use 'load_from_kaggle' or 'load_from_path' before trying to copy the dataset."
            )

        if destination_path.exists():
            raise Exception("Destination path already exists.")

        from shutil import copytree

        destination_path.mkdir(parents=True)
        copytree(self.base_path.as_posix(), str(destination_path))


class TemporaryDataset(Dataset):
    """
    Dataset implementation for temporary datasets stored in a temporary directory.

    Attributes:
        temporary_directory (Optional[tempfile.TemporaryDirectory]): Reference to the temporary directory for cleanup.
    """

    temporary_directory: Optional[tempfile.TemporaryDirectory] = None

    @deprecated("Use 'Template.create_temporary' to instantiate FileSystemDataset.")
    def __init__(self, *_, **__):
        """
        Prevents direct instantiation of TemporaryDataset.
        Use the provided class methods for proper initialization.
        """
        raise RuntimeError(
            "Use 'Template.create_temporary' to instantiate FileSystemDataset."
        )

    def __del__(self):
        """
        Cleans up the temporary directory if it exists when the object is deleted.
        """
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()


if __name__ == "__main__":
    dataset = Dataset.load_from_path(Path("resources/dataset"))
    batch = dataset.pick_random_samples(20)
    temporary_dataset = TemporaryDataset.create_temporary(batch)
