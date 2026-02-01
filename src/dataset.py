"""
dataset.py
This module defines a structured framework for handling datasets of images and their associated labels,
using Pydantic for parameter validation and type safety. It provides classes for representing image labels,
individual samples, and the dataset as a whole, including utilities for loading from disk or Kaggle,
caching images, and random sampling. The design supports extensibility and clear documentation for each
component and its parameters.

Main Components:
- Label: Represents the coordinates for the four corners of an image label.
- Sample: Encapsulates an image and its labels.
- StoredSample: Sample stored on filesystem with lazy image loading.
- TransientSample: Sample with image data in memory.
- Dataset: Abstract base class for managing collections of samples.
- StoredDataset: Dataset stored on filesystem.
- StagingDataset: Temporary dataset for staging samples before persistence.
- TransientDataset: In-memory dataset for volatile data operations.

Each class exposes its fields with descriptions, and methods are documented for clarity and maintainability.
"""

import tempfile
from abc import ABC, abstractmethod
from enum import IntEnum, StrEnum, auto
from pathlib import Path
from typing import Generic, List, Optional, Self, TypeVar
from warnings import deprecated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class MalariaStage(IntEnum):
    """
    Enumeration of malaria stages for labeling images.

    Values:
        SCHIZONT (int): Schizont stage.
        GAMETOCYTE (int): Gametocyte stage.
        RING (int): Ring stage.
        TROPHOZOITE (int): Trophozoite stage.
    """

    SCHIZONT = 0
    GAMETOCYTE = auto()
    RING = auto()
    TROPHOZOITE = auto()


class DatasetSplit(StrEnum):
    """
    Enum for dataset splits.

    Values:
        TRAIN (str): Training split.
        TEST (str): Test split.
        VAL (str): Validation split.
    """

    TRAIN = "train"
    TEST = "test"
    VAL = "val"


class Label(BaseModel):
    """
    Represents a label for an image in YOLO format.

    Attributes:
        malaria_stage (MalariaStage): Malaria stage associated with the label.
        x_center (float): Normalized x-coordinate of bounding box center.
        y_center (float): Normalized y-coordinate of bounding box center.
        width (float): Normalized width of bounding box.
        height (float): Normalized height of bounding box.
    """

    malaria_stage: MalariaStage = Field(
        ..., description="Malaria stage associated with the label."
    )
    x_center: float = Field(
        ..., description="Normalized x-coordinate of bounding box center."
    )
    y_center: float = Field(
        ..., description="Normalized y-coordinate of bounding box center."
    )
    width: float = Field(..., description="Normalized width of bounding box.")
    height: float = Field(..., description="Normalized height of bounding box.")

    @classmethod
    def from_string_line(cls, line: str) -> Self:
        """
        Creates a Label instance from a YOLO format string.

        Args:
            line (str): YOLO formatted string for the label.

        Returns:
            Label: The created Label instance.

        Raises:
            Exception: If the format string is invalid.
        """

        format_parts = line.strip().split()
        if len(format_parts) != 5:
            raise Exception(
                "Invalid YOLO format string. Expected 5 space-separated values."
            )

        return cls(
            malaria_stage=MalariaStage(int(format_parts[0])),
            x_center=float(format_parts[1]),
            y_center=float(format_parts[2]),
            width=float(format_parts[3]),
            height=float(format_parts[4]),
        )

    def to_string_line(self) -> str:
        """
        Converts the label to YOLO format string.

        Returns:
            str: YOLO formatted string for the label.
        """
        return f"{self.malaria_stage.value} {self.x_center} {self.y_center} {self.width} {self.height}"


class Magnitude(StrEnum):
    """
    Enum for sample magnitudes.

    Attributes:
        HCM (str): High-content magnitude.
        LCM (str): Low-content magnitude.
    """

    HCM = "hcm"
    LCM = "lcm"


class Sample(ABC, BaseModel):
    """
    Abstract base class representing a single dataset sample.

    Attributes:
        labels (List[Label]): List of labels for the image.
    """

    labels: List[Label] = Field(description="List of labels for the image.")

    @abstractmethod
    def load_image(self) -> np.ndarray:
        """
        Loads the image as a numpy array.

        Returns:
            np.ndarray: The loaded image.

        Raises:
            Exception: If the image cannot be loaded.
        """
        ...

    def apply_transform(self, filters: List["ParametrizedFilter"]) -> "TransientSample":
        """
        Applies a sequence of filters to the image and returns a TransientSample.

        Args:
            filters (List[ParametrizedFilter]): List of filters to apply sequentially.

        Returns:
            TransientSample: A new transient sample with the transformed image.
        """
        image = self.load_image()
        for filter in filters:
            image = filter.apply(image)

        return TransientSample(
            numpy_image=image,
            labels=self.labels,
        )


class StoredSample(Sample):
    """
    Represents a sample stored on the filesystem with lazy image loading.

    Attributes:
        image_path (Path): Path to the image file.
        labels_path (Path): Path to the labels file.
        magnitude (Optional[Magnitude]): Sample magnitude (HCM or LCM).
        split (Optional[DatasetSplit]): Dataset split (train, test, val).
    """

    image_path: Path = Field(..., description="Path to the image file.")
    labels_path: Path = Field(..., description="Path to the labels file.")
    magnitude: Optional[Magnitude] = Field(
        default=None, description="Sample magnitude (HCM or LCM)."
    )
    dataset_split: Optional[DatasetSplit] = Field(
        default=None, description="Dataset split (train, test, val)."
    )
    _cached_numpy_image: Optional[np.ndarray] = PrivateAttr(default=None)

    def load_image(self) -> np.ndarray:
        """
        Loads the image from disk as a numpy array and caches it.

        Returns:
            np.ndarray: A copy of the loaded image.

        Raises:
            Exception: If the image cannot be loaded.
        """
        if self._cached_numpy_image is None:
            import cv2

            loaded_image = cv2.imread(str(self.image_path))
            if loaded_image is None:
                raise Exception(f"Failed to load image '{self.image_path}'.")
            self._cached_numpy_image = loaded_image

        return self._cached_numpy_image.copy()

    def unload_image(self):
        """
        Unloads the cached image from memory.
        """
        self._numpy_image = None

    def to_transient_sample(
        self, unload_after_conversion: bool = True
    ) -> "TransientSample":
        """
        Converts the StoredSample to a TransientSample by loading the image into memory.

        Args:
            unload_after_conversion (bool): If True, unloads the cached image after conversion. Default is True.

        Returns:
            TransientSample: The converted in-memory sample.
        """
        loaded_image = self.load_image()
        transient_sample = TransientSample(
            labels=self.labels,
            numpy_image=loaded_image,
        )
        if unload_after_conversion:
            self.unload_image()
        return transient_sample


class TransientSample(Sample):
    """
    Represents a sample with image data held in memory (volatile/non-persistent).

    Attributes:
        numpy_image (np.ndarray): The image data as a numpy array.
    """

    numpy_image: np.ndarray = Field(...)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_image(self) -> np.ndarray:
        """
        Returns a copy of the in-memory image.

        Returns:
            np.ndarray: A copy of the image.
        """

        return self.numpy_image.copy()

    def store_image(self, image_path: Path, labels_path: Path):
        """
        Stores the image and labels to disk.

        Args:
            image_path (Path): Destination path for the image file.
            labels_path (Path): Destination path for the labels file.

        Raises:
            Exception: If paths already exist or are directories.
        """
        import cv2

        if image_path.exists():
            raise Exception(f"Image path '{image_path}' already exists.")

        if image_path.is_dir():
            raise Exception(f"Image path '{image_path}' is a directory.")

        if labels_path.exists():
            raise Exception(f"Labels path '{labels_path}' already exists.")

        if labels_path.is_dir():
            raise Exception(f"Labels path '{labels_path}' is a directory.")

        cv2.imwrite(str(image_path), self.numpy_image)

        labels_yolo = "\n".join([label.to_string_line() for label in self.labels])
        labels_path.write_text(labels_yolo)


SampleT = TypeVar("SampleT", bound=Sample)


class Dataset(ABC, Generic[SampleT]):
    """
    Abstract base class representing a dataset containing multiple samples.

    Attributes:
        base_path (Optional[Path]): Base directory of the dataset.
        samples (List[SampleT]): List of loaded samples.
    """

    base_path: Optional[Path] = None
    samples: List[SampleT] = []

    def __init__(self):
        """
        Initializes a new Dataset instance.
        """
        self.base_path: Optional[Path] = None
        self.samples: List[SampleT] = []

    @staticmethod
    def load_from_kaggle(destination_path: Optional[Path] = None) -> "StoredDataset":
        """
        Downloads and loads the dataset from Kaggle.

        Args:
            destination_path (Optional[Path]): Where to download the dataset.

        Returns:
            StoredDataset: The loaded dataset from Kaggle.
        """
        import kagglehub

        download_path = kagglehub.dataset_download(
            "davidesenette/malaria-hcm-lcm-1000",
            path=str(destination_path) if destination_path else None,
        )

        instance = object.__new__(StoredDataset)
        instance.base_path = Path(download_path)
        instance._load_samples()
        return instance

    @staticmethod
    def load_from_directory(path: Path) -> "StoredDataset":
        """
        Loads the dataset from a local directory.

        Args:
            path (Path): Path to the dataset directory.

        Returns:
            StoredDataset: The loaded dataset.

        Raises:
            Exception: If the path does not exist or is not a directory.
        """
        if not path.exists():
            raise Exception(f"Given path '{path}' doesn't exist.")

        if not path.is_dir():
            raise Exception(f"Given path '{path}' is not a directory.")

        instance = object.__new__(StoredDataset)
        instance.base_path = path
        instance._load_samples()
        return instance

    @staticmethod
    def create_staging(samples: List[StoredSample]) -> "StagingDataset":
        """
        Creates a staging dataset from a list of StoredSample instances by copying their files to a temporary directory.

        Args:
            samples (List[StoredSample]): List of stored samples to include in the staging dataset.

        Returns:
            StagingDataset: The created staging dataset.

        Raises:
            Exception: If a sample does not have a valid labels_path.
        """
        from tqdm import tqdm

        staging_temporary_directory = tempfile.TemporaryDirectory()
        staging_directory_path = Path(staging_temporary_directory.name)

        staging_images_directory = staging_directory_path / "images"
        staging_labels_directory = staging_directory_path / "labels"

        staging_images_directory.mkdir(parents=True, exist_ok=True)
        staging_labels_directory.mkdir(parents=True, exist_ok=True)

        staging_samples: List[StoredSample] = []

        for source_sample in tqdm(
            samples, desc="Copying samples to staging directory", unit="sample"
        ):
            if source_sample.labels_path is None:
                raise Exception(
                    "Cannot copy sample to staging dataset: labels_path is None."
                )

            staged_image_path = staging_images_directory / source_sample.image_path.name
            staged_labels_path = (
                staging_labels_directory / source_sample.labels_path.name
            )

            from shutil import copy2

            copy2(source_sample.image_path, staged_image_path)
            copy2(source_sample.labels_path, staged_labels_path)

            staging_samples.append(
                StoredSample(
                    image_path=staged_image_path,
                    labels_path=staged_labels_path,
                    labels=source_sample.labels,
                    magnitude=source_sample.magnitude,
                    dataset_split=getattr(source_sample, "split", None),
                )
            )

        instance = object.__new__(StagingDataset)
        instance.base_path = staging_directory_path
        instance.temporary_directory = staging_temporary_directory
        instance.samples = staging_samples
        return instance

    @staticmethod
    def create_transient_dataset(samples: List[TransientSample]) -> "TransientDataset":
        """
        Creates a transient (in-memory) dataset from a list of TransientSample instances.

        Args:
            samples (List[TransientSample]): List of transient samples.

        Returns:
            TransientDataset: The created in-memory dataset.
        """
        instance = object.__new__(TransientDataset)
        instance.base_path = None
        instance.samples = samples
        return instance

    def unload_cached_images(self):
        """
        Unloads all cached images from memory for all StoredSample instances.
        """
        from tqdm import tqdm

        for sample in tqdm(self.samples, desc="Unloading cached images", unit="sample"):
            if isinstance(sample, StoredSample):
                sample.unload_image()

    def pick_random_samples(
        self,
        sample_count: Optional[int] = None,
        magnitude: Optional[Magnitude] = None,
        split: Optional[DatasetSplit] = None,
    ) -> List[SampleT]:
        """
        Picks random samples from the dataset with optional filtering.

        Args:
            sample_count (Optional[int]): Number of samples to pick. If None, returns all matching samples.
            magnitude (Optional[Magnitude]): Filter by magnitude (HCM or LCM). Only applies to StoredSample instances.
            split (Optional[DatasetSplit]): Filter by dataset split (train, test, val). Only applies to StoredSample instances.

        Returns:
            List[SampleT]: List of randomly picked samples.

        Raises:
            Exception: If no filtering parameters are provided or sample_count exceeds available samples.
        """
        if sample_count is None and magnitude is None and split is None:
            raise Exception(
                "At least one filtering parameter (sample_count, magnitude, or split) must be provided."
            )

        import random

        filtered_samples = self.samples

        if magnitude is not None:
            filtered_samples = [
                sample
                for sample in filtered_samples
                if isinstance(sample, StoredSample) and sample.magnitude == magnitude
            ]

        if split is not None:
            filtered_samples = [
                sample
                for sample in filtered_samples
                if isinstance(sample, StoredSample) and sample.dataset_split == split
            ]

        if sample_count is not None:
            if sample_count > len(filtered_samples):
                raise Exception(
                    f"Requested {sample_count} samples but only {len(filtered_samples)} are available after filtering."
                )
            filtered_samples = random.sample(filtered_samples, sample_count)

        return filtered_samples


class StoredDataset(Dataset[StoredSample]):
    """
    Dataset implementation for datasets stored on the filesystem.

    Provides methods for loading, copying, and managing persistent datasets.
    """

    @deprecated(
        "Use 'Dataset.load_from_kaggle' or 'Dataset.load_from_directory' to instantiate StoredDataset."
    )
    def __init__(self):
        """
        Prevent direct instantiation; use class methods instead.

        Raises:
            RuntimeError: Always raised to prevent direct instantiation.
        """
        raise RuntimeError(
            "Use 'Dataset.load_from_kaggle' or 'Dataset.load_from_directory' to instantiate StoredDataset."
        )

    def _load_samples(self):
        """
        Loads all samples from the dataset directory into memory.

        Raises:
            Exception: If base_path is not set, samples are already loaded, or dataset structure is invalid.
        """
        from tqdm import tqdm

        if self.base_path is None:
            raise Exception("Cannot load samples: base_path is not set.")

        image_paths_generator = self.base_path.rglob("*.png")

        for image_path_absolute in tqdm(
            image_paths_generator, desc="Loading samples from disk", unit="sample"
        ):
            image_path_relative = image_path_absolute.relative_to(self.base_path)

            magnitude: Magnitude
            match image_path_relative.parts[0]:
                case "source":
                    magnitude = Magnitude.HCM
                case "target":
                    magnitude = Magnitude.LCM
                case _:
                    raise Exception(
                        f"Invalid dataset structure: unexpected directory '{image_path_relative.parts[0]}'."
                    )

            dataset_split = DatasetSplit(image_path_relative.parts[2])

            labels_file_path = (
                self.base_path
                / image_path_relative.parts[0]
                / "labels"
                / Path(*image_path_relative.parts[2:])
            )

            labels_file_path = labels_file_path.with_suffix(".txt")
            labels_file_content = labels_file_path.read_text()
            parsed_labels: List[Label] = []

            for label_line in labels_file_content.splitlines():
                parsed_labels.append(Label.from_string_line(label_line))

            self.samples.append(
                StoredSample(
                    image_path=image_path_absolute,
                    labels_path=labels_file_path,
                    labels=parsed_labels,
                    magnitude=magnitude,
                    dataset_split=dataset_split,
                )
            )

    def store(self, destination_path: Path):
        """
        Copies the dataset to a new destination directory.

        Args:
            destination_path (Path): The path to copy the dataset to.

        Raises:
            Exception: If base_path is not set or destination already exists.
        """
        if self.base_path is None:
            raise Exception(
                "Cannot store dataset: base_path is not set. Use 'load_from_kaggle' or 'load_from_directory' first."
            )

        if destination_path.exists():
            raise Exception(f"Destination path '{destination_path}' already exists.")

        from shutil import copytree

        destination_path.mkdir(parents=True)
        copytree(str(self.base_path), str(destination_path))


class StagingDataset(StoredDataset):
    """
    Dataset implementation for staging datasets stored in a temporary directory.

    This dataset type is used as an intermediate stage before persisting data,
    automatically cleaning up temporary files when no longer needed.

    Attributes:
        temporary_directory (tempfile.TemporaryDirectory): Reference to the temporary directory for cleanup.
    """

    temporary_directory: tempfile.TemporaryDirectory = PrivateAttr()

    @deprecated("Use 'Dataset.create_staging' to instantiate StagingDataset.")
    def __init__(self):
        """
        Prevents direct instantiation of StagingDataset.

        Raises:
            RuntimeError: Always raised to prevent direct instantiation.
        """
        raise RuntimeError(
            "Use 'Dataset.create_staging' to instantiate StagingDataset."
        )

    def __del__(self):
        """
        Cleans up the temporary directory when the object is garbage collected.
        """
        if (
            hasattr(self, "temporary_directory")
            and self.temporary_directory is not None
        ):
            self.temporary_directory.cleanup()


class TransientDataset(Dataset[TransientSample]):
    """
    Dataset implementation for in-memory (transient/volatile) datasets.

    Stores all sample data in memory for fast access and transformation operations.
    Data is lost when the object is destroyed unless explicitly persisted.
    """

    @deprecated(
        "Use 'Dataset.create_transient_dataset' to instantiate TransientDataset."
    )
    def __init__(self):
        """
        Prevent direct instantiation; use class methods instead.

        Raises:
            RuntimeError: Always raised to prevent direct instantiation.
        """
        raise RuntimeError(
            "Use 'Dataset.create_transient_dataset' to instantiate TransientDataset."
        )

    def to_staging(self) -> StagingDataset:
        """
        Converts the transient dataset to a staging dataset by writing samples to a temporary directory.

        Returns:
            StagingDataset: A staging dataset with samples stored in a temporary directory.
        """
        import tempfile
        import uuid

        import tqdm

        stored_samples: List[StoredSample] = []
        staging_temporary_directory = tempfile.TemporaryDirectory()
        staging_directory_path = Path(staging_temporary_directory.name)

        for transient_sample in tqdm.tqdm(
            self.samples, desc="Storing samples in staging directory", unit="sample"
        ):
            unique_sample_name = uuid.uuid4().hex

            staged_image_path = (
                staging_directory_path.joinpath("images")
                .joinpath(unique_sample_name)
                .with_suffix(".png")
            )
            staged_image_path.parent.mkdir(parents=True, exist_ok=True)

            staged_labels_path = (
                staging_directory_path.joinpath("labels")
                .joinpath(unique_sample_name)
                .with_suffix(".txt")
            )
            staged_labels_path.parent.mkdir(parents=True, exist_ok=True)

            transient_sample.store_image(
                image_path=staged_image_path,
                labels_path=staged_labels_path,
            )

            stored_samples.append(
                StoredSample(
                    image_path=staged_image_path,
                    labels_path=staged_labels_path,
                    labels=transient_sample.labels,
                )
            )

        instance = object.__new__(StagingDataset)
        instance.base_path = staging_directory_path
        instance.temporary_directory = staging_temporary_directory
        instance.samples = stored_samples
        return instance

    def store(self, destination_path: Path) -> StoredDataset:
        """
        Persists the transient dataset to disk as a StoredDataset.

        Args:
            destination_path (Path): Directory where the dataset will be stored.

        Returns:
            StoredDataset: The newly created persistent dataset.

        Raises:
            Exception: If destination path already exists or is a file.
        """
        import uuid

        import cv2
        from tqdm import tqdm

        if destination_path.exists():
            raise Exception(f"Destination path '{destination_path}' already exists.")

        if destination_path.is_file():
            raise Exception(
                f"Destination path '{destination_path}' is a file, expected directory."
            )

        destination_path.mkdir(parents=True, exist_ok=True)

        destination_images_directory = destination_path / "images"
        destination_labels_directory = destination_path / "labels"

        destination_images_directory.mkdir(parents=True, exist_ok=True)
        destination_labels_directory.mkdir(parents=True, exist_ok=True)

        persisted_samples: List[StoredSample] = []

        for transient_sample in tqdm(
            self.samples, desc="Persisting samples to disk", unit="sample"
        ):
            unique_sample_name = uuid.uuid4().hex

            persisted_image_path = destination_images_directory.joinpath(
                unique_sample_name
            ).with_suffix(".png")
            persisted_labels_path = destination_labels_directory.joinpath(
                unique_sample_name
            ).with_suffix(".txt")

            cv2.imwrite(str(persisted_image_path), transient_sample.numpy_image)
            labels_yolo_format = "\n".join(
                [label.to_string_line() for label in transient_sample.labels]
            )
            persisted_labels_path.write_text(labels_yolo_format)

            persisted_samples.append(
                StoredSample(
                    image_path=persisted_image_path,
                    labels_path=persisted_labels_path,
                    labels=transient_sample.labels,
                )
            )

        instance = object.__new__(StoredDataset)
        instance.base_path = destination_path
        instance.samples = persisted_samples
        return instance


if __name__ == "__main__":
    from filters import (
        ParametrizedFilter,
        SoftClaheFilterAdapter,
    )

    source_dataset = Dataset.load_from_directory(Path("resources/dataset"))
    random_samples: List[StoredSample] = source_dataset.pick_random_samples(10)
    original_transient_dataset = Dataset.create_transient_dataset(
        [sample.to_transient_sample() for sample in random_samples]
    )
    original_transient_dataset.store(Path("source_images/"))
    transformed_samples: List[TransientSample] = []
    for sample in random_samples:
        transformed_sample = sample.apply_transform(
            [
                SoftClaheFilterAdapter.parametrized(
                    SoftClaheFilterAdapter.initial_hyperparameters
                )
            ]
        )
        transformed_samples.append(transformed_sample)
    transformed_transient_dataset = Dataset.create_transient_dataset(
        transformed_samples
    )
    persisted_dataset = transformed_transient_dataset.store(Path("./saved_dataset"))
    print("Finished processing dataset.")
