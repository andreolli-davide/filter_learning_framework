from pathlib import Path
from typing import List, Literal, Self
from warnings import deprecated

from pydantic import BaseModel, Field
from ultralytics.models import YOLO

from dataset import (
    Dataset,
    DatasetSplit,
    Magnitude,
    MalariaStage,
    StagingDataset,
    TransientSample,
)


class YoloConfig(BaseModel):
    path: Path = Field(..., description="Dataset root path.")
    train: str = Field(default="images", description="Training data directory.")
    test: str = Field(default="images", description="Testing data directory.")
    val: str = Field(default="images", description="Validation data directory.")
    nc: int = Field(default=len(MalariaStage), description="Number of classes.")
    names: List[str] = Field(
        default=[stage.name for stage in MalariaStage],
        description="Names of the classes.",
    )

    def store_yaml(self) -> Path:
        config_path = self.path / "yolo_config.yaml"
        config_path.write_text(self.model_dump_yaml())
        return config_path

    def model_dump_yaml(self) -> str:
        import yaml

        dump = self.model_dump()
        dump["path"] = dump["path"].as_posix()
        return yaml.dump(dump)


class Yolo:
    """
    Wrapper for YOLO model operations including loading, evaluation, and configuration.

    Attributes:
        model_path (Path): Path to the YOLO model weights.
        yolo_model (YOLO): The loaded YOLO model instance.
        device (Literal["cpu", "mps", "cuda"]): Device to run the model on.
    """

    model_path: Path
    yolo_model: YOLO
    device: Literal["cpu", "mps", "cuda"] = "cpu"

    @deprecated("Yolo class should be instantiated via 'load_model' method.")
    def __init__(self, *_, **__):
        """
        Deprecated. Use 'load_model' class method to instantiate.
        """
        ...

    @classmethod
    def load_model(
        cls, model_path: Path, device: Literal["cpu", "mps", "cuda"] = "cpu"
    ) -> Self:
        """
        Loads a YOLO model from the specified path.

        Args:
            model_path (Path): Path to the YOLO model weights.
            device (Literal["cpu", "mps", "cuda"]): Device to run the model on.

        Returns:
            Yolo: An instance of the Yolo class with the model loaded.

        Raises:
            Exception: If the model path does not exist.
        """
        if not model_path.exists():
            raise Exception("Given YOLO path doesn't exist.")

        instance = object.__new__(cls)
        instance.model_path = model_path
        instance.yolo_model = YOLO(model_path.as_posix())
        instance.device = device

        return instance

    def evaluate(self, dataset: StagingDataset) -> float:
        """
        Evaluates the YOLO model on a temporary dataset.

        Args:
            dataset (TemporaryDataset): The dataset to evaluate on.

        Returns:
            float: The mAP@0.5 metric for the evaluation.
        """
        from ultralytics.utils.metrics import DetMetrics

        config = YoloConfig(
            path=Path(dataset.temporary_directory.name),
        )
        config_path = config.store_yaml()

        results: DetMetrics = self.yolo_model.val(
            data=config_path.as_posix(),
            split="test",
            save_json=False,
            plots=False,
            verbose=False,
            device=self.device,
        )

        return results.box.map50


if __name__ == "__main__":
    from filters import (
        ClaheParameters,
        MedianBlurFilterAdapter,
        MedianBlurParameters,
        ParametrizedFilter,
        SaturationBoostFilterAdapter,
        SaturationBoostParameters,
        ClaheFilterAdapter,
    )

    dataset = Dataset.load_from_directory(Path("resources/dataset"))
    model = Yolo.load_model(Path("resources/model.pt"))

    samples = dataset.pick_random_samples(
        magnitude=Magnitude.LCM, split=DatasetSplit.VAL
    )

    best_parameters: List[ParametrizedFilter] = [
        MedianBlurFilterAdapter.parametrized(MedianBlurParameters(kernel_size=5)),
        SaturationBoostFilterAdapter.parametrized(
            SaturationBoostParameters(boost_factor=1.6)
        ),
        ClaheFilterAdapter.parametrized(
            ClaheParameters(
                clip_limit=2.0,
                tile_grid_size=12,
            )
        ),
    ]

    transformed_samples: List[TransientSample] = []
    for sample in samples:
        transformed_sample = sample.apply_transform(best_parameters)
        transformed_samples.append(transformed_sample)

    transient_dataset = Dataset.create_transient_dataset(transformed_samples)

    temporary_dataset = transient_dataset.to_staging_dataset()
    result = model.evaluate(temporary_dataset)
    print(result)
