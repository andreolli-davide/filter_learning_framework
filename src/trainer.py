import time
from typing import List, Type
from optuna import Trial

import numpy as np
import optuna
from pydantic import BaseModel

from filters import (
    FilterHyperparametersHint,
    FilterType,
    IsFilterAdapter,
    LaplacianSharpenFilterAdapter,
    MedianBlurFilterAdapter,
    SaturationBoostFilterAdapter,
    SoftClaheFilterAdapter,
)


class Trainer:
    """
    Trainer class for optimizing a sequence of image preprocessing filters
    using Optuna for hyperparameter search.
    """

    filters_path: List[Type[IsFilterAdapter]]
    dataset: List[np.ndarray]

    class Task(BaseModel):
        """
        Represents a single optimization task, storing the iteration and parameters.
        """

        iteration: int
        parameters: List[BaseModel]

    def __init__(
        self, filters_path: List[Type[IsFilterAdapter]], dataset: List[np.ndarray]
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            filters_path: List of filter adapter classes in the order to be applied.
            dataset: List of images (as numpy arrays) to process.
        """
        # Ensure filters are ordered correctly and no NO_OP filters are in the path
        for i, filter_class in enumerate(filters_path):
            if filter_class.filter_type == FilterType.NO_OP:
                continue
            elif filter_class.filter_type != i + 1:
                raise ValueError(
                    f"Violation of ordering constraint at layer {i}: found '{filter_class.filter_type.name}'."
                )
        self.filters_path = filters_path
        self.dataset = dataset

    def objective(self, trial: Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            Simulated YOLO accuracy (float).
        """
        hyperparameters: List[BaseModel] = []

        # Suggest hyperparameters for each filter in the path
        for filter_class in self.filters_path:
            filter_parameters_class = filter_class.parameters_class
            filter_suggested_parameters = {}

            # Iterate over each hyperparameter field in the filter's parameter class
            for field_name, field_info in filter_parameters_class.model_fields.items():
                field_type = field_info.annotation
                type_hints = filter_parameters_class.__annotations__[field_name]

                # Extract hyperparameter hints from type metadata
                for field_hint in type_hints.__metadata__:
                    if isinstance(field_hint, FilterHyperparametersHint):
                        # Suggest integer hyperparameters
                        if field_type is int:
                            filter_suggested_parameters[field_name] = trial.suggest_int(
                                name=f"{filter_class.filter_type.name}_{field_name}",
                                low=field_hint.lower_bound,
                                high=field_hint.upper_bound,
                                step=field_hint.step
                                if field_hint.step is not None
                                else 1,
                            )
                        # Suggest float hyperparameters
                        if field_type is float:
                            filter_suggested_parameters[field_name] = (
                                trial.suggest_float(
                                    name=f"{filter_class.filter_type.name}_{field_name}",
                                    low=field_hint.lower_bound,
                                    high=field_hint.upper_bound,
                                    step=field_hint.step,
                                )
                            )

            # Store the suggested parameters for this filter
            hyperparameters.append(
                filter_class.parameters_class(**filter_suggested_parameters)
            )

        # Apply the filter pipeline to each image in the dataset
        for image in self.dataset:
            preprocessing_result = image.copy()
            for layer_index, filter_class in enumerate(self.filters_path):
                preprocessing_result = filter_class.apply_filter(
                    image=preprocessing_result,
                    parameters=hyperparameters[layer_index],
                )

        # Simulate YOLO inference latency and accuracy evaluation
        time.sleep(0.1)  # Simulate processing time
        return np.random.uniform(0.2, 0.7)  # Simulated accuracy


if __name__ == "__main__":
    # Create or load an Optuna study for YOLO preprocessing optimization
    study = optuna.create_study(
        study_name="yolo_preprocessing_optimization",
        storage="sqlite:///yolo_preprocessing_optimization.db",
        load_if_exists=True,
    )
    # Instantiate the Trainer with a sequence of filter adapters and a random dataset
    trainer = Trainer(
        filters_path=[
            MedianBlurFilterAdapter,
            SaturationBoostFilterAdapter,
            SoftClaheFilterAdapter,
            LaplacianSharpenFilterAdapter,
        ],
        dataset=[
            np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(256)
        ],
    )

    # Run the optimization for 100 trials
    study.optimize(trainer.objective, n_trials=100)

    # Print the best hyperparameters found
    print(study.best_params)
