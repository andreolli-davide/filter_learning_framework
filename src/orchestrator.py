"""
orchestrator.py

This module provides a comprehensive framework for orchestrating the optimization of image
preprocessing filter combinations using Optuna. The orchestrator implements a layer-by-layer
optimization strategy where filters are selected and optimized sequentially, building upon the
best combination found in previous layers.

Key Features:
- Checkpoint support: Save training progress at any point
- Full customization: Configure filter layers, optimization parameters, and behavior
- Comprehensive logging: Track all trials, best results, and progress
- Layer-by-layer optimization: Build optimal filter pipelines incrementally

Main Components:
- OrchestratorConfig: Configuration class for customizing orchestrator behavior
- TrialResult: Represents a single optimization trial result
- FilterOptimizationStudy: Tracks optimization progress for a filter combination
- OrchestratorLog: Main log structure for checkpointing and progress tracking
- Orchestrator: Main orchestrator class that manages the optimization process

The orchestrator can be fully customized through the OrchestratorConfig class, which allows
users to specify filter layers, optimization parameters, paths, and other behavior settings.
All progress can be saved to checkpoint files to track optimization results.
"""

import json
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
)

import optuna
from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer

from dataset import Dataset, DatasetSplit, Magnitude, StoredDataset
from filters import (
    BilateralFilterAdapter,
    ClaheFilterAdapter,
    FilterAdapter,
    GammaCorrectionFilterAdapter,
    LaplacianSharpenFilterAdapter,
    MedianBlurFilterAdapter,
    NoOpFilterAdapter,
    ParametrizedFilter,
    SaturationBoostFilterAdapter,
    UnsharpMaskFilterAdapter,
)
from trainer import Trainer
from yolo import Yolo


class TrialResult(BaseModel):
    """
    Represents the result of a single optimization trial.

    Attributes:
        trial_number: The trial number within the Optuna study.
        map_50: Mean Average Precision at IoU threshold 0.5, the optimization metric.
        filters: List of parametrized filters used in this trial.
    """

    trial_number: Annotated[
        int, Field(..., description="Trial number in the Optuna study")
    ]
    map_50: Annotated[
        float,
        Field(..., description="Mean Average Precision at IoU 0.5 for this trial"),
    ]
    filters: Annotated[
        List[ParametrizedFilter],
        Field(..., description="Parametrized filters used in this trial"),
        PlainSerializer(
            lambda filters: [
                {
                    "name": f.adapter.name,
                    "parameters": f.parameters.model_dump(),
                }
                for f in filters
            ],
            List[Dict[str, Any]],
        ),
    ]


class FilterOptimizationStudy(BaseModel):
    """
    Tracks optimization progress for a specific filter combination.

    Attributes:
        study_name: Unique name for the Optuna study.
        filters: List of filter adapter classes in this combination.
        best_trial: The best trial result found so far (None if no trials completed).
        all_trials: List of all completed trial results.
        completed_trials_count: Number of completed trials for this combination.
    """

    study_name: Annotated[
        str, Field(..., description="Unique name for the Optuna study")
    ]
    filters: Annotated[
        List[Type[FilterAdapter]],
        Field(..., description="Filter adapter classes in this combination"),
        PlainSerializer(
            lambda filters: [f.name for f in filters],
            List[str],
        ),
    ]
    best_trial: Annotated[
        Optional[TrialResult],
        Field(default=None, description="Best trial result found so far"),
    ] = None
    all_trials: Annotated[
        List[TrialResult],
        Field(..., description="All completed trial results"),
    ] = []
    completed_trials_count: Annotated[
        int,
        Field(default=0, description="Number of completed trials for this combination"),
    ] = 0

    def add_trial(self, trial_result: TrialResult) -> None:
        """
        Adds a new trial result and updates the best trial if needed.

        Args:
            trial_result: The trial result to add.
        """
        self.all_trials.append(trial_result)
        self.completed_trials_count += 1

        if self.best_trial is None or trial_result.map_50 > self.best_trial.map_50:
            self.best_trial = trial_result


class OrchestratorLog(BaseModel):
    """
    Main log structure for tracking orchestrator state and progress.

    This class serves as the checkpoint format, allowing the orchestrator to save
    and resume its state. It tracks all optimization studies, the current layer
    being processed, and the best results found so far.

    Attributes:
        reports: List of optimization studies for different filter combinations.
        current_layer_index: Index of the current layer being processed (0-based).
        best_map_50: Best mAP@50 value found across all trials.
        best_filters_combination: Best filter combination found so far.
    """

    reports: Annotated[
        List[FilterOptimizationStudy],
        Field(default_factory=list, description="All optimization studies"),
    ]
    current_layer_index: Annotated[
        int,
        Field(
            default=0,
            description="Current layer index being processed (0-based, resumable)",
        ),
    ] = 0
    best_map_50: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Best mean Average Precision at IoU 0.5 found so far",
        ),
    ] = None
    best_filters_combination: Annotated[
        List[ParametrizedFilter],
        Field(
            default_factory=list,
            description="Best filter combination found so far",
        ),
        PlainSerializer(
            lambda filters: [
                {
                    "name": f.adapter.name,
                    "parameters": f.parameters.model_dump(),
                }
                for f in filters
            ],
            List[Dict[str, Any]],
        ),
    ]

    def save(self, log_file_path: Path) -> None:
        """
        Saves the orchestrator log to disk as JSON.

        Args:
            log_file_path: Path where the log should be saved.
        """
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_path.write_text(self.model_dump_json(indent=2))

    def get_report(
        self, filters: List[Type[FilterAdapter]]
    ) -> Optional[FilterOptimizationStudy]:
        """
        Retrieves a report for the given filter combination, if it exists.

        Args:
            filters: List of filter adapter classes to search for.

        Returns:
            FilterOptimizationStudy if found, None otherwise.
        """
        filters_without_noop = [f for f in filters if f != NoOpFilterAdapter]
        filter_names = [f.name for f in filters_without_noop]

        for report in self.reports:
            report_filters_without_noop = [
                name for name in report.filters if name != "NoOpFilterAdapter"
            ]
            if report_filters_without_noop == filter_names:
                return report
        return None

    def add_or_get_report(
        self, filters: List[Type[FilterAdapter]], study_name: str
    ) -> FilterOptimizationStudy:
        """
        Gets existing report or creates a new one for the given filter combination.

        Args:
            filters: List of filter adapter classes.
            study_name: Name for the Optuna study.

        Returns:
            FilterOptimizationStudy: Existing or newly created study report.
        """
        existing_report = self.get_report(filters)
        if existing_report is not None:
            return existing_report

        new_report = FilterOptimizationStudy(
            filters=filters,
            study_name=study_name,
        )
        self.reports.append(new_report)
        return new_report

    def update_best_if_needed(self, report: FilterOptimizationStudy) -> bool:
        """
        Updates the best combination if the report has a better result.

        Args:
            report: The optimization study to check.

        Returns:
            True if the best combination was updated, False otherwise.
        """
        if report.best_trial is None:
            return False

        if self.best_map_50 is None or report.best_trial.map_50 > self.best_map_50:
            self.best_map_50 = report.best_trial.map_50
            self.best_filters_combination = report.best_trial.filters
            return True
        return False


class OrchestratorConfig(BaseModel):
    """
    Configuration class for customizing orchestrator behavior.

    This class allows full customization of the orchestrator's behavior, including
    filter layers, optimization parameters, paths, and callback functions. All
    settings can be saved to JSON for reproducibility.

    Attributes:
        filter_layers: List of filter layers, where each layer is a list of
            filter adapter classes to choose from. The orchestrator will optimize
            one filter from each layer sequentially.
        n_trials_per_combination: Number of Optuna trials to run for each
            filter combination.
        optuna_db_path: Path to the Optuna SQLite database for storing studies.
        checkpoint_path: Path to save orchestrator checkpoints.
        skip_all_noop: If True, skip combinations where all filters are NoOp.
        study_name_prefix: Prefix to use for Optuna study names.
        trial_callback: Optional callback function called after each trial completes.
        layer_callback: Optional callback function called after each layer completes.
        optuna_study_kwargs: Additional keyword arguments to pass to optuna.create_study.
    """

    filter_layers: Annotated[
        List[List[Type[FilterAdapter]]],
        Field(
            ...,
            description=(
                "List of filter layers. Each layer is a list of filter adapter classes. "
                "The orchestrator optimizes one filter from each layer sequentially."
            ),
        ),
        PlainSerializer(
            lambda layers: [[f.name for f in layer] for layer in layers],
        ),
    ]
    n_trials_per_combination: Annotated[
        int,
        Field(
            default=100,
            ge=1,
            description="Number of Optuna trials to run for each filter combination",
        ),
    ] = 100
    optuna_db_path: Annotated[
        Path,
        Field(..., description="Path to Optuna SQLite database for storing studies"),
    ]
    checkpoint_path: Annotated[
        Path,
        Field(..., description="Path to save/load orchestrator checkpoints"),
    ]
    skip_all_noop: Annotated[
        bool,
        Field(
            default=True,
            description="If True, skip combinations where all filters are NoOp",
        ),
    ] = True
    study_name_prefix: Annotated[
        str,
        Field(
            default="",
            description="Prefix to use for Optuna study names (empty for no prefix)",
        ),
    ] = ""
    trial_callback: Annotated[
        Optional[Callable[[TrialResult, FilterOptimizationStudy], None]],
        Field(
            default=None,
            description=(
                "Optional callback function called after each trial completes. "
                "Receives the trial result and the study report."
            ),
        ),
    ] = None
    layer_callback: Annotated[
        Optional[
            Callable[
                [
                    int,
                    List[Type[FilterAdapter]],
                    Optional[float],
                    List[ParametrizedFilter],
                ],
                None,
            ]
        ],
        Field(
            default=None,
            description=(
                "Optional callback function called after each layer completes. "
                "Receives: layer_index, best_filters, best_map_50, best_parametrized_filters."
            ),
        ),
    ] = None
    optuna_study_kwargs: Annotated[
        Dict[str, Any],
        Field(
            description=(
                "Additional keyword arguments to pass to optuna.create_study. "
                "Common options include 'sampler', 'pruner', etc."
            ),
        ),
    ] = {}

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create_default(
        cls,
        optuna_db_path: Path,
        checkpoint_path: Path,
        n_trials_per_combination: int = 100,
    ) -> "OrchestratorConfig":
        """
        Creates a default configuration with standard filter layers.

        The default configuration uses a 4-layer structure:
        1. Noise reduction: NoOp, MedianBlur, BilateralFilter
        2. Color correction: NoOp, SaturationBoost
        3. Contrast enhancement: NoOp, CLAHE, GammaCorrection
        4. Edge sharpening: NoOp, UnsharpMask, LaplacianSharpen

        Args:
            optuna_db_path: Path to Optuna SQLite database.
            checkpoint_path: Path to save/load checkpoints.
            n_trials_per_combination: Number of trials per filter combination.

        Returns:
            OrchestratorConfig: A default configuration instance.
        """
        return cls(
            filter_layers=[
                [NoOpFilterAdapter, MedianBlurFilterAdapter, BilateralFilterAdapter],
                [
                    NoOpFilterAdapter,
                    SaturationBoostFilterAdapter,
                    GammaCorrectionFilterAdapter,
                ],
                [
                    NoOpFilterAdapter,
                    ClaheFilterAdapter,
                ],
                [
                    NoOpFilterAdapter,
                    UnsharpMaskFilterAdapter,
                    LaplacianSharpenFilterAdapter,
                ],
            ],
            n_trials_per_combination=n_trials_per_combination,
            optuna_db_path=optuna_db_path,
            checkpoint_path=checkpoint_path,
        )

    def save(self, config_path: Path) -> None:
        """
        Saves the configuration to disk as JSON.

        Args:
            config_path: Path where the configuration should be saved.
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Use model_dump with mode='json' to properly serialize Path objects and use PlainSerializers
        serializable_data = self.model_dump(mode="json")
        config_path.write_text(json.dumps(serializable_data, indent=2))


class Orchestrator:
    """
    Main orchestrator class for optimizing filter combinations layer by layer.

    The orchestrator implements a sequential optimization strategy where filters
    are selected and optimized one layer at a time. For each layer, it tests all
    available filter options, optimizes their parameters using Optuna, and selects
    the best performing combination to carry forward to the next layer.

    The orchestrator supports:
    - Comprehensive progress tracking and logging
    - Full customization through OrchestratorConfig
    - Checkpoint saving for tracking progress
    - Callback functions for custom behavior

    Example:
        ```python
        from pathlib import Path
        from orchestrator import Orchestrator, OrchestratorConfig
        from yolo import Yolo
        from dataset import Dataset

        # Create configuration
        config = OrchestratorConfig.create_default(
            optuna_db_path=Path("optuna.db"),
            checkpoint_path=Path("checkpoint.json"),
            n_trials_per_combination=50,
        )

        # Load model and dataset
        model = Yolo.load_model(Path("model.pt"))
        dataset = Dataset.load_from_directory(Path("dataset"))

        # Run optimization
        log = Orchestrator.train(
            model=model,
            dataset=dataset,
            config=config,
        )
        ```
    """

    @staticmethod
    def train(
        model: Yolo,
        dataset: StoredDataset,
        config: OrchestratorConfig,
    ) -> OrchestratorLog:
        """
        Trains the model by optimizing filter combinations layer by layer.

        This method implements the main optimization loop. It processes each filter
        layer sequentially, testing all available filters in each layer, optimizing
        their parameters, and selecting the best combination to carry forward.

        Args:
            model: YOLO model to use for evaluation.
            dataset: Training dataset containing samples to optimize on.
            config: Orchestrator configuration specifying behavior and parameters.

        Returns:
            OrchestratorLog: Final optimization log with all results and state.
        """
        # Create new log for this run
        log = OrchestratorLog()
        print("Starting new optimization run...")

        print(f"Total layers to process: {len(config.filter_layers)}")
        print(f"Trials per combination: {config.n_trials_per_combination}")

        # Process each layer sequentially
        for layer_index in range(log.current_layer_index, len(config.filter_layers)):
            current_layer = config.filter_layers[layer_index]

            # Base combination from previous layers (must stay fixed for this layer)
            base_combination = [f.adapter for f in log.best_filters_combination]
            base_parametrized_filters = log.best_filters_combination.copy()

            # Best result within the current layer
            layer_best_combination: Optional[List[Type[FilterAdapter]]] = None
            layer_best_parametrized_filters: Optional[List[ParametrizedFilter]] = None
            layer_best_map_50: Optional[float] = None

            print(f"\n{'=' * 60}")
            print(f"Processing Layer {layer_index + 1}/{len(config.filter_layers)}")
            print(f"Available filters: {', '.join([f.name for f in current_layer])}")
            print(f"{'=' * 60}\n")

            # Test each filter in the current layer
            for filter_class in current_layer:
                trial_filters_combination: List[Type[FilterAdapter]] = (
                    base_combination + [filter_class]
                )

                # If NoOp doesn't change the chain, treat base result as candidate
                if filter_class == NoOpFilterAdapter and base_combination:
                    print("Skipping redundant NoOp combination (using base result)")
                    if log.best_map_50 is not None and (
                        layer_best_map_50 is None or log.best_map_50 > layer_best_map_50
                    ):
                        layer_best_map_50 = log.best_map_50
                        layer_best_combination = trial_filters_combination.copy()
                        layer_best_parametrized_filters = (
                            base_parametrized_filters.copy()
                        )
                    continue

                # Skip if all filters are NoOp (if configured)
                if config.skip_all_noop and all(
                    f == NoOpFilterAdapter for f in trial_filters_combination
                ):
                    print("Skipping all-NoOp combination")
                    continue

                # Generate study name
                study_name = " + ".join([f.name for f in trial_filters_combination])
                if config.study_name_prefix:
                    study_name = f"{config.study_name_prefix}_{study_name}"

                print(f"\nOptimizing: {study_name}")

                # Get or create report for this combination
                report = log.add_or_get_report(trial_filters_combination, study_name)

                # Check if we need to continue this study
                remaining_trials = (
                    config.n_trials_per_combination - report.completed_trials_count
                )
                if remaining_trials <= 0:
                    print(
                        f"Study already completed "
                        f"({report.completed_trials_count}/{config.n_trials_per_combination} trials)"
                    )
                    print(
                        f"Best mAP@50: {report.best_trial.map_50 if report.best_trial else 'N/A'}"
                    )

                    # Update layer best if this existing report is better
                    if report.best_trial and (
                        layer_best_map_50 is None
                        or report.best_trial.map_50 > layer_best_map_50
                    ):
                        layer_best_map_50 = report.best_trial.map_50
                        layer_best_combination = trial_filters_combination.copy()
                        layer_best_parametrized_filters = (
                            report.best_trial.filters.copy()
                        )
                    continue

                print(
                    f"Resuming study: {report.completed_trials_count}/"
                    f"{config.n_trials_per_combination} trials completed"
                )

                # Create or load Optuna study
                study_kwargs = {
                    "study_name": study_name,
                    "storage": f"sqlite:///{config.optuna_db_path.as_posix()}",
                    "load_if_exists": True,
                    "direction": "maximize",
                    **config.optuna_study_kwargs,
                }
                study = optuna.create_study(**study_kwargs)

                # Create trainer for this filter combination
                trainer = Trainer(
                    model=model,
                    samples=dataset.samples,
                    filters_path=trial_filters_combination,
                )

                # Define callback to log each trial
                # Create a closure factory to properly capture loop variables
                def create_trial_callback(
                    captured_report: FilterOptimizationStudy,
                    captured_filters: List[Type[FilterAdapter]],
                ) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
                    def trial_callback(
                        _study: optuna.Study, trial: optuna.trial.FrozenTrial
                    ) -> None:
                        if trial.state == optuna.trial.TrialState.COMPLETE:
                            # Get filter parameters from trial user attributes
                            parametrized_filters_dict: Optional[Dict[str, Dict]] = (
                                trial.user_attrs.get("filter_parameters")
                            )

                            if trial.value is None:
                                raise ValueError("Trial is missing 'value'.")
                            if parametrized_filters_dict is None:
                                raise ValueError(
                                    "Trial is missing 'filter_parameters' in user attributes."
                                )

                            # Convert dict back to ParametrizedFilter list
                            parametrized_filters = []
                            for (
                                filter_name,
                                parameters,
                            ) in parametrized_filters_dict.items():
                                base_name = filter_name.rsplit("_", 1)[0]
                                filter_adapter = FilterAdapter.from_name(base_name)
                                # Get parameters_class safely using getattr
                                parameters_cls = getattr(
                                    filter_adapter, "parameters_class"
                                )  # type: ignore
                                parametrized_filter = ParametrizedFilter(
                                    adapter=filter_adapter,
                                    parameters=parameters_cls(**parameters),  # type: ignore
                                )
                                parametrized_filters.append(parametrized_filter)

                            trial_result = TrialResult(
                                trial_number=trial.number,
                                map_50=trial.value,
                                filters=parametrized_filters,
                            )

                            # Add trial to report
                            captured_report.add_trial(trial_result)

                            # Update global best if needed
                            was_updated = log.update_best_if_needed(captured_report)

                            # Update layer best if needed
                            nonlocal \
                                layer_best_map_50, \
                                layer_best_combination, \
                                layer_best_parametrized_filters
                            if (
                                layer_best_map_50 is None
                                or trial_result.map_50 > layer_best_map_50
                            ):
                                layer_best_map_50 = trial_result.map_50
                                layer_best_combination = captured_filters.copy()
                                layer_best_parametrized_filters = (
                                    trial_result.filters.copy()
                                )
                                print(
                                    f"  → New layer best! mAP@50: {layer_best_map_50:.4f}"
                                )

                            if was_updated:
                                print(
                                    f"  → New global best! mAP@50: {log.best_map_50:.4f}"
                                )

                            # Call user-defined trial callback if provided
                            if config.trial_callback is not None:
                                config.trial_callback(trial_result, captured_report)

                            # Save progress after each trial
                            log.save(config.checkpoint_path)

                            print(
                                f"  Trial {trial.number} complete: mAP@50 = {trial.value:.4f}"
                            )
                            print(
                                f"  Progress: {captured_report.completed_trials_count}/"
                                f"{config.n_trials_per_combination} trials"
                            )

                    return trial_callback

                # Create the callback with captured values
                trial_callback = create_trial_callback(
                    report, trial_filters_combination
                )

                # Run optimization with callback
                study.optimize(
                    trainer.objective,
                    n_trials=remaining_trials,
                    callbacks=[trial_callback],
                )

                print(f"\nStudy complete for {study_name}")
                if report.best_trial:
                    print(f"Best mAP@50: {report.best_trial.map_50:.4f}")

            # Update log with layer results (fallback to base if no trials ran)
            if layer_best_parametrized_filters is None:
                layer_best_parametrized_filters = base_parametrized_filters
            if layer_best_map_50 is None:
                layer_best_map_50 = log.best_map_50
            if layer_best_combination is None:
                layer_best_combination = base_combination

            log.best_filters_combination = layer_best_parametrized_filters
            log.best_map_50 = layer_best_map_50
            log.current_layer_index = layer_index + 1
            log.save(config.checkpoint_path)

            # Call user-defined layer callback if provided
            if config.layer_callback is not None:
                config.layer_callback(
                    layer_index,
                    layer_best_combination,
                    layer_best_map_50,
                    log.best_filters_combination,
                )

            print(f"\nLayer {layer_index + 1} complete!")
            print(
                f"Best combination so far: {[f.name for f in layer_best_combination]}"
            )
            if log.best_map_50 is None:
                print("Best mAP@50: N/A")
            else:
                print(f"Best mAP@50: {log.best_map_50:.4f}")

        print(f"\n{'=' * 60}")
        print("Training complete!")
        print(
            f"Final best combination: {[f.adapter.name for f in log.best_filters_combination]}"
        )
        if log.best_map_50 is None:
            print("Final best mAP@50: N/A")
        else:
            print(f"Final best mAP@50: {log.best_map_50:.4f}")
        print(f"{'=' * 60}\n")

        return log


if __name__ == "__main__":
    # Example usage
    model = Yolo.load_model(Path("resources/model.pt"))
    dataset = Dataset.load_from_directory(Path("resources/dataset"))
    samples = dataset.pick_random_samples(
        magnitude=Magnitude.LCM, split=DatasetSplit.VAL, sample_count=10
    )
    staging_dataset = Dataset.create_staging_dataset(samples)

    # Create configuration
    config = OrchestratorConfig.create_default(
        optuna_db_path=Path("optuna_studies.db"),
        checkpoint_path=Path("orchestrator_checkpoint.json"),
        n_trials_per_combination=4,
    )

    # Run optimization
    log = Orchestrator.train(
        model=model,
        dataset=staging_dataset,
        config=config,
    )
