# Filter Learning Framework

A Python framework for **automated image preprocessing optimization** using YOLO object detection. This project focuses on optimizing image preprocessing pipelines for malaria parasite detection by finding the best combination of image filters and their hyperparameters using Optuna.

## 🎯 Overview

SIV (Smart Image Vision) is designed to automatically discover optimal image preprocessing pipelines for object detection tasks. The framework:

1. **Defines filter layers** - Organizes image filters into sequential processing layers (noise reduction → color correction → contrast enhancement → edge sharpening)
2. **Optimizes hyperparameters** - Uses Optuna to find optimal parameters for each filter
3. **Evaluates performance** - Measures detection accuracy using YOLO's mAP@50 metric
4. **Tracks progress** - Saves checkpoints and logs all optimization trials

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd siv

# Install dependencies using pip
pip install -e .
```

### Dependencies

- Python ≥ 3.12
- PyTorch ≥ 2.10.0
- Ultralytics (YOLO) ≥ 8.4.9
- Optuna ≥ 4.7.0
- OpenCV ≥ 4.13.0
- Pydantic ≥ 2.12.5
- NumPy ≥ 2.4.1
- TorchMetrics ≥ 1.8.2

## 📁 Project Structure

```
siv/
├── src/
│   ├── orchestrator.py   # Main optimization orchestrator
│   ├── dataset.py        # Dataset management classes
│   ├── filters.py        # Image filter implementations
│   ├── trainer.py        # Optuna training integration
│   └── yolo.py           # YOLO model wrapper
├── utils.py              # Utility functions
├── resources/
│   ├── model.pt          # Pre-trained YOLO model
│   ├── model_da.pt       # Domain-adapted YOLO model
│   └── dataset/          # Training/validation datasets
├── pyproject.toml        # Project configuration
└── README.md
```

---

## 🔧 Core Modules

### 1. Dataset Module (`src/dataset.py`)

Provides a structured framework for handling image datasets with YOLO-format labels.

#### Classes

| Class | Description |
|-------|-------------|
| `Label` | Represents a bounding box label in YOLO format (class, x_center, y_center, width, height) |
| `Sample` | Abstract base class for dataset samples |
| `StoredSample` | Sample stored on filesystem with lazy image loading and caching |
| `TransientSample` | In-memory sample with image data held as NumPy array |
| `Dataset` | Abstract base class for dataset collections |
| `StoredDataset` | Filesystem-based dataset with persistent storage |
| `StagingDataset` | Temporary dataset in a temp directory (auto-cleanup) |
| `TransientDataset` | In-memory dataset for volatile operations |

#### Enums

| Enum | Values | Description |
|------|--------|-------------|
| `MalariaStage` | `SCHIZONT`, `GAMETOCYTE`, `RING`, `TROPHOZOITE` | Malaria parasite life stages (class labels) |
| `DatasetSplit` | `TRAIN`, `TEST`, `VAL` | Dataset partitioning |
| `Magnitude` | `HCM`, `LCM` | Image magnification level (High/Low Content Microscopy) |

#### Key Methods

```python
# Load dataset from directory
dataset = Dataset.load_from_directory(Path("resources/dataset"))

# Pick random samples with filtering
samples = dataset.pick_random_samples(
    sample_count=100,
    magnitude=Magnitude.LCM,
    split=DatasetSplit.VAL
)

# Create staging dataset from samples
staging_dataset = Dataset.create_staging_dataset(samples)

# Create in-memory dataset
transient_dataset = Dataset.create_transient_dataset(transient_samples)

# Apply filter transforms to a sample
transformed = sample.apply_transform([filter1, filter2])
```

---

### 2. Filters Module (`src/filters.py`)

Extensible image filtering framework using an adapter pattern with Pydantic validation.

#### Filter Types

| Type | Description |
|------|-------------|
| `NO_OP` | Identity filter (no modification) |
| `NOISE_REDUCTION` | Smoothing and denoising filters |
| `COLOR_CORRECTION` | Color/intensity adjustment filters |
| `CONTRAST_ENHANCEMENT` | Contrast improvement filters |
| `EDGE_SHARPENING` | Edge enhancement filters |

#### Available Filters

| Filter | Type | Parameters | Description |
|--------|------|------------|-------------|
| `NoOpFilterAdapter` | NO_OP | None | Returns image unchanged |
| `MedianBlurFilterAdapter` | NOISE_REDUCTION | `kernel_size` (3-5) | Salt-and-pepper noise removal |
| `BilateralFilterAdapter` | NOISE_REDUCTION | `diameter`, `sigmaColor`, `sigmaSpace` | Edge-preserving smoothing |
| `SaturationBoostFilterAdapter` | COLOR_CORRECTION | `boost_factor` (1.0-2.0) | Adjust color saturation in HSV space |
| `GammaCorrectionFilterAdapter` | COLOR_CORRECTION | `gamma` (0.8-1.4) | Brightness/contrast adjustment |
| `ClaheFilterAdapter` | CONTRAST_ENHANCEMENT | `clip_limit`, `tile_grid_size` | Adaptive histogram equalization |
| `UnsharpMaskFilterAdapter` | EDGE_SHARPENING | `sigma`, `strength` | Edge enhancement via Gaussian subtraction |
| `LaplacianSharpenFilterAdapter` | EDGE_SHARPENING | `ksize`, `scale`, `delta` | Laplacian-based edge sharpening |

#### Key Classes

```python
class FilterAdapter(ABC, Generic[FilterParameters]):
    """Abstract base class for filter adapters."""

    name: str                              # Unique filter identifier
    filter_type: FilterType                # Category of filter
    parameters_class: Type[FilterParameters]  # Pydantic model for parameters
    initial_parameters: FilterParameters   # Default parameter values

    @staticmethod
    @abstractmethod
    def apply_filter(image: np.ndarray, parameters: FilterParameters) -> np.ndarray:
        """Apply the filter to an image."""
        pass

class ParametrizedFilter(BaseModel):
    """A filter with specific parameter values."""
    adapter: Type[FilterAdapter]
    parameters: FilterParameters

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the parametrized filter to an image."""
        return self.adapter.apply_filter(image, self.parameters)
```

#### Creating Parametrized Filters

```python
from src.filters import BilateralFilterAdapter, BilateralFilterParameters

# Create a filter with specific parameters
bilateral_filter = BilateralFilterAdapter.parametrized(
    BilateralFilterParameters(
        diameter=8,
        sigmaColor=65,
        sigmaSpace=45
    )
)

# Apply to image
filtered_image = bilateral_filter.apply(image)
```

---

### 3. YOLO Module (`src/yolo.py`)

Wrapper for Ultralytics YOLO model operations.

#### Classes

```python
class YoloConfig(BaseModel):
    """YOLO dataset configuration for validation."""
    path: Path          # Dataset root path
    train: str          # Training data directory
    test: str           # Testing data directory
    val: str            # Validation data directory
    nc: int             # Number of classes
    names: List[str]    # Class names

class Yolo:
    """YOLO model wrapper for loading and evaluation."""
    model_path: Path
    yolo_model: YOLO
    device: Literal["cpu", "mps", "cuda"]
```

#### Key Methods

```python
# Load a YOLO model
model = Yolo.load_model(
    model_path=Path("resources/model.pt"),
    device="mps"  # or "cpu", "cuda"
)

# Evaluate on a staging dataset
map50 = model.evaluate(staging_dataset)
```

---

### 4. Trainer Module (`src/trainer.py`)

Integrates filter pipelines with Optuna for hyperparameter optimization.

#### Class: `Trainer`

```python
class Trainer:
    """Optimizes filter hyperparameters using Optuna."""

    def __init__(
        self,
        model: Yolo,
        filters_path: List[Type[FilterAdapter]],
        samples: List[StoredSample],
        locked_filters: Optional[List[ParametrizedFilter]] = None
    ):
        """
        Args:
            model: YOLO model for evaluation
            filters_path: Sequence of filter types to apply
            samples: Dataset samples to process
            locked_filters: Filters with fixed parameters (won't be optimized)
        """

    def objective(self, trial: Trial) -> float:
        """Optuna objective function returning mAP@50."""
```

#### Usage

```python
from src.trainer import Trainer

trainer = Trainer(
    model=yolo_model,
    filters_path=[
        MedianBlurFilterAdapter,
        SaturationBoostFilterAdapter,
        ClaheFilterAdapter,
        LaplacianSharpenFilterAdapter,
    ],
    samples=samples,
)

study = optuna.create_study(direction="maximize")
study.optimize(trainer.objective, n_trials=100)

print(study.best_params)
```

---

### 5. Orchestrator Module (`src/orchestrator.py`)

The main optimization engine implementing layer-by-layer filter pipeline optimization.

#### Classes

| Class | Description |
|-------|-------------|
| `TrialResult` | Single optimization trial result (trial number, mAP@50, filters) |
| `FilterOptimizationStudy` | Tracks optimization progress for a filter combination |
| `OrchestratorLog` | Checkpoint format for saving/resuming optimization state |
| `OrchestratorConfig` | Full configuration for orchestrator behavior |
| `Orchestrator` | Main orchestration class |

#### Configuration

```python
class OrchestratorConfig(BaseModel):
    filter_layers: List[List[Type[FilterAdapter]]]  # Available filters per layer
    n_trials_per_combination: int = 100             # Trials per filter combo
    optuna_db_path: Path                            # SQLite storage path
    checkpoint_path: Path                           # Checkpoint file path
    skip_all_noop: bool = True                      # Skip all-NoOp combinations
    study_name_prefix: str = ""                     # Optuna study name prefix
    trial_callback: Optional[Callable]              # Per-trial callback
    layer_callback: Optional[Callable]              # Per-layer callback
    optuna_study_kwargs: Dict[str, Any] = {}        # Extra Optuna options
```

#### Default Filter Layers

The default configuration uses a 4-layer architecture:

| Layer | Filter Options |
|-------|---------------|
| 1. Noise Reduction | NoOp, MedianBlur, BilateralFilter |
| 2. Color Correction | NoOp, SaturationBoost, GammaCorrection |
| 3. Contrast Enhancement | NoOp, CLAHE |
| 4. Edge Sharpening | NoOp, UnsharpMask, LaplacianSharpen |

#### Usage

```python
from pathlib import Path
from src.orchestrator import Orchestrator, OrchestratorConfig
from src.yolo import Yolo
from src.dataset import Dataset, Magnitude, DatasetSplit

# Load model and dataset
model = Yolo.load_model(Path("resources/model.pt"))
dataset = Dataset.load_from_directory(Path("resources/dataset"))
samples = dataset.pick_random_samples(
    magnitude=Magnitude.LCM,
    split=DatasetSplit.VAL
)
staging_dataset = Dataset.create_staging_dataset(samples)

# Create configuration
config = OrchestratorConfig.create_default(
    optuna_db_path=Path("optuna_studies.db"),
    checkpoint_path=Path("orchestrator_checkpoint.json"),
    n_trials_per_combination=50,
)

# Run optimization
log = Orchestrator.train(
    model=model,
    dataset=staging_dataset,
    config=config,
)

# Access results
print(f"Best mAP@50: {log.best_map_50}")
print(f"Best filters: {[f.adapter.name for f in log.best_filters_combination]}")
```

---

### 6. Utils Module (`utils.py`)

Utility functions for coordinate conversion and experiment execution.

#### Functions

```python
def yolo_to_xyxy(label, img_w, img_h) -> Tuple[int, List[float]]:
    """
    Convert YOLO format (x_center, y_center, w, h) to corner coordinates.

    Returns:
        Tuple of (class_id, [x1, y1, x2, y2])
    """

def execute_experiment(dataset: list, model) -> float:
    """
    Execute a validation experiment with a dataset.

    Args:
        dataset: List of sample dicts with 'image', 'image_name', 'labels'
        model: YOLO model instance

    Returns:
        mAP@50 score
    """

def create_yaml_config(tmp_dir: str) -> str:
    """Create a YOLO config YAML file in the given directory."""
```

---

## 🚀 Quick Start

### Basic Optimization Example

```python
from pathlib import Path
from src.orchestrator import Orchestrator, OrchestratorConfig
from src.yolo import Yolo
from src.dataset import Dataset, Magnitude, DatasetSplit

# 1. Load the YOLO model
model = Yolo.load_model(Path("resources/model.pt"), device="mps")

# 2. Load and prepare dataset
dataset = Dataset.load_from_directory(Path("resources/dataset"))
samples = dataset.pick_random_samples(
    sample_count=100,
    magnitude=Magnitude.LCM,
    split=DatasetSplit.VAL
)
staging_dataset = Dataset.create_staging_dataset(samples)

# 3. Configure the orchestrator
config = OrchestratorConfig.create_default(
    optuna_db_path=Path("optimization.db"),
    checkpoint_path=Path("checkpoint.json"),
    n_trials_per_combination=25,
)

# 4. Run optimization
log = Orchestrator.train(
    model=model,
    dataset=staging_dataset,
    config=config,
)

# 5. Use the best filter pipeline
best_filters = log.best_filters_combination
for sample in samples:
    transformed = sample.apply_transform(best_filters)
```

### Custom Filter Pipeline

```python
from src.filters import (
    BilateralFilterAdapter,
    ClaheFilterAdapter,
    UnsharpMaskFilterAdapter,
)

# Define a custom filter sequence
custom_filters = [
    BilateralFilterAdapter.parametrized(
        BilateralFilterParameters(diameter=7, sigmaColor=50, sigmaSpace=50)
    ),
    ClaheFilterAdapter.parametrized(
        ClaheParameters(clip_limit=2.0, tile_grid_size=8)
    ),
    UnsharpMaskFilterAdapter.parametrized(
        UnsharpMaskParameters(sigma=1.5, strength=1.2)
    ),
]

# Apply to samples
for sample in samples:
    transformed = sample.apply_transform(custom_filters)
```

---

## 📊 Malaria Detection Classes

The framework is designed for malaria parasite detection with four life stages:

| Class ID | Stage | Description |
|----------|-------|-------------|
| 0 | Schizont | Mature stage with multiple nuclei |
| 1 | Gametocyte | Sexual stage, crescent-shaped |
| 2 | Ring | Early stage, ring-like appearance |
| 3 | Trophozoite | Growing stage, irregular shape |

---

## 🔍 Monitoring & Visualization

### Optuna Dashboard

```bash
# Launch the Optuna dashboard to visualize optimization
optuna-dashboard sqlite:///optuna_studies.db
```

### Checkpoint Analysis

```python
import json
from pathlib import Path

# Load checkpoint
checkpoint = json.loads(Path("checkpoint.json").read_text())

print(f"Current layer: {checkpoint['current_layer_index']}")
print(f"Best mAP@50: {checkpoint['best_map_50']}")
print(f"Best filters: {checkpoint['best_filters_combination']}")
```

---

## 📄 License

This project is provided as-is for research and educational purposes.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📚 References

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Optuna Hyperparameter Optimization](https://optuna.org/)
- [OpenCV Image Processing](https://docs.opencv.org/)
- [Pydantic Data Validation](https://docs.pydantic.dev/)
