import optuna
from filters import (
    NoOpFilterAdapter,
    MedianBlurFilterAdapter,
    BilateralFilterAdapter,
    SaturationBoostFilterAdapter,
    SoftClaheFilterAdapter,
    HardClaheFilterAdapter,
    GammaCorrectionFilterAdapter,
    UnsharpMaskFilterAdapter,
    LaplacianSharpenFilterAdapter,
)

import trainer
from utils import load_model, load_dataset
from trainer import Trainer

# Definizione dei livelli di operazioni
livelli_operazioni = [
    [NoOpFilterAdapter, MedianBlurFilterAdapter, BilateralFilterAdapter],  # Livello 1
    [NoOpFilterAdapter, SaturationBoostFilterAdapter],  # Livello 2
    [NoOpFilterAdapter, SoftClaheFilterAdapter, HardClaheFilterAdapter, GammaCorrectionFilterAdapter],  # Livello 3
    [NoOpFilterAdapter, UnsharpMaskFilterAdapter, LaplacianSharpenFilterAdapter ] # Livello 4
]

# Configurazione migliore iniziale
best_config = {
    "operazioni": [],
    "valore": float("-inf")
}

PATH_IMAGE_DATASET = "Dataset/target/images/val"
PATH_LABEL_DATASET = "Dataset/target/labels/val"

dataset = load_dataset(PATH_IMAGE_DATASET, PATH_LABEL_DATASET)
model = load_model("best.pt")

# Create or load an Optuna study for YOLO preprocessing optimization
study = optuna.create_study(
        study_name="yolo_preprocessing_optimization_TEST",
        storage="sqlite:///yolo_preprocessing_optimization.db",
        load_if_exists=True,
        direction="maximize",
    )

for idx, livello in enumerate(livelli_operazioni, 1):
    print(f"--- Livello {idx} ---")
    print(f"Configurazione corrente: {best_config['operazioni']}")
    print(f"Performance corrente: {best_config['valore']:.4f}\n")
    
    # Trova la migliore combinazione per questo livello
    best_level = {
        "operazioni": best_config["operazioni"].copy(),
        "valore": float("-inf")
    }
    
    for operazione in livello:
        # Aggiunge l'operazione corrente a quelle precedenti
        combinazione = best_config["operazioni"].copy()
        combinazione.append(operazione)
        
        # Instantiate the Trainer with a sequence of filter adapters and a random dataset
        Trainer_instance = Trainer(
            filters_path=combinazione,
            dataset=dataset, 
            model=model
        )
        

        # Run the optimization for 100 trials
        study.optimize(Trainer_instance.objective, n_trials=100)

        best_result = study.best_value
        best_params = study.best_params
        
        # Aggiorna se è la migliore per questo livello
        if best_result > best_level["valore"]:
            best_level["operazioni"] = combinazione
            best_level["valore"] = best_result
    
    # Aggiorna la configurazione globale con la migliore di questo livello
    best_config = best_level
    print(f"\n  → MIGLIORE: {best_config['operazioni']}")
    print(f"     Performance: {best_config['valore']:.4f}\n")

print("=" * 70)
print(f"CONFIGURAZIONE FINALE:")
print(f"  Pipeline: {' → '.join(best_config['operazioni'])}")
print(f"  Performance finale: {best_config['valore']:.4f}")