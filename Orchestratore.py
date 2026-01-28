import random

def simula_performance_combinata(operazioni_applicate):
    """
    Simula la performance di una combinazione di operazioni
    """
    return random.uniform(0, 1)

# Definizione dei livelli di operazioni
livelli_operazioni = [
    ['no denoise', 'median blur', 'bilateral filter'],  # Livello 1
    ['no contrast', 'CLAHE soft', 'CLAHE hard', 'Gamma correction'],  # Livello 2
    ['No sharpen', 'Unsharp mask', 'kernel Sharpen']  # Livello 3
]

# Configurazione migliore iniziale
best_config = {
    "operazioni": [],
    "valore": float("-inf")
}


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
        
        # Simula la performance di questa combinazione
        valore_combinazione = simula_performance_combinata(combinazione)
        
        print(f"  {combinazione}: {valore_combinazione:.4f}")
        
        # Aggiorna se è la migliore per questo livello
        if valore_combinazione > best_level["valore"]:
            best_level["operazioni"] = combinazione
            best_level["valore"] = valore_combinazione
    
    # Aggiorna la configurazione globale con la migliore di questo livello
    best_config = best_level
    print(f"\n  → MIGLIORE: {best_config['operazioni']}")
    print(f"     Performance: {best_config['valore']:.4f}\n")

print("=" * 70)
print(f"CONFIGURAZIONE FINALE:")
print(f"  Pipeline: {' → '.join(best_config['operazioni'])}")
print(f"  Performance finale: {best_config['valore']:.4f}")