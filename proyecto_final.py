"""Proyecto Final - Aprendizaje Automático.

Detección no supervisada de fraude en transacciones de dinero móvil (PaySim):
comparación de k-Means, GMM, DBSCAN e Isolation Forest, más un k-Means propio
implementado desde cero.

Uso:
    python proyecto_final.py              # flujo rápido (hiperparámetros finales)
    python proyecto_final.py --completo   # añade exploración, estabilidad y ablación

Partición cronológica: ajuste (step <= 500), validación (501-600), prueba (> 600).
La etiqueta isFraud nunca se usa para entrenar; solo para elegir hiperparámetros
en validación y para evaluar.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

# Semilla global fija (reproducibilidad); cada módulo fija además su random_state.
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

import ablacion  # noqa: E402
import download_data  # noqa: E402
import eda  # noqa: E402
import figuras  # noqa: E402
import final_eval  # noqa: E402
import robustez  # noqa: E402
import run_experiments  # noqa: E402
import validate_kmeans  # noqa: E402

ETAPAS_RAPIDAS = [
    ("Descarga del dataset PaySim", download_data.descargar_paysim),
    ("Análisis exploratorio (EDA)", eda.main),
    ("Validación del k-Means propio contra scikit-learn", validate_kmeans.main),
    ("Evaluación final de los modelos", final_eval.main),
    ("Figuras complementarias (codo y PCA)", figuras.main),
]

ETAPAS_COMPLETAS = [
    ("Descarga del dataset PaySim", download_data.descargar_paysim),
    ("Análisis exploratorio (EDA)", eda.main),
    ("Validación del k-Means propio contra scikit-learn", validate_kmeans.main),
    ("Exploración de hiperparámetros (una semilla)", run_experiments.main),
    ("Estabilidad con 5 semillas", robustez.main),
    ("Evaluación final de los modelos", final_eval.main),
    ("Figuras complementarias (codo y PCA)", figuras.main),
    ("Ablación de variables", ablacion.main),
]


def ejecutar(etapas) -> None:
    """Ejecuta las etapas en orden e informa el tiempo de cada una."""
    inicio = time.time()
    for i, (nombre, funcion) in enumerate(etapas, start=1):
        print(f"\n{'=' * 72}\n[{i}/{len(etapas)}] {nombre}\n{'=' * 72}")
        t0 = time.time()
        funcion()
        print(f"-> {nombre}: {time.time() - t0:.0f} s")
    print(f"\nFlujo terminado en {(time.time() - inicio) / 60:.1f} min. "
          f"Resultados en {RAIZ / 'reports'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--completo", action="store_true",
                        help="incluye exploración, estabilidad y ablación (lento)")
    args = parser.parse_args()
    ejecutar(ETAPAS_COMPLETAS if args.completo else ETAPAS_RAPIDAS)


if __name__ == "__main__":
    main()
