"""Proyecto Final - Aprendizaje Automático.

Detección no supervisada de fraude en transacciones de dinero móvil (PaySim).
Autor: Leonardo Negrete
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Semilla fija para reproducibilidad exacta (requisito de la rúbrica)
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_PATH = Path(__file__).resolve().parent / "data" / "paysim.csv"


def cargar_datos(ruta: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset PaySim."""
    return pd.read_csv(ruta)


def resumen_inicial(df: pd.DataFrame) -> None:
    """Imprime un resumen básico del dataset."""
    print("Dimensiones:", df.shape)
    print("\nTipos de datos:\n", df.dtypes)
    print("\nValores nulos por columna:\n", df.isnull().sum())
    print(f"\nTasa de fraude: {df['isFraud'].mean():.4%}")
    print("\nTransacciones y fraudes por tipo:")
    print(df.groupby("type")["isFraud"].agg(["count", "sum"]))


def main() -> None:
    df = cargar_datos()
    resumen_inicial(df)


if __name__ == "__main__":
    main()
