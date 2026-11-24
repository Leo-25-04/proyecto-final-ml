"""Preprocesamiento: filtrado, características y partición cronológica.

La etiqueta isFraud NO se usa para entrenar: solo se devuelve para evaluar.
El escalador se ajusta únicamente con el conjunto de entrenamiento.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
RAIZ = Path(__file__).resolve().parents[1]
DATA_PATH = RAIZ / "data" / "paysim.csv"

# Regla de negocio observada en el EDA: el fraude solo ocurre en estos tipos
TIPOS_RELEVANTES = ["CASH_OUT", "TRANSFER"]
CORTE_STEP = 600  # train: step <= corte | test: step > corte

COLUMNAS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFraud"]


def cargar_y_filtrar(ruta: Path = DATA_PATH) -> pd.DataFrame:
    """Carga PaySim, conserva CASH_OUT y TRANSFER y ordena por tiempo."""
    df = pd.read_csv(ruta, usecols=COLUMNAS)
    df = df[df["type"].isin(TIPOS_RELEVANTES)].copy()
    return df.sort_values("step", kind="stable").reset_index(drop=True)


def _log_con_signo(x: pd.Series) -> pd.Series:
    """Logaritmo que conserva el signo (para errores contables)."""
    return np.sign(x) * np.log1p(np.abs(x))


def construir_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de características (sin usar isFraud)."""
    out = pd.DataFrame(index=df.index)
    out["es_transfer"] = (df["type"] == "TRANSFER").astype(int)
    out["log_amount"] = np.log1p(df["amount"])
    for col in ["oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest"]:
        out[f"log_{col}"] = np.log1p(df[col])
    out["error_orig"] = _log_con_signo(
        df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"])
    out["error_dest"] = _log_con_signo(
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"])
    out["dest_vacio"] = ((df["oldbalanceDest"] == 0)
                         & (df["newbalanceDest"] == 0)).astype(int)
    out["orig_vaciada"] = ((df["oldbalanceOrg"] > 0)
                           & (df["newbalanceOrig"] == 0)).astype(int)
    hora = df["step"] % 24
    out["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    out["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    return out


def preparar_datos(corte: int = CORTE_STEP):
    """Devuelve X_train, X_test (escalados), y_train, y_test y el escalador."""
    df = cargar_y_filtrar()
    es_train = df["step"] <= corte
    X = construir_caracteristicas(df)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X[es_train]),
                           columns=X.columns, index=X[es_train].index)
    X_test = pd.DataFrame(scaler.transform(X[~es_train]),
                          columns=X.columns, index=X[~es_train].index)
    y_train = df.loc[es_train, "isFraud"]  # solo para evaluar
    y_test = df.loc[~es_train, "isFraud"]
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    df = cargar_y_filtrar()
    print("Filas:", df.shape, "| step máximo:", df["step"].max())
    print("\nFraudes en train/test según el punto de corte:")
    for corte in (500, 550, 600, 650):
        tr = df[df["step"] <= corte]
        te = df[df["step"] > corte]
        print(f"corte={corte}: train={len(tr):,} "
              f"(fraudes={tr['isFraud'].sum():,}) | test={len(te):,} "
              f"(fraudes={te['isFraud'].sum():,}, "
              f"tasa={te['isFraud'].mean():.3%})")

    X_train, X_test, y_train, y_test, _ = preparar_datos()
    print(f"\nCorte usado: {CORTE_STEP}")
    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    print("Medias de X_train (deben ser ~0):\n",
          X_train.mean().round(3).to_string())
