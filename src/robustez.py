"""Estabilidad: cada configuración candidata con varias semillas (media y desv.).

Misma partición que run_experiments.py (ajuste / validación / prueba).
La semilla cambia la muestra de ajuste y la inicialización de cada modelo.
"""
import pandas as pd
import numpy as np

from evaluation import metricas_anomalia, umbral_max_f1
from models import (DBSCANDetector, GMM, IsolationForestDetector,
                    KMeansPropio, KMeansSklearn)
from preprocessing import cargar_y_filtrar, preparar_datos
from run_experiments import CORTE_FIT, GRIDS, N_AJUSTE, RAIZ, _muestra

SEMILLAS = [42, 43, 44, 45, 46]
METRICAS = ["pr_val", "roc_val", "pr_test", "roc_test", "f1_test"]


def construir_semilla(nombre, p, semilla):
    if nombre == "KMeans (sklearn)":
        return KMeansSklearn(p["k"], semilla)
    if nombre == "KMeans (propio)":
        return KMeansPropio(p["k"], semilla)
    if nombre == "GMM":
        return GMM(p["k"], semilla)
    if nombre == "DBSCAN":
        return DBSCANDetector(p["eps"], p["min_samples"])
    return IsolationForestDetector(p["max_samples"], semilla)


def estabilidad(X_fit, X_val, y_val, X_test, y_test, grids=GRIDS,
                n_ajuste=N_AJUSTE, semillas=SEMILLAS):
    filas = []
    for nombre, grid in grids.items():
        for params in grid:
            for semilla in semillas:
                rng = np.random.default_rng(semilla)
                X_aj = _muestra(X_fit, n_ajuste[nombre], rng)
                try:
                    modelo = construir_semilla(nombre, params, semilla).fit(X_aj)
                except ValueError:
                    continue
                s_val, s_test = modelo.score(X_val), modelo.score(X_test)
                umbral = umbral_max_f1(y_val, s_val)
                mv = metricas_anomalia(y_val, s_val, umbral)
                mt = metricas_anomalia(y_test, s_test, umbral)
                filas.append({"modelo": nombre, "hiperparametros": str(params),
                              "semilla": semilla, "pr_val": mv["pr_auc"],
                              "roc_val": mv["roc_auc"], "pr_test": mt["pr_auc"],
                              "roc_test": mt["roc_auc"], "f1_test": mt["f1"]})
            print(f"  {nombre} {params}: listo", flush=True)
    return pd.DataFrame(filas)


def resumir(df):
    res = df.groupby(["modelo", "hiperparametros"], sort=False)[METRICAS].agg(
        ["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    res = res.reset_index()
    mejores = res.loc[res.groupby("modelo", sort=False)["pr_val_mean"].idxmax()]
    return res, mejores


def main() -> None:
    df = cargar_y_filtrar()
    X_train, X_test, y_train, y_test, _ = preparar_datos()
    steps = df.loc[X_train.index, "step"].to_numpy()
    es_fit = steps <= CORTE_FIT
    X_tr, y_tr = X_train.to_numpy(), y_train.to_numpy()

    crudo = estabilidad(X_tr[es_fit], X_tr[~es_fit], y_tr[~es_fit],
                        X_test.to_numpy(), y_test.to_numpy())
    res, mejores = resumir(crudo)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== Todas las configuraciones (media y desv. sobre semillas) ===")
    print(res.round(4).to_string(index=False))
    print("\n=== Mejor configuración por modelo (por PR-AUC medio en validación) ===")
    print(mejores.round(4).to_string(index=False))
    crudo.to_csv(RAIZ / "reports" / "estabilidad_completa.csv", index=False)
    res.to_csv(RAIZ / "reports" / "estabilidad_resumen.csv", index=False)


if __name__ == "__main__":
    main()
