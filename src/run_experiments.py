"""Experimentos: ajuste de hiperparámetros en validación y evaluación en prueba.

Partición cronológica (sin mezclar el tiempo):
    ajuste     : step <= CORTE_FIT
    validación : CORTE_FIT < step <= CORTE_STEP  (elige hiperparámetros y umbral)
    prueba     : step > CORTE_STEP               (solo evaluación final)

isFraud NUNCA se usa para entrenar: solo para elegir hiperparámetros en
validación y para evaluar.
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from evaluation import metricas_anomalia, metricas_cluster, umbral_max_f1
from models import (DBSCANDetector, GMM, IsolationForestDetector,
                    KMeansPropio, KMeansSklearn)
from preprocessing import CORTE_STEP, cargar_y_filtrar, preparar_datos

RANDOM_STATE = 42
CORTE_FIT = 500
N_EVAL_TRAIN = 100_000
RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

N_AJUSTE = {"KMeans (sklearn)": 300_000, "KMeans (propio)": 300_000,
            "GMM": 200_000, "DBSCAN": 30_000, "Isolation Forest": 300_000}

GRIDS = {
    "KMeans (sklearn)": [dict(k=k) for k in (2, 3, 4, 5, 8)],
    "KMeans (propio)": [dict(k=k) for k in (2, 3, 4, 5, 8)],
    "GMM": [dict(k=k) for k in (2, 3, 4, 6, 8)],
    "DBSCAN": [dict(eps=e, min_samples=m) for e in (0.5, 1.0, 2.0)
               for m in (10, 50)],
    "Isolation Forest": [dict(max_samples=m) for m in (1024, 4096, 16384, 65536)],
}


def construir(nombre, p):
    if nombre == "KMeans (sklearn)":
        return KMeansSklearn(p["k"], RANDOM_STATE)
    if nombre == "KMeans (propio)":
        return KMeansPropio(p["k"], RANDOM_STATE)
    if nombre == "GMM":
        return GMM(p["k"], RANDOM_STATE)
    if nombre == "DBSCAN":
        return DBSCANDetector(p["eps"], p["min_samples"])
    return IsolationForestDetector(p["max_samples"], RANDOM_STATE)


def _muestra(X, n, rng):
    if len(X) <= n:
        return X
    return X[rng.choice(len(X), n, replace=False)]


def experimento(X_fit, y_fit, X_val, y_val, X_test, y_test,
                grids=GRIDS, n_ajuste=N_AJUSTE, n_eval_train=N_EVAL_TRAIN):
    rng = np.random.default_rng(RANDOM_STATE)
    filas, resumen, puntajes = [], [], {}

    # muestra fija de ajuste para evaluar "entrenamiento"
    idx_tr = rng.choice(len(X_fit), min(n_eval_train, len(X_fit)), replace=False)
    X_tr_eval, y_tr_eval = X_fit[idx_tr], y_fit[idx_tr]

    for nombre, grid in grids.items():
        X_aj = _muestra(X_fit, n_ajuste[nombre], rng)
        mejor = None
        for params in grid:
            t0 = time.time()
            try:
                modelo = construir(nombre, params).fit(X_aj)
                s_val = modelo.score(X_val)
                pr = metricas_anomalia(y_val, s_val, umbral_max_f1(y_val, s_val))["pr_auc"]
            except ValueError as e:
                print(f"  {nombre} {params}: omitido ({e})")
                continue
            print(f"  {nombre} {params}: PR-AUC validación = {pr:.4f} "
                  f"({time.time() - t0:.0f}s)")
            if mejor is None or pr > mejor[0]:
                mejor = (pr, params, modelo, s_val)
        if mejor is None:
            continue
        pr_val, params, modelo, s_val = mejor
        umbral = umbral_max_f1(y_val, s_val)
        print(f">> {nombre}: mejores parámetros {params} "
              f"(PR-AUC validación {pr_val:.4f})\n")
        resumen.append({"modelo": nombre, "hiperparametros": str(params),
                        "pr_auc_validacion": pr_val})

        s_test = modelo.score(X_test)
        puntajes[nombre] = (s_test, umbral)
        for conjunto, X, y in (("entrenamiento", X_tr_eval, y_tr_eval),
                               ("prueba", X_test, y_test)):
            s = s_test if conjunto == "prueba" else modelo.score(X)
            etiquetas = modelo.labels(X)
            if etiquetas is None:  # Isolation Forest: partición normal/anómalo
                etiquetas = (s >= umbral).astype(int)
            fila = {"modelo": nombre, "conjunto": conjunto,
                    **metricas_anomalia(y, s, umbral),
                    **metricas_cluster(X, etiquetas)}
            filas.append(fila)

    resultados = pd.DataFrame(filas)
    return resultados, pd.DataFrame(resumen), puntajes


def graficar(puntajes, y_test):
    # Curvas PR y ROC
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for nombre, (s, _) in puntajes.items():
        p, r, _ = precision_recall_curve(y_test, s)
        fpr, tpr, _ = roc_curve(y_test, s)
        axs[0].plot(r, p, label=nombre)
        axs[1].plot(fpr, tpr, label=nombre)
    axs[0].axhline(np.mean(y_test), color="gray", ls="--", label="azar (prevalencia)")
    axs[0].set(title="Curvas Precision-Recall (prueba)", xlabel="Recall",
               ylabel="Precision")
    axs[1].plot([0, 1], [0, 1], color="gray", ls="--")
    axs[1].set(title="Curvas ROC (prueba)", xlabel="Tasa de falsos positivos",
               ylabel="Tasa de verdaderos positivos")
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "05_curvas_pr_roc.png", dpi=150)
    plt.close()

    # Matrices de confusión normalizadas
    fig, axs = plt.subplots(1, len(puntajes), figsize=(3.6 * len(puntajes), 3.8))
    axs = np.atleast_1d(axs)
    for ax, (nombre, (s, umbral)) in zip(axs, puntajes.items()):
        cm = confusion_matrix(y_test, (s >= umbral).astype(int), normalize="true")
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
        ax.set(title=nombre, xticks=[0, 1], yticks=[0, 1], xticklabels=["Legítima", "Fraude"], yticklabels=["Legítima", "Fraude"],
               xlabel="Predicho", ylabel="Real")
    plt.tight_layout()
    plt.savefig(FIG / "06_matrices_confusion.png", dpi=150)
    plt.close()


def main() -> None:
    df = cargar_y_filtrar()
    X_train, X_test, y_train, y_test, _ = preparar_datos()
    steps = df.loc[X_train.index, "step"].to_numpy()
    es_fit = steps <= CORTE_FIT

    X_tr, y_tr = X_train.to_numpy(), y_train.to_numpy()
    X_fit, y_fit = X_tr[es_fit], y_tr[es_fit]
    X_val, y_val = X_tr[~es_fit], y_tr[~es_fit]
    print(f"Ajuste: {len(X_fit):,} | validación: {len(X_val):,} "
          f"(fraudes {int(y_val.sum()):,}) | prueba: {len(X_test):,} "
          f"(fraudes {int(y_test.sum()):,})\n")

    resultados, resumen, puntajes = experimento(
        X_fit, y_fit, X_val, y_val, X_test.to_numpy(), y_test.to_numpy())

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== Mejores hiperparámetros (elegidos en validación) ===")
    print(resumen.to_string(index=False))
    print("\n=== Resultados ===")
    print(resultados.round(4).to_string(index=False))
    resultados.to_csv(RAIZ / "reports" / "resultados_modelos.csv", index=False)
    resumen.to_csv(RAIZ / "reports" / "hiperparametros.csv", index=False)
    graficar(puntajes, y_test.to_numpy())
    print(f"\nTablas y figuras guardadas en {RAIZ / 'reports'}")


if __name__ == "__main__":
    main()


