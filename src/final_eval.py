"""Evaluación final con los hiperparámetros elegidos por estabilidad (robustez.py).

Criterio: mejor PR-AUC medio en validación sobre 5 semillas. Se vuelve a correr
con la semilla base para obtener tablas completas y figuras definitivas.
"""
import pandas as pd

from preprocessing import cargar_y_filtrar, preparar_datos
from run_experiments import CORTE_FIT, N_AJUSTE, RAIZ, experimento, graficar

GRIDS_FINALES = {
    "KMeans (sklearn)": [dict(k=3)],
    "KMeans (propio)": [dict(k=3)],
    "GMM": [dict(k=2)],
    "DBSCAN": [dict(eps=1.0, min_samples=10)],
    "Isolation Forest": [dict(max_samples=65536)],
}


def main() -> None:
    df = cargar_y_filtrar()
    X_train, X_test, y_train, y_test, _ = preparar_datos()
    steps = df.loc[X_train.index, "step"].to_numpy()
    es_fit = steps <= CORTE_FIT
    X_tr, y_tr = X_train.to_numpy(), y_train.to_numpy()

    resultados, resumen, puntajes = experimento(
        X_tr[es_fit], y_tr[es_fit], X_tr[~es_fit], y_tr[~es_fit],
        X_test.to_numpy(), y_test.to_numpy(),
        grids=GRIDS_FINALES, n_ajuste=N_AJUSTE)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== Resultados finales ===")
    print(resultados.round(4).to_string(index=False))
    resultados.to_csv(RAIZ / "reports" / "resultados_finales.csv", index=False)
    graficar(puntajes, y_test.to_numpy())
    print("\nTabla y figuras finales guardadas en reports/")


if __name__ == "__main__":
    main()
