"""Ablación de variables: ¿cuánto depende el desempeño de cada grupo de variables?

Se usan los hiperparámetros finales (elegidos con todas las variables) y la
misma partición cronológica. Solo cambia el conjunto de variables de entrada.
"""
import pandas as pd

from final_eval import GRIDS_FINALES
from preprocessing import cargar_y_filtrar, preparar_datos
from run_experiments import CORTE_FIT, N_AJUSTE, RAIZ, experimento

SUBCONJUNTOS = {
    "completo": [],
    "sin_destino": ["log_oldbalanceDest", "log_newbalanceDest",
                    "error_dest", "dest_vacio"],
    "sin_derivadas": ["error_orig", "error_dest", "dest_vacio", "orig_vaciada"],
    "sin_hora": ["hora_sin", "hora_cos"],
}


def ablacion(X_tr_df, y_tr, es_fit, X_te_df, y_te, n_ajuste=N_AJUSTE):
    filas = []
    for nombre, excluidas in SUBCONJUNTOS.items():
        cols = [c for c in X_tr_df.columns if c not in excluidas]
        A, T = X_tr_df[cols].to_numpy(), X_te_df[cols].to_numpy()
        print(f"\n--- Subconjunto: {nombre} ({len(cols)} variables) ---")
        res, _, _ = experimento(A[es_fit], y_tr[es_fit], A[~es_fit], y_tr[~es_fit],
                                T, y_te, grids=GRIDS_FINALES, n_ajuste=n_ajuste)
        prueba = res[res["conjunto"] == "prueba"][
            ["modelo", "pr_auc", "roc_auc", "recall", "precision", "f1"]].copy()
        prueba.insert(0, "subconjunto", nombre)
        filas.append(prueba)
    return pd.concat(filas, ignore_index=True)


def main() -> None:
    df = cargar_y_filtrar()
    X_train, X_test, y_train, y_test, _ = preparar_datos()
    steps = df.loc[X_train.index, "step"].to_numpy()
    es_fit = steps <= CORTE_FIT

    tabla = ablacion(X_train, y_train.to_numpy(), es_fit, X_test, y_test.to_numpy())

    pd.set_option("display.width", 250)
    orden = list(SUBCONJUNTOS)
    for metrica in ("pr_auc", "roc_auc", "recall"):
        print(f"\n=== {metrica} en prueba ===")
        print(tabla.pivot(index="modelo", columns="subconjunto",
                          values=metrica)[orden].round(4).to_string())
    tabla.to_csv(RAIZ / "reports" / "ablacion.csv", index=False)
    print("\nTabla guardada en reports/ablacion.csv")


if __name__ == "__main__":
    main()
