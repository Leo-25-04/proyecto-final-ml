"""EDA de PaySim: imprime tablas y guarda figuras en reports/figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RANDOM_STATE = 42
RAIZ = Path(__file__).resolve().parents[1]
DATA_PATH = RAIZ / "data" / "paysim.csv"
FIG = RAIZ / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]


def guardar(nombre: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / nombre, dpi=150)
    plt.close()


def main() -> None:
    df = pd.read_csv(DATA_PATH, usecols=COLS)

    # 1. Fraude por tipo de transacción
    por_tipo = df.groupby("type")["isFraud"].agg(["count", "sum", "mean"])
    print("Fraude por tipo:\n", por_tipo, "\n")
    por_tipo["sum"].plot(kind="bar", color="steelblue")
    plt.title("Casos de fraude por tipo de transacción")
    plt.ylabel("Fraudes")
    guardar("01_fraude_por_tipo.png")

    # 2. Relación con isFlaggedFraud (variable a descartar)
    print("isFlaggedFraud vs isFraud:\n",
          pd.crosstab(df["isFlaggedFraud"], df["isFraud"]), "\n")

    # Subconjunto donde ocurre el fraude
    sub = df[df["type"].isin(["CASH_OUT", "TRANSFER"])].copy()
    print("Subconjunto CASH_OUT + TRANSFER:", sub.shape,
          f"| tasa de fraude: {sub['isFraud'].mean():.4%}\n")

    # 3. Variables derivadas de inconsistencia contable
    sub["errorOrig"] = sub["newbalanceOrig"] + sub["amount"] - sub["oldbalanceOrg"]
    sub["errorDest"] = sub["oldbalanceDest"] + sub["amount"] - sub["newbalanceDest"]
    sub["destVacio"] = ((sub["oldbalanceDest"] == 0)
                        & (sub["newbalanceDest"] == 0)).astype(int)
    sub["origVaciada"] = ((sub["oldbalanceOrg"] > 0)
                          & (sub["newbalanceOrig"] == 0)).astype(int)
    print("Medias por clase:\n",
          sub.groupby("isFraud")[["amount", "errorOrig", "errorDest",
                                  "destVacio", "origVaciada"]].mean(), "\n")

    # 4. Outliers en el monto
    print("Cuantiles de amount:\n", sub["amount"].quantile([.5, .9, .99, .999]), "\n")
    q99 = sub["amount"].quantile(0.99)
    print(f"Fraudes con monto sobre el percentil 99: "
          f"{(sub.loc[sub['isFraud'] == 1, 'amount'] > q99).mean():.2%}\n")

    muestra = sub.sample(500_000, random_state=RANDOM_STATE)
    muestra = muestra.assign(log_monto=np.log1p(muestra["amount"]))
    sns.histplot(data=muestra, x="log_monto", hue="isFraud", bins=60,
                 stat="density", common_norm=False)
    plt.title("Distribución de log(1 + monto) por clase")
    guardar("02_monto_por_clase.png")

    # 5. Fraude a lo largo del tiempo
    sub.groupby("step")["isFraud"].sum().plot(figsize=(10, 4))
    plt.title("Fraudes por paso de tiempo (1 paso = 1 hora)")
    plt.xlabel("step")
    plt.ylabel("Fraudes")
    guardar("03_fraude_por_step.png")

    # 6. Correlaciones
    numericas = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                 "oldbalanceDest", "newbalanceDest", "errorOrig",
                 "errorDest", "destVacio", "origVaciada", "isFraud"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(sub[numericas].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de correlación (CASH_OUT + TRANSFER)")
    guardar("04_correlacion.png")

    print(f"Figuras guardadas en: {FIG}")


if __name__ == "__main__":
    main()
