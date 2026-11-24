"""Figuras complementarias de clustering: método del codo y proyección PCA 2D.

La proyección usa isFraud solo para colorear (evaluación), no para ajustar.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from preprocessing import cargar_y_filtrar, preparar_datos
from run_experiments import CORTE_FIT, FIG, RANDOM_STATE

K_ELEGIDO = 3  # k-Means (elegido por estabilidad)
K_GMM = 2      # GMM (elegido por estabilidad)


def figura_codo(muestra):
    ks = list(range(2, 13))
    inercias = [KMeans(k, n_init=3, random_state=RANDOM_STATE).fit(muestra).inertia_
                for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inercias, "o-")
    plt.axvline(K_ELEGIDO, color="gray", ls="--", label=f"k elegido ({K_ELEGIDO})")
    plt.xlabel("k")
    plt.ylabel("Inercia (WCSS)")
    plt.title("Método del codo (muestra de ajuste)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "07_codo.png", dpi=150)
    plt.close()


def figura_pca(muestra, X_test, y_test, n_graf=15_000):
    rng = np.random.default_rng(RANDOM_STATE)
    pca = PCA(2, random_state=RANDOM_STATE).fit(muestra)  # ajustado sin prueba
    print("Varianza explicada por PC1 y PC2:",
          np.round(pca.explained_variance_ratio_, 3))
    idx = rng.choice(len(X_test), min(n_graf, len(X_test)), replace=False)
    Z, y = pca.transform(X_test[idx]), y_test[idx]

    km = KMeans(K_ELEGIDO, n_init=10, random_state=RANDOM_STATE).fit(muestra)
    gmm = GaussianMixture(K_GMM, covariance_type="full", reg_covar=1e-3,
                          max_iter=200, n_init=3, random_state=RANDOM_STATE)
    gmm.fit(muestra[:200_000])
    anom = -gmm.score_samples(X_test[idx])
    anom = np.log1p(anom - anom.min())

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    axs[0].scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, alpha=0.4, label="legítima")
    axs[0].scatter(Z[y == 1, 0], Z[y == 1, 1], s=8, c="red", label="fraude")
    axs[0].set_title("Etiqueta real (solo evaluación)")
    axs[0].legend()
    axs[1].scatter(Z[:, 0], Z[:, 1], c=km.predict(X_test[idx]), s=4, cmap="tab10")
    axs[1].set_title(f"Clústeres de k-Means (k={K_ELEGIDO})")
    sc = axs[2].scatter(Z[:, 0], Z[:, 1], c=anom, s=4, cmap="viridis")
    axs[2].set_title(f"Puntuación de anomalía GMM (K={K_GMM})")
    fig.colorbar(sc, ax=axs[2], label="log(1 + -loglik - min)")
    for ax in axs:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(FIG / "08_pca_2d.png", dpi=150)
    plt.close()


def main() -> None:
    df = cargar_y_filtrar()
    X_train, X_test, y_train, y_test, _ = preparar_datos()
    steps = df.loc[X_train.index, "step"].to_numpy()
    X_fit = X_train.to_numpy()[steps <= CORTE_FIT]
    rng = np.random.default_rng(RANDOM_STATE)
    muestra = X_fit[rng.choice(len(X_fit), 300_000, replace=False)]

    figura_codo(muestra)
    figura_pca(muestra, X_test.to_numpy(), y_test.to_numpy())
    print(f"Figuras guardadas en {FIG}")


if __name__ == "__main__":
    main()
