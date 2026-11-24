"""Valida KMeansScratch contra scikit-learn sobre una muestra de PaySim."""
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from kmeans_scratch import KMeansScratch, silhouette_global
from preprocessing import preparar_datos

RANDOM_STATE = 42
K = 5


def main() -> None:
    X_train, _, _, _, _ = preparar_datos()
    muestra = X_train.sample(200_000, random_state=RANDOM_STATE).to_numpy()

    # Prueba 1: mismos centroides iniciales -> Lloyd determinista, debe coincidir
    rng = np.random.default_rng(RANDOM_STATE)
    init = muestra[rng.choice(len(muestra), K, replace=False)]
    mio = KMeansScratch(K, init="manual", init_centroids=init,
                        max_iter=300, tol=1e-6).fit(muestra)
    sk = KMeans(K, init=init, n_init=1, max_iter=300, tol=1e-6,
                algorithm="lloyd").fit(muestra)
    dif = abs(mio.inertia_ - sk.inertia_) / sk.inertia_
    print("PRUEBA 1: mismos centroides iniciales")
    print(f"  Convergió: {mio.converged_} | iteraciones: {mio.n_iter_} "
          f"(sklearn: {sk.n_iter_})")
    print(f"  WCSS propio: {mio.inertia_:,.1f} | sklearn: {sk.inertia_:,.1f} "
          f"| diferencia relativa: {dif:.6%}")
    print(f"  ARI: {adjusted_rand_score(mio.labels_, sk.labels_):.6f}")

    # Prueba 2: k-means++ con 10 reinicios cada uno
    t0 = time.time()
    mio2 = KMeansScratch(K, init="kmeans++", n_init=10, max_iter=300,
                         random_state=RANDOM_STATE).fit(muestra)
    t_mio = time.time() - t0
    t0 = time.time()
    sk2 = KMeans(K, init="k-means++", n_init=10, max_iter=300,
                 random_state=RANDOM_STATE).fit(muestra)
    t_sk = time.time() - t0
    dif2 = abs(mio2.inertia_ - sk2.inertia_) / sk2.inertia_
    print("\nPRUEBA 2: k-means++ con 10 reinicios")
    print(f"  WCSS propio: {mio2.inertia_:,.1f} | sklearn: {sk2.inertia_:,.1f} "
          f"| diferencia relativa: {dif2:.4%}")
    print(f"  ARI: {adjusted_rand_score(mio2.labels_, sk2.labels_):.4f}")
    print(f"  Tiempo propio: {t_mio:.1f}s | sklearn: {t_sk:.1f}s")

    # Silueta sobre submuestra (la versión propia es O(N^2))
    sub = np.random.default_rng(RANDOM_STATE).choice(len(muestra), 3000, replace=False)
    print("\nSILUETA (mismas etiquetas, dos implementaciones)")
    print(f"  propia: {silhouette_global(muestra[sub], mio.labels_[sub]):.4f} "
          f"| sklearn: {silhouette_score(muestra[sub], mio.labels_[sub]):.4f}")


if __name__ == "__main__":
    main()
