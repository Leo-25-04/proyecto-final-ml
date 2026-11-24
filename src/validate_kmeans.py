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

    t0 = time.time()
    mio = KMeansScratch(K, init="kmeans++", n_init=5,
                        random_state=RANDOM_STATE).fit(muestra)
    t_mio = time.time() - t0

    t0 = time.time()
    sk = KMeans(K, init="k-means++", n_init=5, random_state=RANDOM_STATE).fit(muestra)
    t_sk = time.time() - t0

    dif = abs(mio.inertia_ - sk.inertia_) / sk.inertia_
    print(f"WCSS propio: {mio.inertia_:,.1f} | sklearn: {sk.inertia_:,.1f} "
          f"| diferencia relativa: {dif:.4%}")
    print(f"ARI entre ambas asignaciones: "
          f"{adjusted_rand_score(mio.labels_, sk.labels_):.4f}")
    print(f"Tiempo propio: {t_mio:.1f}s | sklearn: {t_sk:.1f}s")
    print("Tamaños propio:", np.sort(np.bincount(mio.labels_, minlength=K)))
    print("Tamaños sklearn:", np.sort(np.bincount(sk.labels_, minlength=K)))

    # Silueta sobre submuestra (la versión propia es O(N^2))
    sub = np.random.default_rng(RANDOM_STATE).choice(len(muestra), 3000, replace=False)
    print(f"Silueta propia: {silhouette_global(muestra[sub], mio.labels_[sub]):.4f} "
          f"| sklearn: {silhouette_score(muestra[sub], mio.labels_[sub]):.4f}")


if __name__ == "__main__":
    main()
