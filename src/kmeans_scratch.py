"""K-Means desde cero (algoritmo de Lloyd), adaptado a PaySim.

Basado en el laboratorio de Leonardo Negrete. Cambios para escalar a millones
de filas: distancias por bloques sin tensor (N, K, D), k-means++, varios
reinicios, historial opcional y puntuación de anomalía.
"""
import numpy as np


def distancias_euclidianas(X, C, bloque=200_000):
    """Matriz (N, K) de distancias, usando ||x||^2 - 2 x.c + ||c||^2 por bloques."""
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    c2 = np.sum(C ** 2, axis=1)
    out = np.empty((X.shape[0], C.shape[0]))
    for i in range(0, X.shape[0], bloque):
        b = X[i:i + bloque]
        d2 = np.sum(b ** 2, axis=1)[:, None] - 2.0 * (b @ C.T) + c2[None, :]
        np.maximum(d2, 0.0, out=d2)  # evita negativos por redondeo
        out[i:i + bloque] = np.sqrt(d2)
    return out


def wcss(X, labels, centroids):
    """Suma de distancias cuadráticas de cada punto a su centroide."""
    total = 0.0
    for k in range(centroids.shape[0]):
        pts = X[labels == k]
        total += np.sum((pts - centroids[k]) ** 2)
    return total


def silhouette_samples_scratch(X, labels):
    """Silueta s(i) punto a punto (matriz N x N: usar solo con muestras)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    D = distancias_euclidianas(X, X)
    clusters = np.unique(labels)
    if len(clusters) < 2:
        raise ValueError("La silueta requiere al menos 2 clústeres no vacíos")
    s = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        propio = labels == labels[i]
        n_propio = propio.sum()
        if n_propio == 1:
            continue
        a = D[i, propio].sum() / (n_propio - 1)
        b = min(D[i, labels == k].mean() for k in clusters if k != labels[i])
        s[i] = (b - a) / max(a, b)
    return s


def silhouette_global(X, labels):
    return silhouette_samples_scratch(X, labels).mean()


class KMeansScratch:
    """K-Means (Lloyd) solo con numpy. init: 'random', 'kmeans++' o 'manual'."""

    def __init__(self, n_clusters=3, init="kmeans++", init_centroids=None,
                 n_init=1, max_iter=100, tol=1e-4, random_state=None,
                 guardar_historial=False):
        if init not in ("manual", "random", "kmeans++"):
            raise ValueError("init debe ser 'manual', 'random' o 'kmeans++'")
        self.n_clusters = n_clusters
        self.init = init
        self.init_centroids = init_centroids
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.guardar_historial = guardar_historial

    # ---------- inicialización ----------
    def _inicializar(self, X, rng):
        if self.init == "manual":
            C = np.asarray(self.init_centroids, dtype=float).copy()
            if C.shape != (self.n_clusters, X.shape[1]):
                raise ValueError("init_centroids con forma incorrecta")
            return C
        if self.init == "random":
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[idx].astype(float).copy()
        # k-means++: cada centroide nuevo se elige con prob. proporcional a d^2
        n = X.shape[0]
        C = [X[rng.integers(n)]]
        d2 = np.sum((X - C[0]) ** 2, axis=1)
        for _ in range(1, self.n_clusters):
            idx = rng.choice(n, p=d2 / d2.sum())
            C.append(X[idx])
            d2 = np.minimum(d2, np.sum((X - X[idx]) ** 2, axis=1))
        return np.array(C, dtype=float)

    # ---------- pasos del algoritmo ----------
    @staticmethod
    def _asignar(X, C):
        return np.argmin(distancias_euclidianas(X, C), axis=1)

    def _actualizar(self, X, labels, C, rng):
        K, D = C.shape
        cuentas = np.bincount(labels, minlength=K)
        nuevo = C.copy()
        for j in range(D):
            sumas = np.bincount(labels, weights=X[:, j], minlength=K)
            nuevo[:, j] = np.where(cuentas > 0, sumas / np.maximum(cuentas, 1),
                                   C[:, j])
        for k in np.where(cuentas == 0)[0]:  # clúster vacío -> obs. aleatoria
            nuevo[k] = X[rng.integers(X.shape[0])]
            self.n_empty_ += 1
        return nuevo

    def _ajustar_una_vez(self, X, rng):
        C = self._inicializar(X, rng)
        historial = []
        convergio = False
        for it in range(1, self.max_iter + 1):
            labels = self._asignar(X, C)
            C_nuevo = self._actualizar(X, labels, C, rng)
            shift = np.max(np.linalg.norm(C_nuevo - C, axis=1))
            if self.guardar_historial:
                historial.append({"it": it, "C_prev": C.copy(),
                                  "labels": labels.copy(), "C_new": C_nuevo.copy(),
                                  "shift": shift})
            C = C_nuevo
            if shift < self.tol:
                convergio = True
                break
        labels = self._asignar(X, C)
        return C, labels, wcss(X, labels, C), it, convergio, historial

    # ---------- API pública ----------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.n_empty_ = 0
        mejor = None
        n_init = 1 if self.init == "manual" else self.n_init
        base = 0 if self.random_state is None else self.random_state
        for s in range(n_init):
            rng = np.random.default_rng(base + s)
            res = self._ajustar_una_vez(X, rng)
            if mejor is None or res[2] < mejor[2]:
                mejor = res
        (self.centroids_, self.labels_, self.inertia_,
         self.n_iter_, self.converged_, self.history_) = mejor
        return self

    def predict(self, X):
        return self._asignar(np.asarray(X, dtype=float), self.centroids_)

    def score_samples(self, X):
        """Puntuación de anomalía: distancia al centroide más cercano."""
        return distancias_euclidianas(np.asarray(X, dtype=float),
                                      self.centroids_).min(axis=1)
