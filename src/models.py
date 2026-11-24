"""Detectores de anomalías con interfaz común.

Cada detector tiene fit(X), score(X) (mayor = más anómalo) y labels(X)
(etiqueta de clúster; None si el modelo no define clústeres).
"""
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from kmeans_scratch import KMeansScratch


class KMeansSklearn:
    def __init__(self, k, seed):
        self.m = KMeans(k, n_init=10, random_state=seed)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return self.m.transform(X).min(axis=1)

    def labels(self, X):
        return self.m.predict(X)


class KMeansPropio:
    def __init__(self, k, seed):
        self.m = KMeansScratch(k, init="kmeans++", n_init=5, max_iter=300,
                               random_state=seed)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return self.m.score_samples(X)

    def labels(self, X):
        return self.m.predict(X)


class GMM:
    def __init__(self, k, seed):
        self.m = GaussianMixture(k, covariance_type="full", reg_covar=1e-3,
                                 max_iter=200, n_init=3, random_state=seed)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)  # -log verosimilitud

    def labels(self, X):
        return self.m.predict(X)


class DBSCANDetector:
    """DBSCAN no tiene predict: los puntos nuevos se asignan por su punto
    núcleo más cercano; si está a más de eps, son ruido (-1)."""

    def __init__(self, eps, min_samples):
        self.m = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.eps = eps

    def fit(self, X):
        self.m.fit(X)
        nucleos = self.m.core_sample_indices_
        if len(nucleos) == 0:
            raise ValueError("DBSCAN sin puntos núcleo con estos parámetros")
        self.core_ = np.asarray(X)[nucleos]
        self.core_labels_ = self.m.labels_[nucleos]
        self.nn_ = NearestNeighbors(n_neighbors=1).fit(self.core_)
        return self

    def score(self, X):
        return self.nn_.kneighbors(X)[0][:, 0]

    def labels(self, X):
        dist, idx = self.nn_.kneighbors(X)
        etiquetas = self.core_labels_[idx[:, 0]].copy()
        etiquetas[dist[:, 0] > self.eps] = -1
        return etiquetas


class IsolationForestDetector:
    def __init__(self, max_samples, seed):
        self.m = IsolationForest(n_estimators=200, max_samples=max_samples,
                                 random_state=seed, n_jobs=-1)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)

    def labels(self, X):
        return None

