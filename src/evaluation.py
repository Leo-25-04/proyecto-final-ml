"""Métricas de evaluación: anomalías (con isFraud solo para evaluar) y clustering."""
import numpy as np
from sklearn.metrics import (average_precision_score, calinski_harabasz_score,
                             davies_bouldin_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)


def umbral_max_f1(y, score):
    """Umbral que maximiza F1 (se elige en VALIDACIÓN y se aplica a prueba)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f1)])


def metricas_anomalia(y, score, umbral):
    """Métricas de clasificación sobre la puntuación de anomalía de un modelo."""
    y = np.asarray(y)
    pred = (score >= umbral).astype(int)
    top_k = np.argsort(-score)[: int(y.sum())]
    return {
        "prevalencia": y.mean(),
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "accuracy": (pred == y).mean(),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "precision_top_k": y[top_k].mean(),
    }


def metricas_cluster(X, labels, n_muestra=10_000, seed=42):
    """Silhouette, Calinski-Harabasz y Davies-Bouldin sobre una muestra.

    Excluye el ruido (etiqueta -1). Devuelve NaN si hay menos de 2 clústeres.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    ok = labels != -1
    X, labels = X[ok], labels[ok]
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan}
    if len(X) > n_muestra:
        idx = np.random.default_rng(seed).choice(len(X), n_muestra, replace=False)
        X, labels = X[idx], labels[idx]
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan}
    return {"silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels)}
