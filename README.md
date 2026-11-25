# Detección no supervisada de fraude en PaySim

Proyecto final de Aprendizaje Automático (Dra. Grissel Rodríguez Roldán).
Autor: Leonardo Negrete.

Se compara cómo detectan fraude en transacciones de dinero móvil cuatro modelos
no supervisados (k-Means, GMM, DBSCAN e Isolation Forest) más un k-Means propio
implementado desde cero con NumPy. La etiqueta `isFraud` **nunca se usa para
entrenar**: solo para elegir hiperparámetros en validación y para evaluar.

## Datos

PaySim (Kaggle: `ealaxi/paysim1`), 6,362,620 transacciones, 743 pasos de tiempo.
El fraude solo aparece en CASH_OUT y TRANSFER, así que se modela ese subconjunto
(2,770,409 filas, 0.30 % de fraude). El CSV no se incluye en el repositorio:
`src/download_data.py` lo descarga solo.

## Partición cronológica

- Ajuste: `step <= 500`
- Validación: `501 <= step <= 600` (elige hiperparámetros y umbral)
- Prueba: `step > 600` (solo evaluación final)

El escalador se ajusta solo con los datos de entrenamiento (sin fuga de datos).

## Cómo ejecutarlo (Windows, PowerShell)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python proyecto_final.py

Esto descarga el dato, hace el EDA, valida el k-Means propio contra scikit-learn,
evalúa los modelos con los hiperparámetros finales y genera las figuras
(unos minutos en un equipo de escritorio). Para repetir también la exploración
de hiperparámetros, las 5 semillas y la ablación (más lento):

    python proyecto_final.py --completo

La semilla fija es `random_state = 42`. Si Kaggle pide credenciales, descarga
`paysim1` a mano y guárdalo como `data/paysim.csv`.

## Estructura

- `proyecto_final.py`: orquesta todo el flujo.
- `src/`: EDA, preprocesamiento, modelos, k-Means propio, evaluación, estabilidad y ablación.
- `reports/`: tablas (`.csv`) y figuras (`figures/`) generadas.

## Resultados en prueba (prevalencia 3.67 %)

| Modelo | PR-AUC | ROC-AUC |
|---|---|---|
| GMM (K=2) | 0.9988 | 1.0000 |
| DBSCAN (eps=1.0, min_samples=10) | 0.9659 | 0.9987 |
| Isolation Forest (max_samples=65536) | 0.9248 | 0.9964 |
| k-Means sklearn (k=3) | 0.8669 | 0.9928 |
| k-Means propio (k=3) | 0.8674 | 0.9928 |

El detalle (estabilidad con 5 semillas, ablación de variables y limitaciones)
está en el reporte técnico.
