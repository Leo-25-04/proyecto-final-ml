"""Descarga automatizada del dataset PaySim desde Kaggle."""
import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DESTINO = DATA_DIR / "paysim.csv"


def descargar_paysim() -> Path:
    """Descarga PaySim y lo copia a data/paysim.csv (si no existe)."""
    if DESTINO.exists():
        print(f"El archivo ya existe: {DESTINO}")
        return DESTINO
    carpeta = Path(kagglehub.dataset_download("ealaxi/paysim1"))
    csv_origen = next(carpeta.glob("*.csv"))
    DATA_DIR.mkdir(exist_ok=True)
    shutil.copy(csv_origen, DESTINO)
    print(f"Dataset guardado en: {DESTINO}")
    return DESTINO


if __name__ == "__main__":
    descargar_paysim()
