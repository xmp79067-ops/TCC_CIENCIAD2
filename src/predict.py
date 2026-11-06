"""Predicción por lotes: valida, limpia y puntúa un archivo con clientes nuevos.

Lo usa la app (sección "Predicción por lotes") y las pruebas; no depende de Streamlit.
"""
import numpy as np
import pandas as pd

from src import config
from src.data_prep import clean_data, engineer_features

REQUIRED = ["Edad", "Antiguedad_Meses", "Soporte_Tickets", "Uso_Diario_Min",
            "Monto_Mensual", "Genero", "Region"]


def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Valida columnas, aplica la misma limpieza del entrenamiento y crea las variables derivadas."""
    missing = [c for c in REQUIRED if c not in df_raw.columns]
    if missing:
        raise ValueError("faltan columnas obligatorias: " + ", ".join(missing))
    if df_raw.empty:
        raise ValueError("el archivo no contiene filas")
    df = df_raw.copy()
    if "ID_Cliente" not in df.columns:
        df.insert(0, "ID_Cliente", [f"fila_{i + 1}" for i in range(len(df))])
    return engineer_features(clean_data(df)).reset_index(drop=True)


def score(feats: pd.DataFrame, model, cluster: dict, umbral: float = 0.5) -> pd.DataFrame:
    """Devuelve puntaje, decisión, cluster y perfil por cliente, ordenado de mayor a menor riesgo."""
    X = feats[config.MODEL_INPUTS]
    proba = model.predict_proba(X)[:, 1]
    k = cluster["kmeans"].predict(cluster["pre"].transform(X[cluster["cols"]]))
    out = pd.DataFrame({
        "ID_Cliente": feats["ID_Cliente"].to_numpy(),
        "Puntaje_Riesgo": proba.round(4),
        "Decision": np.where(proba >= umbral, "Riesgo ALTO", "Riesgo bajo"),
        "Cluster": k.astype(int),
        "Perfil": [cluster["names"][int(i)] for i in k],
    })
    return out.sort_values("Puntaje_Riesgo", ascending=False).reset_index(drop=True)
