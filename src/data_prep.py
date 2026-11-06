"""Fase 2: limpieza, calidad de datos e ingeniería de características."""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


# ----------------------------------------------------------------------------
# Limpieza (reglas deterministas; la imputación estadística va dentro del Pipeline
# para ajustarse solo con datos de entrenamiento y evitar fuga de información)
# ----------------------------------------------------------------------------
def parse_antiguedad(v):
    """Convierte '12', '12 meses', '12m', '2.0 años' a meses (float)."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().lower().replace(",", ".")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan
    x = float(m.group(1))
    if "año" in s or "ano" in s or "year" in s:
        x *= 12
    return x


def clean_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)
    n_null, n_dup, n_edad, n_monto = 0, 0, 0, 0

    # ID: sin ID no hay trazabilidad -> se elimina; duplicados -> se conserva el primero
    if "ID_Cliente" in df.columns:
        df = df.dropna(subset=["ID_Cliente"])
        n_null = n0 - len(df)
        df = df.drop_duplicates(subset="ID_Cliente", keep="first")
        n_dup = n0 - n_null - len(df)

    # Antiguedad_Meses: unificar formato y descartar valores imposibles
    if "Antiguedad_Meses" in df.columns:
        df["Antiguedad_Meses"] = df["Antiguedad_Meses"].map(parse_antiguedad)
        df.loc[(df["Antiguedad_Meses"] < 0) | (df["Antiguedad_Meses"] > 600), "Antiguedad_Meses"] = np.nan

    # Edad: rango plausible 18-90; fuera de rango -> NaN (se imputa con la mediana)
    if "Edad" in df.columns:
        df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")
        n_edad = int(((df["Edad"] < 18) | (df["Edad"] > 90)).sum())
        df.loc[(df["Edad"] < 18) | (df["Edad"] > 90), "Edad"] = np.nan

    # Monto_Mensual: registros en centavos (> 1000) se llevan a USD; no positivos -> NaN
    if "Monto_Mensual" in df.columns:
        df["Monto_Mensual"] = pd.to_numeric(df["Monto_Mensual"], errors="coerce")
        n_monto = int((df["Monto_Mensual"] > 1000).sum())
        df.loc[df["Monto_Mensual"] > 1000, "Monto_Mensual"] /= 100
        df.loc[df["Monto_Mensual"] <= 0, "Monto_Mensual"] = np.nan

    if "Uso_Diario_Min" in df.columns:
        df["Uso_Diario_Min"] = pd.to_numeric(df["Uso_Diario_Min"], errors="coerce")
        df.loc[df["Uso_Diario_Min"] < 0, "Uso_Diario_Min"] = np.nan
    if "Soporte_Tickets" in df.columns:
        df["Soporte_Tickets"] = pd.to_numeric(df["Soporte_Tickets"], errors="coerce")
        df.loc[df["Soporte_Tickets"] < 0, "Soporte_Tickets"] = np.nan

    for c in config.CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), "Desconocido").astype(str).str.strip()

    if verbose:
        print(f"Filas originales: {n0} | ID nulo: {n_null} | duplicados: {n_dup} | "
              f"Edad fuera de rango: {n_edad} | Monto en centavos corregido: {n_monto} | "
              f"filas finales: {len(df)}")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Ingeniería de características
# ----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Costo_Por_Minuto = Monto_Mensual / (Uso_Diario_Min * 30). Se acota el uso a >= 1 min
    # solo en el denominador para evitar divisiones por cero / valores infinitos.
    uso = df["Uso_Diario_Min"].clip(lower=1)
    df["Costo_Por_Minuto"] = df["Monto_Mensual"] / (uso * 30)
    # Atencion_Requerida = 1 si Soporte_Tickets > 5 (un NaN no cumple la condición -> 0)
    df["Atencion_Requerida"] = (df["Soporte_Tickets"] > 5).astype(int)
    # Indicador de dato faltante: los NAs no son aleatorios, el hecho de faltar es informativo
    df["Soporte_Faltante"] = df["Soporte_Tickets"].isna().astype(int)
    return df


def load_clean(path=config.DATA_RAW, verbose: bool = False) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return engineer_features(clean_data(raw, verbose=verbose))


# ----------------------------------------------------------------------------
# Preprocesamiento: imputación + log1p (sesgadas) + StandardScaler + One-Hot
# ----------------------------------------------------------------------------
def build_preprocessor(skew=None, normal=None, categorical=None, binary=None) -> ColumnTransformer:
    skew = config.NUM_SKEW if skew is None else skew
    normal = config.NUM_NORMAL if normal is None else normal
    categorical = config.CATEGORICAL if categorical is None else categorical
    binary = config.BINARY if binary is None else binary

    skew_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("sc", StandardScaler()),
    ])
    norm_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    parts = []
    if skew:
        parts.append(("skew", skew_pipe, skew))
    if normal:
        parts.append(("num", norm_pipe, normal))
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    if binary:
        parts.append(("bin", "passthrough", binary))
    return ColumnTransformer(parts, verbose_feature_names_out=False)
