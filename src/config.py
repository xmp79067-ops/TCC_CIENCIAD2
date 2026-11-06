"""Configuración central del proyecto: rutas, semilla y definición de columnas."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "clientes.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET = "Churn"

# Variables continuas fuertemente sesgadas -> log1p + StandardScaler
NUM_SKEW = ["Uso_Diario_Min", "Monto_Mensual", "Costo_Por_Minuto"]
# Variables continuas / conteos de distribución moderada -> StandardScaler
NUM_NORMAL = ["Edad", "Antiguedad_Meses", "Soporte_Tickets"]
# Banderas binarias (se pasan sin transformar)
BINARY = ["Atencion_Requerida", "Soporte_Faltante"]
# Categóricas -> One-Hot Encoding
CATEGORICAL = ["Genero", "Region"]

MODEL_INPUTS = NUM_SKEW + NUM_NORMAL + BINARY + CATEGORICAL

# Variables usadas para segmentación (comportamiento de uso y soporte, sin demografía)
CLUSTER_SKEW = ["Uso_Diario_Min", "Monto_Mensual", "Costo_Por_Minuto"]
CLUSTER_NORMAL = ["Antiguedad_Meses", "Soporte_Tickets"]

GENERO_OPCIONES = ["M", "F", "Otro"]
REGION_OPCIONES = ["Norte", "Sur", "Centro", "Este", "Oeste"]


def ensure_dirs() -> None:
    for p in (DATA_RAW.parent, DATA_PROCESSED, MODELS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
