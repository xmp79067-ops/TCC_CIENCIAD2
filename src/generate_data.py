"""Genera un dataset sintético de 10,000 registros con los problemas de calidad del enunciado.

Úsalo SOLO si no cuentas con el CSV original. Si tienes el dataset real, colócalo en
data/raw/clientes.csv con las mismas columnas y omite este paso.

Uso:  python -m src.generate_data
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src import config


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def generate(n_unique: int = 9800, n_dups: int = 200, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ids = np.array([f"C{i:05d}" for i in range(1, n_unique + 1)], dtype=object)
    genero = rng.choice(config.GENERO_OPCIONES, n_unique, p=[0.58, 0.38, 0.04])
    region = rng.choice(config.REGION_OPCIONES, n_unique, p=[0.40, 0.25, 0.18, 0.12, 0.05])
    edad = np.clip(rng.normal(41, 10, n_unique), 22, 70).round()
    antig = np.clip(rng.gamma(2.2, 10, n_unique), 1, 96).round()
    uso = np.clip(rng.lognormal(3.2, 0.8, n_unique), 0.5, 600).round(1)  # sesgada a la derecha

    # Plan contratado según el nivel de uso, con ruido en la tarifa
    tier = np.digitize(uso, np.quantile(uso, [0.40, 0.75, 0.93]))
    monto = np.array([49, 99, 199, 499])[tier] * rng.normal(1, 0.03, n_unique)
    monto = monto.round(2)

    tickets = rng.poisson(rng.gamma(2.0, 1.2, n_unique)).astype(float)

    # Proceso generador del churn (menor uso -> mayor riesgo; más tickets -> mayor riesgo)
    region_eff = pd.Series(region).map({"Este": 0.50}).fillna(0.0).to_numpy()
    base = (-1.5 * (np.log(uso) - 3.2) + 0.40 * tickets - 0.020 * antig
            + region_eff + rng.normal(0, 0.35, n_unique))
    b = brentq(lambda c: _sigmoid(c + base).mean() - 0.20, -10, 10)
    churn = (rng.random(n_unique) < _sigmoid(b + base)).astype(int)

    df = pd.DataFrame({
        "ID_Cliente": ids, "Edad": edad, "Antiguedad_Meses": antig,
        "Soporte_Tickets": tickets, "Uso_Diario_Min": uso, "Monto_Mensual": monto,
        "Genero": genero, "Region": region, "Churn": churn,
    })

    # ---- Problemas de calidad intencionales ----
    # Edad: outliers (negativos, 0, 150+)
    idx = rng.choice(n_unique, int(0.012 * n_unique), replace=False)
    df.loc[idx, "Edad"] = rng.choice([-5, -1, 0, 150, 200], len(idx))

    # Antiguedad_Meses: formatos mezclados (número, "12 meses", "12m", "2.0 años")
    def fmt(v):
        r = rng.random()
        if r < 0.85:
            return str(int(v))
        if r < 0.93:
            return f"{int(v)} meses"
        if r < 0.97:
            return f"{int(v)}m"
        return f"{round(v / 12, 1)} años" if v >= 12 else str(int(v))
    df["Antiguedad_Meses"] = [fmt(v) for v in df["Antiguedad_Meses"]]

    # Soporte_Tickets: faltantes NO aleatorios (más probables con más tickets)
    p_miss = np.clip(0.04 + 0.05 * df["Soporte_Tickets"].to_numpy(), 0, 0.5)
    df.loc[rng.random(n_unique) < p_miss, "Soporte_Tickets"] = np.nan

    # Monto_Mensual: escalas distintas (algunos registros en centavos)
    idx = rng.choice(n_unique, int(0.03 * n_unique), replace=False)
    df.loc[idx, "Monto_Mensual"] = (df.loc[idx, "Monto_Mensual"] * 100).round(0)

    # ID nulo ocasional y registros duplicados
    idx = rng.choice(n_unique, int(0.008 * n_unique), replace=False)
    df.loc[idx, "ID_Cliente"] = np.nan
    dups = df.sample(n_dups, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    config.ensure_dirs()
    data = generate()
    data.to_csv(config.DATA_RAW, index=False)
    print(f"Dataset sintético guardado en {config.DATA_RAW}  {data.shape}")
    print(f"Tasa de churn: {data['Churn'].mean():.1%}")
