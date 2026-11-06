"""Fase 1: Análisis Exploratorio (EDA) y verificación de la hipótesis de uso vs. abandono.

Uso:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import config
from src.data_prep import load_clean


def main():
    config.ensure_dirs()
    raw = pd.read_csv(config.DATA_RAW)
    print("=== Calidad de datos (crudo) ===")
    print(f"Registros: {len(raw)} | ID duplicados: {raw['ID_Cliente'].duplicated(keep=False).sum()} "
          f"| ID nulos: {raw['ID_Cliente'].isna().sum()}")
    print("Nulos por columna:\n", raw.isna().sum()[lambda s: s > 0])

    df = load_clean(verbose=True)
    print("\n=== Balance de clases ===")
    print(df[config.TARGET].value_counts(normalize=True).round(3))

    num_cols = config.NUM_SKEW + config.NUM_NORMAL
    print("\n=== Estadísticos descriptivos ===")
    print(df[num_cols].describe().round(2).T)
    print("\nAsimetría (skewness):\n", df[num_cols].skew().round(2))

    # Distribuciones
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), num_cols):
        sns.histplot(df[c].dropna(), bins=40, kde=True, ax=ax)
        ax.set_title(c)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "eda_distribuciones.png", dpi=130)
    plt.close(fig)

    # Correlaciones
    corr = df[num_cols + [config.TARGET]].corr(method="spearman")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlación de Spearman")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "eda_correlacion.png", dpi=130)
    plt.close()
    print("\n=== Correlación con Churn (Spearman) ===")
    print(corr[config.TARGET].drop(config.TARGET).round(3).sort_values())

    # Hipótesis: a mayor uso diario, menor probabilidad de abandono
    print("\n=== Hipótesis: mayor uso diario -> menor abandono ===")
    d = df[["Uso_Diario_Min", config.TARGET]].dropna()
    rho, p_rho = stats.spearmanr(d["Uso_Diario_Min"], d[config.TARGET])
    u, p_u = stats.mannwhitneyu(d.loc[d[config.TARGET] == 1, "Uso_Diario_Min"],
                                d.loc[d[config.TARGET] == 0, "Uso_Diario_Min"], alternative="less")
    med = d.groupby(config.TARGET)["Uso_Diario_Min"].median()
    print(f"Mediana de uso | activos: {med[0]:.1f} min | cancelaron: {med[1]:.1f} min")
    print(f"Spearman rho = {rho:.3f} (p = {p_rho:.2e})")
    print(f"Mann-Whitney U (uso de churners < uso de activos): p = {p_u:.2e}")
    d["Quintil_Uso"] = pd.qcut(d["Uso_Diario_Min"], 5, labels=["Q1 (bajo)", "Q2", "Q3", "Q4", "Q5 (alto)"])
    rate = d.groupby("Quintil_Uso", observed=True)[config.TARGET].mean()
    print("Tasa de churn por quintil de uso:\n", (rate * 100).round(1).astype(str) + "%")
    print("Conclusión:", "hipótesis SOPORTADA" if (rho < 0 and p_rho < 0.05) else "hipótesis NO soportada")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=d, x=config.TARGET, y="Uso_Diario_Min", ax=axes[0], showfliers=False)
    axes[0].set_title("Uso diario según Churn")
    rate.plot(kind="bar", ax=axes[1], color="tab:red")
    axes[1].set_title("Tasa de churn por quintil de uso")
    axes[1].set_ylabel("Churn")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "eda_hipotesis_uso.png", dpi=130)
    plt.close(fig)

    # Categóricas
    print("\n=== Churn por categoría ===")
    for c in config.CATEGORICAL:
        t = df.groupby(c)[config.TARGET].agg(["count", "mean"]).round(3)
        print(f"\n{c}:\n{t}")


if __name__ == "__main__":
    main()
