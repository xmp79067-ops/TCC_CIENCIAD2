"""Fase 4: Segmentación no supervisada con K-Means (Elbow + Silueta + perfiles).

Se agrupa por comportamiento de uso, facturación y soporte (sin Churn ni demografía).
Churn se usa SOLO después, para describir el riesgo de cada perfil.

Uso:  python -m src.clustering
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src import config
from src.data_prep import build_preprocessor, load_clean

K_RANGE = range(2, 9)
MIN_K = 3  # para que la segmentación sea accionable (k=2 sería solo "alto/bajo")


def make_cluster_preprocessor():
    return build_preprocessor(skew=config.CLUSTER_SKEW, normal=config.CLUSTER_NORMAL,
                              categorical=[], binary=[])


def name_profile(row: pd.Series, g: pd.Series) -> str:
    tags = []
    if row["Uso_Diario_Min"] > 1.2 * g["Uso_Diario_Min"]:
        tags.append("uso intensivo")
    elif row["Uso_Diario_Min"] < 0.8 * g["Uso_Diario_Min"]:
        tags.append("uso bajo")
    if row["Monto_Mensual"] > 1.2 * g["Monto_Mensual"]:
        tags.append("tarifa alta")
    elif row["Monto_Mensual"] < 0.8 * g["Monto_Mensual"]:
        tags.append("tarifa baja")
    if row["Soporte_Tickets"] > 1.3 * g["Soporte_Tickets"]:
        tags.append("alto soporte")
    if row["Antiguedad_Meses"] > 1.3 * g["Antiguedad_Meses"]:
        tags.append("antiguos")
    elif row["Antiguedad_Meses"] < 0.7 * g["Antiguedad_Meses"]:
        tags.append("recientes")
    label = ", ".join(tags) if tags else "perfil promedio"
    if row["Churn"] > 1.3 * g["Churn"]:
        label += " [RIESGO ALTO]"
    elif row["Churn"] < 0.7 * g["Churn"]:
        label += " [RIESGO BAJO]"
    return label


def main():
    config.ensure_dirs()
    df = load_clean(verbose=False)
    cols = config.CLUSTER_SKEW + config.CLUSTER_NORMAL
    pre = make_cluster_preprocessor()
    Z = pre.fit_transform(df[cols])
    names = list(pre.get_feature_names_out())

    # --- Elbow + Silueta ---
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=4000, random_state=config.RANDOM_STATE)
        rows.append({"k": k, "inercia": km.inertia_, "silueta": sil})
    sel = pd.DataFrame(rows)
    print(sel.round(3).to_string(index=False))
    k_opt = int(sel[sel["k"] >= MIN_K].sort_values("silueta", ascending=False).iloc[0]["k"])
    print(f"\nk óptimo (máxima silueta con k >= {MIN_K}): {k_opt}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sel["k"], sel["inercia"], "o-")
    axes[0].axvline(k_opt, color="r", ls="--")
    axes[0].set(title="Método del Codo", xlabel="k", ylabel="Inercia (WCSS)")
    axes[1].plot(sel["k"], sel["silueta"], "o-", color="tab:green")
    axes[1].axvline(k_opt, color="r", ls="--")
    axes[1].set(title="Coeficiente de Silueta", xlabel="k", ylabel="Silueta media")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "clustering_codo_silueta.png", dpi=130)
    plt.close(fig)

    # --- Modelo final y perfiles ---
    km = KMeans(n_clusters=k_opt, n_init=20, random_state=config.RANDOM_STATE).fit(Z)
    df["Cluster"] = km.labels_
    profile_cols = ["Uso_Diario_Min", "Monto_Mensual", "Costo_Por_Minuto", "Soporte_Tickets",
                    "Antiguedad_Meses", "Edad", "Churn"]
    prof = df.groupby("Cluster")[profile_cols].mean()
    prof.insert(0, "Tamaño_%", df["Cluster"].value_counts(normalize=True).sort_index() * 100)
    g = df[profile_cols].mean()
    prof["Perfil"] = prof.apply(lambda r: name_profile(r, g), axis=1)
    pd.set_option("display.width", 200)
    print("\n=== Perfiles (promedios en unidades originales) ===")
    print(prof.round(2).to_string())
    prof.to_csv(config.DATA_PROCESSED / "perfiles_clusters.csv")

    # Centroides estandarizados
    cent = pd.DataFrame(km.cluster_centers_, columns=names)
    plt.figure(figsize=(8, 0.9 * k_opt + 2))
    sns.heatmap(cent, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Centroides (variables transformadas y estandarizadas)")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "clustering_centroides.png", dpi=130)
    plt.close()

    # Proyección PCA 2D
    pca = PCA(2, random_state=config.RANDOM_STATE)
    P = pca.fit_transform(Z)
    plt.figure(figsize=(7, 5.5))
    sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=df["Cluster"], palette="tab10", s=8, alpha=0.6)
    plt.title(f"Clusters en el plano PCA (varianza explicada {pca.explained_variance_ratio_.sum():.0%})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "clustering_pca.png", dpi=130)
    plt.close()

    joblib.dump({"pre": pre, "kmeans": km, "cols": cols,
                 "names": prof["Perfil"].to_dict(), "profiles": prof}, config.MODELS / "cluster_model.joblib")
    print("\nModelo de clustering guardado en models/cluster_model.joblib")


if __name__ == "__main__":
    main()
