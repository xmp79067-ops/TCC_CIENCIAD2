"""Fase 5 (Ética): análisis de equidad del modelo según Genero y Region.

Métricas por grupo: tasa de selección (predicho churn), TPR (recall), FPR, precisión.
Métricas agregadas: diferencia de paridad demográfica, diferencia de oportunidad igual
(TPR), diferencia de odds igualadas y ratio de impacto dispar (regla del 80%).

Uso:  python -m src.fairness
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

MIN_GROUP = 30  # grupos más pequeños dan métricas poco fiables


def group_metrics(y, pred, groups) -> pd.DataFrame:
    rows = []
    unique_groups = [g for g in groups.dropna().unique()]
    try:
        unique_groups = sorted(unique_groups)
    except TypeError:
        pass
    for g in unique_groups:
        m = (groups == g).to_numpy()
        yt, yp = y[m], pred[m]
        tp, fp = ((yt == 1) & (yp == 1)).sum(), ((yt == 0) & (yp == 1)).sum()
        fn, tn = ((yt == 1) & (yp == 0)).sum(), ((yt == 0) & (yp == 0)).sum()
        rows.append({
            "grupo": g, "n": int(m.sum()), "tasa_churn_real": yt.mean(),
            "tasa_seleccion": yp.mean(),
            "TPR_recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        })
    return pd.DataFrame(rows).set_index("grupo")


def summarize(t: pd.DataFrame) -> dict:
    t = t[t["n"] >= MIN_GROUP]
    sel = t["tasa_seleccion"]
    return {
        "dif_paridad_demografica": sel.max() - sel.min(),
        "ratio_impacto_dispar": sel.min() / sel.max() if sel.max() > 0 else np.nan,
        "dif_oportunidad_igual_TPR": t["TPR_recall"].max() - t["TPR_recall"].min(),
        "dif_FPR": t["FPR"].max() - t["FPR"].min(),
    }


def main():
    config.ensure_dirs()
    model = joblib.load(config.MODELS / "best_model.joblib")
    test = pd.read_csv(config.DATA_PROCESSED / "test_set.csv")
    y = test[config.TARGET].to_numpy()
    pred = model.predict(test[config.MODEL_INPUTS])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    summary = {}
    for ax, attr in zip(axes, config.CATEGORICAL):
        t = group_metrics(y, pred, test[attr])
        print(f"\n=== Equidad por {attr} ===")
        print(t.round(3).to_string())
        s = summarize(t)
        summary[attr] = s
        for k, v in s.items():
            print(f"  {k}: {v:.3f}")
        flag = "CUMPLE" if (pd.notna(s["ratio_impacto_dispar"]) and s["ratio_impacto_dispar"] >= 0.8) else "NO CUMPLE"
        print(f"  Regla del 80% (impacto dispar): {flag}")
        t[["tasa_churn_real", "tasa_seleccion", "TPR_recall", "FPR"]].plot(kind="bar", ax=ax)
        ax.set_title(f"Métricas por {attr}")
        ax.tick_params(axis="x", rotation=0)
        ax.set_ylim(0, 1)
        t.to_csv(config.DATA_PROCESSED / f"fairness_{attr}.csv")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fairness.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(summary).T.to_csv(config.DATA_PROCESSED / "fairness_resumen.csv")


if __name__ == "__main__":
    main()
