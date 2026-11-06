"""Fase 5 (XAI): explicabilidad con SHAP del modelo supervisado.

Genera: (1) top-3 de variables globales, (2) gráficos globales (barras y beeswarm),
(3) explicación individual (waterfall) de 2 clientes del conjunto de prueba.

Requiere:  pip install shap
Uso:  python -m src.explain
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config


def transform_for_clf(model, X: pd.DataFrame):
    """Aplica preprocesamiento + selección (sin SMOTE ni clasificador) y devuelve matriz y nombres."""
    Xt = X
    feature_names = None
    for name, step in model.named_steps.items():
        if name in ("clf", "smote"):
            continue
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)
            if hasattr(step, "get_feature_names_out"):
                try:
                    if feature_names is not None:
                        feature_names = list(step.get_feature_names_out(feature_names))
                    else:
                        feature_names = list(step.get_feature_names_out())
                except Exception:
                    feature_names = list(step.get_feature_names_out())
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(Xt.shape[1])]
    return np.asarray(Xt, dtype=float), feature_names


def make_explainer(clf, background):
    import shap
    if hasattr(clf, "estimators_"):  # Random Forest / ensembles de árboles
        return shap.TreeExplainer(clf)
    return shap.LinearExplainer(clf, background)


def shap_for_class1(explainer, Xt):
    """Devuelve (valores SHAP clase 1, valor base clase 1) con forma estable entre versiones."""
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    base = np.atleast_1d(explainer.expected_value)
    base = float(base[1] if len(base) > 1 else base[0])
    return np.asarray(sv), base


def main():
    try:
        import shap
    except ImportError:
        raise SystemExit("Instala SHAP primero:  pip install shap")

    config.ensure_dirs()
    model = joblib.load(config.MODELS / "best_model.joblib")
    test = pd.read_csv(config.DATA_PROCESSED / "test_set.csv")
    X = test[config.MODEL_INPUTS]
    Xt, names = transform_for_clf(model, X)
    clf = model.named_steps["clf"]
    explainer = make_explainer(clf, Xt)
    sv, base = shap_for_class1(explainer, Xt)

    # ---- Global ----
    imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    print("=== Importancia global (media |SHAP|) ===")
    print(imp.round(4).to_string())
    print("\nTOP 3 variables más determinantes del abandono:", list(imp.index[:3]))

    shap.summary_plot(sv, Xt, feature_names=names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES / "shap_global_barras.png", dpi=130, bbox_inches="tight")
    plt.close()
    shap.summary_plot(sv, Xt, feature_names=names, show=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES / "shap_global_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ---- Individual: 1 cliente de alto riesgo que canceló y 1 de bajo riesgo activo ----
    proba = model.predict_proba(X)[:, 1]
    hi = int(np.argmax(np.where(test[config.TARGET] == 1, proba, -1)))
    lo = int(np.argmin(np.where(test[config.TARGET] == 0, proba, 2)))
    for tag, i in (("alto_riesgo", hi), ("bajo_riesgo", lo)):
        print(f"\n--- Cliente {tag} (fila {i}) | P(churn) = {proba[i]:.1%} | real = {test[config.TARGET][i]} ---")
        print(X.iloc[i].to_string())
        contrib = pd.Series(sv[i], index=names).sort_values(key=np.abs, ascending=False)
        print("Contribuciones SHAP (log-odds/prob. según modelo):\n", contrib.round(3).head(5).to_string())
        exp = shap.Explanation(values=sv[i], base_values=base, data=Xt[i], feature_names=names)
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        plt.savefig(config.FIGURES / f"shap_individual_{tag}.png", dpi=130, bbox_inches="tight")
        plt.close()
    print("\nGráficos guardados en reports/figures/")


if __name__ == "__main__":
    main()
