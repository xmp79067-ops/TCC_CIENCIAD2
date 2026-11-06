"""Fase 3: Modelado supervisado (Regresión Logística vs. Random Forest).

- Split estratificado 80/20 (hold-out) + StratifiedKFold(5) dentro de GridSearchCV.
- Desbalance: class_weight='balanced' (por defecto) o SMOTE con --smote (requiere imbalanced-learn).
- Selección de características por importancia de Random Forest dentro del Pipeline.

Uso:  python -m src.train [--smote]
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             average_precision_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data_prep import build_preprocessor, load_clean


def make_pipeline(clf, use_smote: bool = False):
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                               class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=1),
        threshold="0.5*mean",
    )
    steps = [("prep", build_preprocessor()), ("select", selector)]
    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
        steps.append(("clf", clf))
        return ImbPipeline(steps)  # SMOTE solo se aplica a los folds de entrenamiento
    steps.append(("clf", clf))
    return Pipeline(steps)


def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "modelo": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }, pred, proba


def main(use_smote: bool = False):
    config.ensure_dirs()
    df = load_clean(verbose=True).dropna(subset=[config.TARGET])
    X, y = df[config.MODEL_INPUTS], df[config.TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.RANDOM_STATE)
    print(f"Train: {len(X_train)} ({y_train.mean():.1%} churn) | Test: {len(X_test)} ({y_test.mean():.1%} churn)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    spec = {
        "Regresion_Logistica": (
            LogisticRegression(max_iter=2000, class_weight=None if use_smote else "balanced",
                               random_state=config.RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Random_Forest": (
            RandomForestClassifier(class_weight=None if use_smote else "balanced_subsample",
                                   random_state=config.RANDOM_STATE, n_jobs=-1),
            {"clf__n_estimators": [200, 300], "clf__max_depth": [4, 6, 10],
             "clf__min_samples_leaf": [5, 20]},
        ),
    }

    results, fitted, preds = [], {}, {}
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    print(f"\nLínea base 'siempre activo': accuracy = {accuracy_score(y_test, dummy.predict(X_test)):.3f}, "
          f"recall = 0.000  -> por eso Accuracy no sirve con 80/20")

    for name, (clf, grid) in spec.items():
        print(f"\n>>> GridSearchCV: {name}")
        gs = GridSearchCV(make_pipeline(clf, use_smote), grid, cv=cv, n_jobs=-1,
                          scoring={"f1": "f1", "recall": "recall", "roc_auc": "roc_auc"}, refit="f1")
        gs.fit(X_train, y_train)
        print("Mejores hiperparámetros:", gs.best_params_)
        m, pred, proba = evaluate(name, gs.best_estimator_, X_test, y_test)
        m.update({"cv_f1": gs.best_score_,
                  "cv_recall": gs.cv_results_["mean_test_recall"][gs.best_index_],
                  "cv_roc_auc": gs.cv_results_["mean_test_roc_auc"][gs.best_index_],
                  "mejores_params": json.dumps(gs.best_params_)})
        results.append(m)
        fitted[name], preds[name] = gs.best_estimator_, (pred, proba)
        print(classification_report(y_test, pred, target_names=["Activo", "Churn"], digits=3))

    res = pd.DataFrame(results).set_index("modelo")
    print("\n=== Comparación en el conjunto de prueba ===")
    print(res.drop(columns="mejores_params").round(3).T)
    res.to_csv(config.DATA_PROCESSED / "metricas_modelos.csv")

    # Mejor modelo según F1 en validación cruzada (NO en test, para no filtrar información)
    best_name = res["cv_f1"].idxmax()
    best = fitted[best_name]
    print(f"\nMejor modelo (por F1 en CV): {best_name}")

    all_names = list(best.named_steps["prep"].get_feature_names_out())
    kept = list(best.named_steps["select"].get_feature_names_out(all_names))
    print(f"Variables de entrada tras preprocesar: {len(all_names)} | seleccionadas: {len(kept)}")
    print("Descartadas:", sorted(set(all_names) - set(kept)))

    # Gráficos: matrices de confusión y curvas ROC
    fig, axes = plt.subplots(1, len(fitted), figsize=(5.5 * len(fitted), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (n, (pred, _)) in zip(axes, preds.items()):
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Activo", "Churn"],
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(n)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "matrices_confusion.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for n, (_, proba) in preds.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=n, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("Curvas ROC (conjunto de prueba)")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "curvas_roc.png", dpi=130)
    plt.close(fig)

    # Artefactos
    joblib.dump(best, config.MODELS / "best_model.joblib")
    joblib.dump(fitted, config.MODELS / "all_models.joblib")
    X_test.assign(**{config.TARGET: y_test}).to_csv(config.DATA_PROCESSED / "test_set.csv", index=False)
    X_train.assign(**{config.TARGET: y_train}).to_csv(config.DATA_PROCESSED / "train_set.csv", index=False)
    json.dump({"best_model": best_name, "features_selected": kept, "features_all": all_names,
               "smote": use_smote},
              open(config.MODELS / "model_info.json", "w"), indent=2, ensure_ascii=False)
    print("\nArtefactos guardados en models/ y data/processed/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smote", action="store_true", help="usar SMOTE en lugar de class_weight")
    main(ap.parse_args().smote)
