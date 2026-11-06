"""Pruebas automáticas del proyecto (unittest; también las detecta pytest).

Ejecutar desde la raíz del proyecto, DESPUÉS de correr train.py, clustering.py y fairness.py:

    python -m unittest discover -s tests -v

Las pruebas que dependen de librerías opcionales (shap) o de artefactos aún no generados
se omiten con un mensaje en lugar de fallar.
"""
import importlib.util
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import config  # noqa: E402
from src.data_prep import (build_preprocessor, clean_data, engineer_features,  # noqa: E402
                           parse_antiguedad)
from src.fairness import group_metrics, summarize  # noqa: E402

HAS_MODELS = (config.MODELS / "best_model.joblib").exists() and (config.MODELS / "cluster_model.joblib").exists()
HAS_TEST_SET = (config.DATA_PROCESSED / "test_set.csv").exists()
HAS_SHAP = importlib.util.find_spec("shap") is not None


class TestLimpieza(unittest.TestCase):
    def test_parse_antiguedad_formatos(self):
        casos = {"12": 12, "12 meses": 12, "12m": 12, "2.0 años": 24, "1,5 años": 18, "7": 7}
        for entrada, esperado in casos.items():
            self.assertAlmostEqual(parse_antiguedad(entrada), esperado, msg=entrada)
        self.assertTrue(np.isnan(parse_antiguedad(np.nan)))
        self.assertTrue(np.isnan(parse_antiguedad("sin dato")))

    def test_clean_data_reglas(self):
        df = pd.DataFrame({
            "ID_Cliente": ["A", "A", None, "B", "C"],
            "Edad": [30, 30, 40, 150, -5],
            "Antiguedad_Meses": ["12", "12", "6 meses", "2 años", "-12 meses"],
            "Soporte_Tickets": [1, 1, 2, np.nan, 3],
            "Uso_Diario_Min": [10, 10, 20, 30, -1],
            "Monto_Mensual": [99, 99, 49, 9900, 0],
            "Genero": ["M", "M", "F", None, "Otro"],
            "Region": ["Norte"] * 5,
            "Churn": [0, 0, 1, 0, 1],
        })
        out = clean_data(df)
        self.assertEqual(list(out["ID_Cliente"]), ["A", "B", "C"])  # sin nulos ni duplicados
        b = out.set_index("ID_Cliente")
        self.assertTrue(np.isnan(b.loc["B", "Edad"]) and np.isnan(b.loc["C", "Edad"]))  # outliers -> NaN
        self.assertEqual(b.loc["B", "Antiguedad_Meses"], 24)
        self.assertTrue(np.isnan(b.loc["C", "Antiguedad_Meses"]))  # negativa -> NaN
        self.assertEqual(b.loc["B", "Monto_Mensual"], 99)  # centavos -> USD
        self.assertTrue(np.isnan(b.loc["C", "Monto_Mensual"]))  # no positivo -> NaN
        self.assertTrue(np.isnan(b.loc["C", "Uso_Diario_Min"]))  # negativo -> NaN
        self.assertEqual(b.loc["B", "Genero"], "Desconocido")

    def test_clean_data_columnas_faltantes(self):
        """Debe tolerar subconjuntos de columnas (p. ej. un registro individual sin ID)."""
        out = clean_data(pd.DataFrame({"Edad": [30, 200], "Uso_Diario_Min": [5, 6]}))
        self.assertEqual(len(out), 2)


class TestFeatures(unittest.TestCase):
    def test_costo_por_minuto(self):
        df = engineer_features(pd.DataFrame({"Monto_Mensual": [99.0], "Uso_Diario_Min": [30.0],
                                             "Soporte_Tickets": [1.0]}))
        self.assertAlmostEqual(df.loc[0, "Costo_Por_Minuto"], 99 / (30 * 30))

    def test_costo_por_minuto_uso_cero_no_es_infinito(self):
        df = engineer_features(pd.DataFrame({"Monto_Mensual": [99.0], "Uso_Diario_Min": [0.0],
                                             "Soporte_Tickets": [1.0]}))
        self.assertTrue(np.isfinite(df.loc[0, "Costo_Por_Minuto"]))

    def test_atencion_requerida_umbral(self):
        df = engineer_features(pd.DataFrame({"Monto_Mensual": [1.0] * 4, "Uso_Diario_Min": [1.0] * 4,
                                             "Soporte_Tickets": [5, 6, 0, np.nan]}))
        self.assertEqual(list(df["Atencion_Requerida"]), [0, 1, 0, 0])  # > 5 estricto; NaN -> 0
        self.assertEqual(list(df["Soporte_Faltante"]), [0, 0, 0, 1])

    def test_preprocesador_sin_nan_ni_inf(self):
        from src.data_prep import load_clean
        if not config.DATA_RAW.exists():
            self.skipTest("falta data/raw/clientes.csv")
        df = load_clean().head(500)
        Z = build_preprocessor().fit_transform(df[config.MODEL_INPUTS])
        self.assertTrue(np.isfinite(np.asarray(Z, dtype=float)).all())
        self.assertEqual(Z.shape[0], 500)


@unittest.skipUnless(HAS_MODELS and HAS_TEST_SET, "ejecuta primero train.py y clustering.py")
class TestArtefactos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import joblib
        cls.model = joblib.load(config.MODELS / "best_model.joblib")
        cls.cluster = joblib.load(config.MODELS / "cluster_model.joblib")
        cls.test = pd.read_csv(config.DATA_PROCESSED / "test_set.csv")
        cls.train = pd.read_csv(config.DATA_PROCESSED / "train_set.csv")

    def _cliente(self, **kw):
        base = {"Edad": 40, "Antiguedad_Meses": 18, "Soporte_Tickets": 2, "Uso_Diario_Min": 25.0,
                "Monto_Mensual": 99.0, "Genero": "F", "Region": "Norte"}
        base.update(kw)
        return engineer_features(pd.DataFrame([base]))[config.MODEL_INPUTS]

    def test_split_estratificado(self):
        diff = abs(self.train[config.TARGET].mean() - self.test[config.TARGET].mean())
        self.assertLess(diff, 0.01)
        self.assertAlmostEqual(len(self.test) / (len(self.test) + len(self.train)), 0.20, places=2)

    def test_probabilidades_validas(self):
        p = self.model.predict_proba(self.test[config.MODEL_INPUTS])[:, 1]
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_sentido_economico(self):
        """Más tickets y menos uso deben elevar el riesgo (coherente con el EDA)."""
        riesgo = self.model.predict_proba(self._cliente(Soporte_Tickets=9, Uso_Diario_Min=5, Monto_Mensual=49))[0, 1]
        leal = self.model.predict_proba(self._cliente(Soporte_Tickets=0, Uso_Diario_Min=150, Monto_Mensual=299))[0, 1]
        self.assertGreater(riesgo, leal + 0.3)

    def test_modelo_supera_al_azar(self):
        from sklearn.metrics import roc_auc_score
        p = self.model.predict_proba(self.test[config.MODEL_INPUTS])[:, 1]
        self.assertGreater(roc_auc_score(self.test[config.TARGET], p), 0.7)

    def test_cluster_asigna_etiqueta_valida(self):
        row = self._cliente()
        Z = self.cluster["pre"].transform(row[self.cluster["cols"]])
        k = int(self.cluster["kmeans"].predict(Z)[0])
        self.assertIn(k, self.cluster["names"])

    def test_transformacion_coincide_con_nombres(self):
        from src.explain import transform_for_clf
        Xt, names = transform_for_clf(self.model, self.test[config.MODEL_INPUTS].head(20))
        self.assertEqual(Xt.shape[1], len(names))
        self.assertTrue(np.allclose(Xt, self.model[:-1].transform(self.test[config.MODEL_INPUTS].head(20))))

    def test_aviso_calibracion(self):
        """Documenta un hecho: con class_weight='balanced' las probabilidades NO están calibradas."""
        p = self.model.predict_proba(self.test[config.MODEL_INPUTS])[:, 1]
        real = self.test[config.TARGET].mean()
        if "class_weight" in str(self.model.named_steps["clf"].get_params()) and p.mean() > real + 0.1:
            print(f"\n  [info] P media predicha = {p.mean():.2f} vs. churn real = {real:.2f} "
                  "(scores inflados; ver informe, sección de limitaciones)")


@unittest.skipUnless(HAS_MODELS and HAS_SHAP, "requiere shap y modelos entrenados")
class TestShap(unittest.TestCase):
    def test_aditividad(self):
        import joblib
        from src.explain import make_explainer, shap_for_class1, transform_for_clf
        model = joblib.load(config.MODELS / "best_model.joblib")
        X = pd.read_csv(config.DATA_PROCESSED / "test_set.csv")[config.MODEL_INPUTS].head(50)
        Xt, names = transform_for_clf(model, X)
        sv, base = shap_for_class1(make_explainer(model.named_steps["clf"], Xt), Xt)
        self.assertEqual(sv.shape, (50, len(names)))
        pred = model.named_steps["clf"].predict_proba(Xt)[:, 1]
        np.testing.assert_allclose(base + sv.sum(axis=1), pred, atol=1e-3)  # f(x) = phi0 + sum(phi_j)


class TestEquidad(unittest.TestCase):
    def test_metricas_por_grupo(self):
        y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        pred = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        g = pd.Series(["a"] * 4 + ["b"] * 4)
        t = group_metrics(y, pred, g)
        self.assertAlmostEqual(t.loc["a", "TPR_recall"], 0.5)
        self.assertAlmostEqual(t.loc["a", "FPR"], 0.5)
        self.assertAlmostEqual(t.loc["b", "tasa_seleccion"], 0.5)

    def test_resumen_regla_80(self):
        t = pd.DataFrame({"n": [100, 100], "tasa_seleccion": [0.4, 0.2], "TPR_recall": [0.8, 0.7], "FPR": [0.2, 0.1]},
                         index=["x", "y"])
        s = summarize(t)
        self.assertAlmostEqual(s["ratio_impacto_dispar"], 0.5)
        self.assertAlmostEqual(s["dif_paridad_demografica"], 0.2)

    def test_grupos_pequenos_se_excluyen(self):
        t = pd.DataFrame({"n": [100, 5], "tasa_seleccion": [0.4, 0.9], "TPR_recall": [0.8, 1.0], "FPR": [0.2, 0.0]},
                         index=["x", "y"])
        self.assertAlmostEqual(summarize(t)["dif_paridad_demografica"], 0.0)


@unittest.skipUnless((config.DATA_PROCESSED / "metricas_modelos.csv").exists()
                     and (ROOT / "reports" / "informe_tecnico.md").exists(),
                     "faltan resultados o informe")
class TestConsistenciaInforme(unittest.TestCase):
    """Verifica que las cifras escritas en el informe coincidan con los resultados guardados."""

    @classmethod
    def setUpClass(cls):
        cls.texto = (ROOT / "reports" / "informe_tecnico.md").read_text(encoding="utf-8")

    def test_metricas_de_modelos(self):
        m = pd.read_csv(config.DATA_PROCESSED / "metricas_modelos.csv", index_col=0)
        for modelo, fila in m.iterrows():
            for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cv_f1"]:
                self.assertIn(f"{fila[col]:.3f}", self.texto, f"{modelo}.{col}={fila[col]:.3f} no aparece en el informe")

    def test_equidad(self):
        f = pd.read_csv(config.DATA_PROCESSED / "fairness_resumen.csv", index_col=0)
        for attr in f.index:
            for col in ["dif_paridad_demografica", "ratio_impacto_dispar", "dif_oportunidad_igual_TPR", "dif_FPR"]:
                self.assertIn(f"{f.loc[attr, col]:.3f}", self.texto, f"{attr}.{col}")

    def test_perfiles_de_clusters(self):
        p = pd.read_csv(config.DATA_PROCESSED / "perfiles_clusters.csv", index_col=0)
        for k, fila in p.iterrows():
            self.assertIn(f"{fila['Churn'] * 100:.0f}%", self.texto, f"churn del cluster {k}")
            self.assertIn(f"{fila['Tamaño_%']:.1f}%", self.texto, f"tamaño del cluster {k}")

    def test_no_hay_pendientes(self):
        self.assertNotIn("⚠️ **Pendiente", self.texto)
        self.assertNotIn("TODO", self.texto)


if __name__ == "__main__":
    unittest.main(verbosity=2)
