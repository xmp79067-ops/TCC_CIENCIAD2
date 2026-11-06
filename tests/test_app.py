"""Prueba de la app Streamlit con el probador oficial (streamlit.testing.v1.AppTest).

Se omite si streamlit o los modelos no están disponibles.
Ejecutar desde la raíz:  python -m unittest tests.test_app -v   (o el discover completo)
"""
import importlib.util
import sys
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import config  # noqa: E402

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None
HAS_MODELS = (config.MODELS / "best_model.joblib").exists() and (config.MODELS / "cluster_model.joblib").exists()


def _widget(collection, prefijo):
    return next(w for w in collection if w.label.startswith(prefijo))


def _puntaje(at) -> float:
    m = next(m for m in at.metric if m.label.startswith("Puntaje"))
    return float(m.value.strip("%")) / 100


@unittest.skipUnless(HAS_STREAMLIT and HAS_MODELS, "requiere streamlit y modelos entrenados")
class TestApp(unittest.TestCase):
    def _run(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()

    def test_carga_sin_errores_y_muestra_resultados(self):
        at = self._run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        etiquetas = [m.label for m in at.metric]
        self.assertTrue(any(e.startswith("Puntaje") for e in etiquetas))
        self.assertIn("Decisión", etiquetas)
        self.assertIn("Cluster", etiquetas)
        self.assertTrue(0 <= _puntaje(at) <= 1)

    def test_cliente_riesgoso_tiene_mas_puntaje_que_cliente_leal(self):
        at = self._run()
        sb = at.sidebar.number_input
        _widget(sb, "Tickets").set_value(9)
        _widget(sb, "Uso diario").set_value(5.0)
        _widget(sb, "Monto").set_value(49.0)
        at.run()
        self.assertFalse(at.exception)
        riesgo = _puntaje(at)

        _widget(sb, "Tickets").set_value(0)
        _widget(sb, "Uso diario").set_value(150.0)
        _widget(sb, "Monto").set_value(299.0)
        at.run()
        self.assertFalse(at.exception)
        self.assertGreater(riesgo, _puntaje(at) + 0.3)

    def test_umbral_cambia_la_decision(self):
        at = self._run()
        _widget(at.sidebar.slider, "Umbral").set_value(0.1)
        at.run()
        decision = next(m for m in at.metric if m.label == "Decisión")
        self.assertIn("ALTO", decision.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
