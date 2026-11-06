f"No se pudo procesar el archivo: {exc}")
    else:
        res = score(feats, model, cluster, umbral)
        n_desc = len(crudo) - len(res)
        m1, m2, m"""Dashboard ejecutivo: predicción de churn, perfil (cluster) y explicación SHAP.

Ejecutar desde la raíz del proyecto:  streamlit run app/app.py
(antes: python -m src.generate_data, src.train, src.clustering)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.data_prep import engineer_features
from src.explain import make_explainer, shap_for_class1, transform_for_clf
from src.predict import REQUIRED, prepare, score

st.set_page_config(page_title="DataCloud Churn", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(config.MODELS / "best_model.joblib")
    cluster = joblib.load(config.MODELS / "cluster_model.joblib")
    test = pd.read_csv(config.DATA_PROCESSED / "test_set.csv")
    Xt, _ = transform_for_clf(model, test[config.MODEL_INPUTS])
    explainer = None
    try:
        explainer = make_explainer(model.named_steps["clf"], Xt)
    except Exception:
        pass
    return model, cluster, explainer


try:
    model, cluster, explainer = load_artifacts()
except FileNotFoundError:
    st.error("Faltan los modelos. Ejecuta primero: python -m src.train y python -m src.clustering")
    st.stop()
except Exception as exc:  # p. ej. modelos guardados con otra versión de scikit-learn
    st.error(f"No se pudieron cargar los modelos ({type(exc).__name__}). Vuelve a generarlos en este "
             "entorno con: python -m src.train y python -m src.clustering")
    st.stop()

st.title("📉 Sistema predictivo y prescriptivo de abandono — DataCloud Solutions")
st.caption("Introduce los datos de un cliente corporativo para estimar su riesgo de cancelación en 30 días.")

with st.sidebar:
    st.header("Datos del cliente")
    edad = st.number_input("Edad del usuario principal", 18, 90, 40)
    antig = st.number_input("Antigüedad (meses)", 1, 240, 18)
    tickets = st.number_input("Tickets de soporte (último trimestre)", 0, 50, 2)
    uso = st.number_input("Uso diario promedio (min)", 0.0, 1000.0, 25.0, step=1.0)
    monto = st.number_input("Monto mensual (USD)", 1.0, 5000.0, 99.0, step=1.0)
    genero = st.selectbox("Género", config.GENERO_OPCIONES)
    region = st.selectbox("Región", config.REGION_OPCIONES)
    umbral = st.slider("Umbral de decisión (churn si P ≥ umbral)", 0.1, 0.9, 0.5, 0.05)

row = pd.DataFrame([{
    "Edad": edad, "Antiguedad_Meses": antig, "Soporte_Tickets": tickets,
    "Uso_Diario_Min": uso, "Monto_Mensual": monto, "Genero": genero, "Region": region,
}])
row = engineer_features(row)[config.MODEL_INPUTS]

proba = float(model.predict_proba(row)[0, 1])
c1, c2, c3 = st.columns(3)
c1.metric("Puntaje de riesgo de churn (30 días)", f"{proba:.1%}")
c2.metric("Decisión", "🔴 Riesgo ALTO" if proba >= umbral else "🟢 Riesgo bajo")
st.caption("El puntaje sirve para ordenar clientes por riesgo; el modelo usa class_weight='balanced', por lo que no es una probabilidad calibrada (ver informe, sección 3).")

Zc = cluster["pre"].transform(row[cluster["cols"]])
k = int(cluster["kmeans"].predict(Zc)[0])
c3.metric("Cluster", f"{k}")
st.info(f"**Perfil del cliente:** {cluster['names'][k]}")

st.subheader("Acción de retención sugerida")
triggered = False
if row["Atencion_Requerida"].iloc[0] == 1:
    st.write("• Alto volumen de tickets: asignar un *customer success manager* y revisar incidencias abiertas.")
    triggered = True
if uso < 15:
    st.write("• Bajo uso: campaña de *onboarding* / capacitación sobre funcionalidades clave.")
    triggered = True
if proba >= umbral and antig < 12:
    st.write("• Cliente reciente en riesgo: llamada proactiva y revisión del plan contratado.")
    triggered = True
if proba >= umbral and not triggered:
    st.write("• Alerta de riesgo de cancelación: contacto preventivo del equipo de retención para evaluar satisfacción y propuesta de renovación.")
if proba < umbral:
    st.write("• Sin acción urgente: monitorear y considerar *upselling*.")

st.subheader("¿Por qué este resultado? (SHAP)")
if explainer is None:
    st.warning("Instala `shap` (pip install shap) para ver la explicación.")
else:
    import shap
    Xr, names = transform_for_clf(model, row)
    sv, base = shap_for_class1(explainer, Xr)
    exp = shap.Explanation(values=sv[0], base_values=base, data=Xr[0], feature_names=names)
    fig = plt.figure()
    shap.plots.waterfall(exp, show=False)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Barras rojas empujan hacia churn; azules, hacia permanencia. Los valores mostrados "
               "son las variables ya transformadas y estandarizadas.")

with st.expander("Perfiles de clientes (segmentación K-Means)"):
    st.dataframe(cluster["profiles"].round(2))

# ----------------------------------------------------------------------------
# Predicción por lotes: subir un CSV con varios clientes
# ----------------------------------------------------------------------------
st.divider()
st.header("📂 Predicción por lotes (archivo CSV)")
st.caption("Sube un CSV con una fila por cliente. Columnas obligatorias: " + ", ".join(REQUIRED)
           + ". Opcional: ID_Cliente. Se aplica la misma limpieza del entrenamiento "
           "(formatos de antigüedad, edades imposibles, faltantes, etc.). Usa el umbral de la barra lateral.")

ejemplo = config.ROOT / "data" / "ejemplo" / "clientes_nuevos_ejemplo.csv"
if ejemplo.exists():
    st.download_button("⬇️ Descargar archivo de ejemplo", ejemplo.read_bytes(),
                       file_name=ejemplo.name, mime="text/csv")

archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if archivo is not None:
    try:
        crudo = pd.read_csv(archivo)
        feats = prepare(crudo)
    except Exception as exc:
        st.error(3 = st.columns(3)
        m1.metric("Clientes procesados", f"{len(res)}")
        m2.metric("En riesgo ALTO", f"{int((res['Decision'] == 'Riesgo ALTO').sum())}")
        m3.metric("% en riesgo ALTO", f"{(res['Decision'] == 'Riesgo ALTO').mean():.0%}")
        if n_desc:
            st.warning(f"Se descartaron {n_desc} fila(s) por ID nulo o duplicado.")
        st.dataframe(res)
        st.download_button("⬇️ Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8"),
                           file_name="resultados_churn.csv", mime="text/csv")

        if explainer is not None:
            import shap
            elegido = st.selectbox("Explicar el riesgo de un cliente del archivo (SHAP)", res["ID_Cliente"].tolist())
            fila = feats.loc[feats["ID_Cliente"] == elegido, config.MODEL_INPUTS].head(1)
            Xb, nombres = transform_for_clf(model, fila)
            svb, baseb = shap_for_class1(explainer, Xb)
            expb = shap.Explanation(values=svb[0], base_values=baseb, data=Xb[0], feature_names=nombres)
            figb = plt.figure()
            shap.plots.waterfall(expb, show=False)
            st.pyplot(figb)
            plt.close(figb)
