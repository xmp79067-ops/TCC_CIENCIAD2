# Sistema predictivo y prescriptivo de abandono de clientes (DataCloud Solutions)

Proyecto reproducible de Ciencia de Datos II: EDA, calidad de datos, modelo supervisado,
segmentación, XAI (SHAP), análisis de equidad y app interactiva en Streamlit.

## Estructura
```
src/config.py          rutas, semilla y columnas
src/generate_data.py   dataset sintético (solo si no tienes el CSV real)
src/data_prep.py       limpieza, feature engineering y preprocesador
src/eda.py             Fase 1: EDA e hipótesis uso vs. churn
src/train.py           Fase 3: LogReg vs. Random Forest, GridSearchCV, métricas
src/clustering.py      Fase 4: K-Means, codo, silueta y perfiles
src/fairness.py        Fase 5: equidad por Género y Región
src/explain.py         Fase 5: SHAP global e individual
app/app.py             Fase 5: demo Streamlit
reports/informe_tecnico.md
reports/registro_de_pruebas.md
tests/test_pipeline.py  pruebas automáticas (unittest)
```

## Ejecución
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Si tienes el CSV real, colócalo en data/raw/clientes.csv y omite este paso:
python -m src.generate_data

python -m src.eda
python -m src.train            # o: python -m src.train --smote
python -m src.clustering
python -m src.fairness
python -m src.explain
streamlit run app/app.py
```
Todo se ejecuta desde la raíz del proyecto. Semilla fija (`RANDOM_STATE = 42`).

## Mejoras y Correcciones Aplicadas

- **Calidad de datos (`src/data_prep.py`):**
  - Corrección en la expresión regular de `parse_antiguedad`: ahora captura signos negativos (`r"(-?\d+(?:\.\d+)?)"`), permitiendo que valores inválidos como `"-12 meses"` se detecten y limpien según la regla de rango (`Antiguedad_Meses < 0`).
  - Validación defensiva de columnas en `clean_data` para evitar fallos por `KeyError` al operar sobre subconjuntos de datos o registros individuales sin `ID_Cliente`.
- **Compatibilidad con SMOTE y XAI (`src/explain.py` y `src/train.py`):**
  - `transform_for_clf` adaptado para filtrar pasos de sobremuestreo (`smote`) y clasificador (`clf`), encadenando únicamente los transformadores compatibles con `.transform()` y propagando nombres de variables de forma robusta con `.get_feature_names_out()`. Esto permite alternar entre `class_weight='balanced'` y `--smote` sin provocar excepciones `AttributeError: This 'Pipeline' has no attribute 'transform'`.
  - Corrección en la obtención de nombres de variables seleccionadas en `train.py` usando el paso explícito `select`.
- **Estabilidad y memoria en Streamlit (`app/app.py`):**
  - Cierre explícito de figuras Matplotlib (`plt.close(fig)`) tras el renderizado de gráficos SHAP para evitar fugas de memoria en sesiones prolongadas.
  - Recomendación prescriptiva completa para clientes de alto riesgo que no activan banderas extremas individuales.
- **Informe técnico (`informe_tecnico.md` y `reports/informe_tecnico.md`):**
  - Incorporación de los resultados numéricos reales de SHAP (Top 3: `Uso_Diario_Min`, `Soporte_Tickets`, `Monto_Mensual`), contraste con importancia Gini y análisis de los dos perfiles individuales evaluados en el conjunto de prueba.

## Pruebas
```bash
python -m unittest discover -s tests -v
```
Ejecutar después de `train.py`, `clustering.py` y `fairness.py`. Verifican la limpieza de datos, las variables
creadas, el preprocesamiento, los artefactos (modelo, cluster), las métricas de equidad y que **las cifras del
informe coincidan con los resultados guardados**. La prueba de SHAP se omite si `shap` no está instalado.
El detalle de lo verificado está en `reports/registro_de_pruebas.md`.

## Ejecutar la app localmente
```bash
pip install -r requirements.txt
python -m src.train && python -m src.clustering     # genera models/ (≈ 2 min); omitir si ya existen
streamlit run app/app.py                            # abre http://localhost:8501
```
Prueba rápida en la interfaz: con los valores por defecto el puntaje ronda 44% (decisión "riesgo bajo" con umbral 0.5);
con 8 tickets, 6 min de uso, US$ 49 y 5 meses de antigüedad sube a ≈ 92% (cluster 2, "alto soporte").
Prueba automática de la app: `python -m unittest tests.test_app -v` (requiere `streamlit`).

## Despliegue en Streamlit Community Cloud
1. **Fija las versiones** con las que entrenaste (los `.joblib` son *pickles* y exigen la misma versión de scikit-learn):
   `pip show scikit-learn shap streamlit` y escribe esas versiones exactas en `requirements.txt` (p. ej. `scikit-learn==X.Y.Z`).
2. Sube el proyecto a GitHub. La app necesita estos archivos en el repositorio:
   `models/best_model.joblib`, `models/cluster_model.joblib` y `data/processed/test_set.csv`
   (el `.gitignore` ya los conserva; excluye `all_models.joblib`, `data/raw/` y `train_set.csv`).
   ```bash
   git init && git add . && git commit -m "Proyecto churn DataCloud"
   git branch -M main && git remote add origin https://github.com/<usuario>/<repo>.git && git push -u origin main
   ```
3. En https://share.streamlit.io → *Create app* → elige el repositorio, rama `main`, archivo principal `app/app.py`.
   En *Advanced settings* selecciona una versión de Python compatible con tus librerías.
4. Si la app muestra "No se pudieron cargar los modelos", reentrena en el mismo entorno
   (`python -m src.train && python -m src.clustering`) y vuelve a subir `models/`.

## Nota sobre el puntaje de riesgo
El modelo se entrena con `class_weight='balanced'`, por lo que `predict_proba` **no es una probabilidad calibrada**
(media predicha 0.41 vs. 20% real). Úsalo para ordenar clientes por riesgo, no como probabilidad literal
(ver sección 3 del informe).
