# Registro de pruebas

Entorno: Python 3, scikit-learn 1.9, pandas 3.0. Datos: `data/raw/clientes.csv` (10,000 filas, sintético).
Semilla fija `RANDOM_STATE = 42`.

## 1. Reproducibilidad
| Verificación | Resultado |
|---|---|
| Reejecutar `train.py`, `clustering.py`, `fairness.py` desde cero y comparar con los resultados guardados | Métricas, perfiles de cluster y métricas de equidad **idénticos** (solo difieren los saltos de línea CRLF/LF entre Windows y Linux) |
| Mismo split y mismas variables seleccionadas | Sí: 7 de 16 variables (se descartan Genero, Region y Soporte_Faltante) |

## 2. Ejecución de extremo a extremo
| Script | Estado |
|---|---|
| `src.eda`, `src.train`, `src.clustering`, `src.fairness` | Ejecutan sin errores |
| `src.explain` (SHAP) | Ejecutado por el autor; sus figuras se contrastaron con el informe (ver sección 4) |
| `app/app.py` | Probada en tres escenarios (bajo riesgo, alto riesgo, alto riesgo sin banderas) simulando la interfaz de Streamlit; devuelve puntaje, decisión y cluster en todos. **No se ejecutó con Streamlit real** (no disponible en el entorno de desarrollo): hacerlo con `streamlit run app/app.py` y `python -m unittest tests.test_app -v` |
| `train.py --smote` | **No probado** (requiere `imbalanced-learn`); la lógica de `transform_for_clf` ya omite el paso `smote` |

## 3. Pruebas automáticas (`tests/test_pipeline.py`)
25 pruebas en `tests/` (`test_pipeline.py`: 22; `test_app.py`: 3). En el entorno de desarrollo pasaron las 21 ejecutables; se omiten 4 que requieren `shap` y `streamlit` (aditividad de SHAP y las 3 pruebas de la app con `AppTest`), que deben correrse en el equipo del autor. Cubren: formatos de antigüedad, reglas de limpieza
(duplicados, IDs nulos, edades fuera de rango, montos en centavos), `Costo_Por_Minuto` y `Atencion_Requerida`,
preprocesador sin NaN/inf, split estratificado, probabilidades en [0, 1], coherencia económica
(más tickets y menos uso ⇒ más riesgo), AUC > 0.7, asignación de cluster, métricas de equidad,
aditividad de SHAP (cuando `shap` está disponible) y **consistencia entre el informe y los resultados**.

## 4. Verificación de las cifras de SHAP del informe
| Dato del informe | Verificación |
|---|---|
| Media \|SHAP\| (0.0917, 0.0870, 0.0710, …) y orden del top 3 | Coincide con `shap_global_barras.png` |
| Cliente alto riesgo (índice 1154): P = 93.5%, perfil, contribuciones +0.157/+0.116/+0.080/+0.062 | Coincide con el modelo reentrenado y con `shap_individual_alto_riesgo.png` |
| Cliente bajo riesgo (índice 1269): P = 6.6%, perfil | Coincide con el modelo reentrenado |

## 5. Hallazgos y correcciones aplicadas al informe
1. **Corregido:** la interpretación de `Monto_Mensual` decía que los costos altos aceleran la deserción; el beeswarm y los datos (churn de 33% en el cuartil de menor tarifa vs. 6% en el de mayor) muestran lo contrario.
2. **Agregado:** el puntaje no está calibrado (media predicha 0.41 vs. 0.20 real; valor base de SHAP 0.5).
3. **Precisado:** contraste Gini vs. SHAP (qué mide cada uno) y escala de los valores SHAP.
4. **Agregado:** valores numéricos de las correlaciones del EDA.

## 6. Cambios finales
1. `app/app.py`: la métrica se renombró a "Puntaje de riesgo de churn" con aviso de calibración, y la carga de modelos muestra un mensaje claro si falla (p. ej. por versión de scikit-learn).
2. `reports/informe_tecnico.pdf`: 8 páginas (límite del enunciado), con fórmulas y 6 figuras; se genera desde `informe_tecnico.md`.
3. `requirements.txt`: `xgboost` comentado (no se usa en el código).
