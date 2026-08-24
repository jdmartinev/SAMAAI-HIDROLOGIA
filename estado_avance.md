# Estado de avance — SAMAAI Hidrología
## Sistema de alerta temprana para crecientes · Quebrada La Oca

---

## Dónde estamos en la hoja de ruta

![Figura 5](figs/fig5.png)
*Figura 5. Hoja de ruta de implementación. El equipo completó las Fases 1 y 2, y está actualmente ejecutando la Fase 3.*

| Fase | Descripción | Estado |
| :--- | :--- | :---: |
| **1** — Línea de base | Persistencia, ARIMA, ARIMAX | ✅ Completada |
| **2** — ML clásico | Random Forest, XGBoost | ✅ Completada |
| **3** — Deep Learning puro | LSTM, GRU | 🔄 En ejecución |
| **4** — Extensión del lead time | NWP / ERA5 como forzante | ⏳ Pendiente |
| **5** — Híbridos física + ML | GR4J-DL, surrogates | ⏳ Pendiente |

---

## Lo que se hizo: Fases 1 y 2

El estudio de la Quebrada La Oca (noviembre 2025 – junio 2026) evaluó siete familias de modelos sobre el sensor SN_1007 y el pluviómetro SP_108, con horizontes de pronóstico de 1, 2 y 3 horas a resolución de 5 minutos.

![Figura 1](figs/fig1.png)
*Figura 1. Taxonomía de enfoques. El estudio cubrió las familias 1 y 2 (ML clásico y DL secuencial).*

### Resultados globales (conjunto TEST fuera de muestra)

**Horizonte 1 h**

| Modelo | MAE (cm) | NSE | KGE |
| :--- | :---: | :---: | :---: |
| XGBoost | **1,354** | **0,983** | 0,971 |
| Random Forest | 1,370 | 0,982 | 0,971 |
| LSTM | 1,489 | 0,975 | **0,985** |
| GRU | 1,592 | 0,977 | 0,975 |
| ARIMA | 2,031 | 0,952 | 0,966 |
| Persistencia | 2,107 | 0,925 | 0,963 |

**Horizonte 3 h**

| Modelo | MAE (cm) | NSE | KGE |
| :--- | :---: | :---: | :---: |
| XGBoost | **3,465** | 0,832 | 0,830 |
| Random Forest | 3,491 | **0,834** | 0,848 |
| GRU | 3,746 | 0,829 | **0,887** |
| LSTM | 4,066 | 0,817 | 0,880 |
| ARIMA | 5,330 | 0,653 | 0,828 |
| Persistencia | 5,444 | 0,593 | 0,796 |

La mejora frente a persistencia fue del **36–39 %** en MAE para los mejores modelos.

### Hallazgo central

Las métricas globales ocultan el comportamiento durante eventos críticos. En condiciones de creciente ≥ 20 cm, el mejor modelo (Random Forest) alcanzó un MAE de **17 cm a 1 h** y **37 cm a 3 h**, con subestimación en más del 85 % de los casos.

| Dimensión evaluada | Mejor modelo | Observación |
| :--- | :--- | :--- |
| Error global | XGBoost / Random Forest | MAE 1–3,5 cm |
| Crecientes ≥ 20 cm | Random Forest | MAE 17–37 cm; subestimación dominante |
| Alertamiento ROJO (CSI) | **GRU** | CSI 0,950 a 1 h; 0,630 a 3 h |
| Picos severos (≥ 310 cm) | GRU a 3 h | MAE 32 cm; más cercano al pico real en los dos eventos ROJO |

> **Implicación directa para la siguiente fase:** la GRU captura estructuras temporales que XGBoost y RF no representan completamente, especialmente en los momentos que más importan operacionalmente. Esto motiva la profundización en arquitecturas recurrentes.

---

## En qué estamos: Fase 3 — Deep Learning puro

![Figura 3](figs/fig3.png)
*Figura 3. Ecosistema moderno de flood forecasting. El equipo está trabajando en el bloque C (Forecast Engine), familia Deep Learning.*

Las líneas activas de trabajo son:

- **Arquitecturas recurrentes:** explorar variantes de GRU con atención, Encoder-Decoder para predicción multi-step explícita, y ConvLSTM si se dispone de datos espaciales.
- **Función de pérdida orientada a extremos:** reemplazar MSE estándar por funciones que penalicen más la subestimación de crecientes (pérdida asimétrica, pérdida ponderada por régimen hidrológico).
- **Ventana de contexto:** el estudio usó 3 h (37 pasos a 5 min). Explorar ventanas de 6–12 h puede mejorar la captura de la dinámica antecedente.
- **Información de precipitación:** el bajo beneficio de ARIMAX no implica que la lluvia sea irrelevante — sí lo es de forma no lineal. El LSTM/GRU puede capturarlo si se construyen secuencias correctas de precipitación acumulada.

### Qué NO cambiar respecto al diseño anterior

- División cronológica estricta sin shuffle.
- Purga temporal entre bloques de entrenamiento / validación / TEST.
- Evaluación separada por régimen (global, crecientes, alertamiento, picos) — no solo MAE global.
- Umbrales de alerta (260 / 310 / 350 cm) como referencia experimental hasta validación oficial con el equipo de hidrología.

---

## Qué viene: Fases 4 y 5

![Figura 2](figs/fig2.png)
*Figura 2. Evolución histórica. Las fases 4 y 5 corresponden a los nodos 6–8 de la línea de tiempo.*

**Fase 4 — Extensión del lead time con NWP**

El tiempo de concentración de La Oca limita el lead time útil a ~1–3 h con solo datos observados. Para extender ese horizonte es necesario incorporar pronóstico de precipitación:

- **ERA5-Land** (reanálisis, gratuito vía Copernicus CDS) — para entrenamiento histórico.
- **ECMWF IFS / GFS / WRF regional** — para operación en tiempo real.

Esto permitiría evaluar si es posible anticipar una creciente 6–12 h antes de que ocurra.

**Fase 5 — Híbridos física + ML**

Si el equipo cuenta con o puede calibrar un modelo conceptual (HEC-HMS, GR4J), el patrón más directo es usarlo como generador de variables de estado (humedad de suelo, escorrentía acumulada) que alimenten al DL. Esto mejora la generalización a eventos fuera de distribución y agrega interpretabilidad.

---

## Referencias del informe base

- García Surianu, S. (2026). *Predicción del nivel hidrométrico de la Quebrada La Oca mediante modelos estadísticos, aprendizaje automático y redes neuronales*. Periodo nov 2025 – jun 2026.
- Kratzert et al. (2018, 2019) — HESS.
- Nearing et al. (2024) — Nature 627. doi:10.1038/s41586-024-07145-1
- Li et al. (2024) — Scientific Reports. doi:10.1038/s41598-024-62127-7
- NeuralHydrology: https://neuralhydrology.github.io
