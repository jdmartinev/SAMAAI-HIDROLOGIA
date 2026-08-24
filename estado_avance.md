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

![Figura 1](figs/fig1.png)
*Figura 1. Taxonomía de enfoques. El estudio cubrió las familias 1 y 2: modelos estadísticos clásicos y ML basado en árboles.*

El punto de partida fue instrumentar el problema correctamente. Se construyó una serie temporal común a partir de los datos del sensor de nivel y el pluviómetro de la cuenca, resolviendo problemas de frecuencia irregular, vacíos temporales y etiquetas de anomalía que, de haberse eliminado sin criterio, habrían descartado precisamente los eventos hidrológicos más relevantes. La resolución temporal adoptada fue de cinco minutos, con tres horizontes de pronóstico: una, dos y tres horas.

Un aspecto metodológico central fue el diseño para prevenir fuga de información. La división de los datos fue estrictamente cronológica, se aplicó purga temporal entre bloques y todos los modelos fueron configurados antes de acceder al conjunto de prueba. Esto garantiza que los resultados reflejen el desempeño real fuera de muestra y no una optimización retrospectiva.

La primera familia evaluada fue la de **modelos estadísticos**: persistencia, ARIMA y ARIMAX. La persistencia —asumir que el nivel no cambiará durante el horizonte— funcionó como un baseline exigente dada la alta autocorrelación natural de la serie, pero se deterioró notablemente durante crecientes. ARIMA mejoró sobre la persistencia, especialmente en términos de error cuadrático. ARIMAX incorporó la precipitación de forma causal, pero la mejora fue marginal, lo que sugiere que la relación lluvia–nivel no puede representarse adecuadamente mediante un modelo lineal.

La segunda familia fue la de **ML clásico basado en árboles**: Random Forest y XGBoost. Ambos modelos recibieron como entrada un conjunto de variables construidas manualmente: retardos del nivel, cambios recientes, acumulados de precipitación en distintas ventanas temporales y el instante actual. Esta familia mostró el mejor desempeño global de la fase, con mejoras sustanciales frente a la persistencia en los tres horizontes. Random Forest destacó especialmente durante las crecientes rápidas, mientras que XGBoost fue más consistente en error promedio.

Sin embargo, la evaluación reveló un hallazgo estructural importante: **las métricas globales ocultan el comportamiento durante los eventos críticos**. El error en condiciones de nivel elevado o ascenso rápido fue considerablemente mayor que el error promedio general, y todos los modelos mostraron una tendencia sistemática a subestimar la magnitud de los picos. Este desequilibrio entre el rendimiento cotidiano y el rendimiento durante crecientes es la principal motivación para continuar hacia las fases siguientes.

---

## En qué estamos: Fase 3 — Deep Learning puro

![Figura 3](figs/fig3.png)
*Figura 3. Ecosistema moderno de flood forecasting. El equipo está trabajando en el bloque C (Forecast Engine), familia Deep Learning.*

Las Fases 1 y 2 dejaron claro que los modelos tabulares, aunque precisos en condiciones normales, no capturan plenamente la dinámica temporal del sistema durante eventos extremos. La hipótesis de trabajo es que arquitecturas recurrentes —que procesan la serie como una secuencia real y no como un vector de features ingeniadas— pueden representar mejor esa dinámica.

En esta fase se están evaluando redes **LSTM** y **GRU**. A diferencia de los modelos de árbol, estas redes reciben directamente las secuencias temporales de nivel y precipitación, sin necesidad de construir manualmente los retardos y acumulados. El contexto histórico utilizado abarca aproximadamente tres horas de observaciones pasadas, lo que captura la dinámica reciente del sistema sin reducir excesivamente el número de muestras disponibles dado el patrón de vacíos en los datos.

Las líneas activas de exploración incluyen el ajuste de la función de pérdida para penalizar más los errores durante crecientes, la búsqueda de arquitecturas que mejoren la anticipación de picos, y el análisis de si la GRU —que en la fase anterior mostró señales interesantes en eventos extremos— mantiene esa ventaja cuando se entrena de forma más sistemática como red recurrente pura.

El criterio de evaluación no se limita al error promedio global. Se evalúa por separado el comportamiento durante crecientes rápidas, la capacidad de alertamiento ante niveles elevados y la predicción de la magnitud de los picos, siguiendo el mismo protocolo de evaluación multi-dimensión establecido en las fases anteriores.

---

## Qué viene: Fases 4 y 5

![Figura 2](figs/fig2.png)
*Figura 2. Evolución histórica. Las fases 4 y 5 corresponden a la incorporación de información meteorológica externa y al acoplamiento con modelos físicos.*

**Fase 4 — Extensión del lead time con pronóstico meteorológico**

El horizonte máximo alcanzable con solo datos observados está acotado por el tiempo de concentración de la cuenca: el tiempo que tarda el agua en llegar al sensor desde que comienza a llover. Para cuencas pequeñas y montañosas como las del área de estudio, ese tiempo puede ser inferior a tres horas, lo que limita estructuralmente la anticipación posible.

La vía para extender ese horizonte es incorporar **pronóstico numérico del tiempo (NWP)**: información sobre la precipitación esperada en las próximas horas, proveniente de modelos como ERA5-Land (disponible gratuitamente para entrenamiento histórico vía Copernicus CDS), ECMWF IFS o modelos regionales tipo WRF. Con esa información, el modelo puede anticipar una creciente antes de que la lluvia haya comenzado a caer sobre la cuenca.

**Fase 5 — Híbridos física + ML**

Si el equipo cuenta con o puede calibrar un modelo hidrológico conceptual (HEC-HMS, GR4J), el paso natural es usarlo como fuente de variables de estado —humedad de suelo simulada, escorrentía acumulada, déficit de almacenamiento— que complementen los datos observados. Este enfoque híbrido mejora la generalización a eventos fuera de distribución y agrega interpretabilidad física al sistema, dos aspectos críticos para el uso operacional en gestión del riesgo.

---

## Referencias del informe base

- Kratzert et al. (2018, 2019) — HESS.
- Nearing et al. (2024) — Nature 627. doi:10.1038/s41586-024-07145-1
- Li et al. (2024) — Scientific Reports. doi:10.1038/s41598-024-62127-7
- NeuralHydrology: https://neuralhydrology.github.io
