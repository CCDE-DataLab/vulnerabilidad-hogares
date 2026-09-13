# Dinámica y Vulnerabilidad de la Pobreza en Hogares Peruanos (2015-2019)

## 📌 Resumen del Proyecto
Este proyecto de investigación aplicada busca identificar los determinantes socioeconómicos que explican la transición de hogares peruanos desde una situación de "no pobreza" hacia la "pobreza". En lugar de un enfoque transversal tradicional, el proyecto explora la **dinámica de la pobreza** utilizando datos longitudinales para entender la vulnerabilidad ante choques económicos y la resiliencia de los hogares.

**Pregunta de investigación:** 
¿Qué características socioeconómicas, demográficas y habitacionales permiten anticipar que un hogar no pobre caerá en situación de pobreza el año siguiente?

## 📊 Fuente de Datos
Se utilizan exclusivamente los microdatos de la **Encuesta Nacional de Hogares (ENAHO) Panel 2015-2019** del INEI, específicamente los módulos:
*   **Módulo 100:** Características de la Vivienda y el Hogar.
*   **Módulo 200:** Características de los Miembros del Hogar.
*   **Módulo 500:** Empleo e Ingresos.
*   **Módulo 34 (Sumaria):** Variables calculadas de ingreso, gasto y pobreza.

## ⚙️ Metodología
1.  **Diseño Longitudinal (Pooled Panel):** Transformación de la base de datos de formato *Wide* a *Long* para construir un panel apilado de transiciones de 1 año (2015-2016, 2016-2017, etc.).
2.  **Ingeniería de Características (Feature Engineering):** Agregación de variables a nivel individual (Módulos 200 y 500) para construir indicadores a nivel de hogar (ej. Tasa de dependencia demográfica, características del jefe de hogar).
3.  **Variable Objetivo (Target):** Matriz de transición de 4 estados. El enfoque principal será el modelamiento de la transición `0 -> 1` (De No Pobre en $t$ a Pobre en $t+1$).
4.  **Modelamiento (Machine Learning):** Se evaluarán modelos interpretables (Regresión Logística) y basados en árboles (Random Forest / XGBoost) aislando el *data leakage* temporal.

## 🗂 Selección Inicial de Variables (Features)
**Vulnerabilidad Estructural (Módulo 100):**
*   Tipo de vivienda (`P101`) y Materiales predominantes.
*   Acceso a servicios básicos: Agua (`P110`), Electricidad (`P1121`).
*   Necesidades Básicas Insatisfechas (Módulo NBI).

**Demografía y Estructura del Hogar (Módulo 200 agregado):**
*   Ratio de Dependencia Demográfica (Menores y ancianos vs. Población en edad de trabajar).
*   Características del Jefe de Hogar (`P203 = 1`): Sexo, Edad, Estado Civil.

**Choques y Capacidad Generadora (Módulo 500 agregado - *En desarrollo*):**
*   Tasa de ocupación del hogar.
*   Dependencia de ingresos independientes vs. dependientes.