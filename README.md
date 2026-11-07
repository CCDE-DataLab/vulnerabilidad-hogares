# Vulnerabilidad y Dinámica Intertemporal de la Pobreza en Hogares Peruanos (ENAHO Panel 2015–2019)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: INEI ENAHO Panel](https://img.shields.io/badge/data-INEI%20ENAHO%20Panel-green.svg)](https://proyectos.inei.gob.pe/microdatos/)

Proyecto de investigación aplicada del **Círculo de Ciencia de Datos y Econometría (CCDE)** enfocado en modelar y anticipar la **vulnerabilidad socioeconómica** y las transiciones interanuales hacia y desde la pobreza en los hogares peruanos. En lugar del enfoque estático tradicional de corte transversal, este estudio modela la **dinámica intertemporal de la pobreza** aprovechando la estructura longitudinal de la **Encuesta Nacional de Hogares (ENAHO) Panel 2015–2019** del INEI ($N = 37,985$ pares de transición anual en $18,485$ hogares únicos).

📄 **[Ver presentación del estudio (`presentacion-vulnerabilidad-hogares.pdf`)](presentacion-vulnerabilidad-hogares.pdf)**

---

## 📌 Pregunta de Investigación y Justificación

> **¿Qué factores demográficos, habitacionales y de consumo en el periodo $t$ permiten anticipar que un hogar no pobre caerá en situación de pobreza en el periodo $t+1$?**

La medición convencional evalúa el gasto del hogar frente a la línea en un instante fijo ($t$), tratando a la población no pobre como un bloque homogéneo y seguro. Sin embargo, los datos longitudinales muestran una alta rotación interanual:
* **8.9%** de los hogares clasificados como *no pobres* en $t$ cae en la pobreza monetaria en $t+1$.
* **42.2%** de los hogares *pobres* en $t$ logra superar la línea monetaria en $t+1$.

Para superar la inercia de la focalización reactiva, este proyecto implementa una **estrategia de identificación dual** que combina un modelo econométrico **Logit Multinomial con errores estándar agrupados por hogar** y algoritmos de **Machine Learning (Regresión Logística L2, Random Forest balanceado y Gradient Boosting)**.

---

## 📂 Estructura del Repositorio

```text
vulnerabilidad-hogares/
├── .vscode/                              # Configuración de workspace (PowerShell, Jupyter Interactive)
├── data/
│   ├── raw/                              # [Ignorado en git] Microdatos crudos .dta de ENAHO Panel
│   └── processed/
│       └── data_pobreza.parquet          # Base de datos panel procesada (~1 MB, lista para modelar)
├── notebooks/
│   └── analisis_exploratorio.ipynb       # Exploración de variables y matriz de transición
├── reports/
│   └── modelo_pobreza_stata.log          # Bitácora auxiliar de pruebas en Stata
├── src/
│   ├── procesar_datos.py                 # Pipeline ETL longitudinal (extracción, reshape y agregación)
│   └── explotando_modelo.py              # Ingeniería de features, MNLogit agrupado y modelos ML (CV 5-fold)
├── presentacion-vulnerabilidad-hogares.pdf # Diapositivas de la publicación (LaTeX/Beamer)
├── .gitignore                            # Exclusión de microdatos crudos y entornos virtuales
├── requirements.txt                      # Dependencias del proyecto
└── README.md                             # Documentación general del repositorio
```

> **Nota sobre reproducibilidad inmediata:** La base procesada `data/processed/data_pobreza.parquet` está versionada en el repositorio para permitir la reproducción directa de las estimaciones econométricas y de Machine Learning sin necesidad de descargar los microdatos crudos de ENAHO.

---

## 🎯 Matriz de Transición y Variable Objetivo (`target_multinomial`)

Para modelar la dinámica panel de 1 año ($t \to t+1$), se restringe la muestra a pares anuales estrictamente consecutivos ($t+1 - t = 1$, $N = 37,985$). Se definen cuatro estados dinámicos mutuamente excluyentes ($Y_{it}$):

| Valor | Estado Dinámico | Transición ($t \to t+1$) | Participación Muestral | Interpretación Socioeconómica |
| :---: | :--- | :---: | :---: | :--- |
| **`0`** | **Resiliente (Base)** | $\text{No Pobre} \to \text{No Pobre}$ | **73.7%** ($28,210$) | Hogar que mantiene su condición de no pobre en ambos años. Categoría de referencia. |
| **`1`** | **Vulnerable (Caída)** | $\text{No Pobre} \to \text{Pobre}$ | **7.2%** ($2,751$) | **Foco preventivo:** Hogar no pobre en $t$ que cae bajo la línea en $t+1$ (8.9% de los no pobres). |
| **`2`** | **Escape / Salida** | $\text{Pobre} \to \text{No Pobre}$ | **8.1%** ($3,094$) | Hogar pobre en $t$ que supera la línea monetaria en $t+1$ (42.2% de los pobres). |
| **`3`** | **Pobre Crónico** | $\text{Pobre} \to \text{Pobre}$ | **11.1%** ($4,231$) | Hogar que permanece en pobreza monetaria en ambos periodos (57.8% de los pobres). |

---

## 📊 Diccionario de Variables y Feature Engineering

### 1. Variables Base Extraídas del Panel ENAHO (`src/procesar_datos.py`)
* **Identificación y panel:** `numpanh` (ID único del hogar panel), `year` (`15` a `19`), `conglome`, `vivienda`, `hogar`.
* **Vivienda y servicios (Módulo 100):** `p101` (tipo de vivienda), `p110` (abastecimiento de agua).
* **Agregados demográficos y laborales (Módulos 200 y 500):** `n_miembros` (residentes habituales), `n_ocupados` (miembros ocupados), `sexo_jefe`, `edad_jefe`.
* **Agregados monetarios y línea oficial (Módulo 34 - Sumaria):** `gashog2d` (gasto bruto anual), `inghog2d` (ingreso bruto anual), `linea` (línea de pobreza mensual per cápita), `pobreza_bin` (condición en $t$), `pobreza_futura` (condición en $t+1$).

### 2. Variables Derivadas (`ingenieria_features` en `src/explotando_modelo.py`)
* **Capacidad económica per cápita:** `log_gasto_pc` ($\ln(1 + \text{gasto}_{pc})$), `log_ingreso_pc` ($\ln(1 + \text{ingreso}_{pc})$), `ratio_gasto_ingreso` ($\text{gashog2d}/\text{inghog2d}$).
* **Estructura familiar y ciclo vital:** `tasa_ocupacion` ($\text{ocupados}/\text{miembros}$), `tasa_dependencia` ($(\text{miembros}-\text{ocupados})/\text{ocupados}$), `hogar_grande` ($>5$ miembros), `edad_jefe_sq`, `jefe_joven` ($<30$ años), `jefe_adulto_mayor` ($\ge 65$ años).
* **Holgura respecto al umbral oficial (exclusivo para ML):** `ratio_gasto_linea` ($\text{gasto}_{pc}/\text{línea}$) y `ratio_ingreso_linea` ($\text{ingreso}_{pc}/\text{línea}$).

---

## 🧠 Estrategia de Identificación Dual y Principales Hallazgos

### 1. Separación Cuasi-Completa y Diseño en Dos Etapas
Dado que el INEI define la pobreza monetaria en $t$ mediante la regla determinística $\text{gasto}_{pc,it} < \text{línea}_{it}$, incluir el cociente $\text{gasto}_{pc}/\text{línea}$ en un modelo de máxima verosimilitud no penalizado induce **separación cuasi-completa** entre los estados $\{0, 1\}$ y $\{2, 3\}$ (Albert & Anderson, 1984). Por ello, `src/explotando_modelo.py` implementa dos especificaciones complementarias:

1. **Modelo Econométrico (`entrenar_mnlogit`):**
   * Ajusta un **Logit Multinomial (`statsmodels.MNLogit`)** excluyendo los ratios respecto a la línea y empleando magnitudes en logaritmos, estructura demográfica y vivienda (`VARS_MNLOGIT`).
   * Corrige la correlación serial intra-hogar mediante **errores estándar agrupados por hogar (`cov_type="cluster"`, $18,485$ clusters)** y calcula Efectos Marginales Promedio (AME) y Odds Ratios ($\text{Pseudo } R^2 = 0.142$).
   * **Hallazgos estructurales:** Mayor capacidad de consumo base ($\ln \text{gasto}_{pc}$, $p < 0.001$) y acceso a agua por red pública ($-2.4\text{ pp}$ hacia pobreza crónica) actúan como factores protectores, mientras que la tasa de dependencia y el tamaño del hogar elevan significativamente el riesgo de caída interanual ($+1.8\text{ pp}$).

2. **Modelos de Aprendizaje Estadístico (`entrenar_ml`):**
   * Incorpora el margen relativo sobre la línea (`VARS_ML`) controlando el ajuste mediante regularización L2 y poda de árboles, evaluados fuera de muestra con **Validación Cruzada Estratificada ($K = 5$)**.

| Especificación | F1-Macro (Balance entre clases) | F1-Weighted (Desempeño global) |
| :--- | :---: | :---: |
| **Regresión Logística (L2)** | `0.5427` | `0.8191` |
| **Gradient Boosting** | `0.5817` | **`0.8314`** |
| **Random Forest (Balanceado)** | **`0.6081`** | `0.7578` |

* **Jerarquía de importancia (Gini en Random Forest):** El margen sobre la línea (`ratio_gasto_linea`: **45.7%**), la capacidad monetaria per cápita (`log_gasto_pc` + `log_ingreso_pc`: **34.4%**), el ratio ingreso/línea (**8.9%**), la propensión al gasto (**3.7%**) y la presión demográfica permiten identificar como grupo prioritario para **sistemas de alerta temprana** a los hogares con bajo colchón de consumo ($1.0 < \text{gasto}_{pc}/\text{línea} < 1.25$) y alta carga familiar.

---

## 🚀 Instalación y Reproducción

### 1. Clonar el repositorio
```bash
git clone https://github.com/CCDE-DataLab/vulnerabilidad-hogares.git
cd vulnerabilidad-hogares
```

### 2. Crear y activar el entorno virtual
* **En Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **En Linux/macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar el pipeline de modelado econométrico y Machine Learning
```bash
python src/explotando_modelo.py
```

*(Opcional)* Si dispones de los microdatos crudos `.dta` de la ENAHO Panel 2015–2019 en `data/raw/`, puedes regenerar el archivo Parquet desde cero ejecutando:
```bash
python src/procesar_datos.py
```

---

## 📄 Licencia
Este proyecto se distribuye bajo la licencia **MIT**.
