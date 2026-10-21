# Dinámica y Vulnerabilidad de la Pobreza en Hogares Peruanos (ENAHO Panel 2015-2019)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: INEI ENAHO Panel](https://img.shields.io/badge/data-INEI%20ENAHO%20Panel-green.svg)](https://proyectos.inei.gob.pe/microdatos/)

Proyecto de investigación aplicada enfocado en modelar y anticipar la **vulnerabilidad socioeconómica** y las transiciones hacia la pobreza de los hogares peruanos. En lugar del enfoque estático tradicional de corte transversal, este estudio modela la **dinámica intertemporal de la pobreza** aprovechando la estructura longitudinal de la **Encuesta Nacional de Hogares (ENAHO) Panel 2015-2019** del INEI.

---

## 📌 Pregunta de Investigación y Justificación

> **¿Qué características habitacionales, demográficas y socioeconómicas en el periodo $t$ permiten anticipar que un hogar no pobre caerá en situación de pobreza en el periodo $t+1$?**

La política social tradicional suele focalizarse en quienes ya se encuentran en situación de pobreza monetaria (estáticos). Sin embargo, una fracción considerable de la población se encuentra en condición de **vulnerabilidad**: hogares clasificados como "no pobres" pero con una alta probabilidad de caer en pobreza ante choques económicos (salud, desempleo, inflación, choques climáticos). Este proyecto busca construir un modelo econométrico multinomial y de machine learning para predecir y caracterizar dichas transiciones.

---

## 📂 Estructura del Repositorio

```text
vulnerabilidad-hogares/
├── .vscode/                     # Configuración de workspace (PowerShell, Jupyter Interactive)
├── data/
│   ├── raw/                     # [Ignorado en git] Microdatos crudos .dta de ENAHO Panel
│   └── processed/
│       └── data_pobreza.parquet # Base de datos final procesada (~1 MB, lista para modelar)
├── notebooks/
│   └── analisis_exploratorio.ipynb # Exploración inicial de variables y transiciones
├── src/
│   ├── procesar_datos.py        # Pipeline ETL completo (extracción, reshape, agregación y exportación)
│   └── entrenar_modelo.py       # Modelo Multinomial Logit (MNLogit) con clustering a nivel de hogar
├── bibliography/                # [Ignorado en git] Documentación y cuestionarios ENAHO
├── .gitignore                   # Exclusión de microdatos crudos, entornos virtuales y temporales
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación exhaustiva del proyecto
```

> **Nota sobre los datos:** La base procesada `data/processed/data_pobreza.parquet` se encuentra versionada en el repositorio para que cualquier colaborador pueda comenzar a modelar de inmediato sin necesidad de descargar gigabytes de microdatos crudos de ENAHO.

---

## 📊 Diccionario de Variables

### 1. Variables de Identificación y Panel
| Variable | Descripción | Fuente | Tipo |
| :--- | :--- | :--- | :--- |
| `numpanh` | Identificador único del hogar panel a lo largo de los años | INEI | Categórica / ID numérico |
| `year` | Año de la observación (`15` = 2015, ..., `19` = 2019) | ENAHO Panel | Entero |
| `conglome` | Conglomerado censal | ENAHO | Categórica |
| `vivienda` | Número de selección de la vivienda | ENAHO | Categórica |
| `hogar` | Número de hogar dentro de la vivienda | ENAHO | Categórica |

### 2. Variables de Características de la Vivienda (Módulo 100)
| Variable | Descripción | Categorías / Valores de referencia |
| :--- | :--- | :--- |
| `p101` | Tipo de vivienda | Casa independiente, departamento, vivienda en quinta, casa de vecindad, choza/cabaña, otra vivienda |
| `p110` | Abastecimiento de agua | Red pública (dentro/fuera), pilón, camión cisterna, pozo, río/acequia, etc. |

### 3. Variables de Ingreso, Gasto y Pobreza (Módulo 34 - Sumaria)
| Variable | Descripción | Unidad |
| :--- | :--- | :--- |
| `linea` | Línea de pobreza total mensual per cápita | Soles constantes mensuales por persona |
| `gashog2d` | Gasto bruto anual del hogar | Soles anuales |
| `inghog2d` | Ingreso bruto anual del hogar | Soles anuales |
| `pobreza` | Condición oficial de pobreza monetaria | `no pobre`, `pobre no extremo`, `pobre extremo` |
| `pobreza_bin` | Pobreza binarizada en periodo actual $t$ | `0` = No pobre, `1` = Pobre |
| `pobreza_futura`| Condición de pobreza en periodo siguiente $t+1$ | `0` = No pobre en $t+1$, `1` = Pobre en $t+1$ |

### 4. Variables Agregadas a Nivel Hogar (Módulos 200 y 500)
| Variable | Descripción | Método de Construcción |
| :--- | :--- | :--- |
| `n_miembros` | Cantidad de residentes habituales del hogar | Conteo de personas con `p204 == 'si'/'1'` en el hogar |
| `n_ocupados` | Cantidad de miembros ocupados en el mercado laboral | Conteo de personas con `p507` válido (trabajó o tuvo empleo) |
| `sexo_jefe` | Sexo del jefe de hogar (`p203 == 'jefe'/'1'`) | Hombre / Mujer |
| `edad_jefe` | Edad del jefe de hogar en años cumplidos | Numérico (`p208a`) |

---

## 🎯 Variable Objetivo: La Matriz de Transición (`target_multinomial`)

Para modelar la dinámica panel de 1 año ($t \to t+1$), filtramos estrictamente las observaciones donde el hogar estuvo presente consecutivamente en ambos periodos (`year_futuro - year == 1`).

Se construye la variable discreta multiclase `target_multinomial` con 4 estados mutuamente excluyentes:

| Valor | Estado de Transición | Condición ($t \to t+1$) | Interpretación Socioeconómica |
| :---: | :--- | :---: | :--- |
| **`0`** | **Resiliente (Base)** | $0 \to 0$ | Hogar que era No Pobre y se mantiene No Pobre al año siguiente. Categoría de referencia. |
| **`1`** | **Vulnerable (Caída)**| $0 \to 1$ | **Foco principal:** Hogar que era No Pobre y cae en Pobreza al año siguiente. |
| **`2`** | **Escape / Rescate** | $1 \to 0$ | Hogar que era Pobre y logra salir de la pobreza monetaria. |
| **`3`** | **Pobre Crónico** | $1 \to 1$ | Hogar que era Pobre y permanece atrapado en la pobreza. |

---

## ⚙️ Arquitectura del Código y Funcionamiento de los Scripts

### 1. `src/procesar_datos.py` (Pipeline ETL)

Este script orquesta la lectura de microdatos en formato Stata (`.dta`), su reestructuración longitudinal y la consolidación de la base de datos analítica:

* **`load_modulo100(raw_path)`:** Lee el Módulo 100 en formato wide (`conglome_15`, ..., `p101_19`). Extrae solo las columnas requeridas para optimizar memoria RAM.
* **`load_modulo200(raw_path)`:** Lee el Módulo 200 (personas). Mapea el identificador longitudinal de personas `numper` y el del hogar `numpanh`.
* **`load_modulo500(raw_path)`:** Lee el Módulo 500 (empleo). Extrae la condición de ocupación (`p507`) y relación laboral.
* **`load_sumaria(raw_path)`:** Lee el Módulo Sumaria con los agregados económicos y la clasificación oficial de pobreza.
* **`transform_wide_to_long(df, id_cols)`:** Aplica `pd.wide_to_long` detectando automáticamente los sufijos temporales `_15`, `_16`, etc., transformando las columnas en una fila por combinación `(hogar/persona, year)`.
* **Bloque de Agregación (`df_miembros` -> `df_hogar_agregado`):**
  1. Filtra a los residentes habituales del hogar (`p204`).
  2. Determina el estado de ocupación laboral a partir de `p507`.
  3. Agrupa a nivel `["numpanh", "year"]` calculando `n_miembros` y `n_ocupados`.
  4. Extrae la fila del jefe del hogar (`p203`) para asignar `sexo_jefe` y `edad_jefe` a nivel de hogar.
* **Cruce Maestro y Creación de Target:**
  * Realiza los `pd.merge` longitudinales por `["numpanh", "year"]`.
  * Calcula la variable adelantada `pobreza_futura` mediante `shift(-1)`.
  * Valida que la distancia temporal sea exactamente de 1 año (`year_futuro - year == 1`) para evitar sesgos de atrición o saltos de años no encuestados.
  * Exporta el resultado final a `data/processed/data_pobreza.parquet`.

---

### 2. `src/entrenar_modelo.py` (Modelado Econométrico MLogit)

Este script entrena el modelo de elección discreta multinomial considerando la naturaleza panel de los datos:

* **`cargar_datos()`:**
  * Carga `data_pobreza.parquet`.
  * Agrupa categorías con escasas observaciones en `p101` para evitar separaciones cuasi-completas en la regresión.
  * **Prevención de Data Leakage:** Excluye rigurosamente `pobreza_bin`, `pobreza_futura`, `target_multinomial` e identificadores del conjunto de regresores $X$.
  * Retorna la matriz de predictores $X$, la variable objetivo $y$ y la serie de identificación de hogares `grupos` (`numpanh`).
* **`entrenar_modelo(X, y, grupos_hogar)`:**
  * Transforma las variables categóricas en variables dicotómicas (dummies) mediante `pd.get_dummies(..., drop_first=True)`.
  * Agrega el término de intercepción constante (`sm.add_constant`).
  * Ajusta un **Logit Multinomial (`sm.MNLogit`)**.
  * **Corrección de Errores Estándar:** Aplica errores estándar robustos con **clustering a nivel de hogar (`cov_type='cluster'`, `groups=grupos_hogar`)**, controlando la correlación serial intra-hogar propia de los datos de panel repetidos en el tiempo.
* **Métricas y Post-Estimación:**
  * Imprime el resumen econométrico (`summary()`).
  * Calcula los **Efectos Marginales Promedio (AME)** mediante `resultados_modelo.get_margeff()`, indispensables para la interpretación cuantitativa de probabilidades.
  * Reporta **Pseudo $R^2$ de McFadden**, **AIC**, **BIC** y los **Odds Ratios** ($\exp(\beta)$).

---

## 🛠️ GUÍA PARA ABEL: ¿Dónde y cómo editar para seleccionar variables y mejorar el modelo?

Tu misión principal es **probar nuevas especificaciones, seleccionar variables predictoras y mejorar el ajuste y capacidad predictiva del modelo MLogit**. 

Dependiendo del tipo de cambio que vayas a realizar, debes editar en un lugar o en otro:

---

### Escenario A: Trabajar con variables que YA están en la base procesada

Si deseas probar transformaciones (logaritmos, ratios, interacciones) o seleccionar un subconjunto específico de variables para $X$:

👉 **Archivo a editar:** [`src/entrenar_modelo.py`](file:///c:/Users/mauri/OneDrive/Documentos/proyectos-personales/vulnerabilidad-hogares/src/entrenar_modelo.py)  
👉 **Sección exacta:** Función `cargar_datos()` (líneas 19 a 40)

#### Ejemplo práctico:

```python
# En src/entrenar_modelo.py -> cargar_datos():

def cargar_datos():
    df = pd.read_parquet(PROCESSED_PATH / "data_pobreza.parquet")

    # 1. Agrupación de categorías
    mapa_cambios = {
        "local no destinado para habitación humana": "otra vivienda",
        "otro": "otra vivienda",
        "vivienda improvisada": "otra vivienda"
    }
    df["p101"] = df["p101"].replace(mapa_cambios)
    df = df[df["p101"] != "nan"]

    # 2. INGENIERÍA DE FEATURES (Aquí puedes crear nuevas variables):
    # Ratio de ocupación del hogar:
    df["tasa_ocupacion"] = df["n_ocupados"] / df["n_miembros"]
    # Gasto e ingreso per cápita mensual aproximado:
    df["log_gashog"] = np.log1p(df["gashog2d"] / df["n_miembros"])
    df["log_inghog"] = np.log1p(df["inghog2d"] / df["n_miembros"])
    # Relación del ingreso con respecto a la línea de pobreza:
    df["ratio_ingreso_linea"] = (df["inghog2d"] / (df["n_miembros"] * 12)) / df["linea"]

    # 3. SELECCIÓN DE VARIABLES PREDICTORAS (Elige qué entra y qué no):
    # Opción 1: Mantener la lista de exclusión si solo quieres quitar cosas específicas
    columnas_excluir = [
        "numpanh", "year", "conglome", "vivienda", 
        "hogar", "pobreza", "pobreza_futura", "pobreza_bin"
    ]
    X = df.drop(columns=columnas_excluir + ["target_multinomial"])

    # Opción 2 (RECOMENDADA para control total): Elegir explícitamente las columnas de X
    # variables_seleccionadas = [
    #     "tasa_ocupacion", "log_gashog", "ratio_ingreso_linea",
    #     "edad_jefe", "sexo_jefe", "p101", "p110"
    # ]
    # X = df[variables_seleccionadas]

    y = df["target_multinomial"]
    grupos = df["numpanh"]

    return X, y, grupos
```

---

### Escenario B: Extraer NUEVAS variables desde los microdatos crudos de ENAHO

Si necesitas incorporar variables adicionales desde los archivos `.dta` originales (por ejemplo: nivel educativo del jefe en el Módulo 300, tenencia de activos en el Módulo 100, formalidad en el Módulo 500, etc.):

👉 **Archivo a editar:** [`src/procesar_datos.py`](file:///c:/Users/mauri/OneDrive/Documentos/proyectos-personales/vulnerabilidad-hogares/src/procesar_datos.py)

#### Pasos a seguir:

1. **Paso 1: Agregar la variable a la lista de columnas en el módulo correspondiente:**
   * Si es del Módulo 100 (vivienda/servicios): Edita `load_modulo100()` (línea 18):
     ```python
     formato_base = ["conglome", "vivienda", "hogar", "p101", "p110", "p111a", "p1121"] # ejemplo: alumbrado eléctrico
     ```
   * Si es del Módulo 200 (educación/demografía): Edita `load_modulo200()` (línea 30):
     ```python
     formato_base = ["conglome", "vivienda", "hogar", "p203", "p204", "p207", "p208a", "p209"] # estado civil
     ```
   * Si es del Módulo 500 (laboral): Edita `load_modulo500()` (línea 47):
     ```python
     formato_base = ["conglome", "vivienda", "hogar", "p507", "p512b", "p510a"] # formalidad / contrato
     ```
2. **Paso 2: Si la variable es a nivel individual (Módulos 200 o 500), definir cómo se agrega al hogar:**
   * En la sección `[5] Agregación a Nivel Hogar`:
     * ¿Se calculará un promedio/tasa en el hogar? (e.g. `df_hogar_agregado = df_miembros.groupby(...).agg(...)`).
     * ¿O corresponde a un atributo del jefe de hogar? (e.g. `df_jefes[["numpanh", "year", "edad_jefe", "educ_jefe"]]`).
3. **Paso 3: Regenerar el dataset procesado:**
   * Ejecuta en terminal:
     ```bash
     python src/procesar_datos.py
     ```
   * Esto actualizará automáticamente `data/processed/data_pobreza.parquet`.

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

### 4. Ejecutar el entrenamiento del modelo
```bash
python src/entrenar_modelo.py
```

---

## 🤝 Flujo de Trabajo con Git y GitHub para Colaboradores

1. **Crear una rama para tus avances:**
   ```bash
   git checkout -b avances-abel
   ```
2. **Realizar modificaciones y commits atómicos:**
   ```bash
   git add src/entrenar_modelo.py
   git commit -m "feat(model): agregar ratios de ingreso e indicadores de formalidad"
   ```
3. **Publicar tu rama y solicitar revisión:**
   ```bash
   git push origin avances-abel
   ```
   Luego abre un **Pull Request (PR)** hacia la rama `main` describiendo los cambios en las variables, el valor del Pseudo $R^2$ alcanzado y las mejoras en los criterios AIC/BIC.

---

## 📄 Licencia
Este proyecto se distribuye bajo la licencia **MIT**. Consulta el archivo `LICENSE` para más información.