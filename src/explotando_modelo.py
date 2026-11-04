# %% [1] Importaciones y Configuración de Rutas
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import io

warnings.filterwarnings("ignore")

# Forzar UTF-8 en stdout para evitar errores de codificación en Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Modelado econométrico
import statsmodels.api as sm

# Machine Learning
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed"

ETIQUETAS_TARGET = {
    0: "Resiliente",
    1: "Cae en pobreza",
    2: "Sale de pobreza",
    3: "Pobre cronico",
}


# ============================================================================
# %% [2] Ingeniería de Features
# ============================================================================
def ingenieria_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables derivadas que capturan señales de vulnerabilidad
    económica del hogar sin incurrir en data leakage ni separación perfecta.

    NOTA METODOLÓGICA:
    -  La pobreza en ENAHO se define como gasto_pc_mensual < linea.
       Por lo tanto, ratio_gasto_linea y ratio_ingreso_linea son predictores
       PERFECTOS de pobreza_bin y no deben usarse en el MNLogit (causan
       separación completa → coeficientes NaN). Se crean aquí solo para
       los modelos de ML como señal de cercanía económica.
    -  Las variables "log_gasto_pc" y "log_ingreso_pc" sí son válidas para
       el MNLogit porque capturan el nivel económico general sin ser
       determinísticos de la definición de pobreza.
    """
    df = df.copy()

    # --- Agrupación de categorías raras en p101 ---
    mapa_vivienda = {
        "local no destinado para habitación humana": "otra vivienda",
        "otro": "otra vivienda",
        "vivienda improvisada": "otra vivienda",
    }
    df["p101"] = df["p101"].replace(mapa_vivienda)
    df = df[~df["p101"].isin(["nan", ""])]
    df = df.dropna(subset=["p101"])

    # --- Agrupación de categorías raras en p110 ---
    mapa_agua = {"8.0": "otra"}
    df["p110"] = df["p110"].replace(mapa_agua)

    # --- Variables económicas per cápita (log-transformadas) ---
    df["gasto_pc_mensual"] = (df["gashog2d"] / df["n_miembros"]) / 12
    df["ingreso_pc_mensual"] = (df["inghog2d"] / df["n_miembros"]) / 12
    df["log_gasto_pc"] = np.log1p(df["gasto_pc_mensual"])
    df["log_ingreso_pc"] = np.log1p(df["ingreso_pc_mensual"])

    # --- Ratios respecto a la línea (SOLO para ML, NO para MNLogit) ---
    df["ratio_ingreso_linea"] = df["ingreso_pc_mensual"] / df["linea"]
    df["ratio_gasto_linea"] = df["gasto_pc_mensual"] / df["linea"]

    # --- Variables de estructura del hogar ---
    df["tasa_ocupacion"] = df["n_ocupados"] / df["n_miembros"]
    df["tasa_dependencia"] = np.where(
        df["n_ocupados"] > 0,
        (df["n_miembros"] - df["n_ocupados"]) / df["n_ocupados"],
        df["n_miembros"],
    )
    df["hogar_grande"] = (df["n_miembros"] > 5).astype(int)

    # --- Variables del jefe de hogar ---
    df["edad_jefe_sq"] = df["edad_jefe"] ** 2
    df["jefe_joven"] = (df["edad_jefe"] < 30).astype(int)
    df["jefe_adulto_mayor"] = (df["edad_jefe"] >= 65).astype(int)

    # --- Ratio gasto/ingreso (indicador de capacidad de ahorro) ---
    df["ratio_gasto_ingreso"] = np.where(
        df["inghog2d"] > 0,
        df["gashog2d"] / df["inghog2d"],
        1.0,
    )

    return df


# ============================================================================
# %% [3] Función para Cargar y Preparar Datos
# ============================================================================

# Variables para el MNLogit econométrico:
# Excluimos ratio_ingreso_linea y ratio_gasto_linea porque causan separación
# perfecta (son la definición misma de la pobreza monetaria en ENAHO).
VARS_MNLOGIT = [
    # Categoricas - vivienda
    "p101",
    "p110",
    # Estructura del hogar
    "n_miembros",
    "tasa_ocupacion",
    "tasa_dependencia",
    "hogar_grande",
    # Jefe de hogar
    "sexo_jefe",
    "edad_jefe",
    "edad_jefe_sq",
    "jefe_joven",
    "jefe_adulto_mayor",
    # Económicas (nivel general, sin ser definición de pobreza)
    "log_gasto_pc",
    "log_ingreso_pc",
    "ratio_gasto_ingreso",
]

# Variables para los modelos de ML:
# Incluyen los ratios respecto a la línea como señal de cercanía al umbral.
VARS_ML = VARS_MNLOGIT + [
    "ratio_ingreso_linea",
    "ratio_gasto_linea",
]


def cargar_datos():
    """
    Carga la base procesada, aplica feature engineering y devuelve
    X_mnlogit, X_ml, y (target multinomial) y grupos (numpanh).
    """
    print("=" * 70)
    print("  CARGA Y PREPARACION DE DATOS")
    print("=" * 70)

    df = pd.read_parquet(PROCESSED_PATH / "data_pobreza.parquet")
    print(f"  Observaciones crudas: {len(df):,}")

    df = ingenieria_features(df)
    print(f"  Observaciones post-limpieza: {len(df):,}")

    X_mnlogit = df[VARS_MNLOGIT].copy()
    X_ml = df[VARS_ML].copy()
    y = df["target_multinomial"].astype(int)
    grupos = df["numpanh"]

    print(f"\n  Variables MNLogit ({len(VARS_MNLOGIT)}): {VARS_MNLOGIT}")
    print(f"  Variables ML      ({len(VARS_ML)}): {VARS_ML}")
    print(f"\n  Distribucion del target:")
    for val, cnt in y.value_counts().sort_index().items():
        print(f"    {val} ({ETIQUETAS_TARGET[val]}): {cnt:,} ({cnt/len(y)*100:.1f}%)")
    print()

    return X_mnlogit, X_ml, y, grupos


# ============================================================================
# %% [4] Modelo 1: Multinomial Logit con Errores Clustered (Econométrico)
# ============================================================================
def entrenar_mnlogit(X: pd.DataFrame, y: pd.Series, grupos_hogar: pd.Series = None):
    """
    Ajusta un MNLogit de statsmodels con errores estándar clustered
    a nivel de hogar panel. Reporta efectos marginales, pseudo-R²,
    AIC/BIC y odds ratios.

    IMPORTANTE: No recibe ratio_ingreso_linea ni ratio_gasto_linea
    porque causan separación perfecta (pobreza = gasto_pc < linea).
    """
    print("=" * 70)
    print("  MODELO 1: MULTINOMIAL LOGIT (Econometrico - statsmodels)")
    print("=" * 70)

    X_work = X.copy()
    y_work = y.copy()

    # Dummies para categóricas
    X_dummies = pd.get_dummies(X_work, drop_first=True, dtype=float)

    # Verificar multicolinealidad: eliminar columnas con varianza 0
    low_var = X_dummies.columns[X_dummies.std() < 1e-10]
    if len(low_var) > 0:
        print(f"  [WARN] Eliminando {len(low_var)} columnas con varianza ~0: {low_var.tolist()}")
        X_dummies = X_dummies.drop(columns=low_var)

    X_con_constante = sm.add_constant(X_dummies)

    print(f"  Regresores: {X_con_constante.shape[1]} columnas")
    print(f"  Observaciones: {len(y_work):,}")
    print(f"  Clusters (hogares unicos): {grupos_hogar.nunique():,}" if grupos_hogar is not None else "")

    modelo = sm.MNLogit(endog=y_work, exog=X_con_constante)

    if grupos_hogar is not None:
        resultados = modelo.fit(
            cov_type="cluster",
            cov_kwds={"groups": grupos_hogar},
            maxiter=500,
            disp=True,
        )
    else:
        resultados = modelo.fit(maxiter=500, disp=True)

    # Verificar convergencia real
    if np.isnan(resultados.llf):
        print("\n  [ERROR] Log-Likelihood es NaN. El modelo no convergio correctamente.")
        print("  Posible causa: separacion perfecta o multicolinealidad extrema.")
        print("  Revise las variables predictoras.")
        return resultados

    # --- Resumen ---
    print("\n" + resultados.summary().as_text())

    # --- Efectos Marginales Promedio ---
    print("\n" + "-" * 50)
    print("  Efectos Marginales Promedio (AME)")
    print("-" * 50)
    try:
        margeff = resultados.get_margeff()
        print(margeff.summary())
    except Exception as e:
        print(f"  [WARN] No se pudieron calcular AME: {e}")

    # --- Métricas de Ajuste ---
    print("\n" + "-" * 50)
    print("  Metricas de Ajuste del Modelo Econometrico")
    print("-" * 50)
    print(f"  Pseudo R2 (McFadden): {resultados.prsquared:.4f}")
    print(f"  AIC:                  {resultados.aic:.2f}")
    print(f"  BIC:                  {resultados.bic:.2f}")
    print(f"  Log-Likelihood:       {resultados.llf:.2f}")
    print(f"  Log-Lik (Null):       {resultados.llnull:.2f}")

    # --- Odds Ratios ---
    print("\n" + "-" * 50)
    print("  Odds Ratios (exp(coef))")
    print("-" * 50)
    odds_ratios = np.exp(resultados.params)
    print(odds_ratios.to_string())

    # --- Predicción in-sample y clasificación ---
    print("\n" + "-" * 50)
    print("  Clasificacion In-Sample (MNLogit)")
    print("-" * 50)
    y_pred_probs = resultados.predict(X_con_constante)
    y_pred = y_pred_probs.values.argmax(axis=1)
    report = classification_report(
        y_work, y_pred,
        target_names=list(ETIQUETAS_TARGET.values()),
        digits=4,
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(y_work, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real: {e}" for e in ETIQUETAS_TARGET.values()],
        columns=[f"Pred: {e}" for e in ETIQUETAS_TARGET.values()],
    )
    print("  Matriz de Confusion:")
    print(cm_df.to_string())

    return resultados


# ============================================================================
# %% [5] Modelos de Machine Learning (con Validación Cruzada Estratificada)
# ============================================================================
def entrenar_ml(X: pd.DataFrame, y: pd.Series):
    """
    Entrena modelos de ML con validación cruzada estratificada (5-fold).
    Incluye Logistic Regression, Random Forest y Gradient Boosting.
    """
    print("\n" + "=" * 70)
    print("  MODELOS DE MACHINE LEARNING (Validacion Cruzada Estratificada)")
    print("=" * 70)

    cols_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    cols_num = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols_num),
            ("cat", OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist"), cols_cat),
        ],
        remainder="drop",
    )

    modelos = {
        "Logistic Regression (L2)": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_leaf=30,
                subsample=0.8,
                random_state=42,
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    etiquetas_lista = list(ETIQUETAS_TARGET.values())

    resultados_ml = {}

    for nombre, pipeline in modelos.items():
        print(f"\n{'_' * 50}")
        print(f"  {nombre}")
        print(f"{'_' * 50}")

        y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")

        report = classification_report(
            y, y_pred,
            target_names=etiquetas_lista,
            digits=4,
            zero_division=0,
        )
        print(report)

        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Real: {e}" for e in etiquetas_lista],
            columns=[f"Pred: {e}" for e in etiquetas_lista],
        )
        print("  Matriz de Confusion:")
        print(cm_df.to_string())

        f1_macro = f1_score(y, y_pred, average="macro")
        f1_weighted = f1_score(y, y_pred, average="weighted")
        print(f"\n  F1-macro:    {f1_macro:.4f}")
        print(f"  F1-weighted: {f1_weighted:.4f}")

        resultados_ml[nombre] = {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "y_pred": y_pred,
        }

    # --- Feature importance del Random Forest ---
    print(f"\n{'=' * 70}")
    print("  IMPORTANCIA DE VARIABLES (Random Forest - Ajuste Completo)")
    print(f"{'=' * 70}")
    rf_pipeline = modelos["Random Forest"]
    rf_pipeline.fit(X, y)
    rf_clf = rf_pipeline.named_steps["clf"]
    prep = rf_pipeline.named_steps["prep"]

    feature_names = (
        cols_num
        + list(prep.named_transformers_["cat"].get_feature_names_out(cols_cat))
    )
    importances = pd.Series(rf_clf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    print("\n  Top 15 variables mas importantes:")
    for i, (feat, imp) in enumerate(importances.head(15).items(), 1):
        bar = "#" * int(imp * 200)
        print(f"  {i:2d}. {feat:<45s} {imp:.4f} {bar}")

    # --- Tabla comparativa ---
    print(f"\n{'=' * 70}")
    print("  TABLA COMPARATIVA DE MODELOS ML")
    print(f"{'=' * 70}")
    comp = pd.DataFrame(
        {k: {m: v[m] for m in ["f1_macro", "f1_weighted"]} for k, v in resultados_ml.items()}
    ).T
    comp.columns = ["F1 Macro", "F1 Weighted"]
    print(comp.to_string())

    return resultados_ml


# ============================================================================
# %% [6] Ejecución Principal
# ============================================================================
if __name__ == "__main__":
    # Carga de datos con ingeniería de features
    X_mnlogit, X_ml, y, grupos = cargar_datos()

    # --- Modelo Econométrico (sin ratios respecto a la linea) ---
    resultados_mnlogit = entrenar_mnlogit(X_mnlogit, y, grupos)

    # --- Modelos de Machine Learning (con ratios respecto a la linea) ---
    resultados_ml = entrenar_ml(X_ml, y)

    print("\n" + "=" * 70)
    print("  EJECUCION COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print("  Modelos entrenados:")
    print("    1. MNLogit con errores clustered (econometrico)")
    print("    2. Logistic Regression L2 (ML baseline)")
    print("    3. Random Forest (ML - no lineal)")
    print("    4. Gradient Boosting (ML - ensemble)")
    print("=" * 70)
