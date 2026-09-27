# %% [1] Importaciones y Configuración de Rutas
import pandas as pd
import numpy as np
from pathlib import Path

# Importamos las herramientas de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed"

# %% [2] Función para Cargar y Preparar Datos
def cargar_datos():
    print("Cargando base de datos procesada...\n")
    df = pd.read_parquet(PROCESSED_PATH / "data_pobreza.parquet")

    mapa_cambios = {
        "local no destinado para habitación humana": "otra vivienda",
        "otro": "otra vivienda",
        "vivienda improvisada": "otra vivienda"
    }
    df["p101"] = df["p101"].replace(mapa_cambios)
    df = df[df["p101"] != "nan"]

    columnas_excluir = [
        "numpanh", "year", "conglome", "vivienda", 
        "hogar", "pobreza", "pobreza_futura", "pobreza_bin"
    ]
    # Se remueve pobreza_futura y pobreza_bin para evitar data leakage
    X = df.drop(columns=columnas_excluir + ["target_multinomial"])
    y = df["target_multinomial"]
    grupos = df["numpanh"]

    return X, y, grupos

# %% [3] Función para Entrenar el Modelo Multinomial (Panel Data)
def entrenar_modelo(X: pd.DataFrame, y: pd.Series, grupos_hogar: pd.Series = None):
    X = X.copy()
    y = y.copy()
    X_dummies = pd.get_dummies(X, drop_first=True, dtype=float)
    X_con_constante = sm.add_constant(X_dummies)

    modelo = sm.MNLogit(endog=y, exog=X_con_constante)

    if grupos_hogar is not None:
        resultados = modelo.fit(
            cov_type="cluster",
            cov_kwds={"groups": grupos_hogar}
        )
    else:
        resultados = modelo.fit()
    return resultados

# %% [4] Ejecución Principal
if __name__ == "__main__":
    # Carga de datos
    X, y, grupos = cargar_datos()
    print("\nVariables predictoras disponibles (X):")
    print(X.columns.tolist())
    print(f"\nDimensiones de X: {X.shape}\n")

    # %% [5] Entrenamiento y Resumen del Modelo
    print("Entrenando modelo de panel:\n")
    resultados_modelo = entrenar_modelo(X, y, grupos)

    print("Resumen de resultados:\n")
    print(resultados_modelo.summary())

    # %% [6] Efectos Marginales Promedio (AME)
    print("Cálculo de los Efectos Marginales Promedio (AME):\n")
    margeff = resultados_modelo.get_margeff()
    print(margeff.summary())

    # %% [7] Métricas de Ajuste y Odds Ratios
    print(f"Pseudo R-cuadrado (McFadden): {resultados_modelo.prsquared:.4f}")
    print(f"Criterio de Información de Akaike (AIC): {resultados_modelo.aic:.2f}")
    print(f"Criterio Bayesiano (BIC): {resultados_modelo.bic:.2f}")

    odds_ratios = np.exp(resultados_modelo.params)
    print("\nOdds Ratios:\n")
    print(odds_ratios)
