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

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR /  "data" / "processed"

def cargar_datos():
    print("Cargando base de datos procesada...\n")

    df = pd.read_parquet(PROCESSED_PATH/"data_pobreza.parquet")

    columnas_excluir = ["numpanh", "year", "conglome", "vivienda", "hogar", "pobreza", "pobreza_futura","pobreza_bin"]
    #se remueve pobreza_futura para evitar dataleak
    X = df.drop(columns=columnas_excluir + ["target_multinomial"])
    y = df["target_multinomial"]

    return X, y


if __name__ == "__main__":
    X, y = cargar_datos()
    print("\nVariables predictoras disponibles (X):")
    print(X.columns.tolist())
    print(f"\nDimensiones de X: {X.shape}")
