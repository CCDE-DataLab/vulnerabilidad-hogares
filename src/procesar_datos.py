import pandas as pd
import numpy as np
import os 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
RAW_PATH = DATA_PATH / "raw"
PROCESSED_PATH = DATA_PATH / "processed"


#función ETL para el módulo 100:
def load_modulo100(raw_path: Path) -> pd.DataFrame:
    """
    Carga el módulo 100 de la ENAHO Panel, filtrando solo las columnas necesarias para el proyecto
    """
    with pd.read_stata(RAW_PATH / "enaho01-2015-2019-100-panel.dta", iterator=True) as reader:
        # Leer solo las columnas necesarias para el proyecto
        todas_columnas = reader.variable_labels()
        print(list(todas_columnas.keys()))

    file_path = raw_path / "enaho01-2015-2019-100-panel.dta"

    formato_base = ["conglome","vivienda","hogar","p101","p110"]
    columns_to_use = [f"{col}_{year}" for year in range(15, 20) for col in formato_base]
    columns_to_use.insert(0, "numpanh")
    print(f"Cargando los datos desde {file_path}\n Columnas a usar: {columns_to_use}")
    df = pd.read_stata(file_path, columns=columns_to_use)
    return df


if __name__ == "__main__":
    print("Iniciando bloque ETL para procesamiento de datos...")

    df_mod100 = load_modulo100(RAW_PATH)

    print(f"Módulo 100 cargado con éxito. Dimensiones: {df_mod100.shape}")
    print(df_mod100.head())