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
    #with pd.read_stata(RAW_PATH / "enaho01-2015-2019-100-panel.dta", iterator=True) as reader:
        # Leer solo las columnas necesarias para el proyecto
        #todas_columnas = reader.variable_labels()
        #print(list(todas_columnas.keys()))

    file_path = raw_path / "enaho01-2015-2019-100-panel.dta"

    formato_base = ["conglome","vivienda","hogar","p101","p110"]
    columns_to_use = [f"{col}_{year}" for year in range(15, 20) for col in formato_base]
    columns_to_use.insert(0, "numpanh")
    print(f"Cargando los datos del módulo 100desde {file_path}\n Columnas a usar: {columns_to_use}")
    df = pd.read_stata(file_path, columns=columns_to_use)
    return df

def load_sumaria(raw_path: Path) -> pd.DataFrame:
    """
    Carga el modulo sumaria de la ENAHO Panel
    """
    file_path = raw_path / "sumaria-2015-2019-panel.dta"
    formato_base = ["pobreza","linea","gashog2d","inghog2d"]
    columns_to_use = [f"{col}_{year}" for year in range(15, 20) for col in formato_base]
    columns_to_use.insert(0, "numpanh")
    print(f"Cargando los datos de la sumaria desde {file_path}\n Columnas a usar: {columns_to_use}")
    df = pd.read_stata(file_path, columns=columns_to_use)
    return df


def transform_wide_to_long(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    """
    Transforma el DataFrame de formato ancho a formato largo

    Args:
        df (pd.DataFrame): DataFrame en formato ancho
        id_cols (list): Lista de columnas que se mantendrán como identificadores
    """
    columnas_temporales = [col for col in df.columns if col not in id_cols]
    stubs = list(set([col.rsplit("_",1)[0] for col in columnas_temporales if "_" in col]))

    df = pd.wide_to_long(
        df,
        stubnames = stubs,
        i = id_cols,
        j = "year",
        sep = "_",
        suffix = r"\d+"
    ).reset_index()
    return df





if __name__ == "__main__":
    print("Iniciando bloque ETL para procesamiento de datos...")
    print("Cargando y transformando el módulo 100")
    df_mod100_wide = load_modulo100(RAW_PATH)
    df_mod100_long = transform_wide_to_long(df_mod100_wide, id_cols=["numpanh"])

    print("Cargando y transformando el módulo sumaria")
    df_sumaria_wide = load_sumaria(RAW_PATH)
    df_sumaria_long = transform_wide_to_long(df_sumaria_wide, id_cols=["numpanh"])

    df_final = pd.merge(df_mod100_long, df_sumaria_long, on=["numpanh","year"], how="left")

    print(df_final.head())
