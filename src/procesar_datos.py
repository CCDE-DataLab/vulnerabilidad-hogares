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
    print(f"Cargando los datos del módulo 100 desde:\n {file_path}\n Columnas a usar: {columns_to_use}")
    df = pd.read_stata(file_path, columns=columns_to_use)

    return df

#Función ETL para el módulo 200:
def load_modulo200(raw_path: Path) -> pd.DataFrame:
    """
    Carga el módulo 200 de la ENAHO Panel, filtrando solo las columnas necesarias para el proyecto
    """

    file_path = raw_path / "enaho01-2015-2019-200-panel.dta"

    formato_base = ["conglome","vivienda","hogar","p203","p204","p207","p208a"]
    columns_to_use = [f"{col}_{year}" for year in range(15, 20) for col in formato_base]

    numpanh_cols = [f"numpanh{year}" for year in range(15, 20)]
    columns_to_use.extend(numpanh_cols)

    columns_to_use.insert(0, "numper")

    print(f"Cargando los datos del módulo 200 desde:\n {file_path}\n Columnas a usar: {columns_to_use}")
    df = pd.read_stata(file_path, columns=columns_to_use)

    rename_dict = {f"numpanh{year}": f"numpanh_{year}" for year in range(15, 20)}
    df.rename(columns=rename_dict, inplace=True)

    return df


# Función ETL para el módulo sumaria:
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

    print("Cargando y transformando el módulo 100\n")
    df_mod100_wide = load_modulo100(RAW_PATH)
    df_mod100_long = transform_wide_to_long(df_mod100_wide, id_cols=["numpanh"])


    print("Cargando y transformando el módulo 200\n")
    df_mod200_wide = load_modulo200(RAW_PATH)
    df_mod200_long = transform_wide_to_long(df_mod200_wide, id_cols=["numper"])


    print("Cargando y transformando el módulo sumaria")
    df_sumaria_wide = load_sumaria(RAW_PATH)
    df_sumaria_long = transform_wide_to_long(df_sumaria_wide, id_cols=["numpanh"])


    ##########################################
    ####AGREGAR MÓDULO 200 (nivel persona a nivel hogar)
    ##########################################
    print("Agregando el módulo 200 a nivel persona a nivel hogar")
    df_miembros = df_mod200_long[df_mod200_long["p204"] == 1].copy()

    df_demografia = df_miembros.groupby(["numpanh","year"]).agg(
        n_miembros = ("numper","count")
    ).reset_index()

    ###########################################
    ##### Merge final de las bases de datos
    ###########################################
    print("Cruzando las bases de datos")

    df_final = pd.merge(df_mod100_long, df_sumaria_long, on=["numpanh","year"], how="left")
    df_final = pd.merge(df_final, df_demografia, on=["numpanh","year"], how="left")

    print(df_final.head())
