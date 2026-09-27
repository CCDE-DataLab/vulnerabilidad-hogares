# %% [1] Importaciones y Configuración de Rutas
import pandas as pd
import numpy as np
import os 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
RAW_PATH = DATA_PATH / "raw"
PROCESSED_PATH = DATA_PATH / "processed"

# %% [2] Funciones de Carga de Módulos ENAHO
def load_modulo100(raw_path: Path) -> pd.DataFrame:
    """
    Carga el módulo 100 de la ENAHO Panel, filtrando solo las columnas necesarias para el proyecto
    """
    file_path = raw_path / "enaho01-2015-2019-100-panel.dta"
    formato_base = ["conglome", "vivienda", "hogar", "p101", "p110"]
    columns_to_use = [f"{col}_{year}" for year in range(15, 20) for col in formato_base]
    columns_to_use.insert(0, "numpanh")
    print(f"Cargando los datos del módulo 100 desde:\n {file_path}\n Columnas a usar: {columns_to_use}")
    df = pd.read_stata(file_path, columns=columns_to_use)
    return df

def load_modulo200(raw_path: Path) -> pd.DataFrame:
    """
    Carga el módulo 200 de la ENAHO Panel, filtrando solo las columnas necesarias para el proyecto
    """
    file_path = raw_path / "enaho01-2015-2019-200-panel.dta"
    formato_base = ["conglome", "vivienda", "hogar", "p203", "p204", "p207", "p208a"]
    columns_to_use = [f"{col}_{year}" for year in range(15, 20) for col in formato_base]
    numpanh_cols = [f"numpanh{year}" for year in range(15, 20)]
    columns_to_use.extend(numpanh_cols)
    columns_to_use.insert(0, "numper")

    print(f"Cargando los datos del módulo 200 desde:\n {file_path}\n Columnas a usar: {columns_to_use}")
    df = pd.read_stata(file_path, columns=columns_to_use)
    rename_dict = {f"numpanh{year}": f"numpanh_{year}" for year in range(15, 20)}
    df.rename(columns=rename_dict, inplace=True)
    return df

def load_modulo500(raw_path: Path) -> pd.DataFrame:
    """
    Carga el módulo 500 de la ENAHO Panel, filtrando solo las columnas necesarias para el proyecto
    """
    file_path = raw_path / "enaho01a-2015-2019-500-panel_01.dta"
    formato_base = ["conglome", "vivienda", "hogar", "p507", "p512b"]
    columns_to_use = [f"{col}_{year}" for year in range(15, 20) for col in formato_base]
    numpanh_cols = [f"numpanh{year}" for year in range(15, 20)]
    columns_to_use.extend(numpanh_cols)
    columns_to_use.insert(0, "numper")

    print(f"Cargando los datos del módulo 500 desde:\n {file_path}\n Columnas a usar: {columns_to_use}")
    df = pd.read_stata(file_path, columns=columns_to_use)
    rename_dict = {f"numpanh{year}": f"numpanh_{year}" for year in range(15, 20)}
    df.rename(columns=rename_dict, inplace=True)
    return df

def load_sumaria(raw_path: Path) -> pd.DataFrame:
    """
    Carga el módulo sumaria de la ENAHO Panel
    """
    file_path = raw_path / "sumaria-2015-2019-panel.dta"
    formato_base = ["pobreza", "linea", "gashog2d", "inghog2d"]
    columns_to_use = [f"{col}_{year}" for year in range(15, 20) for col in formato_base]
    columns_to_use.insert(0, "numpanh")
    print(f"Cargando los datos de la sumaria desde {file_path}\n Columnas a usar: {columns_to_use}")
    df = pd.read_stata(file_path, columns=columns_to_use)
    return df

# %% [3] Función de Transformación Wide a Long
def transform_wide_to_long(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    """
    Transforma el DataFrame de formato ancho a formato largo

    Args:
        df (pd.DataFrame): DataFrame en formato ancho
        id_cols (list): Lista de columnas que se mantendrán como identificadores
    """
    columnas_temporales = [col for col in df.columns if col not in id_cols]
    stubs = list(set([col.rsplit("_", 1)[0] for col in columnas_temporales if "_" in col]))

    df = pd.wide_to_long(
        df,
        stubnames=stubs,
        i=id_cols,
        j="year",
        sep="_",
        suffix=r"\d+"
    ).reset_index()
    return df

# %% [4] Ejecución del Pipeline ETL
if __name__ == "__main__":
    print("Iniciando bloque ETL para procesamiento de datos...")

    # Carga y transformación modular
    print("Cargando y transformando el módulo 100\n")
    df_mod100_wide = load_modulo100(RAW_PATH)
    df_mod100_long = transform_wide_to_long(df_mod100_wide, id_cols=["numpanh"])

    print("Cargando y transformando el módulo 200\n")
    df_mod200_wide = load_modulo200(RAW_PATH)
    df_mod200_long = transform_wide_to_long(df_mod200_wide, id_cols=["numper"])

    print("Cargando y transformando el módulo 500\n")
    df_mod500_wide = load_modulo500(RAW_PATH)
    df_mod500_long = transform_wide_to_long(df_mod500_wide, id_cols=["numper"])

    print("Cargando y transformando el módulo sumaria")
    df_sumaria_wide = load_sumaria(RAW_PATH)
    df_sumaria_long = transform_wide_to_long(df_sumaria_wide, id_cols=["numpanh"])

    # %% [5] Agregación a Nivel Hogar (Módulos 200 y 500)
    print("Agregando el módulo 200 y 500 a nivel persona a nivel hogar...")
    df_personas = pd.merge(df_mod200_long, df_mod500_long, on=["numpanh", "numper", "year"], how="left")

    df_mod100_long["numpanh"] = pd.to_numeric(df_mod100_long["numpanh"], errors="coerce")
    df_sumaria_long["numpanh"] = pd.to_numeric(df_sumaria_long["numpanh"], errors="coerce")
    df_personas["numpanh"] = pd.to_numeric(df_personas["numpanh"], errors="coerce")

    # Filtrar a los miembros reales
    df_miembros = df_personas[df_personas["p204"].astype(str).str.contains("1|si", case=False, na=False)].copy()
    df_miembros["ocupado"] = df_miembros["p507"].notna().astype(int)

    # Agrupación base (n_miembros y n_ocupados)
    df_hogar_agregado = df_miembros.groupby(["numpanh", "year"]).agg(
        n_miembros=("numper", "count"),
        n_ocupados=("ocupado", "sum")
    ).reset_index()

    # Extracción del jefe de hogar
    es_jefe = df_miembros["p203"].astype(str).str.lower().str.contains("jefe|1", na=False)
    df_jefes = df_miembros[es_jefe].copy()
    df_jefes = df_jefes[["numpanh", "year", "p207", "p208a"]].rename(columns={
        "p207": "sexo_jefe",
        "p208a": "edad_jefe"
    })
    df_jefes = df_jefes.drop_duplicates(subset=["numpanh", "year"])
    df_hogar_agregado = pd.merge(df_hogar_agregado, df_jefes, on=["numpanh", "year"], how="left")

    # %% [6] Cruce Maestro de Módulos (Merge Final)
    print("Cruzando las bases de datos maestras...\n")
    df_final = pd.merge(df_mod100_long, df_sumaria_long, on=["numpanh", "year"], how="left")
    df_final = pd.merge(df_final, df_hogar_agregado, on=["numpanh", "year"], how="left")

    print(f"Columnas finales del modelo:\n{df_final.columns.to_list()}\n")
    print(f"Dimensiones de la base final:\n{df_final.shape}")
    print(df_final.head())

    # %% [7] Construcción del Target Multinomial y Exportación
    print("Limpiando nulos y creando la variable objetivo (Target Multinomial)...")
    df_final = df_final.dropna(subset=["pobreza"])
    df_final["pobreza_bin"] = np.where(df_final["pobreza"] == "no pobre", 0, 1)

    df_final.sort_values(by=["numpanh", "year"], inplace=True)
    df_final["pobreza_futura"] = df_final.groupby("numpanh")["pobreza_bin"].shift(-1)

    # Eliminamos los casos en los que la diferencia entre el año actual y el siguiente no sea 1
    df_final["year_futuro"] = df_final.groupby("numpanh")["year"].shift(-1)
    df_final = df_final[df_final["year_futuro"] - df_final["year"] == 1]
    df_final = df_final.drop(columns=["year_futuro"])

    # Creación de variable target multicategórica
    condiciones = [
        ((df_final["pobreza_bin"] == 0) & (df_final["pobreza_futura"] == 0)),
        ((df_final["pobreza_bin"] == 0) & (df_final["pobreza_futura"] == 1)),
        ((df_final["pobreza_bin"] == 1) & (df_final["pobreza_futura"] == 0)),
        ((df_final["pobreza_bin"] == 1) & (df_final["pobreza_futura"] == 1))
    ]
    # 0: Resiliente, 1: Cae, 2: Escapa, 3: Crónico
    resultados = [0, 1, 2, 3]
    df_final["target_multinomial"] = np.select(condiciones, resultados, default=np.nan)

    df_final.dropna(subset=["pobreza_futura"], inplace=True)

    print(f"\nDimensiones de la base lista para el modelo: {df_final.shape}")
    print("\nDistribución del Target Multinomial:")
    print("0: Se mantiene No Pobre | 1: Cae en Pobreza | 2: Sale de Pobreza | 3: Se mantiene Pobre")
    print(df_final["target_multinomial"].value_counts().sort_index())

    columnas_texto = df_final.select_dtypes(include=["object", "category"]).columns
    df_final[columnas_texto] = df_final[columnas_texto].astype(str)
    ruta_bd = PROCESSED_PATH / "data_pobreza.parquet"
    df_final.to_parquet(ruta_bd, index=False)
    print(f"\nBase exportada con éxito en: {ruta_bd}")
