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

def load_modulo500(raw_path: Path) -> pd.DataFrame:
    """
    Carga el módulo 500 de la Enaho Panel, filtrando solo las columnas necesarias para el proyecto
    """
    file_path = raw_path / "enaho01a-2015-2019-500-panel_01.dta"

    formato_base = ["conglome","vivienda","hogar","p507","p512b"]
    columns_to_use = [f"{col}_{year}" for year in range(15,20) for col in formato_base]

    numpanh_cols = [f"numpanh{year}" for year in range(15,20)]

    columns_to_use.extend(numpanh_cols)
    columns_to_use.insert(0, "numper")

    print(f"Cargando los datos del módulo 500 desde:\n {file_path}\n Columnas a usar: {columns_to_use}")
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

    print("Cargando y transformando el módulo 500\n")
    df_mod500_wide = load_modulo500(RAW_PATH)
    df_mod500_long = transform_wide_to_long(df_mod500_wide, id_cols = ["numper"])


    print("Cargando y transformando el módulo sumaria")
    df_sumaria_wide = load_sumaria(RAW_PATH)
    df_sumaria_long = transform_wide_to_long(df_sumaria_wide, id_cols=["numpanh"])


    ##########################################
    #### AGREGAR MÓDULOS NIVEL PERSONA A NIVEL HOGAR
    ##########################################
    print("Agregando el módulo 200 y 500 a nivel persona a nivel hogar...")

    df_personas = pd.merge(df_mod200_long, df_mod500_long, on=["numpanh", "numper", "year"], how="left")

    # 1. CORRECCIÓN VITAL: Forzar a que la llave de cruce sea numérica en todas las bases 
    # para evitar problemas en los cruces posteriores (el bug de los NaN)
    df_mod100_long["numpanh"] = pd.to_numeric(df_mod100_long["numpanh"], errors="coerce")
    df_sumaria_long["numpanh"] = pd.to_numeric(df_sumaria_long["numpanh"], errors="coerce")
    df_personas["numpanh"] = pd.to_numeric(df_personas["numpanh"], errors="coerce")

    # 2. Filtrar a los miembros reales. 
    # Ojo: pd.read_stata suele cargar los labels como texto. 
    # Usamos .astype(str) para asegurarnos de que atrape el "1" o el "si"
    df_miembros = df_personas[df_personas["p204"].astype(str).str.contains("1|si", case=False, na=False)].copy()
    
    # 3. Variable de ocupación
    df_miembros["ocupado"] = df_miembros["p507"].notna().astype(int)

    # 4. Agrupación base (n_miembros y n_ocupados)
    df_hogar_agregado = df_miembros.groupby(["numpanh", "year"]).agg(
        n_miembros=("numper", "count"),
        n_ocupados=("ocupado", "sum")
    ).reset_index()

    # 5. EXTRACCIÓN DEL JEFE DE HOGAR (Feature Engineering)
    # Filtramos a los jefes (puede venir como '1' o 'jefe/jefa')
    es_jefe = df_miembros["p203"].astype(str).str.lower().str.contains("jefe|1", na=False)
    df_jefes = df_miembros[es_jefe].copy()
    
    # Seleccionamos las variables clave y las renombramos
    df_jefes = df_jefes[["numpanh", "year", "p207", "p208a"]].rename(columns={
        "p207": "sexo_jefe",
        "p208a": "edad_jefe"
    })
    
    # Eliminamos duplicados por si algún hogar encuestado por error tiene 2 jefes
    df_jefes = df_jefes.drop_duplicates(subset=["numpanh", "year"])

    # Agregamos las variables del jefe a nuestro resumen de hogar
    df_hogar_agregado = pd.merge(df_hogar_agregado, df_jefes, on=["numpanh", "year"], how="left")

    ###########################################
    ##### Merge final de las bases de datos
    ###########################################
    print("Cruzando las bases de datos maestras...\n")

    df_final = pd.merge(df_mod100_long, df_sumaria_long, on=["numpanh", "year"], how="left")
    df_final = pd.merge(df_final, df_hogar_agregado, on=["numpanh", "year"], how="left")

    print(f"Columnas finales del modelo:\n{df_final.columns.to_list()}\n")
    print(f"Dimensiones de la base final:\n{df_final.shape}")
    print(df_final.head())


    ###########################################
    #### 4.- Limpieza y creación de la TARGET
    ###########################################
    print("Limpiando nulos y creando la variable objetivo (Target Multinomial)...")
    
    df_final = df_final.dropna(subset=["pobreza"])
    df_final["pobreza_bin"] = np.where(df_final["pobreza"] == "no pobre", 0, 1)

    df_final.sort_values(by=["numpanh", "year"], inplace=True)
    df_final["pobreza_futura"] = df_final.groupby("numpanh")["pobreza_bin"].shift(-1)

    #Eliminamos los casos en los que la diferencia entre el año actual y el siguiente no sea 1
    df_final["year_futuro"] = df_final.groupby("numpanh")["year"].shift(-1)
    df_final = df_final[df_final["year_futuro"] - df_final["year"] == 1]
    df_final = df_final.drop(columns = ["year_futuro"],axis=1)
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


    columnas_texto = df_final.select_dtypes(include=["object","category"]).columns
    df_final[columnas_texto] = df_final[columnas_texto].astype(str)
    ruta_bd = PROCESSED_PATH / "data_pobreza.parquet"
    df_final.to_parquet(ruta_bd,index = False)
    print(f"\nBase exportada con éxito en: {ruta_bd}")