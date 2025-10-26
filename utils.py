import os
import pandas as pd

def read_csv_files(material):
    path = f"dados/{material}"
    all_files = os.listdir(path)
    df_list = []
    for file in all_files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, file))
            df_list.append(df)
    combined_df = pd.concat(df_list)
    combined_df.sort_index(inplace=True)
    return combined_df

def read_all_csv_files():
    base_path = "dados/"
    all_materials = os.listdir(base_path)
    data_list = []
    for material in all_materials:
        material_path = os.path.join(base_path, material)
        if os.path.isdir(material_path):
            data_list.append(read_csv_files(material))
    df = pd.concat(data_list)
    return df

def correcting_dtypes(df):
    df['mes'] = pd.to_datetime(df['mes'], format='%Y-%m-%d')
    df['marca'] = df['marca'].astype(str)
    df['idade_loja'] = df['idade_loja'].astype(int)
    df['clima_loja'] = df['clima_loja'].astype(str)
    df['porte_loja'] = df['porte_loja'].astype(str)
    df['perfil_loja'] = df['perfil_loja'].astype(str)
    return df

def stocks_removing_negatives(df):
    columns = df.columns
    for col in columns:
        if 'estoque' in col:
            df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
    return df