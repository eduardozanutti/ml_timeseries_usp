# src/data/make_dataset.py
import os
import pandas as pd
from hierarchicalforecast.utils import aggregate
import logging
import numpy as np

# Configura logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetCreator:
    """
    Classe para preparação de dados, desde extração até preparação inicial.
    Suporta modo hierárquico opcional.
    """
    def __init__(self, config):
        self.config = config
        self.weather_features = self.config.get('features',{}).get('weather_features',{}).get('api',{}).get('open-meteo',{}).get('features',{})
        self.data_path = self.config.get('paths', {}).get('data', {}).get('raw_path', 'data/raw/')
        self.output_path = self.config.get('paths', {}).get('data', {}).get('interim_path', 'data/interim/')
        self.min_time_series_length = self.config.get('min_time_series_length', 24)
        self.dtypes_dict = self.config.get('dtypes', {})  # ex: {'mes': 'datetime', 'marca': 'str'}
        self.dtypes_dict_enterprise = self.dtypes_dict.get('enterprise',{})
        self.dtypes_dict_weather = self.dtypes_dict.get('weather',{})
        self.hierarchical_spec = self.config.get('hierarchical_spec', [])  # ex: [['total'], ['total/marca'], ...]
        self.hierarchy = self.hierarchical_spec[-1] if self.hierarchical_spec else []
        self.columns_selected = self.config.get('columns_selected', None)
        self.ds_col = self.config.get('ds_col', 'mes')  
        self.y_col = self.config.get('y_col', 'venda_unidades')  
        self.time_col = self.config.get('time_col', 'ds')
        self.target_col = self.config.get('target_col', 'y')
        self.debug = self.config.get('debug', False)
        self.test_cutoff_start = self.config.get('split_dates',{}).get('test',{}).get('start',{})
        self.test_cutoff_end = self.config.get('split_dates',{}).get('test',{}).get('end',{})
        self.weather_col_names = self.config.get('columns_names','').get('weather','')

    def read_csv_files(self, path):
        """
        Lê CSVs de uma pasta de material.
        """
        all_files = os.listdir(path)
        df_list = []
        for file in all_files:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(path, file),dtype=str)####
                df_list.append(df)
        combined_df = pd.concat(df_list)
        combined_df.sort_index(inplace=True)
        return combined_df

    def read_all_material_csv_files(self):
        all_managements = [folder for folder in os.listdir(self.data_path) if folder != 'weather']
        data_list = []
        for management in all_managements:
            management_path = os.path.join(self.data_path,management)
            all_materials = os.listdir(management_path)
            for material in all_materials:
                material_path = os.path.join(management_path, material)
                if os.path.isdir(material_path):
                    data_list.append(self.read_csv_files(material_path))
        df = pd.concat(data_list)
        return df
    
    def read_all_weather_csv_files(self):
        weathers = [folder for folder in os.listdir(self.data_path) if folder == 'weather']
        weathers_path = os.path.join(self.data_path, weathers[0])
        df_weather = self.read_csv_files(weathers_path)
        return df_weather

    def correcting_dtypes(self,df,source='enterprise'):
        """
        Corrige tipos de dados baseado em dtypes_dict.
        """
        if self.debug:
            print("DEBUG: Iniciando correção de tipos de dados.")

        if source == 'enterprise':
            dtypes = self.dtypes_dict_enterprise
        elif source == 'weather':
            dtypes = self.dtypes_dict_weather 
        else:
            pass
        for col, dtype in dtypes.items():
            if col in df.columns:
                try:
                    if dtype == 'datetime':
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
                    elif dtype == 'str':
                        df[col] = df[col].astype(str)
                    elif dtype == 'int':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                    elif dtype == 'float':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                    # Adicione mais tipos se necessário
                except Exception as e:
                    logging.error(f"Erro ao converter {col} para {dtype}: {e}")
        if self.debug:
            print(f"DEBUG: Correção de tipos concluída. Tipos atuais: {df.dtypes}")

        return df
    
    def clean_string_columns(self,df):
        string_cols = [col for col,type in df.dtypes.items() if type == 'O']
        for col in string_cols:
            df[col] = df[col].str.strip()                          # remove espaços no início/fim
            df[col] = df[col].str.lower()                          # tudo minúsculo
            df[col] = df[col].str.normalize('NFKD')                # remove acentos
            df[col] = df[col].str.encode('ascii', errors='ignore') # remove caracteres especiais
            df[col] = df[col].str.decode('utf-8')
            df[col] = df[col].str.replace(r'[^a-z0-9\s]', ' ', regex=True)  # só letras, números e espaço
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)          # múltiplos espaços → 1 só
            df[col] = df[col].str.replace('-', '', regex=False).str.replace('.', '', regex=False)
            if col == 'loja':
                df[col] = df[col].str.replace(' ', '', regex=False)
        return df

    def stocks_removing_negatives(self, df):
        """
        Substitui negativos em colunas de estoque por 0.
        """
        if self.debug:
            print("DEBUG: Iniciando remoção de negativos em colunas de estoque.")

        stock_cols = [col for col in df.columns if 'estoque' in col.lower()]
        for col in stock_cols:
            if self.debug:
                print(f"DEBUG: Processando coluna {col}. Negativos antes: {(df[col] < 0).sum()}")
            df[col] = df[col].clip(lower=0)
        if self.debug:
            print("DEBUG: Remoção de negativos concluída.")
        return df

    def ajustes_preco_features(self,df):
        if self.debug:
            print("DEBUG: Iniciando ajustes de preço para a criação de features.")
    
        df['preco_unitario'] = df['venda_valor_total']/df['venda_unidades']

        # Primeiro preenche com qualquer coisa que tenha (mesmo que 50% vazio)
        df['preco_unitario'] = df['preco_unitario'].fillna(df['preco_atual'])
        df['preco_unitario'] = df['preco_unitario'].fillna(df['preco_original'])

        # Usa mediana POR SKU + LOJA
        mediana_sku_loja = df.groupby(['sku', 'loja'])['preco_unitario'].transform('median')

        df['preco_unitario'] = df['preco_unitario'].fillna(mediana_sku_loja)

        # Se ainda faltar media SKU (poucos casos)
        mediana_sku = df.groupby('sku')['preco_unitario'].transform('median')
        df['preco_unitario'] = df['preco_unitario'].fillna(mediana_sku)

        # Último recurso: mediana geral
        df['preco_unitario'] = df['preco_unitario'].fillna(df['preco_unitario'].median())

        df['desconto_percentual'] = np.where(
        df['preco_original'].notna() & (df['preco_original'] > 0),
        (df['preco_original'] - df['preco_unitario']) / df['preco_original'],
        0   # onde não tinha preco_original → assume zero desconto
        )

        df['desconto_percentual'] = np.where(df['desconto_percentual']<0,0,df['desconto_percentual'])

        df['desconto_absoluto'] = df['preco_original'] - df['preco_unitario']
        df['desconto_absoluto'] = df['desconto_absoluto'].fillna(0)
        #garantir que não tenha desconto negativo
        df['desconto_absoluto'] = np.where(df['desconto_absoluto']<0,0,df['desconto_absoluto'])

        # 2. Features de desconto + dummies (o que realmente importa)
        df['tem_preco_original'] = df['preco_original'].notna().astype(int)

        df.drop(columns=['preco_atual','preco_original','diff_preco','etiqueta'],inplace=True)
        if self.debug:
            print("DEBUG: Ajustes de preço concluídos.")
        return df

    def ajustar_decotes(self,df):
        if self.debug:
            print("DEBUG: Iniciando ajuste de decotes.")
        df['decote'] = df['decote'].replace({
        'decote u': 'u',
        'decote v': 'v'
        })
        df['decote'] = df['decote'].fillna('sem decote')
        if self.debug:
            print("DEBUG: Ajuste de decotes concluído.")
    
        return df
    
    def create_total_hierarchy_level(self, df):
        """
        Cria nível 'total' na hierarquia.
        """
        if self.debug:
            print("DEBUG: Criando nível 'total' na hierarquia.")
        df['total'] = 'total'
        if self.debug:
            print("DEBUG: Nível 'total' criado.")
        return df

    def remove_short_series_hierarchical(
        self,
        df: pd.DataFrame,
        min_length: int = 42
    ) -> pd.DataFrame:
        """
        Remove séries temporais curtas de forma HIERÁRQUICA (bottom-up).
        
        Parameters
        ----------
        df : pd.DataFrame
            Dados com colunas de hierarquia + coluna de tempo.
        hierarchy : list[str]
            Lista ORDENADA do nível mais baixo até o mais alto.
            Ex: ['sku', 'material', 'produto', 'loja', 'clima_loja', 'regiao_loja', 'centro_distribuicao']
        time_col : str
            Nome da coluna de tempo (não usada no groupby, só pra clareza).
        min_length : int
            Número mínimo de observações por série.
            
        Returns
        -------
        pd.DataFrame
            DataFrame filtrado (apenas séries válidas em todos os níveis).
        """
        if self.debug:
            print("DEBUG: Iniciando remoção hierárquica de séries curtas.")

        df = df.copy()  # não modifica o original
        
        # Percorre do nível mais baixo até o mais alto (bottom-up)
        for i in range(len(self.hierarchy) - 1, -1, -1):  # ex: de 6 até 0
            current_level_cols = self.hierarchy[:i + 1]   # ['sku'], ['sku','material'], ..., todas
            
            print(f"Filtrando nível: {current_level_cols} (>= {min_length} obs)")
            
            # Conta observações por combinação atual
            counts = df.groupby(current_level_cols).size()
            
            # Mantém apenas as combinações com pelo menos min_length observações
            valid_combinations = counts[counts >= min_length].index
            
            # Filtra o DataFrame (muito mais rápido que merge!)
            df = df.set_index(current_level_cols).loc[valid_combinations].reset_index()
            
            print(f"   → {len(counts):,} combinações → {len(valid_combinations):,} mantidas "
                f"({100*len(valid_combinations)/max(len(counts),1):.1f}%)")
        if self.debug:
            print(f"\nLimpeza concluída! Linhas finais: {len(df):,}")
        return df
    
    def drop_discontinued_series(self,df):
        max_dates = df.groupby(['sku', 'loja'])['mes'].max()
        
        descontinuados = max_dates[max_dates < self.test_cutoff_start].index

        df = df[~df.set_index(['sku', 'loja']).index.isin(descontinuados)]
        
        return df
    
    def drop_recent_series_with_no_training(self,df):
        min_dates = df.groupby(['sku', 'loja'])['mes'].min()

        series_sem_treino = min_dates[min_dates >= self.test_cutoff_start].index

        df = df[~df.set_index(['sku', 'loja']).index.isin(series_sem_treino)]
        return df
    
    def prepare_dataset_h_forecast(self,df):
            df = df.dropna()
            df = df.rename(columns={self.ds_col: self.time_col, self.y_col: self.target_col})
            df['total'] = 'total'
            df[self.time_col] = pd.to_datetime(df[self.time_col])
            df = df.sort_values(self.time_col)
            return df
    
    def correcting_weather_data_names(self,df_weather):
            df_weather.rename(
                columns = self.weather_col_names,inplace=True)
            return df_weather
    
    def standardizing_columns_to_forecast(self,df):
            rename_dict = {
                            self.ds_col:self.time_col,
                            self.y_col:self.target_col
                            }
            df = df.rename(columns = rename_dict)
            return df


    def save_intermediary(self, df,filename='dataset.parquet'):
        """
        Salva o dataset intermediário em Parquet.
        """
        save_path = os.path.join(self.output_path, filename)

        if self.debug:
            print(f"DEBUG: Iniciando salvamento em {save_path}")

        try:
            df.to_parquet(save_path, compression='snappy')
            if self.debug:
                print("DEBUG: Dataset Salvo.")
        except Exception as e:
            logging.error(f"Erro ao salvar: {e}")

    def load_intermediary(self,filename='dataset.parquet'):
        """
        Carrega o dataset intermediário de Parquet.
        """
        if self.debug:
            print(f"DEBUG: Iniciando carregamento de {self.output_path}")

        load_path = os.path.join(self.output_path, filename)
        
        try:
            df = pd.read_parquet(load_path)
            if self.debug:
                print("DEBUG: Dataset Carregado.")
            return df
        except Exception as e:
            logging.error(f"Erro ao carregar: {e}")
            return None

    def run(self):
        """
        Executa o pipeline completo.
        Retorna DF flat ou (Y_df, S_df, tags) se hierárquico.
        """
        if self.debug:
            print("DEBUG: Iniciando pipeline de preparação de dados.")
        
        ## == MATERIAL FILES == ##
        df = self.read_all_material_csv_files()
        df = self.correcting_dtypes(df,source = 'enterprise')
        df = self.clean_string_columns(df)
        df = self.stocks_removing_negatives(df)
        df = self.ajustes_preco_features(df)
        df = self.ajustar_decotes(df)
        df = self.create_total_hierarchy_level(df)
        df = self.drop_recent_series_with_no_training(df)
        df = self.drop_discontinued_series(df)
        df = self.remove_short_series_hierarchical(df,min_length = self.min_time_series_length)
        
        
        ## == WEATHER FILES == ##
        df_weather = self.read_all_weather_csv_files()
        df_weather = self.correcting_dtypes(df_weather,source = 'weather')
        df_weather = self.correcting_weather_data_names(df_weather)
        df_weather = self.clean_string_columns(df_weather)
        
        ## == ENSEMBLE == ##
        df = df.merge(df_weather,on=['mes','uf_loja','cidade_loja','bairro_loja'],how='left')
        df.drop(columns=['latitude','longitude'],inplace=True)
        df = self.standardizing_columns_to_forecast(df)
        
        ## == SAVE == #

        self.save_intermediary(df,filename='dataset.parquet')
        df = self.load_intermediary(filename='dataset.parquet')

        if self.debug:
            print(f"DEBUG: Shape final após limpeza: {df.shape}")
        return df

if __name__ == "__main__":
    creator = DatasetCreator()
    result = creator.run()






      