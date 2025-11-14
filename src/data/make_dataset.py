# src/data/make_dataset.py
import os
import pandas as pd
from hierarchicalforecast.utils import aggregate
import logging

# Configura logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetCreator:
    """
    Classe para preparação de dados, desde extração até preparação inicial.
    Suporta modo hierárquico opcional.
    """
    def __init__(self, config):
        self.config = config
        self.data_path = self.config.get('data', {}).get('raw_path', 'data/raw/')
        self.output_path = self.config.get('data', {}).get('interim_path', 'data/interim/')
        self.keys = self.config.get('keys', [])
        self.exclude_list_by_materials = self.config.get('exclude_materials', [])
        self.min_time_series_length = self.config.get('min_time_series_length', 24)
        self.dtypes_dict = self.config.get('dtypes', {})  # ex: {'mes': 'datetime', 'marca': 'str'}
        self.hierarchical_spec = self.config.get('hierarchical_spec', [])  # ex: [['total'], ['total/marca'], ...]
        self.hierarchical = self.config.get('hierarchical', False)  # Flag do config
        self.columns_selected = self.config.get('columns_selected', None)
        self.ds_col = self.config.get('ds_col', 'mes')  # Coluna de data
        self.y_col = self.config.get('y_col', 'venda_unidades')  # Coluna alvo
        self.save = self.config.get('save_interim',None)
        self.debug = self.config.get('debug',False)

    def read_csv_files(self, material_path):
        """
        Lê CSVs de uma pasta de material.
        """
        if self.debug:
            print(f"DEBUG: Iniciando leitura de arquivos em {material_path}")
            
        try:
            all_files = [f for f in os.listdir(material_path) if f.endswith('.csv')]
            df_list = [pd.read_csv(os.path.join(material_path, file)) for file in all_files]
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df.sort_index(inplace=True)
            logging.info(f"Lidos {len(df_list)} arquivos para material em {material_path}")
            if self.debug:
                print(f"DEBUG: Lidos {len(all_files)} arquivos. Shape do DF combinado: {combined_df.shape}")
            return combined_df
        except Exception as e:
            logging.error(f"Erro ao ler arquivos em {material_path}: {e}")
            return pd.DataFrame()

    def read_all_csv_files(self):
        """
        Lê todos os materiais (pastas) em data_path.
        """
        if self.debug:
            print(f"DEBUG: Iniciando leitura de todos os materiais em {self.data_path}")

        all_materials = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        data_list = []
        for material in all_materials:
            material_path = os.path.join(self.data_path, material)
            df_material = self.read_csv_files(material_path)
            if not df_material.empty:
                data_list.append(df_material)
        if data_list:
            combined = pd.concat(data_list, ignore_index=True)
            if self.debug:
                print(f"DEBUG: Todos materiais lidos. Shape total: {combined.shape}")
            return combined
        else:
            logging.warning("Nenhum dado lido.")
            if self.debug:
                print("DEBUG: Nenhum dado lido.")
            return pd.DataFrame()

    def correcting_dtypes(self, df):
        """
        Corrige tipos de dados baseado em dtypes_dict.
        """
        if self.debug:
            print("DEBUG: Iniciando correção de tipos de dados.")

        for col, dtype in self.dtypes_dict.items():
            if col in df.columns:
                try:
                    if dtype == 'datetime':
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
                    elif dtype == 'str':
                        df[col] = df[col].astype(str)
                    elif dtype == 'int':
                        df[col] = df[col].astype(int)
                    # Adicione mais tipos se necessário
                except Exception as e:
                    logging.error(f"Erro ao converter {col} para {dtype}: {e}")
        if self.debug:
            print(f"DEBUG: Correção de tipos concluída. Tipos atuais: {df.dtypes}")

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

    def remove_materials_no_sales(self, df):
        """
        Remove materiais da lista de exclusão.
        """
        if self.debug:
            print(f"DEBUG: Iniciando remoção de materiais excluídos: {self.exclude_list_by_materials}")
        if self.exclude_list_by_materials:
            before_shape = df.shape
            df = df[~df['material'].isin(self.exclude_list_by_materials)]
            if self.debug:
                print(f"DEBUG: Shape antes: {before_shape}, após: {df.shape}")
        else:
            if self.debug:
                print("DEBUG: Nenhuma lista de exclusão fornecida.")
        return df

    def remove_time_series_min_length(self, df):
        """
        Remove séries temporais com menos que min_time_series_length observações.
        """
        if self.debug:
            print(f"DEBUG: Iniciando filtro de séries com mínimo {self.min_time_series_length} observações.")
        counts = df.groupby(self.keys).size().reset_index(name='n_obs')
        if self.debug:
            print(f"DEBUG: Séries totais: {len(counts)}. Séries curtas: {(counts['n_obs'] < self.min_time_series_length).sum()}")
        valid = counts[counts['n_obs'] >= self.min_time_series_length][self.keys]
        df = df.merge(valid, on=self.keys, how='inner')
        if self.debug:
            print(f"DEBUG: Filtro concluído. Shape final: {df.shape}")
        return df

    def hierarchical_aggregation(self, df):
        """
        Realiza agregação hierárquica se ativado.
        """
        if self.debug:
            print("DEBUG: Iniciando agregação hierárquica.")

        for col in self.hierarchical_spec:
            hier_col = col[-1]
            if hier_col != 'total':
                self.columns_selected.append(hier_col)
        df_h = df[self.columns_selected].copy()
        df_h = df_h.dropna() ####
        if self.debug:
            print(f"DEBUG: Após dropna, shape: {df_h.shape}")
        df_h = df_h.rename(columns={self.ds_col: 'ds', self.y_col: 'y'}) ###
        df_h['total'] = 'total' ####
        df_h['ds'] = pd.to_datetime(df_h['ds'])
        df_h = df_h.sort_values('ds')

        try:
            Y_df, S_df, tags = aggregate(df=df_h, spec=self.hierarchical_spec)
            logging.info("Agregação hierárquica concluída.")
            if self.debug:
                print(f"DEBUG: Shape Y_df: {Y_df.shape}, S_df: {S_df.shape}")
            print("Níveis criados:")
            for k, v in tags.items():
                print(f"{k}: {len(v)} séries")
            return Y_df, S_df, tags
        except Exception as e:
            logging.error(f"Erro na agregação hierárquica: {e}")
            return None, None, None

    def save_intermediary(self, df_or_tuple, path='data/interim/dataset.parquet'):
        """
        Salva o dataset intermediário em Parquet.
        """
        if self.debug:
            print(f"DEBUG: Iniciando salvamento em {self.output_path}")

        try:
            full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), path)
            if isinstance(df_or_tuple, tuple):  # Modo hierárquico
                Y_df, S_df, _ = df_or_tuple
                Y_df.to_parquet(full_path.replace('.parquet', '_Y.parquet'), compression='snappy')
                S_df.to_parquet(full_path.replace('.parquet', '_S.parquet'), compression='snappy')
                if self.debug:
                    print("DEBUG: Salvos Y_df e S_df em Parquet.")
            else:  # Modo flat
                df_or_tuple.to_parquet(full_path, compression='snappy')
                if self.debug:
                    print("DEBUG: Salvo DF flat em Parquet.")
            logging.info(f"Dataset salvo em {full_path}")
        except Exception as e:
            logging.error(f"Erro ao salvar: {e}")

    def run(self):
        """
        Executa o pipeline completo.
        Retorna DF flat ou (Y_df, S_df, tags) se hierárquico.
        """
        if self.debug:
            print("DEBUG: Iniciando pipeline completo (run).")
        df = self.read_all_csv_files()
        if df.empty:
            return None

        df = self.correcting_dtypes(df)
        df = self.stocks_removing_negatives(df)
        df = self.remove_materials_no_sales(df)
        df = self.remove_time_series_min_length(df)

        if self.hierarchical:
            result = self.hierarchical_aggregation(df)
        else:
            result = df,None,None

        if self.save:
            self.save_intermediary(result)

        if self.debug:
            print("DEBUG: Pipeline concluído.")
        return result

if __name__ == "__main__":
    creator = DatasetCreator()
    result = creator.run()