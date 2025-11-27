# src/data/make_dataset.py
import os
import pandas as pd
from hierarchicalforecast.utils import aggregate
import logging
import numpy as np
import joblib

# Configura logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetHierarchicalAggregator:
    """
    Classe para tornar os dados hierarquicos.
    """
    def __init__(self, config,df):
        self.df = df
        self.columns = self.df.columns
        self.categorical_features = [col for col in self.columns if col[:4] == 'cat_']
        self.numerical_features = [col for col in self.columns if col[:4] == 'num_']
        self.holiday_features = [col for col in self.columns if col[:4] == 'dat_']
        self.config = config
        self.output_path = self.config.get('data', {}).get('interim_path', 'data/interim/')
        self.hierarchical_spec = self.config.get('hierarchical_spec', [])  # ex: [['total'], ['total/marca'], ...]
        self.hierarchy = self.hierarchical_spec[-1] if self.hierarchical_spec else []
        self.columns_selected = self.config.get('columns_selected', None)
        self.ds_col = self.config.get('ds_col', 'mes')  # Coluna de data
        self.y_col = self.config.get('y_col', 'venda_unidades')  # Coluna alvo
        self.debug = self.config.get('debug', False)

    def hierarchical_aggregation(self, df):
        """
        Realiza agregação hierárquica se ativado.
        """
        if self.debug:
            print("DEBUG: Iniciando agregação hierárquica.")

        df_h = df.dropna()
        df_h = df_h.rename(columns={self.ds_col: 'ds', self.y_col: 'y'})
        df_h['total'] = 'total'
        df_h['ds'] = pd.to_datetime(df_h['ds'])
        df_h = df_h.sort_values('ds')
        
        exog_vars = {col:'sum' for col in self.categorical_features}
        exog_vars.update({col:'mean' for col in self.numerical_features if 'num_preco' in col or 'num_desconto' in col or 'num_prop' in col})
        exog_vars.update({col:'sum' for col in self.numerical_features if 'num_preco' not in col and 'num_desconto' not in col and 'num_prop' not in col})
        exog_vars.update({col:'max' for col in self.holiday_features})

        try:
            y_hier, S, tags = aggregate(
                df=df_h,
                spec=self.hierarchical_spec,
                exog_vars=exog_vars
            )
            logging.info("Agregação hierárquica concluída.")
            if self.debug:
                print(f"DEBUG: Shape Y_df: {y_hier.shape}, S_df: {S.shape}")
                for k, v in tags.items():
                    print(f"{k}: {len(v)} séries")
            return y_hier, S, tags

        except Exception as e:
            logging.error(f"Erro na agregação hierárquica: {e}")
            return None, None, None

    def save_intermediary(self, df,filename='dataset.parquet'):
        """
        Salva o dataset intermediário em Parquet.
        """
        
        if self.debug:
            print(f"DEBUG: Iniciando salvamento em {self.output_path}")

        save_path = os.path.join(self.output_path, filename)

        try:
            df.to_parquet(save_path, compression='snappy')
            if self.debug:
                print("DEBUG: Dataset Salvo.")
        except Exception as e:
            logging.error(f"Erro ao salvar: {e}")
    
    def save_tags(self, data, filename='tags.joblib'):
        """
        Salva dados em formato tags.
        """
        if self.debug:
            print(f"DEBUG: Salvando JSON em {filename}")

        save_path = os.path.join(self.output_path, filename)

        try:
           # Salva EXATAMENTE como era (arrays, dtypes, tudo)
            joblib.dump(data, save_path, compress=3)
            print("tags salvo com joblib → 100% fiel ao original")
            print(f"Tamanho: {os.path.getsize(save_path) / 1024:.1f} KB")
        except Exception as e:
            logging.error(f"Erro ao salvar tags: {e}")
        return
    
    def load_tags(self, filename='tags.joblib'):
        """
        Carrega um arquivo tags e retorna como dict.
        """
        if self.debug:
            print(f"DEBUG: Carregando tags de {filename}")

        load_path = os.path.join(self.output_path, filename)

        try:
            tags = joblib.load(load_path)
            return tags
        except Exception as e:
            logging.error(f"Erro ao carregar tags: {e}")
            return None

    def load_intermediary(self, filename='dataset.parquet'):
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
        Y_df, S_df, tags = self.hierarchical_aggregation(self.df)
        self.save_intermediary(Y_df,filename='hierarchical_dataset.parquet')
        self.save_intermediary(S_df,filename='hierarchical_structure.parquet')
        self.save_tags(tags, filename='hierarchical_tags.joblib')
        Y_df = self.load_intermediary(filename='hierarchical_dataset.parquet')
        S_df = self.load_intermediary(filename='hierarchical_structure.parquet')
        tags = self.load_tags(filename='hierarchical_tags.joblib')
        if self.debug:
            print(f"DEBUG: Pipeline concluído.")
        return Y_df, S_df, tags

if __name__ == "__main__":
    aggregator = self.DatasetHierarchicalAggregator()
    result = aggregator.run()