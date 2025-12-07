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
    def __init__(self, config,df=None, dataset_type='global'):
        self.df = df
        self.dataset_type = dataset_type
        self.config = config
        self.features = self.config.get('features',{})
        self.categorical_features = self.features.get('categorical_features',[])
        self.numerical_features = self.features.get('numerical_features',[])
        self.path = self.config.get('paths', {}).get('data', {}).get('processed_path', 'data/processed/')
        self.hierarchical_spec = self.config.get('hierarchical_spec', [])  # ex: [['total'], ['total/marca'], ...]
        self.seen = set()
        self.hierarchy = [col for cols in self.hierarchical_spec for col in cols if col not in self.seen and not self.seen.add(col)]
        self.columns_selected = self.config.get('columns_selected', None)
        self.time_col = self.config.get('time_col', 'ds')  # Coluna de data
        self.target_col = self.config.get('target_col', 'y')  # Coluna alvo
        self.id_col = self.config.get('id_col','unique_id')
        self.debug = self.config.get('debug', False)
        self.aggregation_rules = self.config.get('aggregation_rules',None)
        self.categorical_rules = self.aggregation_rules.get('categorical',None)
        self.numerical_rules = self.aggregation_rules.get('numerical',None)
        self.time_col = config.get('time_col','ds')
        self.lags = config.get('feature_engineering',{}).get('exogen_lag_features',{}).get('lags',[])
        self.split_dates = config.get('split_dates',{})
        self.test_dates = self.split_dates.get('test',{})
        self.test_start = self.test_dates.get('start',{})
        self.test_end = self.test_dates.get('end',{})
        self.test_horizon = len(pd.date_range(start=self.test_start, end=self.test_end, freq='MS'))

    def hierarchical_aggregation(self,df):
        """
        Realiza agregação hierárquica se ativado.
        """
        if self.debug:
            print("DEBUG: Iniciando agregação hierárquica.")

        try:
            y_hier, S, tags = aggregate(
                df=df,
                spec=self.hierarchical_spec
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
    
    def hierarchical_exogen_aggregation(self, df_exogen_features, agg_func,static=None):

        # Gera todos os níveis
        # Gera todos os níveis
        levels = [self.hierarchy [:i+1] for i in range(len(self.hierarchy ))]

        dfs = []

        for level in levels:
            #Agregação a temporal
            if static:
                cols_to_keep = [self.id_col] + list(agg_func.keys())

                agg = (
                    df_exogen_features.groupby(level, observed=True)
                    .agg(agg_func)
                    .reset_index()
                )
                agg[self.id_col] = agg[level].astype(str).agg('/'.join, axis=1)
                dfs.append(agg[cols_to_keep])
                Y_exogen_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
            #agregação temporal
            else:
                cols_to_keep = [self.time_col, self.id_col] + list(agg_func.keys())
                agg = (
                    df_exogen_features.groupby([self.time_col] + level, observed=True)
                    .agg(agg_func)
                    .reset_index()
                )
            # Cria o unique_id bonitinho
                agg[self.id_col] = agg[level].astype(str).agg('/'.join, axis=1)
                dfs.append(agg[cols_to_keep])
                Y_exogen_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

        return Y_exogen_df
    
    def split_target_exog_features(self,Y_df):
            exogen_features = [col for col in Y_df.columns if col not in [self.time_col,self.target_col]+self.hierarchy]
            
            df_exogen_features = Y_df.drop(columns=[self.target_col])
            #CATEGORICAL FEATURES
            df_exogen_categorical_features = df_exogen_features.drop(columns=self.numerical_features)
            
            
            #NUMERIC FEATURES
            df_exogen_numerical_features = df_exogen_features.drop(columns=self.categorical_features)
            
            Y_df = Y_df.drop(columns=exogen_features)
            
            return Y_df,df_exogen_categorical_features,df_exogen_numerical_features
    

    def make_future_dataframe(self,tags):

        ids = [id for tag in tags for id in tags[tag]]

        end_test = pd.Timestamp(self.test_end)

        start_production = end_test + pd.DateOffset(months=1)

        production_dates = pd.date_range(start=start_production, periods=self.test_horizon, freq="MS")

        #CROSS JOIN
        future_dataframe = pd.MultiIndex.from_product(
                                            [ids, production_dates],
                                            names=[self.id_col, self.time_col]
                                            ).to_frame(index=False)           
        return future_dataframe

    def extract_lag_information_to_future(self,Y_exogen_num_df,tags):
           
        production_dataframe = self.make_future_dataframe(tags)

        feature_names = Y_exogen_num_df.drop(columns=[self.id_col,self.time_col])

        for lag in self.lags:
                #EXTRACT LAST LAGS DATES
                lag_feature = Y_exogen_num_df.groupby(self.id_col,group_keys=False).tail(lag)
                #ADJUST DATE TO FUTURE
                lag_feature[self.time_col] = lag_feature[self.time_col] + pd.DateOffset(months=lag)
                #RENAME AS LAG COLUMNS
                lag_feature = lag_feature.rename(columns={feature:feature+f'_lag{lag}'for feature in feature_names})
                #INSERT IN PRODUCTION DATAFRAME
                production_dataframe = production_dataframe.merge(lag_feature,on=[self.id_col,self.time_col],how='inner')
                #The ideia is to exctract h month in lags data to use in production dataset
        return production_dataframe
    
    
    def safe_mode(self,x):
        if len(x) == 0:
            return np.nan
        counts = x.value_counts(dropna=True)
        if len(counts) == 0:
            return np.nan
        return counts.index[0]
    

    def save_processed(self, df,filename='dataset.parquet'):
        """
        Salva o dataset final em Parquet.
        """
        
        if self.debug:
            print(f"DEBUG: Iniciando salvamento em {self.path}")

        dataset_output_path = self.path+self.dataset_type
        save_path = os.path.join(dataset_output_path, filename)

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

        dataset_output_path = self.path+self.dataset_type
        save_path = os.path.join(dataset_output_path, filename)

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
        
        dataset_load_path = self.path+self.dataset_type
        load_path = os.path.join(dataset_load_path, filename)

        try:
            tags = joblib.load(load_path)
            return tags
        except Exception as e:
            logging.error(f"Erro ao carregar tags: {e}")
            return None
    
    def load_processed(self, filename='dataset.parquet'):
        """
        Carrega o dataset intermediário de Parquet.
        """
        dataset_load_path = self.path + self.dataset_type
        load_path = os.path.join(dataset_load_path, filename)

        if self.debug:
            print(f"DEBUG: Iniciando carregamento de {load_path}")

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
        
        Y_df = self.df.copy()


        ## == SPLIT TARGET EXOGEN FEATURES
        Y_df, df_exogen_categorical_features,df_exogen_numerical_features = self.split_target_exog_features(Y_df)


        ## == HIERARCHICAL AGGREGATION ==
        #Rules
        cat_agg_func = {col:rule for rule in self.categorical_rules for col in self.categorical_rules[rule] if rule != 'mode'}
        cat_agg_func.update({col:self.safe_mode for rule in self.categorical_rules for col in self.categorical_rules[rule] if rule == 'mode'})
        num_agg_func = {col:rule for rule in self.numerical_rules for col in self.numerical_rules[rule]}
        #Agregations
        Y_df, S_df, tags = self.hierarchical_aggregation(Y_df) #Target Aggregation
        Y_df[self.id_col] = Y_df[self.id_col].astype('category') #unique_id as category (statis feature)
        #Categorical
        Y_exogen_cat_df = self.hierarchical_exogen_aggregation(df_exogen_categorical_features,agg_func = cat_agg_func, static=True)
        for col in Y_exogen_cat_df.columns:
            Y_exogen_cat_df[col] = Y_exogen_cat_df[col].astype('category') #columns (including unique_id) as category (statis feature)
        #Numerical
        Y_exogen_num_df = self.hierarchical_exogen_aggregation(df_exogen_numerical_features, agg_func = num_agg_func)
        Y_exogen_num_df[self.id_col] = Y_exogen_num_df[self.id_col].astype('category') #unique_id as category (statis feature)

        #FUTURE DATAFRAME
        future_df = self.extract_lag_information_to_future(Y_exogen_num_df,tags)
        future_df = future_df.merge(Y_exogen_cat_df,on=[self.id_col],how='inner')
        
        ## == SAVE DATASET == #
        self.save_processed(Y_df,filename='dataset.parquet')
        self.save_processed(Y_exogen_cat_df,filename='exog_cat_vars.parquet')
        self.save_processed(Y_exogen_num_df,filename='exog_num_vars.parquet')
        self.save_processed(future_df,filename='future.parquet')
        
        if self.dataset_type != 'local':
            self.save_processed(S_df,filename='structure.parquet')
            self.save_tags(tags, filename='tags.joblib')
            
        
        ## == LOAD DATASET == #
        Y_df = self.load_processed(filename='dataset.parquet')
        
        if self.dataset_type != 'local':
            S_df = self.load_processed(filename='structure.parquet')
            tags = self.load_tags(filename='tags.joblib')
        
        Y_exogen_cat_df = self.load_processed(filename='exog_cat_vars.parquet')
        Y_exogen_num_df = self.load_processed(filename='exog_num_vars.parquet')
        future_df = self.load_processed(filename='future.parquet')

        if self.debug:
            print(f"DEBUG: Pipeline concluído.")
        return Y_df, S_df, tags, Y_exogen_cat_df, Y_exogen_num_df, future_df