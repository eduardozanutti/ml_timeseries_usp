# src/data/split.py

import pandas as pd
from mlforecast.feature_engineering import transform_exog
import os
import logging

class HierarchicalTrainTestSplit:
    def __init__(self,Y_df,S_df,tags,Y_exogen_cat_df, Y_exogen_num_df,split_dates,config,dataset_type='global'):
        self.S_df = S_df
        self.tags = tags
        self.Y_df = Y_df
        self.Y_exogen_cat_df = Y_exogen_cat_df
        self.Y_exogen_num_df = Y_exogen_num_df
        self.split_dates = split_dates
        self.train_dates = self.split_dates.get('train',{})
        self.test_dates = self.split_dates.get('test',{})

        self.train_start = self.train_dates.get('start',{})
        self.train_end = self.train_dates.get('end',{})

        self.test_start = self.test_dates.get('start',{})
        self.test_end = self.test_dates.get('end',{})
        self.test_horizon = len(pd.date_range(start=self.test_start, end=self.test_end, freq='MS'))
        
        self.time_col = config.get('time_col','ds')
        self.id_col = config.get('id_col','unique_id')
        self.lags = config.get('feature_engineering',{}).get('exogen_lag_features',{}).get('lags',[])
        self.holidays = config.get('features',{}).get('holiday_features',{})
        self.processed_path = config.get('paths',{}).get('data',{}).get('processed_path','data/processed')
        self.dataset_type = dataset_type
        self.train_features_path = config.get('paths',{}).get('features',{}).get('train','features/train')
        self.test_features_path = config.get('paths',{}).get('features',{}).get('test','features/test')

    def hierarchical_train_test_split(self,Y_df):
      """
      Split hierárquico temporal para backtest real.
      Últimas N observações de cada série → test
      Resto → train
      """
      df = Y_df.sort_values(['unique_id', 'ds']).copy()
      
      test = df.groupby(['unique_id']).tail(self.test_horizon)
      
      train = df.drop(test.index)
      
      print(f"Train: {train[self.time_col].min().date()} → {train[self.time_col].max().date()} "
            f"({train[self.time_col].nunique()} meses)")
      print(f"Test (backtest): {test[self.time_col].min().date()} → {test[self.time_col].max().date()} "
            f"({test[self.time_col].nunique()} meses)")
      
      return train, test
    
    def create_exog_lag_features(self,Y_exogen_num_df):
       before_num_features = Y_exogen_num_df.drop(columns=[self.id_col,self.time_col]).columns
       
       Y_exogen_num_with_lags = transform_exog(
                df = self.Y_exogen_num_df,
                lags=self.lags,
                id_col=self.id_col,
                time_col=self.time_col,
                )
       
       Y_exogen_num_with_lags.drop(columns=before_num_features,inplace=True)
       
       return Y_exogen_num_with_lags
       
    def add_holidays(self,Y_df):

      Y_df[self.time_col] = pd.to_datetime(Y_df[self.time_col])
      
      for event_name, config in self.holidays.items():
            dates = config['dates']
            before = config['window_before']
            after = config['window_after']
            
            # Máscaras iniciais
            before_mask = pd.Series([False] * len(Y_df), index=Y_df.index)
            during_mask = pd.Series([False] * len(Y_df), index=Y_df.index)
            after_mask  = pd.Series([False] * len(Y_df), index=Y_df.index)
            
            for event_date in dates:
                  # Mês do evento
                  during_mask |= (Y_df[self.time_col] == event_date)
                  
                  # Before: meses anteriores (1 mês antes, 2 meses antes, etc.)
                  for i in range(1, before + 1):
                        before_date = event_date - pd.offsets.MonthBegin(i)
                        before_mask |= (Y_df[self.time_col] == before_date)
                  
                  # After: meses posteriores
                  for i in range(1, after + 1):
                        after_date = event_date + pd.offsets.MonthBegin(i)
                        after_mask |= (Y_df[self.time_col] == after_date)
            
            # Cria as colunas (só cria after se window_after > 0)
            Y_df[f'before_{event_name}'] = before_mask.astype(int)
            Y_df[event_name]             = during_mask.astype(int)
            if after > 0:
                  Y_df[f'after_{event_name}'] = after_mask.astype(int)
      
      return Y_df
    
    def save_features(self, df, set='train',filename='train_features.parquet'):
        """
        Salva o dataset final em Parquet.
        """
        dataset_output_path = self.processed_path + self.dataset_type
        if set == 'train':
            train_features_path =  os.path.join(dataset_output_path,self.train_features_path)
            set_train_features_path = os.path.join(train_features_path)
            save_path = os.path.join(set_train_features_path,filename)
        else:
            test_features_path =  os.path.join(dataset_output_path,self.test_features_path)
            set_test_features_path = os.path.join(test_features_path)
            save_path = os.path.join(set_test_features_path,filename)

        try:
            df.to_parquet(save_path, compression='snappy')

        except Exception as e:
            logging.error(f"Erro ao salvar: {e}")

    
    def run(self):
       #init variables
       Y_df = self.Y_df.copy()
       Y_exogen_num_df = self.Y_exogen_num_df.copy()
       Y_exogen_cat_df = self.Y_exogen_cat_df.copy()

       #Add holidays
       Y_df = self.add_holidays(Y_df)
      
       #Train test split
       train, test = self.hierarchical_train_test_split(Y_df)
       
       #Create Exogen Lags
       Y_exogen_num_with_lags = self.create_exog_lag_features(Y_exogen_num_df)
       

       #Create complete trainset
       train = (
                  train
                  .merge(Y_exogen_num_with_lags,on=[self.id_col,self.time_col],how='inner')
                  .merge(Y_exogen_cat_df,on=[self.id_col],how='inner')
                )
       
       #Create complete test
       test = (
                  test
                  .merge(Y_exogen_num_with_lags,on=[self.id_col,self.time_col],how='inner')
            )
       
       self.save_features(Y_exogen_cat_df, set='train',filename='static_features.parquet')
       self.save_features(train, set='train', filename='train_features.parquet')
       self.save_features(test, set='test',filename='test_features.parquet')

       return train, test

       



       

