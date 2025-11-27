from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from pandas.tseries.offsets import DateOffset

class FeaturesBuilder:
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()  # SEMPRE faça copy() na entrada!
        self.config = config
        self.categorical_features = config.get('features', {}).get('categorical_features', [])
        self.numerical_features = config.get('features', {}).get('numerical_features', [])
        self.ds_col = config.get('ds_col',{})
        self.holiday_features = config.get('features', {}).get('holiday_features', {})

        # self.external_features = ...

    def _encode_categorical(self) -> pd.DataFrame:
        if not self.categorical_features:
            return pd.DataFrame(index=self.df.index)
            
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded = encoder.fit_transform(self.df[self.categorical_features])
        cols = encoder.get_feature_names_out(self.categorical_features)
        cols = ['cat_' + col for col in cols]
        return pd.DataFrame(encoded, columns=cols, index=self.df.index)

    def _select_numerical(self) -> pd.DataFrame:
        if not self.numerical_features:
            return pd.DataFrame(index=self.df.index)
        
        num_df = self.df[self.numerical_features].copy()
        num_df.columns = [f"num_{col}" for col in num_df.columns]  # <--- AQUI A MÁGICA
        return num_df
    
    
    def _add_date_features(self):
        df = self.df.copy()

        df['date_month'] = pd.to_datetime(df[self.ds_col]).dt.to_period('M').dt.to_timestamp()

        all_dates = []
        holiday_type_map = {}

        for name, cfg in self.holiday_features.items():
            dates = pd.to_datetime(cfg['dates']).to_period('M').to_timestamp()
            all_dates.extend(dates.tolist())
            for d in dates:
                holiday_type_map[d] = name

        df['dat_is_holiday'] = df['date_month'].isin(all_dates).astype(int)

        # Janelas
        for holiday_name, cfg in self.holiday_features.items():
            dates = pd.to_datetime(cfg['dates']).to_period('M').to_timestamp()
            before = cfg['window_before']
            after = cfg['window_after']

            col_name = f"dat_is_{holiday_name}_window"
            df[col_name] = 0

            for event_date in dates:
                start = event_date - DateOffset(months=before)
                end = event_date + DateOffset(months=after)
                df.loc[df['date_month'].between(start, end), col_name] = 1

            df[f"dat_months_to_{holiday_name}"] = df['date_month'].apply(
                lambda x: min(((d - x).days // 30 for d in dates if d >= x), default=12)
            )

            df[f"dat_months_after_{holiday_name}"] = df['date_month'].apply(
                lambda x: min(((x - d).days // 30 for d in dates if x > d), default=12)
            )

        # remove auxiliar
        df = df.drop(columns=['date_month'])

        # Retorna apenas features criadas
        new_cols = [c for c in df.columns if c.startswith('dat_')]
        return df[new_cols]

    def build_features(self):
        cat_df = self._encode_categorical()
        num_df = self._select_numerical()
        date_df = self._add_date_features()
        
        features_df = pd.concat([cat_df, num_df,date_df], axis=1)
        
        # Atualiza o df principal
        self.df = self.df.drop(columns=self.categorical_features + self.numerical_features)
        self.df = pd.concat([self.df, features_df], axis=1)
        
        return self.df

    def run(self):
        return self.build_features()