# src/models/tuning.py

from re import S
import optuna
import pandas as pd
from mlforecast import MLForecast
from statsforecast import StatsForecast
from mlforecast.optimization import mlforecast_objective
from utilsforecast.losses import rmse, mae, smape, mase,rmsse,bias
# Mapeamento string → objeto real (o coração do sistema)
from src.utils.presets import build_lag_transforms, build_target_transforms
from src.utils.optuna_presets import get_sampler_map, get_pruner_map
from src.utils.model_registry import get_model_class
import logging

class ModelTuning:
    def __init__(self, df, config, model_name,fixed_params,param_space,cv_config,mlforecast_params,tuning_metric):
        self.df = df.copy()
        self.model_name = model_name
        self.model = get_model_class(self.model_name)
        self.fixed_params = fixed_params
        self.param_space = param_space
        self.mlf_n_window = cv_config.get('n_windows',{})
        self.mlf_step_size = cv_config.get('step_size',{})
        self.mlf_h = cv_config.get('h',{})
        self.mlf_id_col = mlforecast_params.get('id_col',{})
        self.mlf_freq = mlforecast_params.get('freq',{})
        self.optuna_direction_cfg = config['optuna']['direction']
        self.optuna_sampler_cfg = get_sampler_map(config['optuna']['sampler'])
        self.optuna_pruner_cfg = get_pruner_map(config['optuna']['pruner'])
        self.optuna_n_trials_cfg = config['optuna']['n_trials']
        self.mlforecast_presets = mlforecast_params['presets']
        self.search_lag_preset = mlforecast_params['tuning']['search_lag_preset']
        self.search_target_preset = mlforecast_params['tuning']['search_target_preset']
        self.search_date_preset = mlforecast_params['tuning']['search_date_preset']
        self.active_lag_preset = mlforecast_params['tuning']['active_lag_preset']
        self.active_data_preset = mlforecast_params['tuning']['active_date_preset']
        self.active_target_preset = mlforecast_params['tuning']['active_target_preset']
        self.active_date_preset = mlforecast_params['tuning']['active_date_preset']
        self.tuning_metric = tuning_metric
        logging.basicConfig(level=logging.DEBUG)  # No init ou global

    def get_mlf_init_params(self):
        if self.search_lag_preset:
            lag_strategy = self.mlforecast_presets['lag_strategies'][self.active_lag_preset]
            lags = lag_strategy['lags']
            transforms = lag_strategy['transforms']
            transforms_dict = {}
            for (lag,transf) in transforms.items():
                lag_transf = build_lag_transforms({lag:transf})
                transforms_dict.update(lag_transf)
        else:
            transforms_dict = {}

        if self.search_target_preset:
            target_strategy = self.mlforecast_presets['target_transforms'][self.active_target_preset]
            target_transforms = build_target_transforms(target_strategy)
        else:
            target_transforms = []

        if self.search_lag_preset:
            date_features = self.mlforecast_presets['date_features'][self.active_date_preset]


        mlf_init_params = {
            'lags': lags,
            'lag_transforms': transforms_dict,
            'target_transforms': target_transforms,
            'date_features': date_features
            }
        return mlf_init_params
            
    def get_mlf_fit_params(self):
        return {'static_features': []}
    
    def get_model_params(self, trial):
        model_params = self.fixed_params.copy()  # Cópia para evitar mutar
        
        for type_, space in self.param_space.items():
            for name, params in space.items():
                param = params.get('param', name)
                low = float(params['low'])  # Force float para '1e-8' str
                high = float(params['high'])
                log = bool(params.get('log', False))
                
                if type_ == 'int':
                    low = int(low)
                    high = int(high)
                    model_params[name] = trial.suggest_int(param, low, high, log=log)
                elif type_ == 'float':
                    model_params[name] = trial.suggest_float(param, low, high, log=log)
                else:
                    raise ValueError(f"Tipo não suportado: {type_}")
        
        return model_params

    def config_fn(self, trial):
        space = {
            'model_params': self.get_model_params(trial),
            'mlf_init_params': self.get_mlf_init_params(),
            'mlf_fit_params': self.get_mlf_fit_params()
        }
        return space
     
    
    def loss(self,cv_df, train_df=None):
        train_df = self.df
        seasonality = 12
        metric_map = {
            'smape': lambda df: smape(df, models=['model'])['model'].mean(),
            'mase':  lambda df: mase(df, models=['model'], seasonality=seasonality)['model'].mean(),
            'rmsse': lambda df: rmsse(df, models=['model'],seasonality=12,train_df=train_df)['model'].mean(),
            'rmse':  lambda df: rmse(df,models=['model'])['model'].mean()
        }
            
        return metric_map[self.tuning_metric](cv_df)
    
    def create_objetive(self):
        
        objective = mlforecast_objective(
        df=self.df,
        config_fn=self.config_fn,
        loss=self.loss,    
        model = self.model(),
        freq='MS',
        n_windows= self.mlf_n_window,
        step_size= self.mlf_step_size,
        id_col= self.mlf_id_col,
        h= self.mlf_h,
    )
        return objective

    def optimize(self,objective):

        study = optuna.create_study(
            direction= self.optuna_direction_cfg,
            sampler= self.optuna_sampler_cfg
            )
    
        study.optimize(objective, n_trials=self.optuna_n_trials_cfg)
        
        return study

    def run(self):
        objective = self.create_objetive()
        print(self.fixed_params)
        study = self.optimize(objective)
        best_value = study.best_value
        best_trial_cfg = study.best_trial.user_attrs['config']
        best_model_params = best_trial_cfg['model_params']
        best_mlforecast_params = best_trial_cfg['mlf_init_params']
        mlf_fit_params = best_trial_cfg['mlf_fit_params']
        return best_value,best_model_params,best_mlforecast_params,mlf_fit_params