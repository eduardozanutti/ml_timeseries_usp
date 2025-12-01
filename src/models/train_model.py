import os
import joblib
from mlforecast import MLForecast
from statsforecast import StatsForecast
from src.utils.model_registry import get_model_class
from src.utils.presets import lag_transforms_to_config,target_transforms_to_config
from utilsforecast.losses import rmse, mae, smape, mase, scaled_crps, mqloss,rmsse
import json

class CreateCanditateModel:
    def __init__(self,df,config,cv_config,type_model,model_name,cv_metric,compare_metrics,metric=None,model_params=None,mlf_params=None,mlf_fit_params=None):
        self.model_path = config['paths']['models']['candidates_model_path']
        self.df = df
        self.cv_config = cv_config
        self.type_model = type_model
        self.model_name = model_name
        self.model = get_model_class(model_name)
        self.compare_metrics = compare_metrics
        self.metric = metric
        self.cv_metric = cv_metric
        self.model_params = model_params
        self.mlf_params = mlf_params
        self.mlf_fit_params = mlf_fit_params
        self.n_windows = cv_config.get('n_windows',{})
        self.step_size = cv_config.get('step_size',{})
        self.h = cv_config.get('h',{})

    def create_mlf_model(self):
        model_artifact = MLForecast(
                    models = [self.model(**self.model_params)],
                    freq='MS',
                    **self.mlf_params
        )
        return model_artifact

    def create_stats_model(self):
        model_artifact = StatsForecast(
            models = [self.model(season_lenght=12)],
            freq = 'MS',
            n_jobs = -1
        )
        return model_artifact
    
    def cross_validation(self,model_artifact,train_df=None,seasonality=None):
        train_df = self.df
        seasonality = 12
        metric_map = {
            'smape': lambda df: smape(df, models=[self.model_name])[self.model_name].mean(),
            'mase':  lambda df: mase(df, models=[self.model_name], seasonality=seasonality,train_df=train_df)[self.model_name].mean(),
            'rmsse': lambda df: rmsse(df, models=[self.model_name],seasonality=12,train_df=train_df)[self.model_name].mean(),
            'rmse':  lambda df: rmse(df,models=[self.model_name])[self.model_name].mean()
        }

        cv_df = model_artifact.cross_validation(
                                     df=self.df,
                                     h=self.h,
                                     n_windows=self.n_windows,
                                     step_size=self.step_size
                                     )
        #update best metric
        self.metric = metric_map[self.cv_metric](cv_df)
        results_metrics = {mtrc: metric_map[mtrc](cv_df) for mtrc in self.compare_metrics}
        return results_metrics
    
    def fit_model(self,model_artifact):
        if self.type_model == 'mlforecast':
            return model_artifact.fit(self.df,**self.mlf_fit_params)
        else:
            return model_artifact.fit(self.df)

    def save_model(self,model_artifact):
        """
        Salva o modelo em plk
        """
        model_path_name = os.path.join(self.model_path,self.model_name)

        if not os.path.exists(model_path_name):
            os.makedirs(model_path_name)
        model_artifact.save(model_path_name)
        return

    def save_metric_params(self,results_metrics,filename='info.json'):
        """
        save CV or Tunning Metrics and params
        """
        model_path_name = os.path.join(self.model_path,self.model_name)
        save_path = os.path.join(model_path_name,filename)
        if self.type_model == 'mlforecast':

            # Converte TUDO pra string legível (pronto pra JSON/YAML)
            mlf_params_json_format = {
                "lags": self.mlf_params['lags'],
                "lag_transforms": lag_transforms_to_config(self.mlf_params['lag_transforms']),
                "target_transforms": target_transforms_to_config(self.mlf_params['target_transforms']),
                "date_features": self.mlf_params['date_features']
            }
            info = {
                    'main_metric': self.metric,
                    'metrics': results_metrics,
                    'model_params': self.model_params,
                    'mlf_params': mlf_params_json_format,
                    'mlf_fit_params': self.mlf_fit_params
                    }
        else:
            info = {
                    'main_metric': self.metric,
                    'metrics': results_metrics,
                    'model_params': self.model_params
                    }
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=4)
        
    def load_model(self):
        """
        Load plk model
        """
        model_path_name = os.path.join(self.model_path,self.model_name)
        model_artifact = MLForecast.load(model_path_name)
        return model_artifact
    

    def load_metric(self,filename='info.json'):
        """
        Load Metrics and Params
        """
        model_path_name = os.path.join(self.model_path,self.model_name)
        load_path = os.path.join(model_path_name,filename)
        with open(load_path, 'r') as f:
            info = json.load(f)
        metric = info['main_metric']
        return metric

    def run(self):
        results_metrics = None

        if self.type_model == 'statsforecast':
            model_artifact = self.create_stats_model()
        else:
            model_artifact = self.create_mlf_model()

        results_metrics = self.cross_validation(model_artifact)
        fitted_model = self.fit_model(model_artifact)
        self.save_model(fitted_model)
        self.save_metric_params(results_metrics=results_metrics,filename='info.json')
        model = self.load_model()
        metric = self.load_metric(filename='info.json')
        return model,metric,results_metrics