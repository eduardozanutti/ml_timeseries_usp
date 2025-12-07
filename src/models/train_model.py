import os
import joblib
from mlforecast import MLForecast
from statsforecast import StatsForecast
from src.utils.model_registry import get_model_class
from src.utils.presets import lag_transforms_to_config,target_transforms_to_config
from utilsforecast.losses import rmse, mae, smape, mase, scaled_crps, mqloss,rmsse
import json
import pandas as pd
from src.evaluation.metrics import get_metric
import matplotlib.pyplot as plt


class CreateCanditateModel:
    def __init__(self,df,config,cv_config,type_model,model_name,cv_metric,compare_metrics,metric=None,model_params=None,mlf_params=None,mlf_fit_params=None):
        self.model_path = config['paths']['models']['candidates_model_path']
        self.df = df
        self.cv_config = cv_config
        self.seasonality = config['seasonality']
        self.type_model = type_model
        self.model_name = model_name
        self.model = get_model_class(model_name)
        self.compare_metrics = compare_metrics
        self.metric = metric
        self.cv_metric = cv_metric
        self.scaled_metrics = config['evaluation']['scaled_metrics']
        self.model_params = model_params
        self.mlf_params = mlf_params
        self.mlf_fit_params = mlf_fit_params
        self.n_windows = cv_config.get('n_windows',{})
        self.step_size = cv_config.get('step_size',{})
        self.h = cv_config.get('h',{})
        self.evaluation_path = config['paths']['evaluation']['evaluation_path']
        self.output_plots_path = config['paths']['plots']['plots_path']
        self.time_col = config['time_col']
        self.target_col = config['target_col']
    
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
    
    def cross_validation(self,model_artifact,static_features=None):
        #Column model
        results_metrics = {'model':self.model_name}

        cv_df = model_artifact.cross_validation(
                                     df=self.df,
                                     h=self.h,
                                     n_windows=self.n_windows,
                                     step_size=self.step_size,
                                     static_features=static_features
                                     )
        #Column for each metric
        for metric in self.compare_metrics:
            if metric in self.scaled_metrics:
                results_metrics.update({metric:get_metric(
                                                df = cv_df,
                                                metric_name = metric,
                                                model_name = self.model_name,
                                                seasonality = self.seasonality,
                                                train_df = self.df
                                                )}
                                        )
            else:
                results_metrics.update({metric:get_metric(
                                                df = cv_df,
                                                metric_name = metric,
                                                model_name = self.model_name,
                                                )}
                                        )
        
        #Column for notes
        results_metrics.update({'notes':'cv_results'})
        
        self.create_cross_validation_plots(cv_df)

        return results_metrics
    
    def plot_cv(self, df_cv,fname,uid, last_n=24 * 14):
        cutoffs = df_cv.query('unique_id == @uid')['cutoff'].unique()
        fig, ax = plt.subplots(nrows=len(cutoffs), ncols=1, figsize=(14, 6), gridspec_kw=dict(hspace=0.8))
        for cutoff, axi in zip(cutoffs, ax.flat):
            self.df.query('unique_id == @uid').tail(last_n).set_index(self.time_col).plot(ax=axi, title=uid, y=self.target_col)
            df_cv.query('unique_id == @uid & cutoff == @cutoff').set_index(self.time_col).plot(ax=axi, title=uid, y=self.model_name)
        fig.savefig(fname, bbox_inches='tight')
        plt.close()
        return
    
    def create_cross_validation_plots(self,df_cv):
        plot_model_path = os.path.join(self.output_plots_path,self.model_name)
        cv_plot_model_path = os.path.join(plot_model_path,'cross_validation')
        if not os.path.exists(cv_plot_model_path):
            os.makedirs(cv_plot_model_path)
        for uid in self.df['unique_id'].unique():
            if len(uid.split('/')) == 1:
                fname = os.path.join(cv_plot_model_path,uid)
            else:
                folder = uid.split('/')[-2]
                file_name = uid.split('/')[-1]
                path_file = os.path.join(cv_plot_model_path,folder)
                if not os.path.exists(path_file):
                    os.mkdir(path_file)
                fname = os.path.join(path_file,file_name)
            try:
                self.plot_cv(df_cv,fname,uid)
            except Exception as e:
                print(e)
                pass
        return
        
    def save_evaluation_metrics(self,results_metrics,filename='metrics_summary.csv'):
        
        if not os.path.exists(self.evaluation_path):
            os.makedirs(self.evaluation_path)
            

        evaluation_file = os.path.join(self.evaluation_path,filename)
        
        if not os.path.exists(evaluation_file):
            evaluation_csv = pd.DataFrame(
                                            columns=['model'] + self.compare_metrics + ['notes']
                                        )
            evaluation_csv.to_csv(evaluation_file, index=False)
        else:
            pass
        
        
        evaluation_csv = pd.DataFrame([results_metrics])
        evaluation_csv.to_csv(evaluation_file, mode='a', header=False, index=False)

        return
    
    def fit_model(self,model_artifact):
        if self.type_model == 'mlforecast':
            return model_artifact.fit(self.df,**self.mlf_fit_params,fitted=True)
        else:
            return model_artifact.fit(self.df,fitted=True)

    def save_model(self,model_artifact):
        """
        Salva o modelo em plk
        """
        model_path_name = os.path.join(self.model_path,self.model_name)

        if not os.path.exists(model_path_name):
            os.makedirs(model_path_name)
        model_artifact.save(model_path_name)
        return
    
    def save_fitted_values(self,model_artifact,filename='fitted_values.parquet'):
        model_path = os.path.join(self.model_path,self.model_name)
        save_path = os.path.join(model_path,filename)
        fitted_values = model_artifact.forecast_fitted_values()
        fitted_values.to_parquet(save_path)
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
    
    def load_fitted_values(self,filename='fitted_values.parquet'):
        model_path = os.path.join(self.model_path,self.model_name)
        load_path = os.path.join(model_path,filename)
        fitted_values = pd.read_parquet(load_path)
        return fitted_values

    def run(self):
        results_metrics = None

        if self.type_model == 'statsforecast':
            model_artifact = self.create_stats_model()
            results_metrics = self.cross_validation(model_artifact)
        else:
            static_features = self.mlf_fit_params['static_features']
            model_artifact = self.create_mlf_model()
            results_metrics = self.cross_validation(model_artifact,static_features=static_features)
            pass
        
        self.save_evaluation_metrics(results_metrics)
        
        fitted_model = self.fit_model(model_artifact)
        
        self.save_model(fitted_model)
        
        self.save_fitted_values(fitted_model)
        
        self.save_metric_params(results_metrics=results_metrics,filename='info.json')
        
        model = self.load_model()
        
        fitted_values = self.load_fitted_values()
        
        metric = self.load_metric(filename='info.json')
        
        return model, fitted_values, metric, results_metrics