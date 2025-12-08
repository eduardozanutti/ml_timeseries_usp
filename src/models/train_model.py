import os
import joblib
import gc  # ← adicionado
from mlforecast import MLForecast
from statsforecast import StatsForecast
from src.utils.model_registry import get_model_class
from src.utils.presets import lag_transforms_to_config, target_transforms_to_config
from utilsforecast.losses import rmse, mae, smape, mase, scaled_crps, mqloss, rmsse
import json
import pandas as pd
from src.evaluation.metrics import get_metric, get_metric_by_unique_id
import matplotlib.pyplot as plt
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class CreateCanditateModel:
    def __init__(self, df, config, cv_config, type_model, model_name, cv_metric, compare_metrics, metric=None,
                 model_params=None, mlf_params=None, mlf_fit_params=None):
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
        self.n_windows = cv_config.get('n_windows', {})
        self.step_size = cv_config.get('step_size', {})
        self.h = cv_config.get('h', {})
        self.train_metrics_path = config['paths']['metrics']['train_path']
        self.output_plots_path = config['paths']['plots']['plots_path']
        self.time_col = config['time_col']
        self.target_col = config['target_col']
        self.split_dates = config.get('split_dates',{})
        self.test_dates = self.split_dates.get('test',{})
        self.test_start = self.test_dates.get('start',{})
        self.test_end = self.test_dates.get('end',{})
        self.test_horizon = len(pd.date_range(start=self.test_start, end=self.test_end, freq='MS'))

    def create_mlf_model(self):
        model_artifact = MLForecast(
                    models = [self.model(**self.model_params)],
                    freq='MS',
                    **self.mlf_params
        )
        return model_artifact

    def create_stats_model(self):
        model_artifact = StatsForecast(
            models = [self.model(season_length=12)],
            freq = 'MS',
            n_jobs = -1
        )
        return model_artifact


    
    def cross_validation(self,model_artifact,static_features=None):
        #Column model
        results_metrics = {'model':self.model_name}
        if static_features:
            cv_df = model_artifact.cross_validation(
                                        df=self.df,
                                        h=self.h,
                                        n_windows=self.n_windows,
                                        step_size=self.step_size,
                                        static_features=static_features
                                        )
        else:
            cv_df = model_artifact.cross_validation(
                                        df=self.df,
                                        h=self.h,
                                        n_windows=self.n_windows,
                                        step_size=self.step_size
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

    def plot_cv(self, df_cv, fname, uid, last_n=24 * 14):
        cutoffs = df_cv.query('unique_id == @uid')['cutoff'].unique()
        n_cutoffs = len(cutoffs)

        if n_cutoffs == 0:
            print(f"[Plot CV] {uid} - Histórico insuficiente para CV.")
            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            hist = self.df.query('unique_id == @uid').set_index(self.time_col)[self.target_col]

            if len(hist) == 0:
                ax.text(0.5, 0.5, f'{uid}\nSem dados históricos', ha='center', va='center', fontsize=16)
            else:
                hist.tail(last_n).plot(ax=ax, title=f'{uid} - Sem CV (histórico curto)')
                ax.text(0.5, 0.1, 'Histórico muito curto para Cross-Validation',
                        ha='center', va='center', transform=ax.transAxes,
                        bbox=dict(facecolor='yellow', alpha=0.7), fontsize=12)

            ax.set_ylabel(self.target_col)
            fig.savefig(f"{fname}.png", bbox_inches='tight', dpi=100)
            plt.close(fig)
            gc.collect()
            return

        # Limita altura máxima da figura (evita figuras gigantes com muitos cutoffs)
        max_height = 30
        height = min(4.5 * n_cutoffs, max_height)

        fig, axs = plt.subplots(n_cutoffs, 1, figsize=(16, height), sharex=True, squeeze=False)
        axs = axs.flatten()

        for i, cutoff in enumerate(cutoffs):
            ax = axs[i]
            hist_df = self.df.query('unique_id == @uid').tail(last_n).set_index(self.time_col)
            hist_df[self.target_col].plot(ax=ax, label='Histórico', color='black', linewidth=1.8)

            pred_df = df_cv.query('unique_id == @uid & cutoff == @cutoff').set_index(self.time_col)
            if not pred_df.empty:
                pred_df[self.model_name].plot(ax=ax, label='Previsão', color='red', linestyle='--')

            ax.set_title(f'{uid} - Cutoff: {pd.to_datetime(cutoff).date()}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{fname}.png", bbox_inches='tight', dpi=100)  # ← dpi=100
        plt.close(fig)
        plt.clf()
        plt.close('all')  # mata tudo que possa ter ficado
        gc.collect()      # ← força liberação imediata da memória

    def create_cross_validation_plots(self,df_cv):
        cv_plot_model_path = os.path.join(self.output_plots_path, self.model_name, 'cross_validation')
        os.makedirs(cv_plot_model_path, exist_ok=True)

        # === Cálculo do erro por série usando SUA função (muito mais correto que MAE manual) ===
        def error_per_series(g):
            uid = g.name  # unique_id da série
            train_series = self.df[self.df['unique_id'] == uid]  # histórico só dessa série (preciso para mase/rmsse)

            metric_df = get_metric_by_unique_id(
                df=g,
                metric_name=self.cv_metric,
                model_name=self.model_name,
                seasonality=self.seasonality if self.cv_metric in self.scaled_metrics else None,
                train_df=train_series if self.cv_metric in self.scaled_metrics else None
            )
            # Blindado contra scalar, DataFrame, Series ou até bug no (df)
            if isinstance(metric_df, (pd.DataFrame, pd.Series)):
                return float(metric_df[self.model_name].iloc[0])
            else:
                return float(metric_df)  # caso raro de já vir scalar

        print(f"\nCalculando erro por série usando métrica principal: {self.cv_metric} ...")
        errors = df_cv.groupby('unique_id').apply(error_per_series).rename('error')

        # === Hierarquia ===
        def get_hierarchy(uid):
            parts = uid.split('/')
            return '/'.join(parts[:-1]) if len(parts) > 1 else 'root'

        uids_df = pd.DataFrame({'unique_id': self.df['unique_id'].unique()})
        uids_df['hierarchy'] = uids_df['unique_id'].apply(get_hierarchy)
        uids_df = uids_df.merge(errors, on='unique_id', how='left')
        uids_df['error'] = uids_df['error'].fillna(float('inf'))  # ← segura e limpa

        total_best = 0
        total_worst = 0

        print("\n=== Séries selecionadas para plot de CV ===")

        for hierarchy, group in uids_df.groupby('hierarchy'):
            n = len(group)
            group = group.sort_values('error')  # menor erro = melhor

            if n <= 10:
                best_uids = group['unique_id'].tolist()
                worst_uids = []
                print(f"Hierarquia '{hierarchy}' → {n} séries → todas em best_cases")
            else:
                best_uids = group.head(5)['unique_id'].tolist()
                worst_uids = group.tail(5)['unique_id'].tolist()
                print(f"Hierarquia '{hierarchy}' → {n} séries → 5 melhores + 5 piores")

            # Pasta da hierarquia (ex: eletronicos/tv ou root)
            hierarchy_path = os.path.join(cv_plot_model_path, hierarchy)
            os.makedirs(hierarchy_path, exist_ok=True)

            # best_cases sempre existe
            best_path = os.path.join(hierarchy_path, 'best_cases')
            os.makedirs(best_path, exist_ok=True)

            # worst_cases só quando tem mais de 10 séries
            worst_path = None
            if worst_uids:
                worst_path = os.path.join(hierarchy_path, 'worst_cases')
                os.makedirs(worst_path, exist_ok=True)

            # === Plot das melhores (ou todas) ===
            for uid in best_uids:
                file_name = uid.split('/')[-1]
                fname = os.path.join(best_path, file_name)
                try:
                    self.plot_cv(df_cv, fname, uid)
                except Exception as e:
                    print(f"Erro best {uid}: {e}")

            total_best += len(best_uids)

            # === Plot das piores ===
            for uid in worst_uids:
                file_name = uid.split('/')[-1]
                fname = os.path.join(worst_path, file_name)
                try:
                    self.plot_cv(df_cv, fname, uid)
                except Exception as e:
                    print(f"Erro worst {uid}: {e}")

            total_worst += len(worst_uids)

        print(f"\nTotal de plots gerados:")
        print(f"→ best_cases:  {total_best} séries")
        print(f"→ worst_cases: {total_worst} séries")
        print("Organização: cross_validation → [hierarquia] → best_cases / worst_cases (só quando >10)\n")

    def save_evaluation_metrics(self,results_metrics,filename='metrics_summary.csv'):
        
        if not os.path.exists(self.train_metrics_path):
            os.makedirs(self.train_metrics_path)
            

        evaluation_file = os.path.join(self.train_metrics_path,filename)
        
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
            return model_artifact.fit(self.df)

    def save_model(self,model_artifact):
        """
        Salva o modelo em pkl
        """
        
        model_path_name = os.path.join(self.model_path,self.model_name)
        
        if not os.path.exists(model_path_name):
            os.makedirs(model_path_name)

        if self.type_model == 'statsforecast':
            model_path_name = os.path.join(model_path_name,'model.pkl')
        else:
            pass

        model_artifact.save(model_path_name)
        return
        
    def save_fitted_values(self,model_artifact,filename='fitted_values.parquet'):
        model_path = os.path.join(self.model_path,self.model_name)
        save_path = os.path.join(model_path,filename)
        if self.type_model == 'mlforecast':
            fitted_values = model_artifact.forecast_fitted_values()
        else:
            #statsforecast
            Y_hat_df = model_artifact.forecast(h=self.test_horizon,df=self.df,fitted=True)
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
            json.dump(info, f, indent=4,cls = NpEncoder)
        
    def load_model(self):
        """
        Load plk model
        """
        model_path_name = os.path.join(self.model_path,self.model_name)
        if self.type_model == 'statsforecast':
            model_path_name =os.path.join(model_path_name,'model.plk')
        else:
            pass
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
        fitted_values = None

        if self.type_model == 'statsforecast':
            model_artifact = self.create_stats_model()
            results_metrics = self.cross_validation(model_artifact)
            pass
        else:
            #mlforecast
            static_features = self.mlf_fit_params['static_features']
            model_artifact = self.create_mlf_model()
            results_metrics = self.cross_validation(model_artifact,static_features=static_features)
            pass

        self.save_evaluation_metrics(results_metrics)
        fitted_model = self.fit_model(model_artifact)
        self.save_model(fitted_model)
        self.save_fitted_values(fitted_model)
        self.save_metric_params(results_metrics=results_metrics,filename='info.json')
        fitted_values = self.load_fitted_values()

        #model = self.load_model()
        metric = self.load_metric(filename='info.json')
        
        return fitted_model, fitted_values, metric, results_metrics

    # ... resto da classe permanece igual (save_evaluation_metrics, fit_model, etc.) ...