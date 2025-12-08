from types import MethodDescriptorType
from networkx import dfs_edges
from utilsforecast.losses import rmse, mae, smape, mase, scaled_crps, mqloss,rmsse
from utilsforecast.plotting import plot_series
import os
import pandas as pd
from src.evaluation.metrics import get_metric
import matplotlib.pyplot as plt
import gc

class ModelEvaluate:
    def __init__(self, model_name, type_model, fitted_values, train, test, validation_metric,compare_metrics,config, candidate_model=None, prediction=False,dataset_type='global'):
        self.candidate_model=candidate_model
        self.model_name=model_name
        self.type_model = type_model
        self.id_col = config.get('id_col','unique_id')
        self.time_col = config.get('time_col','ds')
        self.train=train
        self.test=test
        self.split_dates = config.get('split_dates',{})
        self.test_dates = self.split_dates.get('test',{})
        self.test_start = self.test_dates.get('start',{})
        self.test_end = self.test_dates.get('end',{})
        self.test_horizon = len(pd.date_range(start=self.test_start, end=self.test_end, freq='MS'))
        self.seasonality = config.get('seasonality',12)
        self.scaled_metrics = config['evaluation']['scaled_metrics']
        self.validation_metric=validation_metric
        self.compare_metrics=compare_metrics
        self.output_plots_path = config['paths']['plots']['plots_path']
        self.evaluation_path = config['paths']['metrics']['evaluation_path']
        self.fitted_values = fitted_values
        self.prediction = prediction
        self.paths = config.get('paths',{})
        self.processed_path = self.paths.get('data',{}).get('processed_path','data/processed/')
        self.features_path = self.paths.get('features',{})
        self.test_features_path = self.features_path.get('test','features/test')
        self.dataset_type = dataset_type
        self.static_features = config.get('modeling',{}).get('mlforecast',{}).get('fit_params',{}).get('static_features',[])
        self.target_col = config.get('target_col','y')
    
    def plot_backtest_evaluation(self, df_backtest_with_preds, fname_base=None, last_n_train=36):
        """
        Plota APENAS 5 melhores + 5 piores por hierarquia (ou todas se ≤10)
        → Mesma lógica que usamos no CV e na reconciliação
        → Pastas: backtest_evaluation / [hierarquia] / best_cases  ou  worst_cases
        """
        plot_model_path = os.path.join(self.output_plots_path, self.model_name, 'backtest_evaluation')
        os.makedirs(plot_model_path, exist_ok=True)

        metric_for_ranking = self.validation_metric
        seasonality = self.seasonality
        scaled_metrics = self.scaled_metrics

        print(f"\nCalculando erro por série no backtest usando métrica: {metric_for_ranking} ...")

        # === Calcula erro por série no período de backtest (usando get_metric) ===
        def error_per_series(g):
            uid = g.name
            # Histórico só dessa série (train + test até o ponto que tem real)
            train_series = pd.concat([
                self.train[self.train['unique_id'] == uid],
                self.test[self.test['unique_id'] == uid]
            ]).sort_values(self.time_col)

            metric_val = get_metric(
                df=g,
                metric_name=metric_for_ranking,
                model_name=self.model_name,
                seasonality=seasonality if metric_for_ranking in scaled_metrics else None,
                train_df=train_series if metric_for_ranking in scaled_metrics else None
            )
            # get_metric retorna dict → pegamos o valor da coluna do modelo
            if isinstance(metric_val, (pd.DataFrame, pd.Series, dict)):
                return float(metric_val[self.model_name])
            else:
                return float(metric_val)  # já é scalar (o que está acontecendo no seu caso)

        errors = df_backtest_with_preds.groupby('unique_id').apply(error_per_series).rename('error')

        # === Hierarquia ===
        def get_hierarchy(uid):
            parts = str(uid).split('/')
            return '/'.join(parts[:-1]) if len(parts) > 1 else 'root'

        uids_df = pd.DataFrame({'unique_id': df_backtest_with_preds['unique_id'].unique()})
        uids_df['hierarchy'] = uids_df['unique_id'].apply(get_hierarchy)
        uids_df = uids_df.merge(errors, left_on='unique_id', right_index=True, how='left')
        uids_df['error'] = uids_df['error'].fillna(float('inf'))

        total_best = 0
        total_worst = 0

        print("\n=== Plotando séries selecionadas no backtest (máx 10 por hierarquia) ===")

        for hierarchy, group in uids_df.groupby('hierarchy'):
            n = len(group)
            group = group.sort_values('error')

            if n <= 10:
                best_uids = group['unique_id'].tolist()
                worst_uids = []
                print(f"Hierarquia '{hierarchy}' → {n} séries → todas em best_cases")
            else:
                best_uids = group.head(5)['unique_id'].tolist()
                worst_uids = group.tail(5)['unique_id'].tolist()
                print(f"Hierarquia '{hierarchy}' → {n} séries → 5 melhores + 5 piores")

            # Pasta da hierarquia
            hierarchy_path = os.path.join(plot_model_path, hierarchy)
            os.makedirs(hierarchy_path, exist_ok=True)

            best_path = os.path.join(hierarchy_path, 'best_cases')
            os.makedirs(best_path, exist_ok=True)

            worst_path = None
            if worst_uids:
                worst_path = os.path.join(hierarchy_path, 'worst_cases')
                os.makedirs(worst_path, exist_ok=True)

            # === Plot das séries selecionadas ===
            for uid in best_uids + worst_uids:
                is_worst = uid in worst_uids
                path = worst_path if is_worst else best_path
                
                fname = os.path.join(path, f"backtest_{str(uid).replace('/', '_')}.png")
                try:
                    self._plot_single_backtest(uid, df_backtest_with_preds, fname, last_n_train=last_n_train)
                except Exception as e:
                    print(f"Erro ao plotar {uid}: {e}")

                if is_worst:
                    total_worst += 1
                else:
                    total_best += 1

        print(f"\nPlots de backtest gerados com sucesso!")
        print(f"→ best_cases:  {total_best} séries")
        print(f"→ worst_cases: {total_worst} séries")
        print(f"Local: {plot_model_path}")


    def _plot_single_backtest(self, uid, df_backtest_with_preds, fname, last_n_train=36):
        """
        Plot de uma única série no backtest: histórico + real (test) + previsão
        """
        # 1. Histórico de treino (self.df é o train completo)
        hist_df = self.train.query('unique_id == @uid').set_index(self.time_col)[[self.target_col]].copy()
        hist_df = hist_df.tail(last_n_train) if len(hist_df) > last_n_train else hist_df

        # 2. Dados do backtest (contém tanto o real quanto a previsão no período de test)
        backtest_uid = df_backtest_with_preds.query('unique_id == @uid').set_index(self.time_col).copy()

        if backtest_uid.empty:
            print(f"[Backtest Plot] {uid} - Sem dados de backtest")
            return

        # Colunas esperadas: target_col (real) e model_name (previsão)
        real_test = backtest_uid[[self.target_col]]
        pred_test = backtest_uid[[self.model_name]]

        # Data do split train/test (última data do histórico de treino)
        train_end_date = hist_df.index.max() if not hist_df.empty else real_test.index.min()

        fig, ax = plt.subplots(figsize=(16, 6))

        # Histórico (treino)
        if not hist_df.empty:
            hist_df.plot(ax=ax, color='black', linewidth=2, label='Histórico (treino)')

        # Real no período de test
        if not real_test.empty:
            real_test.plot(ax=ax, color='steelblue', linewidth=2.5, label='Real (backtest)')

        # Previsão do modelo no período de test
        if not pred_test.empty:
            pred_test.plot(ax=ax, color='red', linestyle='--', linewidth=2.5, label='Previsão')

        # Linha vertical separando treino do backtest
        ax.axvline(train_end_date, color='gray', linestyle='-', linewidth=1.8, alpha=0.9, label='Início backtest')

        ax.set_title(f'{uid} - Backtest vs Real - {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel(self.target_col)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # Aviso se histórico for muito curto
        if len(hist_df) < 6:
            ax.text(0.5, 0.5, 'Histórico muito curto', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, color='orange',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'))

        plt.tight_layout()
        fig.savefig(f"{fname}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        
    
    def load_test_features(self):
        data_path = os.path.join(self.processed_path,self.dataset_type)
        
        exogen_test_features_path = os.path.join(data_path,self.test_features_path)
        test_features_path = os.path.join(exogen_test_features_path,'test_features.parquet')
        X_df = pd.read_parquet(test_features_path)

        return X_df
    
    def predict_future(self,X_df=None):
        if self.type_model=='mlforecast':
            prediction = self.candidate_model.predict(self.test_horizon,X_df=X_df)
        else:
            prediction = self.candidate_model.predict(self.test_horizon)
        return prediction
    

    def evaluate(self,df_pred):

        results_metrics = {'model':self.model_name}

        #update best metric
        #Column for each metric
        for metric in self.compare_metrics:
            if metric in self.scaled_metrics:
                results_metrics.update({metric:get_metric(
                                                df = df_pred,
                                                metric_name = metric,
                                                model_name = self.model_name,
                                                seasonality = self.seasonality,
                                                train_df = self.df
                                                )}
                                        )
            else:
                results_metrics.update({metric:get_metric(
                                                df = df_pred,
                                                metric_name = metric,
                                                model_name = self.model_name,
                                                )}
                                        )
        
        #Column for notes
        results_metrics.update({'notes':'backtest_evaluation'})
        return self.validation_metric,results_metrics
    
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
    
    def plot_time_series(self,df):
        fig = plot_series(
                            df,
                            forecasts_df = df.drop(columns='y'),
                            engine = 'matplotlib',
                            #palette = 'reds'
                        )
        return fig

    def run(self):
       X_df = self.load_test_features()
       #get predictions values
       if self.prediction is False: #Hierarquical reconciliaton has prediction
          if self.type_model == 'mlforecast':
            self.prediction = self.predict_future(X_df)
          else:
            self.prediction = self.predict_future()
       else:
           pass
       
       # Join with Test
       df_pred = self.prediction.merge(self.test,on=['unique_id','ds'])
       
       #Get results
       validation_metric,results_metrics = self.evaluate(df_pred)

       self.save_evaluation_metrics(results_metrics)

       self.plot_backtest_evaluation(df_pred, fname_base='backtest_before_reconciliation.png')
       
       return self.prediction, validation_metric, results_metrics

class CompareRecMethods:
    def __init__(self, model_name, df_post_processed, fitted_values, test, config):
        self.df_post_processed = df_post_processed.reset_index(drop=True)
        self.fitted_values = fitted_values.reset_index(drop=True)
        self.test = test.reset_index(drop=True)
        self.config = config
        
        self.id_col = config.get('unique_id_col', 'unique_id')
        self.time_col = config.get('time_col', 'ds')
        self.target_col = config.get('target_col', 'y')
        self.model_name = model_name
        self.validation_metric = config['modeling']['validation_metric']
        self.seasonality = config.get('seasonality', 12)
        self.scaled_metrics = config['evaluation']['scaled_metrics']

        self.output_plots_path = os.path.join(config['paths']['plots']['plots_path'], self.model_name, 'reconciliation_comparison')
        os.makedirs(self.output_plots_path, exist_ok=True)

    def create_full_dataset(self):
        df_test = self.test.merge(self.df_post_processed, on=[self.id_col, self.time_col], how='left')
        df_full = pd.concat([
            self.fitted_values[[self.id_col, self.time_col, self.target_col]],
            df_test
        ], ignore_index=True)
        return df_full.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

    def _detect_model_columns(self, df):
        ignore = {self.id_col, self.time_col, self.target_col}
        model_cols = [col for col in df.columns if col not in ignore]
        base_cols = [col for col in model_cols if '/' not in col]
        rec_cols  = [col for col in model_cols if '/' in col]
        
        if len(base_cols) != 1:
            raise ValueError(f"Esperava exatamente 1 coluna base, encontrou: {base_cols}")
        
        return base_cols[0], rec_cols

    def plot_all_series_individual(self, df_full, last_n=48):
        base_col, rec_cols = self._detect_model_columns(df_full)
        
        print(f"Modelo base → {base_col}")
        print(f"Reconciliações encontradas → {rec_cols}")

        last_train_date = self.fitted_values[self.time_col].max()

        # === Calcula erro por série ===
        errors_by_method = {}
        for col in [base_col] + rec_cols:
            def calc_error(group):
                uid = group.name
                mask = group[self.target_col].notna() & group[col].notna()
                if mask.sum() == 0:
                    return float('inf')
                
                y_true = group.loc[mask, self.target_col].values
                y_pred = group.loc[mask, col].values

                temp_df = pd.DataFrame({
                    'unique_id': uid,
                    'y': y_true,
                    col: y_pred
                })

                full_history = pd.concat([
                    self.fitted_values[self.fitted_values[self.id_col] == uid],
                    self.test[self.test[self.id_col] == uid]
                ]).sort_values(self.time_col)
                full_history = full_history.assign(unique_id=uid)

                metric_val = get_metric(
                    df=temp_df,
                    metric_name=self.validation_metric,
                    model_name=col,
                    seasonality=self.seasonality if self.validation_metric in self.scaled_metrics else None,
                    train_df=full_history if self.validation_metric in self.scaled_metrics else None
                )
                
                return float(metric_val[col]) if isinstance(metric_val, dict) else float(metric_val)

            print(f"Calculando erro para coluna {col}...")
            errors_by_method[col] = df_full.groupby(self.id_col).apply(calc_error)

        # === Hierarquia ===
        def get_hierarchy(uid):
            parts = str(uid).split('/')
            return '/'.join(parts[:-1]) if len(parts) > 1 else 'root'

        uids_all = df_full[self.id_col].unique()
        uids_df = pd.DataFrame({self.id_col: uids_all})
        uids_df['hierarchy'] = uids_df[self.id_col].apply(get_hierarchy)

        # === Plot por método ===
        for method_col in rec_cols:
            method_name = method_col.split('/')[-1]
            method_folder = os.path.join(self.output_plots_path, method_name)
            os.makedirs(method_folder, exist_ok=True)
            print(f"\nGerando plots para método: {method_name}")

            current_errors = errors_by_method[method_col].rename('error')
            temp_uids = uids_df.merge(current_errors, left_on=self.id_col, right_index=True, how='left')
            temp_uids['error'] = temp_uids['error'].fillna(float('inf'))

            for hierarchy, group in temp_uids.groupby('hierarchy'):
                n = len(group)
                group = group.sort_values('error')

                if n <= 10:
                    selected = group[self.id_col].tolist()
                    worst = []
                else:
                    selected = group.head(5)[self.id_col].tolist()
                    worst = group.tail(5)[self.id_col].tolist()
                    selected += worst

                hierarchy_path = os.path.join(method_folder, hierarchy)
                best_path = os.path.join(hierarchy_path, 'best_cases')
                worst_path = os.path.join(hierarchy_path, 'worst_cases') if worst else None

                os.makedirs(best_path, exist_ok=True)
                if worst_path:
                    os.makedirs(worst_path, exist_ok=True)

                for uid in selected:
                    is_worst = uid in worst
                    save_path = worst_path if is_worst else best_path
                    safe_uid = str(uid).replace('/', '_')

                    df_uid = df_full[df_full[self.id_col] == uid].set_index(self.time_col)
                    df_plot = df_uid.tail(last_n + 36) if len(df_uid) > last_n + 36 else df_uid

                    fig, ax = plt.subplots(figsize=(16, 6))

                    # Actual (histórico + real) → sempre atrás (zorder baixo)
                    ax.plot(df_plot.index, df_plot[self.target_col], 
                            color='black', linewidth=2.4, label='Actual', zorder=1, alpha=0.9)

                    # Modelo base → por cima do actual, sempre visível
                    ax.plot(df_plot.index, df_plot[base_col], 
                            color='#1f77b4', linewidth=2.6, label='Base', zorder=8, alpha=1.0)

                    # Outros métodos → cinza bem claro, tracejado fino, atrás
                    for c in rec_cols:
                        if c == method_col:
                            continue
                        if df_plot[c].notna().any():
                            ax.plot(df_plot.index, df_plot[c], 
                                    color='#aaaaaa', linewidth=1.6, linestyle='--', alpha=0.6, zorder=2)

                    # Método atual → cor forte, tracejado grosso, na frente de tudo
                    if df_plot[method_col].notna().any():
                        colors = ['#d62728', '#ff7f0e', '#e377c2', '#9467bd', '#2ca02c', '#bcbd22']
                        color = colors[rec_cols.index(method_col) % len(colors)]
                        ax.plot(df_plot.index, df_plot[method_col], 
                                color=color, linewidth=3.6, linestyle='--', 
                                label=method_name.upper(), zorder=9, alpha=1.0)

                    ax.axvline(last_train_date, color='gray', linestyle='-', linewidth=2, alpha=0.8)
                    ax.set_title(f'{uid} — {method_name.upper()}', fontsize=14, fontweight='bold')
                    ax.set_ylabel(self.target_col)
                    ax.grid(True, alpha=0.3)
                    ax.legend(frameon=True, fancybox=True, loc='upper left')

                    plt.tight_layout()
                    fig.savefig(os.path.join(save_path, f'rec_{safe_uid}.png'), dpi=180, bbox_inches='tight')
                    plt.close(fig)
                    gc.collect()

            print(f"   → Plots do método {method_name} concluídos")

        print(f"\nComparação de reconciliação 100% finalizada! Tudo salvo em:\n{self.output_plots_path}")

    def run(self, last_n=48):
        print("Iniciando comparação visual de métodos de reconciliação hierárquica...")
        df_full = self.create_full_dataset()
        self.plot_all_series_individual(df_full, last_n=last_n)