from types import MethodDescriptorType
from networkx import dfs_edges
from utilsforecast.losses import rmse, mae, smape, mase, scaled_crps, mqloss,rmsse
from utilsforecast.plotting import plot_series
import os
import pandas as pd
from src.evaluation.metrics import get_metric

class ModelEvaluate:
    def __init__(self, model_name, fitted_values, train, test, n_months_test, validation_metric,compare_metrics,config, candidate_model=None, prediction=False):
        self.candidate_model=candidate_model
        self.model_name=model_name
        self.train=train
        self.test=test
        self.n_months_test=n_months_test
        self.validation_metric=validation_metric
        self.compare_metrics=compare_metrics
        self.output_plots_path = config['paths']['plots']['plots_path']
        self.evaluation_path = config['paths']['evaluation']['evaluation_path']
        self.fitted_values = fitted_values
        self.prediction = prediction
    
    def predict_future(self):
        prediction = self.candidate_model.predict(self.n_months_test)
        return prediction
    

    def evaluate(self,df_pred):

        results_metrics = {'model':self.model_name}

        #update best metric
        validation_metric = {self.validation_metric: get_metric(
                                            df = df_pred,
                                            metric_name = self.validation_metric,
                                            model_name = self.model_name,
                                            seasonality = self.seasonality,
                                            train_df = self.df)
        }
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
        results_metrics.update({'notes':'cv_results'})
        return validation_metric,results_metrics
    
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
    
    def save_time_series_plot(self,fig,filename='time_series_plot.png'):
        
        
        model_path = os.path.join(self.output_plots_path, self.model_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        save_path = os.path.join(model_path,filename)

        fig.savefig(save_path)
        
        return

    def run(self):
       #get predictions values
       if self.prediction is False:
          self.prediction = self.predict_future()
       else:
           pass
       
       # Join with Test
       df_pred = self.prediction.merge(self.test,on=['unique_id','ds'])
       
       #Get results
       validation_metric,results_metrics = self.evaluate(df_pred)

       self.save_evaluation_metrics(results_metrics)
       
       fig = self.plot_time_series(df_pred)

       self.save_time_series_plot(fig)
       
       return self.prediction, validation_metric, results_metrics

class CompareRecMethods:
    def __init__(self,df_post_processed,methods,fitted_values,test,config):
        self.df_post_processed = df_post_processed
        self.methods = methods
        self.fitted_values = fitted_values
        self.test = test
        self.output_plots_path = config['paths']['plots']['plots_path']
        self.config = config
    
    def create_full_dataset(self):
        df_full = self.df_post_processed.merge(self.test,on=['unique_id','ds'])
        df_full = pd.concat([self.fitted_values,df_full])
        return df_full
    
    def plot_time_series(self,df):
        # Cores fixas para cada modelo (você escolhe as que quiser)
        # Pega automaticamente TODAS as colunas de previsão (menos 'y' e 'ds')

        
        mints = [col for col in df.columns if col not in ['y', 'ds', 'unique_id']]

        # Dicionário automático de cores (ciclo de 10 cores bonitas do Plotly)
        CORES_CICLO = [
            '#1f77b4',  
            '#ff7f0e',  
            '#2ca02c',  
            '#d62728',  
            '#9467bd',  
            '#8c564b',  
            '#e377c2',  
            '#7f7f7f',  
            '#bcbd22', 
            '#17becf',  
        ]
        fig = plot_series(
                            df,
                            forecasts_df = df.drop(columns='y'),
                            engine='plotly'
                        )
        fig.update_layout(
            height=750,
            width=1480,
            title=dict(
                text='Actual vs Forecast vs Reconciliações',
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top',
                font=dict(size=22)
            ),
            margin=dict(t=140, l=50, r=50, b=80),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1
            )
        )  # tamanho bom

        # Cria dicionário dinâmico: cada previsão recebe uma cor fixa do ciclo
        cor_map = {model: CORES_CICLO[i % len(CORES_CICLO)] for i, model in enumerate(mints)}

        # Aplica no plot
        for trace in fig.data:
            if trace.name == 'y':
                trace.line = dict(color='black', width=2, dash='solid')
            elif trace.name in cor_map:
                trace.line = dict(
                    color=cor_map[trace.name],
                    width=1.5,
                    dash='dash'   # todas as previsões pontilhadas
                )

        # Opcional: deixa a linha do actual mais grossa e preta pra destacar
        #fig.update_traces(selector=dict(name='y'), line=dict(width=4, color='black'))

        return fig
    
    def save_time_series_plot(self,fig,filename='time_series_plot_rec.html'):
        if not os.path.exists(self.output_plots_path):
            os.makedirs(self.output_plots_path)
        save_path = os.path.join(self.output_plots_path, filename)
        print(save_path)
        fig.write_html(save_path)
        return
    
    def save_plot(self):
        df_full = self.create_full_dataset()
        fig = self.plot_time_series(df_full)
        self.save_time_series_plot(fig)
        return

