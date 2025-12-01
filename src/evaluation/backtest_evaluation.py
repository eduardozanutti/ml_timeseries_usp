from utilsforecast.losses import rmse, mae, smape, mase, scaled_crps, mqloss,rmsse
class ModelEvaluate:
    def __init__(self, candidate_model, model_name, train, test, n_months_test, validation_metric,compare_metrics,output_plots_path):
        self.candidate_model=candidate_model
        self.model_name=model_name
        self.train=train
        self.test=test
        self.n_months_test=n_months_test
        self.validation_metric=validation_metric
        self.compare_metrics=compare_metrics
        self.output_plots_path=output_plots_path

    def predict_future(self):
        prediction = self.candidate_model.predict(self.n_months_test)
        df_pred = self.test.merge(prediction,on=['unique_id','ds'])
        return df_pred
    
    def evaluate(self,df_pred):
        seasonality = 12
        metric_map = {
            'smape': lambda df: smape(df, models=[self.model_name])[self.model_name].mean(),
            'mase':  lambda df: mase(df, models=[self.model_name], seasonality=seasonality,train_df=self.train)[self.model_name].mean(),
            'rmsse': lambda df: rmsse(df, models=[self.model_name],seasonality=seasonality,train_df=self.train)[self.model_name].mean(),
            'rmse':  lambda df: rmse(df,models=[self.model_name])[self.model_name].mean()
        }

        #update best metric
        validation_metric = metric_map[self.validation_metric](df_pred)
        results_metrics = {mtrc: metric_map[mtrc](df_pred) for mtrc in self.compare_metrics}
        return validation_metric,results_metrics
    
    def plot_time_series(self,df_pred):
        fig = plot_series(df_pred,df_pred[['unique_id','ds',self.model_name]])
        return fig

    def save_time_series_plot(self,fig,filename='time_series_plot.png'):
        if not os.path.exists(self.output_plots_path):
            os.makedirs(self.output_plots_path)
        save_path = os.path.join(self.output_plots_path, filename)
        print(save_path)
        fig.savefig(save_path)
        return

    def run(self):
       prediction = self.predict_future()
       validation_metric,results_metrics = self.evaluate(prediction)
       fig = self.plot_time_series(prediction)
       self.save_time_series_plot(fig,filename=self.model_name+'_time_series_plot.png')
       return validation_metric,results_metrics

