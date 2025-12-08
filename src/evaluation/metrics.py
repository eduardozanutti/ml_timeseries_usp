
from utilsforecast.losses import nd, smape, mase, mape, rmsse

def get_metric(df,metric_name,model_name,seasonality=None,train_df=None):
    """
    Function que calcula os erros e trás a média global dos folds e séries temporais
    """
    metric_map = {
                #Scaled Mean Absolute Percentual Error
                'smape': lambda df: smape(
                                            df = df,
                                            models = [model_name]
                                        )[model_name].mean(),
                #Mean Absolute Percentual Error
                'mape': lambda df:   mape(
                                            df = df,
                                            models = [model_name]
                                        )[model_name].mean(),
                #Mean Absolute Scaled Error
                'mase':  lambda df: mase(
                                            df = df,
                                            models = [model_name],
                                            seasonality = seasonality,
                                            train_df = train_df
                                        )[model_name].mean(),
                #Root Mean Scaled Squared Error
                'rmsse': lambda df: rmsse(
                                            df = df,
                                            models=[model_name],
                                            seasonality=seasonality,
                                            train_df=train_df
                                        )[model_name].mean(),
                #
                'nd':    lambda df: nd(
                                            df = df,
                                            models=[model_name]
                                        )[model_name].mean()
            }
    return metric_map[metric_name](df)

def get_metric_by_unique_id(df,metric_name,model_name,seasonality=None,train_df=None):
    """
    Function que calcula os erros e trás a média global dos folds e séries temporais
    """
    metric_map = {
                #Scaled Mean Absolute Percentual Error
                'smape': lambda df: smape(
                                            df = df,
                                            models = [model_name]
                                        )[model_name].mean(),
                #Mean Absolute Percentual Error
                'mape': lambda df:   mape(
                                            df = df,
                                            models = [model_name]
                                        )[model_name].mean(),
                #Mean Absolute Scaled Error
                'mase':  lambda df: mase(
                                            df = df,
                                            models = [model_name],
                                            seasonality = seasonality,
                                            train_df = train_df
                                        )[model_name].mean(),
                #Root Mean Scaled Squared Error
                'rmsse': lambda df: rmsse(
                                            df = df,
                                            models=[model_name],
                                            seasonality=seasonality,
                                            train_df=train_df
                                        )[model_name].mean(),
                #
                'nd':    lambda df: nd(
                                            df = df,
                                            models=[model_name]
                                        )[model_name].mean()
            }
    return metric_map[metric_name](df)