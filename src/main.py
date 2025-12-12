import os
from src.data.make_dataset import DatasetCreator
from src.data.make_hierarchical_dataset import DatasetHierarchicalAggregator
from src.data.refine_dataset import HierarchicalTimeSeriesOutlierRemover
from src.data.split import HierarchicalTrainTestSplit
from src.features.build_features import FeaturesBuilder
from src.config import load_config
from src.models.tuning import ModelTuning
from src.models.train_model import CreateCanditateModel
from src.evaluation.backtest_evaluation import ModelEvaluate,CompareRecMethods
from src.data.back_to_origin import DatasetReconciliator
import warnings
warnings.filterwarnings("ignore")
config = load_config()

if config['cache']:

    print('loading data...')
    ## == LOAD AGGREGATED DATASETS AND OBJECTS 
    Hagg = DatasetHierarchicalAggregator(config)
    
    Y_df = Hagg.load_processed(filename='dataset.parquet')
    S_df = Hagg.load_processed(filename='structure.parquet')
    tags = Hagg.load_tags(filename='tags.joblib')
    Y_exogen_cat_df = Hagg.load_processed(filename='exog_cat_vars.parquet')
    Y_exogen_num_df = Hagg.load_processed(filename='exog_num_vars.parquet')
    future_df = Hagg.load_processed(filename='future.parquet')

    pass

else:
    ## == CREATE DATASET
    df = DatasetCreator(config).run()

    ## == HIERARCHICAL AGGREGATION
    Y_df, S_df, tags, Y_exogen_cat_df, Y_exogen_num_df, future_df = DatasetHierarchicalAggregator(config,df=df).run()

## == HIERARCHICAL TRAIN TEST SPLIT
split_dates = config['split_dates']
train , test = HierarchicalTrainTestSplit(Y_df, S_df, tags, Y_exogen_cat_df, Y_exogen_num_df,split_dates,config).run()


models = config['models']
cv_config = config['modeling']['cv_config']
methods = config['reconciliation']['methods']
min_trace_methods = config['reconciliation']['min_trace_methods']
mid_level = config['reconciliation']['middle_level']
train_metrics_path = config['paths']['metrics']['train_path']
evaluation_metrics_path = config['paths']['metrics']['evaluation_path']
time_col = config['time_col']
id_col = config['id_col']
target_col = config['target_col']

if os.path.exists(os.path.join(train_metrics_path,'metrics_summary.csv')):
    os.remove(os.path.join(train_metrics_path,'metrics_summary.csv'))

if os.path.exists(os.path.join(evaluation_metrics_path,'metrics_summary.csv')):
    os.remove(os.path.join(evaluation_metrics_path,'metrics_summary.csv'))

candidate_info = {}
candidate_performances = {}
for model in [m for m in models if m not in ['LightGBM','XGBoost','AutoARIMA']]:

    train_to_model = train.copy()
    test_to_model = test.copy()
    
    is_enable = models[model]['enabled']
    
    #check if model is enable in yaml file
    if is_enable:

        type_model = models[model]['type']
        training_metric = config['modeling']['training_metric']
        validation_metric = config['modeling']['validation_metric']
        train_compare_metrics = config['modeling']['train_compare_metrics']
        test_compare_metrics = config['modeling']['test_compare_metrics']
        model_name = models[model]['name']

        if type_model == 'mlforecast':
            #Get Tuning info
            fixed_params = models[model]['fixed_params']
            param_space = config['parameter_space'][model]
            mlforecast_params = config['modeling']['mlforecast']
            

            #Tuning model
            best_value,best_model_params,best_mlforecast_params,mlf_fit_params =\
                  ModelTuning(
                                df=train_to_model,
                                config=config,
                                model_name=model_name,
                                fixed_params=fixed_params,
                                param_space=param_space,
                                cv_config=cv_config,
                                mlforecast_params=mlforecast_params,
                                tuning_metric = training_metric
                            ).run()
        
            #create model with best tuning parameters
            candidate, fitted_values, metric,results_metrics = CreateCanditateModel(
                                df=train_to_model,
                                config=config,
                                cv_config=cv_config,
                                type_model=type_model,
                                model_name=model_name,
                                metric=best_value,
                                cv_metric = training_metric,
                                compare_metrics=train_compare_metrics,
                                model_params=best_model_params,
                                mlf_params=best_mlforecast_params,
                                mlf_fit_params = mlf_fit_params
                                ).run()
            
        else:#statsforecast
            train_to_model = train_to_model[[id_col,time_col,target_col]]
            test_to_model = test_to_model[[id_col,time_col,target_col]]

            #create model without tuning parameters
            candidate, fitted_values, metric,results_metrics = CreateCanditateModel(
                                df=train_to_model,
                                config=config,
                                cv_config=cv_config,
                                cv_metric = training_metric,
                                type_model=type_model,
                                compare_metrics=train_compare_metrics,
                                model_name=model_name
                                ).run()
                
        candidate_info.update({metric:candidate})
        candidate_performances.update({model:results_metrics})

        #eavluate model
        prediction,validation_metric,results_metrics = ModelEvaluate(
                                candidate_model=candidate,
                                model_name=model_name,
                                fitted_values=fitted_values,
                                type_model=type_model,
                                train=train_to_model,
                                test=test_to_model,
                                validation_metric=validation_metric,
                                compare_metrics=test_compare_metrics,
                                config=config
                                ).run()
        
        #Hierarchical Reconcilier
        df_post_processed,recmethods = DatasetReconciliator(
                        Y_df = Y_df,
                        S_df = S_df,
                        tags = tags,
                        model_name = model_name,
                        methods = methods,
                        mid_level = mid_level,
                        min_trace_methods = min_trace_methods,
                        fitted_values = fitted_values,
                        prediction = prediction
                    ).run()
        
        CompareRecMethods(
                                model_name=model_name,
                                df_post_processed=df_post_processed,      # seu df com as colunas reconciliadas (com "/")
                                fitted_values=fitted_values,       # só unique_id, ds, y do treino
                                test=test_to_model[[id_col,time_col,target_col]],                  # unique_id, ds, y do período de teste
                                config=config
                            ).run()
        
                                    
        for method in recmethods:
            
            df_rec_method = df_post_processed[[id_col,time_col,method]]

            # Evaluate with each hierarchical reconcilier
            prediction,validation_metric,results_metrics = ModelEvaluate(
                                                                candidate_model=None,
                                                                model_name=method,
                                                                fitted_values=fitted_values,
                                                                type_model=type_model,
                                                                train=train_to_model,
                                                                test=test_to_model,
                                                                validation_metric=validation_metric,
                                                                compare_metrics=test_compare_metrics,
                                                                config=config,
                                                                prediction=df_rec_method
                                                                ).run()
        
        