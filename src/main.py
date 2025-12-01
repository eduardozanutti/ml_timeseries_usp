from src.data.make_dataset import DatasetCreator
from src.data.make_hierarchical_dataset import DatasetHierarchicalAggregator
from src.data.refine_dataset import HierarchicalTimeSeriesOutlierRemover
from src.data.split import hierarchical_train_test_split
from src.features.build_features import FeaturesBuilder
from src.config import load_config
from src.models.tuning import ModelTuning
from src.models.train_model import CreateCanditateModel
from src.evaluation.backtest_evaluation import ModelEvaluate

config = load_config()

if config['cache']:
    print('loading data...')
    df = DatasetCreator(config).load_intermediary()
    Hagg = DatasetHierarchicalAggregator(config,df)
    Y_df = Hagg.load_processed(filename='dataset.parquet')
    S_df = Hagg.load_processed(filename='structure.parquet')
    S_df = Hagg.load_tags(filename='tags.joblib')
    pass
else:
    df = DatasetCreator(config).run() 

    if config['exogen_features'] == 'true':
        df = FeaturesBuilder(df,config).run()
    else:
        pass

    Y_df,S_df,tags = DatasetHierarchicalAggregator(config,df).run()

train,test = hierarchical_train_test_split(Y_df)

models = config['models']
cv_config = config['modeling']['cv_config']
candidate_info = {}
candidate_performances = {}
for model in models:
    is_enable = models[model]['enabled']
    #check if model is enable in yaml file
    if is_enable:
        type_model = models[model]['type']
        if type_model == 'mlforecast':
            #Get Tuning info
            model_name = models[model]['regressor']
            fixed_params = models[model]['fixed_params']
            param_space = config['parameter_space'][model]
            mlforecast_params = config['modeling']['mlforecast']
            training_metric = config['modeling']['training_metric']
            compare_metrics = config['modeling']['compare_metrics']
            #Tuning model
            best_value,best_model_params,best_mlforecast_params,mlf_fit_params =\
                  ModelTuning(
                                df=train,
                                config=config,
                                model_name=model_name,
                                fixed_params=fixed_params,
                                param_space=param_space,
                                cv_config=cv_config,
                                mlforecast_params=mlforecast_params,
                                tuning_metric = training_metric
                            ).run()
            
            #create model with best tuning parameters
            candidate,metric,results_metrics = CreateCanditateModel(
                                        df=train,
                                        config=config,
                                        cv_config=cv_config,
                                        type_model=type_model,
                                        model_name=model_name,
                                        metric=best_value,
                                        cv_metric = training_metric,
                                        compare_metrics=compare_metrics,
                                        model_params=best_model_params,
                                        mlf_params=best_mlforecast_params,
                                        mlf_fit_params = mlf_fit_params
                                        ).run()
        else:
            #create model without tuning parameters
            candidate,metric,results_metrics = CreateCanditateModel(
                                        df=train,
                                        config=config,
                                        cv_config=cv_config,
                                        type_model=type_model,
                                        compare_metrics=compare_metrics,
                                        model_name=model_name
                                        ).run()
                
        candidate_info.update({metric:candidate})
        candidate_performances.update({model:results_metrics})

    
    validation_metric,validation_compare_metrics = ModelEvaluate(
                    candidate_model=candidate,
                    model_name=model_name,
                    train=train,
                    test=test,
                    n_months_test=6,
                    validation_metric='rmsse',
                    compare_metrics=compare_metrics,
                    output_plots_path=output_plots_path
                ).run()