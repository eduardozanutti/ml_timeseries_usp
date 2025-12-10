# src/presets.py
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences, LocalBoxCox, LocalStandardScaler

# Presets para LightGBM (baseado no seu config_fn original)
LIGHTGBM_PRESET = {
    'candidate_lags': [
        [1],
        [13],
        [1, 13],
        range(1, 33),
    ],
    'candidate_lag_tfms': [
        {1: [RollingMean(window_size=13)]},
        {1: [RollingMean(window_size=13)], 13: [RollingMean(window_size=13)]},
        {13: [RollingMean(window_size=13)]},
        {4: [ExpandingMean(), RollingMean(window_size=4)], 8: [ExpandingMean(), RollingMean(window_size=4)]},
    ],
    'candidate_targ_tfms': [
        [Differences([1])],
        [LocalBoxCox()],
        [LocalStandardScaler()],
        [LocalBoxCox(), Differences([1])],
        [LocalBoxCox(), LocalStandardScaler()],
        [LocalBoxCox(), Differences([1]), LocalStandardScaler()],
    ],
    'param_ranges': {
        'learning_rate': (0.01, 0.1),
        'n_estimators': (10, 1000, True),
        'num_leaves': (31, 1024, True),
        'lambda_l1': (0.01, 10, True),
        'lambda_l2': (0.01, 10, True),
        'bagging_fraction': (0.75, 1.0),
        'feature_fraction': (0.75, 1.0),
    },
    'fixed_params': {
        'objective': 'regression',  # Corrigido de 'lr' (não existe; use 'poisson' se for counts)
        'bagging_freq': 1,
        'num_threads': 2,
        'verbose': -1,
        'force_col_wise': True,
    }
}

# Presets para XGBoost (exemplo; ajuste conforme necessário)
XGBOOST_PRESET = {
    'candidate_lags': [
        [1],
        [13],
        [1, 13],
        range(1, 33),
    ],
    'candidate_lag_tfms': [  # Mesmo do LGBM, ou customize
        {1: [RollingMean(window_size=13)]},
        {1: [RollingMean(window_size=13)], 13: [RollingMean(window_size=13)]},
        {13: [RollingMean(window_size=13)]},
        {4: [ExpandingMean(), RollingMean(window_size=4)], 8: [ExpandingMean(), RollingMean(window_size=4)]},
    ],
    'candidate_targ_tfms': [  # Mesmo
        [Differences([1])],
        [LocalBoxCox()],
        [LocalStandardScaler()],
        [LocalBoxCox(), Differences([1])],
        [LocalBoxCox(), LocalStandardScaler()],
        [LocalBoxCox(), Differences([1]), LocalStandardScaler()],
    ],
    'param_ranges': {
        'learning_rate': (0.005, 0.2),
        'n_estimators': (50, 2000, True),
        'lambda_l1': (0.001, 20, True),
        'lambda_l2': (0.001, 20, True),
        'max_depth': (3, 15),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
    },
    'fixed_params': {
        'objective': 'reg:squarederror',  # Ou 'count:poisson' se for counts
        'tree_method': 'hist',
        'enable_categorical': True,
        'n_jobs': 2,
        'verbosity': 0,
    }
}

# Função para pegar o preset baseado no model_name
def get_model_preset(model_name):
    model_name = model_name
    if model_name == 'LGBMRegressor':
        return LIGHTGBM_PRESET
    elif model_name == 'XGBRegressor':
        return XGBOOST_PRESET
    else:
        raise ValueError(f"Preset não encontrado para {model_name}. Adicione no presets.py.")