# src/utils/model_registry.py
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoTheta, 
    SeasonalNaive, HistoricAverage
)

MODEL_REGISTRY = {
    "LGBMRegressor": LGBMRegressor,
    "CatBoostRegressor": CatBoostRegressor,
    "XGBRegressor": XGBRegressor,
    "AutoARIMA": AutoARIMA,
    "AutoETS": AutoETS,
    "AutoTheta": AutoTheta,
    "SeasonalNaive": SeasonalNaive,
    "HistoricAverage": HistoricAverage,
}

def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{name}' não encontrado. Disponíveis: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]