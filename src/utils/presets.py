# src/utils/presets.py
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences, LocalBoxCox, LocalStandardScaler

# Mapeamento: string do YAML → objeto real
LAG_TRANSFORM_MAP = {
    "expanding_mean": ExpandingMean(),
    "rolling_mean_3": RollingMean(window_size=3),
    "rolling_mean_6": RollingMean(window_size=6),
    "rolling_mean_12": RollingMean(window_size=12),
}

def build_lag_transforms(transform_cfg: dict) -> dict:
    """Converte config do YAML em dict de objetos reais"""
    if not transform_cfg:
        return {}
    
    result = {}
    for lag_str, transform_names in transform_cfg.items():
        lag = int(lag_str)
        result[lag] = [LAG_TRANSFORM_MAP[name] for name in transform_names]
    return result

TARGET_TRANSFORM_MAP = {
    "differences_1": Differences([1]),
    "boxcox": LocalBoxCox(),
    "standard_scaler": LocalStandardScaler()
}


def build_target_transforms(transform_names: list) -> list:
    """Converte lista de strings em lista de objetos"""
    if not transform_names:
        return []
    return [TARGET_TRANSFORM_MAP[name] for name in transform_names]