# src/utils/presets.py
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences, LocalBoxCox, LocalStandardScaler
from utilsforecast.losses import rmse, mae, smape, mase,rmsse,bias


# Mapeamento: string do YAML → objeto real
LAG_TRANSFORM_MAP = {
    "expanding_mean": ExpandingMean(),
    "rolling_mean_3": RollingMean(window_size=3),
    "rolling_mean_6": RollingMean(window_size=6),
    "rolling_mean_12": RollingMean(window_size=12),
}

def _get_transform_name(obj):
    if isinstance(obj, ExpandingMean):
        return "expanding_mean"
    if isinstance(obj, RollingMean):
        return f"rolling_mean_{obj.window_size}"
    if isinstance(obj, LocalBoxCox):
        return "boxcox"
    if isinstance(obj, LocalStandardScaler):
        return "standard_scaler"
    if isinstance(obj, Differences):
        return f"differences_{obj.differences[0]}"
    return str(obj.__class__.__name__)

def lag_transforms_to_config(d):
    if not d:
        return {}
    result = {}
    for lag, transforms in d.items():
        lag_str = str(lag)
        if len(transforms) == 1:
            result[lag_str] = _get_transform_name(transforms[0])
        else:
            result[lag_str] = [_get_transform_name(t) for t in transforms]
    return result

def target_transforms_to_config(lst):
    if not lst:
        return []
    return [_get_transform_name(t) for t in lst]

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


LOSSES_FUNCTIONS_MAP = {
    'rmse':rmse,
    'rmsse':rmsse,
    'mae':mae,
    'smape':smape,
    'mase':mase,
    'bias':bias,

}

def build_loss(loss_name):
    return LOSSES_FUNCTIONS_MAP[loss_name]