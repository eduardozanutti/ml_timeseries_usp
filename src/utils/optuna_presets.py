from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

def get_sampler_map(sampler_cfg):
    type = sampler_cfg['type']
    seed = sampler_cfg['seed']
    if sampler_cfg['type'] == 'TPESampler':
        SAMPLER_MAP = TPESampler(seed=seed)
    return SAMPLER_MAP

def get_pruner_map(pruner_cfg):
    type = pruner_cfg['type']
    n_startup_trials = pruner_cfg['n_startup_trials']

    if type == 'MedianPruner':
        PRUNER_MAP = MedianPruner(n_startup_trials=n_startup_trials)
    else:
        return {}
    return PRUNER_MAP
