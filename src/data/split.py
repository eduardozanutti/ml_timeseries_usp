# src/data/split.py

import pandas as pd

def hierarchical_train_test_split(
    df: pd.DataFrame,
    test_horizon: int = 6,
    time_col: str = "ds",
    id_col: str = "unique_id"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split hierárquico temporal para backtest real.
    Últimas N observações de cada série → test
    Resto → train
    """
    df = df.sort_values([id_col, time_col]).copy()
    
    test = df.groupby(id_col).tail(test_horizon)
    train = df.drop(test.index)
    
    print(f"Train: {train[time_col].min().date()} → {train[time_col].max().date()} "
          f"({train[time_col].nunique()} meses)")
    print(f"Test (backtest): {test[time_col].min().date()} → {test[time_col].max().date()} "
          f"({test[time_col].nunique()} meses)")
    
    return train, test
