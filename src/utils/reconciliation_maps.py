from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace

REC_MAPS = {
    'BottomUp':BottomUp,
    'TopDown':TopDown,
    'MiddleOut':MiddleOut,
    'MinTrace':MinTrace
}

def get_reconciliation_functions(method):
    return REC_MAPS[method]
