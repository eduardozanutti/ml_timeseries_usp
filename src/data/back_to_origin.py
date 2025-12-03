from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace
from hierarchicalforecast.core import HierarchicalReconciliation
import pandas as pd

methods = ['BottomUp','TopDown','MinTrace']#,'MiddleOut']
min_trace_methods = ['wls_var','wls_struct','mint_shrink']

REC_MAPS = {
    'BottomUp':BottomUp,
    'TopDown':TopDown,
    'MiddleOut':MiddleOut,
    'MinTrace':MinTrace
}

def get_reconciliation_function(method):
    return REC_MAPS[method]
    

class DatasetReconciliator:
    def __init__(self,Y_df,S_df,tags,model_name,methods,min_trace_methods,mid_level,fitted_values,prediction):
        self.Y_df = Y_df
        self.S_df = S_df
        self.tags = tags
        self.methods = methods
        self.min_trace_methods = min_trace_methods
        self.mid_level = mid_level
        self.fitted_values = fitted_values
        self.prediction = prediction
        self.model_name = model_name
    
    def reconciliate(self,Y_hat_df,Y_df):
        reconcilers = []
        for method in self.methods:
            method_function = get_reconciliation_function(method)
            if method == "MinTrace":
                for m in self.min_trace_methods:
                    reconcilers.append(method_function(method=m,nonnegative=False))
            else:
                if method == "TopDown":
                    reconcilers.append(method_function(method='forecast_proportions'))
                elif method == "MiddleOut":
                    reconcilers.append(method_function(top_down_method ='forecast_proportions',middle_level=self.mid_level))
                   
        rec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_rec = rec.reconcile(
                                Y_hat_df = Y_hat_df,
                                Y_df = Y_df,
                                S_df = self.S_df,
                                tags = self.tags
                                )
        return Y_rec
    
    def run(self):
        df_pre_processed = pd.concat([self.fitted_values,self.prediction])
       #Hierarchical reconciliation
        df_post_processed = self.reconciliate(
                                                Y_hat_df = self.prediction,
                                                Y_df = df_pre_processed
                                            )
        methods_names = df_post_processed.drop(columns=['unique_id','ds',self.model_name]).columns
       #crate a plot dataframe
        return df_post_processed, methods_names