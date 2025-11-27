from statsforecast import StatsForecast
from statsforecast.models import MSTL

class HierarchicalTimeSeriesOutlierRemover:
    def __init__(config,y_hier):
        self.remove = config.get('outllier', {}).get('remove', True)
        self.iqr_factor = config.get('outllier', {}).get('iqr_factor', 3.0)
        self.season_length = config.get('outllier', {}).get('season_length', 12)
        self.y_hier = y_hier
        self.iqr_factor = iqr_factor
        self.season_length = season_length
        self.timeseries_list = y_hier['unique_id'].unique().tolist()

    def mstl_fit(self, df):
        sf = StatsForecast(
            models=[MSTL(season_length=self.season_length, stl_kwargs={"robust": True})],
            freq="M"
        )
        sf.fit(df)
        df_decompose = sf.fitted_[0, 0].model_
        return df_decompose

    def identify_outliers(self, df_decompose):
        q1 = df_decompose['remainder'].quantile(0.25)
        q3 = df_decompose['remainder'].quantile(0.75)
        iqr = q3 - q1

        outliers_mask = (
            (df_decompose['remainder'] < (q1 - self.iqr_factor * iqr)) |
            (df_decompose['remainder'] > (q3 + self.iqr_factor * iqr))
        )
        outliers = df_decompose.loc[outliers_mask]
        return outliers, outliers_mask

    def print_outliers_info(self, timeseries, df_decompose, outliers,i):
        print(f"\n📌 Série: {i} - {timeseries}")
        print(f"   → Tamanho total: {len(df_decompose)} pontos")
        print(f"   → Outliers detectados: {len(outliers)}")

        if len(outliers) > 0:
            print("   → Índices dos outliers:")
            for idx, value in outliers['remainder'].items():
                print(f"      - {idx}: remainder={value:.3f}")

    def remove_outliers(self, df_decompose, outliers_mask):
        # Substitui por NaN e interpola
        df_decompose.loc[outliers_mask, 'remainder'] = None
        df_decompose['remainder'] = df_decompose['remainder'].interpolate(method='linear', limit_direction='both')
        return df_decompose
    
    def recompose(self, df_decompose):
        return df_decompose['trend'] + df_decompose['seasonal'] + df_decompose['remainder']
    
    def update_dataframe(self, recomposed_without_outlier, timeseries):
        mask = self.y_hier['unique_id'] == timeseries
        self.y_hier.loc[mask, 'y'] = recomposed_without_outlier.values
    
    def run(self):
        n = len(self.timeseries_list)
        print(f"Iniciando limpeza de {n:,} séries hierárquicas...\n")

        for i, timeseries in enumerate(self.timeseries_list):
            if (i+1) % 500 == 0:
                print(f"   → Progresso: {i+1:,} séries processadas")

            df_ts = self.y_hier[self.y_hier['unique_id'] == timeseries]

            # 1. MSTL decomposition
            df_decompose = self.mstl_fit(df=df_ts)

            # 2. Detect outliers
            outliers, outliers_mask = self.identify_outliers(df_decompose)

            # 3. Print details
            self.print_outliers_info(timeseries, df_decompose, outliers,i)

            # 4. Correction
            if len(outliers) > 0:
                print("   → Corrigindo outliers via interpolação...")
                df_decompose = self.remove_outliers(df_decompose, outliers_mask)
            else:
                print("   → Nenhum outlier encontrado.")

            # 5. Recompose
            recomposed_without_outlier = self.recompose(df_decompose)

            # 6. Update final dataframe
            self.update_dataframe(recomposed_without_outlier, timeseries)

        print("\n✔ Limpeza concluída!")
        return self.y_hier
