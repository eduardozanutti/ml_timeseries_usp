import matplotlib.pyplot as plt
import pandas as pd
import os

class ModelEssembler:
    def __init__(self,config,candidate_info,candidate_performances):
        self.output_plots_path = config['paths']['plots']['plots_path']
        self.candidate_info = candidate_info
        self.candidate_performances = candidate_performances

    def create_candidate_performances_plot(self,title='Model Comparison (Cross-Validation)'):

        df = pd.DataFrame(candidate_performances).T.round(4)

        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.6)
        
        # Remove TODAS as bordas internas
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0)
            cell.set_edgecolor('white')

        # Cores
        cor_cabecalho = "#0D47A1"      # azul escuro
        cor_primeira_coluna = "#E8EAF6"  # azul bem claro

        # === CABEÇALHO (modelos) ===
        for j, modelo in enumerate(df.columns):
            cell = table[(0, j)]
            cell.set_linewidth(2)
            cell.set_edgecolor(cor_cabecalho)
            cell.get_text().set_color("white")
            cell.get_text().set_weight('bold')
            cell.set_facecolor(cor_cabecalho)

        # === PRIMEIRA COLUNA (métricas) ===
        for i, metrica in enumerate(df.index):
            cell = table[(i + 1, -1)]  # última coluna = nome da métrica
            cell.set_linewidth(2)
            cell.set_edgecolor(cor_cabecalho)
            cell.set_facecolor(cor_primeira_coluna)
            cell.get_text().set_weight('bold')
            cell.get_text().set_color("#1A237E")

        # === BORDA EXTERNA SUPERIOR E INFERIOR ===
        n_rows = len(df.index) + 1
        n_cols = len(df.columns)

        # Linha superior (acima do cabeçalho)
        for j in range(n_cols):
            table[(0, j)].set_linewidth(2)
            table[(0, j)].set_edgecolor(cor_cabecalho)

        # Linha inferior (abaixo da última linha)
        for j in range(-1, n_cols):
            table[(i, n_cols-1)].set_linewidth(2)
            table[(n_rows - 1, j)].set_edgecolor(cor_cabecalho)

        # Borda esquerda (só na coluna de métricas)
        for i in range(n_rows):
            table[(i, n_cols-1)].set_linewidth(2)
            table[(i, n_cols-1)].set_edgecolor(cor_cabecalho)

        # Borda direita (última coluna de valores)
        for i in range(n_rows):
            table[(i, n_cols - 1)].set_linewidth(2)
            table[(i, n_cols - 1)].set_edgecolor(cor_cabecalho)

        plt.title(title, fontsize=16, fontweight='bold', pad=35, color="#0D47A1")

        return plt

    def save_candidate_performances_plot(self,plt,filename='model_comparison.png'):
        if not os.path.exists(self.output_plots_path):
            os.makedirs(self.output_plots_path)
        save_path = os.path.join(self.output_plots_path, filename)
        print(save_path)
        plt.savefig(save_path,dpi=400, bbox_inches='tight', facecolor='white')
        print("Tabela salva com sucesso!")
        return

    # Função para Ensemble
    def load_and_ensemble(models_dir, test_df, h):
        models_info = {}
        
        # Ler todos os modelos e métricas
        for file in os.listdir(models_dir):
            if file.endswith('_best.pkl'):
                model_type = file.replace('_best.pkl', '')
                model_path = os.path.join(models_dir, file)
                mlf = joblib.load(model_path)
                
                info_path = os.path.join(models_dir, f'{model_type}_info.json')
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                models_info[model_type] = (mlf, info['metric'])
        
        # Gerar previsões de cada modelo
        preds = {}
        weights = {}
        total_weight = 0
        for model_type, (mlf, metric) in models_info.items():
            pred = mlf.predict(h=h, df=test_df)  # Previsão no test_df
            preds[model_type] = pred['prediction']  # Assuma coluna de pred
            
            # Peso inverso à metric (melhor metric = maior peso)
            weight = 1 / metric if metric != 0 else 1e-10
            weights[model_type] = weight
            total_weight += weight
        
        # Ensemble Weighted Average
        ensemble_pred = np.zeros(len(test_df))
        for model_type, weight in weights.items():
            ensemble_pred += preds[model_type] * (weight / total_weight)
        
        return ensemble_pred, models_info  # Retorna pred e info para análise
    
    def run(self):
        plt = self.create_candidate_performances_plot(title='Model Comparison (Cross-Validation)')
        self.save_candidate_performances_plot(plt,filename='model_comparison.png')
        return