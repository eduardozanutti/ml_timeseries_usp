import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hierarchicalforecast.utils import aggregate


def read_csv_files(material):
    path = f"data/raw/{material}"
    all_files = os.listdir(path)
    df_list = []
    for file in all_files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, file))
            df_list.append(df)
    combined_df = pd.concat(df_list)
    combined_df.sort_index(inplace=True)
    return combined_df

def read_all_csv_files():
    base_path = "data/raw/"
    all_materials = os.listdir(base_path)
    data_list = []
    for material in all_materials:
        material_path = os.path.join(base_path, material)
        if os.path.isdir(material_path):
            data_list.append(read_csv_files(material))
    df = pd.concat(data_list)
    return df

def correcting_dtypes(df):
    df['mes'] = pd.to_datetime(df['mes'], format='%Y-%m-%d')
    df['marca'] = df['marca'].astype(str)
    df['idade_loja'] = df['idade_loja'].astype(int)
    df['clima_loja'] = df['clima_loja'].astype(str)
    df['porte_loja'] = df['porte_loja'].astype(str)
    df['perfil_loja'] = df['perfil_loja'].astype(str)
    return df

def stocks_removing_negatives(df):
    columns = df.columns
    for col in columns:
        if 'estoque' in col:
            df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
    return df

def plot_time_series_by_products(df, x=['mes','produto'], y='venda_unidades', title='Time Series by Products'):
    # Agrupa e prepara dados
    df_grp = df.groupby(x)[y].sum().reset_index()
    products = df_grp[x[-1]].unique()
    n = len(products)
    
    # Ajusta tamanho conforme número de produtos
    fig, axs = plt.subplots(n, 1, figsize=(14, max(3*n, 6)), sharex=True)
    if n == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=16)
    
    # Plota cada produto em um subplot
    for i, prod in enumerate(products):
        sub = df_grp[df_grp[x[-1]] == prod].sort_values(x[0])
        sns.lineplot(data=sub, x=x[0], y=y, ax=axs[i])
        axs[i].set_title(str(prod))
        axs[i].set_ylabel(y)
        axs[i].grid(alpha=0.3)
    
    axs[-1].set_xlabel(x[0])
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    return fig, axs

def plot_products_materials(df, x='mes', product_col='produto', material_col='material', y='venda_unidades', products=None, n_products=2, title='Time Series by Product (materials)'):
    # escolhe produtos (top n se não fornecido)
    if products is None:
        top = df.groupby(product_col)[y].sum().sort_values(ascending=False).head(n_products).index.tolist()
    else:
        top = products if isinstance(products, (list, tuple)) else [products]
    
    n = len(top)
    fig, axs = plt.subplots(n, 1, figsize=(14, max(3*n, 6)), sharex=True)
    if n == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=16)
    
    for i, prod in enumerate(top):
        sub = df[df[product_col] == prod].groupby([x, material_col])[y].sum().reset_index()
        pivot = sub.pivot(index=x, columns=material_col, values=y).fillna(0).sort_index()
        # plota todas as colunas (materiais) como linhas no mesmo subplot
        pivot.plot(ax=axs[i], legend=True, linewidth=1)
        axs[i].set_title(str(prod))
        axs[i].set_ylabel(y)
        axs[i].grid(alpha=0.3)
        axs[i].legend(title='material', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    axs[-1].set_xlabel(x)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig, axs

def plot_produtos_in_cds(df, x='mes', product_col='produto', cd_col='centro_distribuicao', y='venda_unidades', products=None, n_products=2, title='Time Series by Product (CDs)'):
    # escolhe produtos (top n se não fornecido)
    if products is None:
        top = df.groupby(product_col)[y].sum().sort_values(ascending=False).head(n_products).index.tolist()
    else:
        top = products if isinstance(products, (list, tuple)) else [products]
    
    n = len(top)
    fig, axs = plt.subplots(n, 1, figsize=(14, max(3*n, 6)), sharex=True)
    if n == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=16)
    
    for i, prod in enumerate(top):
        sub = df[df[product_col] == prod].groupby([x, cd_col])[y].sum().reset_index()
        pivot = sub.pivot(index=x, columns=cd_col, values=y).fillna(0).sort_index()
        # plota todas as colunas (cds) como linhas no mesmo subplot
        pivot.plot(ax=axs[i], legend=True, linewidth=1)
        axs[i].set_title(str(prod))
        axs[i].set_ylabel(y)
        axs[i].grid(alpha=0.3)
        axs[i].legend(title='cd', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    axs[-1].set_xlabel(x)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig, axs

def remove_materials_no_sales(df):
    df = df[~df['material'].isin(
            [
                'd2d02f574554bb7284161539e60d251b',
                '37227479c0bd1ef74da961eafb87f13b',
                '395be8e13e6a427d3dc713212d59539f'
            ])]
    return df

def plot_time_series_cds_products_by_stores(
    df,
    n_rows=3,
    n_cols=2,
    cds=None,
    products=None,
    top_stores=8,
    x='mes',
    y='venda_unidades',
    show_legend=False,
    mean_line=True,
    alpha_lines=0.6,
    linewidth_lines=0.8,
    linewidth_mean=3.0,
    figsize=(16, 3.8 * 3)
):
    """
    Plota uma grade de séries temporais com n_rows (CDs) x n_cols (produtos).
    
    Melhorias:
    - Linha de média agregada "firme" no centro
    - Legenda opcional (evita poluição)
    - Linhas das lojas mais finas e transparentes
    - Tratamento robusto de dados vazios
    - Foco em padrões agregados para EDA e forecasting
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        Dados com colunas: mes, centro_distribuicao, produto, loja, venda_unidades
    n_rows, n_cols : int
        Dimensões da grade (linhas = CDs, colunas = produtos)
    cds, products : list or None
        Listas específicas; se None, seleciona top por vendas
    top_stores : int
        Número de lojas mais vendidas por (CD, produto)
    show_legend : bool
        Mostrar legenda com IDs das lojas
    mean_line : bool
        Mostrar linha de média agregada (grossa e preta)
    """
    
    # === 1. Seleção automática de CDs e produtos (se não fornecidos) ===
    if products is None:
        products = df.groupby('produto')[y].sum().sort_values(ascending=False).head(n_cols).index.tolist()
    else:
        products = products if isinstance(products, (list, tuple)) else [products]
        products = products[:n_cols]

    if cds is None:
        cds = df.groupby('centro_distribuicao')[y].sum().sort_values(ascending=False).head(n_rows).index.tolist()
    else:
        cds = cds if isinstance(cds, (list, tuple)) else [cds]
        cds = cds[:n_rows]

    # Garantir dimensões
    if len(cds) < n_rows:
        cds.extend([None] * (n_rows - len(cds)))
    if len(products) < n_cols:
        products.extend([None] * (n_cols - len(products)))

    cds = cds[:n_rows]
    products = products[:n_cols]

    # === 2. Criação da figura ===
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    
    # Normaliza axs para indexação 2D
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = np.array([axs])
    elif n_cols == 1:
        axs = np.array([[ax] for ax in axs])
    else:
        axs = np.array(axs)

    # === 3. Coleta de handles para legenda comum (se ativada) ===
    all_handles = []
    all_labels = []

    # === 4. Loop por CD (linha) e Produto (coluna) ===
    for i, cd in enumerate(cds):
        for j, prod in enumerate(products):
            ax = axs[i, j] if n_rows > 1 and n_cols > 1 else axs[max(i, j)]

            if cd is None or prod is None:
                ax.text(0.5, 0.5, 'Configuração inválida', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{cd or 'N/A'} | {prod or 'N/A'}")
                continue

            # Filtra dados
            sub = df[(df['centro_distribuicao'] == cd) & (df['produto'] == prod)].copy()
            if sub.empty:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{cd} | {prod}")
                ax.set_ylabel(y if j == 0 else "")
                ax.grid(alpha=0.3)
                continue

            # Agrega por mês e loja
            grp = sub.groupby([x, 'loja'])[y].sum().reset_index()
            if grp.empty:
                ax.text(0.5, 0.5, 'Sem vendas', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{cd} | {prod}")
                continue

            # Seleciona top lojas
            top_lojas = grp.groupby('loja')[y].sum().nlargest(top_stores).index.tolist()
            pivot = grp[grp['loja'].isin(top_lojas)].pivot(index=x, columns='loja', values=y).fillna(0)
            pivot = pivot.sort_index()

            if pivot.empty:
                ax.text(0.5, 0.5, 'Sem lojas top', ha='center', va='center', transform=ax.transAxes)
            else:
                # Plota linhas das lojas (finas, transparentes)
                colors = plt.cm.tab20(np.linspace(0, 1, len(pivot.columns)))
                for idx, (col, color) in enumerate(zip(pivot.columns, colors)):
                    ax.plot(pivot.index, pivot[col], linewidth=linewidth_lines, alpha=alpha_lines, color=color, label=col)

                # Plota média agregada (linha firme)
                if mean_line and len(pivot.columns) > 1:
                    mean_series = pivot.mean(axis=1)
                    ax.plot(mean_series.index, mean_series, color='black', linewidth=linewidth_mean, label='Média Agregada')

                # Coleta handles/labels para legenda comum
                if show_legend and len(all_handles) == 0:
                    line_handles = [plt.Line2D([0], [0], color=colors[k % len(colors)], linewidth=linewidth_lines) for k in range(len(pivot.columns))]
                    if mean_line and len(pivot.columns) > 1:
                        line_handles.append(plt.Line2D([0], [0], color='black', linewidth=linewidth_mean))
                        all_labels = list(pivot.columns) + ['Média Agregada']
                    else:
                        all_labels = list(pivot.columns)
                    all_handles = line_handles

            # Configurações do subplot
            ax.set_title(f"{cd} | {prod}", fontsize=12, pad=10)
            if j == 0:
                ax.set_ylabel(y, fontsize=11)
            else:
                ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

    # === 5. Legenda comum (opcional) ===
    if show_legend and all_handles:
        fig.legend(
            all_handles, all_labels,
            title='Lojas (top) + Média',
            bbox_to_anchor=(1.02, 0.88),
            loc='upper left',
            ncol=1,
            fontsize='small',
            frameon=True,
            fancybox=True,
            shadow=False
        )

    # === 6. Ajuste final ===
    plt.tight_layout(rect=[0, 0, 0.88 if show_legend else 1.0, 1])
    plt.subplots_adjust(hspace=0.3)
    plt.show()

    return fig, axs

def plot_time_series_cds_products_by_store_size(df, n_rows=3, n_cols=2, cds=None, products=None, top_stores=8, x='mes', y='venda_unidades'):
    """
    Plota uma grade de séries temporais com n_rows x n_cols.
    Linhas = CDs (centro_distribuicao), Colunas = produtos.
    Em cada subplot plota as séries por porte_loja agregadas por mês (em vez de por loja).
    Por clareza limitamos por padrão às top `top_stores` categorias de porte_loja por venda no par (cd,produto).
    """
    if products is None:
        products = df.groupby('produto')[y].sum().sort_values(ascending=False).head(n_cols).index.tolist()
    else:
        products = products if isinstance(products, (list, tuple)) else [products]
        products = products[:n_cols]

    if cds is None:
        cds = df.groupby('centro_distribuicao')[y].sum().sort_values(ascending=False).head(n_rows).index.tolist()
    else:
        cds = cds if isinstance(cds, (list, tuple)) else [cds]
        cds = cds[:n_rows]

    assert len(cds) == n_rows, "Número de CDs deve ser igual a n_rows"
    assert len(products) == n_cols, "Número de produtos deve ser igual a n_cols"

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = np.array([axs])
    elif n_cols == 1:
        axs = np.array([[ax] for ax in axs])

    for i, cd in enumerate(cds):
        for j, prod in enumerate(products):
            ax = axs[i, j]
            sub = df[(df['centro_distribuicao'] == cd) & (df['produto'] == prod)].copy()
            if sub.empty:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center')
                ax.set_title(f"{cd} / {prod}")
                ax.set_ylabel(y if j == 0 else "")
                continue

            # agrega por mes e por porte_loja
            grp = sub.groupby([x, 'porte_loja'])[y].sum().reset_index()
            # seleciona top porte_loja para não poluir o gráfico
            top_portes = grp.groupby('porte_loja')[y].sum().nlargest(top_stores).index.tolist()
            pivot = grp[grp['porte_loja'].isin(top_portes)].pivot(index=x, columns='porte_loja', values=y).fillna(0).sort_index()

            if pivot.shape[1] == 0:
                ax.text(0.5, 0.5, 'Sem portes selecionados', ha='center', va='center')
            else:
                pivot.plot(ax=ax, linewidth=1, legend=False)
            ax.set_title(f"{cd}  |  {prod}")
            if j == 0:
                ax.set_ylabel(y)
            else:
                ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

    # legenda comum à direita (top portes)
    handles = []
    labels = []
    for i in range(n_rows):
        for j in range(n_cols):
            sub = df[(df['centro_distribuicao'] == cds[i]) & (df['produto'] == products[j])].copy()
            if not sub.empty:
                grp = sub.groupby(['porte_loja'])[y].sum()
                top_portes = grp.nlargest(top_stores).index.tolist()
                if top_portes:
                    line_handles = [plt.Line2D([0], [0], color=plt.cm.tab20(k % 20)) for k in range(len(top_portes))]
                    handles = line_handles
                    labels = top_portes
                    break
        if handles:
            break

    if handles:
        fig.legend(handles, labels, title='porte_loja (top)', bbox_to_anchor=(1.02, 0.9), loc='upper left', ncol=1, fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()
    return fig, axs

def plot_time_series_cds_products_by_store_weather(df, n_rows=3, n_cols=2, cds=None, products=None, top_stores=8, x='mes', y='venda_unidades'):
    """
    Plota uma grade de séries temporais com n_rows x n_cols.
    Linhas = CDs (centro_distribuicao), Colunas = produtos.
    Em cada subplot plota as séries por porte_loja agregadas por mês (em vez de por loja).
    Por clareza limitamos por padrão às top `top_stores` categorias de porte_loja por venda no par (cd,produto).
    """
    if products is None:
        products = df.groupby('produto')[y].sum().sort_values(ascending=False).head(n_cols).index.tolist()
    else:
        products = products if isinstance(products, (list, tuple)) else [products]
        products = products[:n_cols]

    if cds is None:
        cds = df.groupby('centro_distribuicao')[y].sum().sort_values(ascending=False).head(n_rows).index.tolist()
    else:
        cds = cds if isinstance(cds, (list, tuple)) else [cds]
        cds = cds[:n_rows]

    assert len(cds) == n_rows, "Número de CDs deve ser igual a n_rows"
    assert len(products) == n_cols, "Número de produtos deve ser igual a n_cols"

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = np.array([axs])
    elif n_cols == 1:
        axs = np.array([[ax] for ax in axs])

    for i, cd in enumerate(cds):
        for j, prod in enumerate(products):
            ax = axs[i, j]
            sub = df[(df['centro_distribuicao'] == cd) & (df['produto'] == prod)].copy()
            if sub.empty:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center')
                ax.set_title(f"{cd} / {prod}")
                ax.set_ylabel(y if j == 0 else "")
                continue

            # agrega por mes e por porte_loja
            grp = sub.groupby([x, 'clima_loja'])[y].sum().reset_index()
            # seleciona top porte_loja para não poluir o gráfico
            top_portes = grp.groupby('clima_loja')[y].sum().nlargest(top_stores).index.tolist()
            pivot = grp[grp['clima_loja'].isin(top_portes)].pivot(index=x, columns='clima_loja', values=y).fillna(0).sort_index()

            if pivot.shape[1] == 0:
                ax.text(0.5, 0.5, 'Sem portes selecionados', ha='center', va='center')
            else:
                pivot.plot(ax=ax, linewidth=1, legend=False)
            ax.set_title(f"{cd}  |  {prod}")
            if j == 0:
                ax.set_ylabel(y)
            else:
                ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

    # legenda comum à direita (top portes)
    handles = []
    labels = []
    for i in range(n_rows):
        for j in range(n_cols):
            sub = df[(df['centro_distribuicao'] == cds[i]) & (df['produto'] == products[j])].copy()
            if not sub.empty:
                grp = sub.groupby(['clima_loja'])[y].sum()
                top_portes = grp.nlargest(top_stores).index.tolist()
                if top_portes:
                    line_handles = [plt.Line2D([0], [0], color=plt.cm.tab20(k % 20)) for k in range(len(top_portes))]
                    handles = line_handles
                    labels = top_portes
                    break
        if handles:
            break

    if handles:
        fig.legend(handles, labels, title='clima_loja (top)', bbox_to_anchor=(1.02, 0.9), loc='upper left', ncol=1, fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()
    return fig, axs

def plot_time_series_cds_products_by_store_profile(df, n_rows=3, n_cols=2, cds=None, products=None, top_stores=8, x='mes', y='venda_unidades'):
    """
    Plota uma grade de séries temporais com n_rows x n_cols.
    Linhas = CDs (centro_distribuicao), Colunas = produtos.
    Em cada subplot plota as séries por porte_loja agregadas por mês (em vez de por loja).
    Por clareza limitamos por padrão às top `top_stores` categorias de porte_loja por venda no par (cd,produto).
    """
    if products is None:
        products = df.groupby('produto')[y].sum().sort_values(ascending=False).head(n_cols).index.tolist()
    else:
        products = products if isinstance(products, (list, tuple)) else [products]
        products = products[:n_cols]

    if cds is None:
        cds = df.groupby('centro_distribuicao')[y].sum().sort_values(ascending=False).head(n_rows).index.tolist()
    else:
        cds = cds if isinstance(cds, (list, tuple)) else [cds]
        cds = cds[:n_rows]

    assert len(cds) == n_rows, "Número de CDs deve ser igual a n_rows"
    assert len(products) == n_cols, "Número de produtos deve ser igual a n_cols"

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = np.array([axs])
    elif n_cols == 1:
        axs = np.array([[ax] for ax in axs])

    for i, cd in enumerate(cds):
        for j, prod in enumerate(products):
            ax = axs[i, j]
            sub = df[(df['centro_distribuicao'] == cd) & (df['produto'] == prod)].copy()
            if sub.empty:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center')
                ax.set_title(f"{cd} / {prod}")
                ax.set_ylabel(y if j == 0 else "")
                continue

            # agrega por mes e por porte_loja
            grp = sub.groupby([x, 'perfil_loja'])[y].sum().reset_index()
            # seleciona top porte_loja para não poluir o gráfico
            top_portes = grp.groupby('perfil_loja')[y].sum().nlargest(top_stores).index.tolist()
            pivot = grp[grp['perfil_loja'].isin(top_portes)].pivot(index=x, columns='perfil_loja', values=y).fillna(0).sort_index()

            if pivot.shape[1] == 0:
                ax.text(0.5, 0.5, 'Sem portes selecionados', ha='center', va='center')
            else:
                pivot.plot(ax=ax, linewidth=1, legend=False)
            ax.set_title(f"{cd}  |  {prod}")
            if j == 0:
                ax.set_ylabel(y)
            else:
                ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

    # legenda comum à direita (top portes)
    handles = []
    labels = []
    for i in range(n_rows):
        for j in range(n_cols):
            sub = df[(df['centro_distribuicao'] == cds[i]) & (df['produto'] == products[j])].copy()
            if not sub.empty:
                grp = sub.groupby(['perfil_loja'])[y].sum()
                top_portes = grp.nlargest(top_stores).index.tolist()
                if top_portes:
                    line_handles = [plt.Line2D([0], [0], color=plt.cm.tab20(k % 20)) for k in range(len(top_portes))]
                    handles = line_handles
                    labels = top_portes
                    break
        if handles:
            break

    if handles:
        fig.legend(handles, labels, title='perfil_loja (top)', bbox_to_anchor=(1.02, 0.9), loc='upper left', ncol=1, fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()
    return fig, axs

def plot_time_series_cds_products_store_region(df, n_rows=3, n_cols=2, cds=None, products=None, top_stores=8, x='mes', y='venda_unidades'):
    """
    Plota uma grade de séries temporais com n_rows x n_cols.
    Linhas = CDs (centro_distribuicao), Colunas = produtos.
    Em cada subplot plota as séries por porte_loja agregadas por mês (em vez de por loja).
    Por clareza limitamos por padrão às top `top_stores` categorias de porte_loja por venda no par (cd,produto).
    """
    if products is None:
        products = df.groupby('produto')[y].sum().sort_values(ascending=False).head(n_cols).index.tolist()
    else:
        products = products if isinstance(products, (list, tuple)) else [products]
        products = products[:n_cols]

    if cds is None:
        cds = df.groupby('centro_distribuicao')[y].sum().sort_values(ascending=False).head(n_rows).index.tolist()
    else:
        cds = cds if isinstance(cds, (list, tuple)) else [cds]
        cds = cds[:n_rows]

    assert len(cds) == n_rows, "Número de CDs deve ser igual a n_rows"
    assert len(products) == n_cols, "Número de produtos deve ser igual a n_cols"

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = np.array([axs])
    elif n_cols == 1:
        axs = np.array([[ax] for ax in axs])

    for i, cd in enumerate(cds):
        for j, prod in enumerate(products):
            ax = axs[i, j]
            sub = df[(df['centro_distribuicao'] == cd) & (df['produto'] == prod)].copy()
            if sub.empty:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center')
                ax.set_title(f"{cd} / {prod}")
                ax.set_ylabel(y if j == 0 else "")
                continue

            # agrega por mes e por porte_loja
            grp = sub.groupby([x, 'regiao_loja'])[y].sum().reset_index()
            # seleciona top porte_loja para não poluir o gráfico
            top_portes = grp.groupby('regiao_loja')[y].sum().nlargest(top_stores).index.tolist()
            pivot = grp[grp['regiao_loja'].isin(top_portes)].pivot(index=x, columns='regiao_loja', values=y).fillna(0).sort_index()

            if pivot.shape[1] == 0:
                ax.text(0.5, 0.5, 'Sem portes selecionados', ha='center', va='center')
            else:
                pivot.plot(ax=ax, linewidth=1, legend=False)
            ax.set_title(f"{cd}  |  {prod}")
            if j == 0:
                ax.set_ylabel(y)
            else:
                ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

    # legenda comum à direita (top portes)
    handles = []
    labels = []
    for i in range(n_rows):
        for j in range(n_cols):
            sub = df[(df['centro_distribuicao'] == cds[i]) & (df['produto'] == products[j])].copy()
            if not sub.empty:
                grp = sub.groupby(['regiao_loja'])[y].sum()
                top_portes = grp.nlargest(top_stores).index.tolist()
                if top_portes:
                    line_handles = [plt.Line2D([0], [0], color=plt.cm.tab20(k % 20)) for k in range(len(top_portes))]
                    handles = line_handles
                    labels = top_portes
                    break
        if handles:
            break

    if handles:
        fig.legend(handles, labels, title='regiao_loja (top)', bbox_to_anchor=(1.02, 0.9), loc='upper left', ncol=1, fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()
    return fig, axs

def violin_plot_material_store_sales(df, product, materials=None, top_n_stores=400, show_points=False, y='venda_unidades'):
    """
    Para cada material do produto dado plota violins por mês da distribuição da variável y (vendas).
    - materials: lista de materiais ou None (usa todos do produto)
    - top_n_stores: limita às lojas mais frequentes para reduzir poluição visual
    - show_points: se True sobrepõe pontos (jitter) por loja
    - y: coluna a ser usada no eixo y (ex: 'venda_unidades' ou 'venda_valor_total')
    Retorna um dict {material: dataframe_filtrado} para inspeção.
    """
    import pandas as pd

    if product not in df['produto'].unique():
        print(f"Nenhum registro encontrado para o produto: {product}")
        return {}

    if y not in df.columns:
        print(f"Coluna '{y}' não encontrada no dataframe.")
        return {}

    df_prod = df[df['produto'] == product].copy()

    if materials is None:
        materials = df_prod['material'].unique().tolist()
    else:
        materials = [m for m in materials if m in df_prod['material'].unique()]

    if len(materials) == 0:
        print("Nenhum material válido encontrado para plotagem.")
        return {}

    pivots = {}
    n = len(materials)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, 4 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, mat in zip(axes, materials):
        data = df_prod[df_prod['material'] == mat].copy()
        data = data.dropna(subset=[y, 'mes', 'loja'])
        if data.empty:
            ax.set_visible(False)
            continue

        # top lojas por número de observações
        top_stores = data['loja'].value_counts().nlargest(top_n_stores).index.tolist()
        data_top = data[data['loja'].isin(top_stores)].copy()
        if data_top.empty:
            ax.set_visible(False)
            continue

        # criar coluna de mês ordenável (YYYY-MM) para o eixo x
        data_top['mes_str'] = data_top['mes'].dt.to_period('M').astype(str)
        # ordenar meses cronologicamente
        order = sorted(data_top['mes_str'].unique(), key=lambda x: pd.Period(x, freq='M'))

        # armazenar dataframe filtrado
        pivots[mat] = data_top

        # plot violino
        sns.violinplot(x='mes_str', y=y, data=data_top, order=order, inner='quartile', cut=0, ax=ax)
        if show_points:
            sns.stripplot(x='mes_str', y=y, data=data_top, order=order, color='k', size=2, jitter=True, alpha=0.35, ax=ax)

        ax.set_title(f'{product} - material: {mat} ({len(top_stores)} lojas)')
        ax.set_xlabel('Mês')
        ax.set_ylabel(y.replace('_', ' ').capitalize())
        ax.tick_params(axis='x', rotation=45)

    plt.show()
    return

def hierarchical_aggregation(df, spec, columns=None, ds='mes', y='venda_unidades'):
    if columns:
        df_h = df[columns].copy()
    df_h = df_h.dropna()
    df_h = df_h.rename(columns={ds: 'ds', y: 'y'})
    df_h['total'] = 'total'
    df_h['ds'] = pd.to_datetime(df_h['ds'])
    df_h = df_h.sort_values('ds')
    
    y_hier, S, tags = aggregate(
        df=df_h,
        spec=spec
    )

    print("Níveis criados:")
    for k, v in tags.items():
        print(f"{k}: {len(v)} séries")
    return y_hier, S, tags

def return_minimum_length(y_hier,S,tags,min_length=24):
    print(f"Filtrando séries com pelo menos {min_length} pontos...")
    series_lengths = y_hier.groupby('unique_id')['ds'].count().sort_values()
    print(f"Número de séries antes do filtro: {len(series_lengths)}")
    valid_ids = series_lengths[series_lengths >= min_length].index
    y_hier_filtered = y_hier[y_hier['unique_id'].isin(valid_ids)]
    print(f"Número de séries após o filtro: {len(valid_ids)}")
    bottom_tag = list(tags.keys())[-1]
    bottom_ids = tags[bottom_tag]
    valid_bottoms = [id_ for id_ in bottom_ids if id_ in valid_ids]
    # Filtra S: assume S.index são todos níveis, S.columns são bottoms
    S_filtered = S[valid_bottoms]  # Filtra colunas
    # Atualiza tags: pra cada level, filtra os que sobraram em valid_ids
    tags_filtered = {k: [id_ for id_ in v if id_ in valid_ids] for k, v in tags.items()}
    return y_hier_filtered,S_filtered,tags_filtered

def remove_time_series_min_length(y, min_length):
    group_cols = ['centro_distribuicao','regiao_loja','clima_loja','loja','produto','material','sku']
    # conta observações por série (combinação das colunas)
    counts = y.groupby(group_cols).size().reset_index(name='n_obs')
    # mantém apenas séries com pelo menos min_length observações
    valid = counts[counts['n_obs'] >= min_length][group_cols]
    # filtra o dataframe original pelas séries válidas
    return y.merge(valid, on=group_cols, how='inner')