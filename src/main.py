from data.make_dataset import DataPreparator

data_path='dados/'
keys = ['centro_distribuicao','regiao_loja','clima_loja','loja','produto','material','sku']
dtypes_dict = []
hierarchical_spec = ['total']
hierarchical = True
columns_selected = None
exclude_list_by_materials = [
                'd2d02f574554bb7284161539e60d251b',
                '37227479c0bd1ef74da961eafb87f13b',
                '395be8e13e6a427d3dc713212d59539f'
            ]

etl = DataPreparator(data_path,keys,dtypes_dict,hierarchical_spec,hierarchical,columns_selected,exclude_list_by_materials,min_time_series_lenght=24)
Y_df, S_df, tags = etl.run()
print(Y_df.head)