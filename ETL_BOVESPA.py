#!/usr/bin/env python
# coding: utf-8
# # Importando as libs
# In[1]:
import pandas as pd
import glob
import os
import numpy as np
# # Importando os dados
# In[ ]:
#colunas necessárias (com base no layout) : data do pregão,código do bdi,sigla da ação,nome da ação, preço de abertura, 
#preço máximo, preço mínimo, preço de fechamento, quantidade de títulos negociados, volume de títulos negociados
nomes_colunas = ["data_pregao", "cod_bdi", "sigla_acao", "nome_acao", "preco_abertura", "preco_max",
                 "preco_min", "preco_fechamento", "qtd_negocios", "vol_negocios"
                ]
posicoes_colunas = [(2,10), (10,12), (12,24), (27,39), (56,69), (69,82), 
                    (82,95), (108,121), (152,170), (170,188)]
cwd = os.getcwd()
todos_os_arquivos =  glob.glob(cwd + "\cota_hist\COTAHIST*.zip")
df_cotacoes =pd.concat((pd.read_fwf(l,
                          compression='zip', colspecs = posicoes_colunas, names = nomes_colunas, skiprows =1, skipfooter=1)
                      for l in todos_os_arquivos), ignore_index=True)
df_cotacoes.sample(5)
# In[ ]:
df_cotacoes["sigla_acao"].unique()
# # Selecionando os dados - lote padrão e índice Ibovespa
# In[ ]:
#somente ações no lote padrão (cod_bdi=2)
df_cotacoes = df_cotacoes[df_cotacoes["cod_bdi"]==2]
df_cotacoes = df_cotacoes.drop(["cod_bdi"],1)
df_cotacoes.sample(5)
# In[ ]:
#pegando somente as siglas das ações do índice Ibovespa
df_ibov_index=pd.read_csv(".\\cota_hist\\ibov.csv",sep=";")
#filtrando somente as cotações do índice Ibovespa
df_cotacoes =  pd.merge(df_cotacoes, df_ibov_index, left_on="sigla_acao", right_on="codigo").drop(["codigo","acao","qtde_teorica","part_percentual"],axis=1)
df_cotacoes.sample(5)
# # Ajustando a formatação
# In[ ]:
#ajuste de data
df_cotacoes["data_pregao"]=pd.to_datetime(df_cotacoes["data_pregao"], format = "%Y%m%d")
df_cotacoes.sample(5)
# In[ ]:
#ajustes preços duas casas decimais
df_cotacoes["preco_abertura"] = (df_cotacoes["preco_abertura"]/100).astype(float)
df_cotacoes["preco_max"] = (df_cotacoes["preco_max"]/100).astype(float)
df_cotacoes["preco_min"] = (df_cotacoes["preco_min"]/100).astype(float)
df_cotacoes["preco_fechamento"] = (df_cotacoes["preco_fechamento"]/100).astype(float)
df_cotacoes.sample(5)
# # Lidando com duplicidades e ausências
# In[ ]:
#removendo possíveis duplicatas
df_cotacoes = df_cotacoes.drop_duplicates()
# In[ ]:
#checando se existem dados vazios ou infinitos
data_to_drop = df_cotacoes.isin([np.inf, -np.inf, np.nan])
  
df_cotacoes[data_to_drop == True].count()
# # Dataframe tratado
# In[ ]:
df_cotacoes.describe()
# In[ ]:
df_cotacoes.to_csv(cwd + "\cota_hist\cotacoes.csv.zip",index_label = "index",compression="zip")
# In[ ]:
df_cotacoes.sample(5)
# In[ ]:
df_cotacoes["sigla_acao"].unique()
# In[ ]:
