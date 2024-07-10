#!/usr/bin/env python
# coding: utf-8
# # Importando o dataframe e as libs
# In[1]:
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error,mean_absolute_error
from sklearn.feature_selection import *
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor 
from kneed import KneeLocator
import seaborn as sns
from sklearn.svm import *
from sklearn.ensemble import *
import numpy as np
from datetime import datetime, date, timedelta
from sklearn import metrics
import glob
import os
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
cwd = os.getcwd()
get_ipython().run_line_magic('matplotlib', 'inline')
# # Verificando o  dataframe
# In[2]:
#importando o dataframe
df = pd.read_csv(".\\cota_hist\\cotacoes.csv.zip",compression='zip',index_col="index")
df.sample(5)
# In[3]:
df.describe()
# # Retorno diário
# In[4]:
get_ipython().run_cell_magic('capture', '', '#dataframe de ações\narray_siglas = df["sigla_acao"].unique()\n\ndf_enriquecido = pd.DataFrame()\nfor i in array_siglas:  \n    #ordenando por index\n    df_sigla = df[df["sigla_acao"] == i].sort_values(["data_pregao"])\n    #retorno\n    df_sigla["variacao"]=df_sigla["preco_fechamento"].diff()\n    #taxas de retorno\n    df_sigla["taxa_de_retorno_simples_diaria"]=df_sigla["variacao"]/df_sigla["preco_fechamento"].shift(1)  \n    df_enriquecido = df_enriquecido.append(df_sigla)')
# In[5]:
df_enriquecido[["data_pregao","sigla_acao","preco_fechamento","variacao","taxa_de_retorno_simples_diaria"]].sample(5)
# In[6]:
df_enriquecido["sigla_acao"].unique()
# # Retorno médio anual e volatilidade
# In[7]:
df_estatistico = df_enriquecido.groupby('sigla_acao').agg(
   preco_fechamento_min=pd.NamedAgg(column="preco_fechamento", aggfunc="min"),
   preco_fechamento_medio=pd.NamedAgg(column="preco_fechamento", aggfunc="mean"),
   preco_fechamento_inicial=pd.NamedAgg(column="preco_fechamento", aggfunc="first"),
   preco_fechamento_final=pd.NamedAgg(column="preco_fechamento", aggfunc="last"),
   preco_fechamento_variancia=pd.NamedAgg(column="preco_fechamento", aggfunc="var"),
   taxa_de_retorno_media_diaria=pd.NamedAgg(column="taxa_de_retorno_simples_diaria", aggfunc='mean'),
   desvpad_medio_diario = pd.NamedAgg(column="taxa_de_retorno_simples_diaria", aggfunc='std'))
df_estatistico["taxa_de_retorno_media_anual"]=df_estatistico["taxa_de_retorno_media_diaria"]*250 #pregões anuais
df_estatistico["taxa_de_retorno_media_anual"] = df_estatistico["taxa_de_retorno_media_anual"]
df_estatistico["volatilidade"]=df_estatistico["desvpad_medio_diario"]*250**0.5 #a volatilidade é o desvio padrão anualizado
    
df_estatistico["percentual_retorno"] = 100*(df_estatistico["preco_fechamento_final"] - df_estatistico["preco_fechamento_inicial"])/df_estatistico["preco_fechamento_inicial"]
df_estatistico= df_estatistico.dropna()
df_estatistico = df_estatistico[df_estatistico["percentual_retorno"]>0]
df_estatistico = df_estatistico[df_estatistico["taxa_de_retorno_media_anual"]>0]
df_estatistico=df_estatistico[["taxa_de_retorno_media_diaria","desvpad_medio_diario","taxa_de_retorno_media_anual","volatilidade"]]
df_estatistico = df_estatistico.round(6)
df_estatistico.sample(5)
# In[8]:
df_estatistico[df_estatistico.columns[0]].count()
# # Vizualizando os dados por volatilidade e taxa de retorno media anual
# In[9]:
#cada ponto em azul significa uma ação diferente.
plt.subplots(figsize=(15, 5)) 
plt.subplot(1, 2, 1)    
sns.scatterplot(data=df_estatistico, x= "volatilidade", y="taxa_de_retorno_media_anual")
plt.subplot(1, 2, 2)    
sns.boxplot(df_estatistico[['volatilidade',"taxa_de_retorno_media_anual"]])
plt.tight_layout() 
plt.show()
# # Removendo outliers - IQR
# In[10]:
#Removendo outliers
def remove_outlier(df_in, col_names):
    df_out=df_in
    for col_name in col_names :
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df_out = df_out.loc[(df_out[col_name] > fence_low) & (df_out[col_name] < fence_high)]
        
    return df_out
df_estatistico = remove_outlier(df_estatistico,["taxa_de_retorno_media_anual","volatilidade"])
sns.scatterplot(data=df_estatistico, x= "volatilidade", y="taxa_de_retorno_media_anual")
plt.show() 
# # Análise de clusters - definindo o número de clusters ideal 
# In[11]:
def elbow_definition(features):
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 600,
        "random_state": 42,
    }
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    kl = KneeLocator(
        range(1, 11), sse, curve="convex", direction="decreasing"
    )
    return kl.elbow
elbow1 = elbow_definition(df_estatistico[["taxa_de_retorno_media_anual","volatilidade"]])
print("elbows",elbow1)
# # Clusterizando pelo número ótimo de clusters (3)
# In[12]:
kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 600,
        "random_state": 42,
    }
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_estatistico[["taxa_de_retorno_media_anual","volatilidade"]])
kmeans = KMeans(n_clusters=elbow1, **kmeans_kwargs)
kmeans.fit(scaled_features)
y_kmeans = kmeans.predict(scaled_features)
sns.scatterplot(data=df_estatistico,  x= "volatilidade", y="taxa_de_retorno_media_anual", hue=y_kmeans, palette="deep",sizes=(400, 400))
plt.show() 
# # Agregando indicadores e cluster ao dataframe de pregões diários
# In[13]:
#craindo uma coluna com o valor do cluster
df_estatistico["cluster"] = y_kmeans
df_estatistico.sample(5)
# In[14]:
df_estatistico[df_estatistico.columns[0]].count()
# In[15]:
#Enriquecendo o dataframe de cotações diárias com os : trazendo para o dataframe via inner join (merge)
df =  pd.merge(df_enriquecido, df_estatistico, left_on="sigla_acao", right_index=True) 
df.sample(5)
# In[16]:
#colunas
df.columns
# In[17]:
#array de ações
array_siglas = df["sigla_acao"].unique()
array_siglas
# # Adicionando indicadores de análise técnica
# In[18]:
get_ipython().run_cell_magic('capture', '', 'df_enriched = pd.DataFrame()\nfor i in array_siglas:\n    #ordenando por data\n    df_sigla = df[df["sigla_acao"] == i].sort_values(["data_pregao"])\n    \n    #Bandas de Bollinger\n    for n in [5,13,60,200]:\n        df_sigla[f"mm_{n}_dias"]=df_sigla["preco_fechamento"].rolling(n).mean()#meio\n        df_sigla[f"desv_pad_{n}_dias"]=df_sigla["preco_fechamento"].rolling(n).std()#desvio padrao\n        k = 2 #padrão\n        df_sigla[f"banda_superior_{n}_dias"]=df_sigla[f"mm_{n}_dias"] + k*df_sigla[f"desv_pad_{n}_dias"] #banda superior\n        df_sigla[f"banda_inferior_{n}_dias"]=df_sigla[f"mm_{n}_dias"] - k*df_sigla[f"desv_pad_{n}_dias"] #banda inferior\n        df_sigla[f"volatilidade_entre_bandas_{n}_dias"] = 2*k*df_sigla[f"desv_pad_{n}_dias"] #diferença entre bandas\n        df_sigla[f"diferenca_fechamento_e_mm_{n}_dias"] = df_sigla["preco_fechamento"] - df_sigla[f"mm_{n}_dias"]  \n        \n        \n    #IFR — Índice de Força Relativa    \n    df_sigla["ganho"] = np.where(df_sigla[\'variacao\'] > 0, df_sigla[\'variacao\'], 0) \n    df_sigla["perda"] = np.where(df_sigla[\'variacao\'] < 0, df_sigla[\'variacao\'].abs(), 0) \n    #De acordo com J. Welles Wilder, criador do indicador, os parâmetros recomendados para análise são um período de 14 dias.\n    #Valores acima de 70 sugerem um ativo sobrecomprado e abaixo de 30 indicam um ativo sobrevendido.\n    n=14\n    df_sigla[f"mm_{n}_dias_ganho"]=df_sigla["ganho"].rolling(n).mean()\n    df_sigla[f"mm_{n}_dias_perda"]=df_sigla["perda"].abs().rolling(n).mean() #em absolutos\n    df_sigla["fr_simples"] =df_sigla[f"mm_{n}_dias_ganho"]/df_sigla[f"mm_{n}_dias_perda"]\n    df_sigla["fr_classica"] = (df_sigla[f"mm_{n}_dias_ganho"].shift(1)*(n-1)+df_sigla["ganho"])/\\\n    (df_sigla[f"mm_{n}_dias_perda"].shift(1)*(n-1)+df_sigla["perda"])\n    df_sigla["ifr_simples"] = 100 - (100/(1+df_sigla["fr_simples"]))\n    df_sigla["ifr_classica"] = 100 - (100/(1+df_sigla["fr_classica"]))\n    \n    #Cálculo de suporte/resistência > num padão de 5 candles, o candle central apresenta máxima ou mínima \n    #maior que os dois candles ao seu extremo\n    df_sigla[\'fractal_resistencia\'] = np.select([(df_sigla[\'preco_max\'] > df_sigla[\'preco_max\'].shift(2)) &\\\n     (df_sigla[\'preco_max\'] > df_sigla[\'preco_max\'].shift(1)) & (df_sigla[\'preco_max\'] > df_sigla[\'preco_max\'].shift(-1)) & \\\n     (df_sigla[\'preco_max\'] > df_sigla[\'preco_max\'].shift(-2))], [df_sigla[\'preco_max\']])\n    df_sigla[\'fractal_suporte\'] = np.select([(df_sigla[\'preco_min\'] < df_sigla[\'preco_min\'].shift(2)) &\\\n     (df_sigla[\'preco_min\'] < df_sigla[\'preco_min\'].shift(1)) & (df_sigla[\'preco_min\'] < df_sigla[\'preco_min\'].shift(-1)) & \\\n     (df_sigla[\'preco_min\'] < df_sigla[\'preco_min\'].shift(-2))], [df_sigla[\'preco_min\']])\n    #substitui 0 pelo valor anterior não nulo\n    df_sigla[\'fractal_resistencia\'] = df_sigla[\'fractal_resistencia\'].replace(to_replace=0, method=\'ffill\')\n    df_sigla[\'fractal_suporte\'] = df_sigla[\'fractal_suporte\'].replace(to_replace=0, method=\'ffill\')    \n    \n    #Estabelecendo linhas de suporte e resistência com base nos rompimentos\n    \n    df_sigla[\'linha_resistencia\'] = np.where((df_sigla["preco_min"] >=  df_sigla[\'fractal_suporte\']), \\\n                                             df_sigla[\'fractal_resistencia\'], df_sigla[\'fractal_suporte\'])\n    df_sigla[\'linha_suporte\'] = np.where((df_sigla["preco_min"] >=  df_sigla[\'fractal_suporte\']), df_sigla[\'fractal_suporte\'],\\\n                                         df_sigla["preco_min"])\n    df_sigla[\'linha_suporte\'] = np.where((df_sigla["preco_max"] <=  df_sigla[\'fractal_resistencia\']),\\\n                                         df_sigla[\'fractal_suporte\'], df_sigla["fractal_resistencia"])\n    df_sigla[\'linha_resistencia\'] = np.where((df_sigla["preco_max"] <=  df_sigla[\'fractal_resistencia\']),\\\n                                             df_sigla[\'fractal_resistencia\'], df_sigla["preco_max"])\n    \n    #Preço de fechamento dias futuros\n    f=1 #um dia\n    df_sigla["preco_{}_dias_futuros".format(f)]=df_sigla["preco_fechamento"].shift(-f)\n        \n    \n    df_enriched = df_enriched.append(df_sigla)\n')
# In[19]:
df_enriched.sample(10) #= df_enriched.drop(['variacao','taxa_de_retorno_simples_diaria'],axis=1)
# ### Checando se há nulos e infinitos
# In[20]:
#checando
data_to_drop = df_enriched.isin([np.inf, -np.inf, np.nan])
data_to_drop[data_to_drop == True].count()  
# In[21]:
# Transformando infinito em nan
df_enriched.replace([np.inf, -np.inf], np.nan, inplace=True)
#dropando esses valores
df_enriched=df_enriched.dropna()
#arredondando o que sobra
df_enriched = df_enriched.round(6)
df_enriched
# ### Codificando as features que são string
# In[22]:
le = LabelEncoder()
le.fit(df_enriched["sigla_acao"])
df_enriched["sigla_acao_label"]=le.transform(df_enriched["sigla_acao"])
le.fit(df_enriched["tipo"])
df_enriched["tipo_label"]=le.transform(df_enriched["tipo"])
# In[23]:
df_enriched
# # Iniciando nosso modelo de machine learning
# # Selecionando as melhores features - Verificando as colunas
# In[24]:
df_enriched.columns
# # Selecionando as melhores features: removendo informações redundantes
# In[25]:
df_enriched = df_enriched.drop(['variacao', 'taxa_de_retorno_simples_diaria','taxa_de_retorno_media_diaria',    'desvpad_medio_diario','ganho', 'perda', 'fr_simples', 'fr_classica','fractal_resistencia', 'fractal_suporte',    'desv_pad_5_dias','desv_pad_13_dias', 'desv_pad_60_dias','desv_pad_200_dias','mm_14_dias_ganho',                                'mm_14_dias_perda'], axis=1)
df_enriched.columns
# ## Separando features, labels e pré processamento
# In[26]:
features = df_enriched[['sigla_acao_label','tipo_label',  'preco_abertura', 'preco_max', 'preco_min', 'preco_fechamento',
       'qtd_negocios', 'vol_negocios','taxa_de_retorno_media_anual', 'volatilidade', 'cluster', 'mm_5_dias',
       'banda_superior_5_dias', 'banda_inferior_5_dias','volatilidade_entre_bandas_5_dias', 
       'diferenca_fechamento_e_mm_5_dias','mm_13_dias', 'banda_superior_13_dias', 'banda_inferior_13_dias',
        'volatilidade_entre_bandas_13_dias','diferenca_fechamento_e_mm_13_dias', 'mm_60_dias','banda_superior_60_dias', 
        'banda_inferior_60_dias','volatilidade_entre_bandas_60_dias', 'diferenca_fechamento_e_mm_60_dias', 'mm_200_dias',
       'banda_superior_200_dias', 'banda_inferior_200_dias','volatilidade_entre_bandas_200_dias',
        'diferenca_fechamento_e_mm_200_dias', 'ifr_simples', 'ifr_classica', 'linha_resistencia', 'linha_suporte']]
label_1 = df_enriched["preco_1_dias_futuros"]
#removendo outliers
scaler = MinMaxScaler().fit(features)
features_normalized = pd.DataFrame(scaler.transform(features),columns=features.columns)
features_normalized.sample(5)
# In[27]:
# melhores features regressão linear e regressão polinomial
f = features_normalized
l=label_1
feature_list = f.columns.values.tolist()
k_best_features = SelectKBest(f_regression , k="all")
k_best_features.fit_transform(f,l)
k_best_features_score = k_best_features.scores_
raw_pairs = zip(feature_list[:],k_best_features_score)
ordered_pairs= list(reversed(sorted(raw_pairs, key = lambda x: x[1])))
k_best_features_final =  dict(ordered_pairs[:])
total = sum(k_best_features_final.values())
k_best_features_percent = {key: value / total for key, value in k_best_features_final.items()}
best_features = k_best_features_final.keys()
print("para ---> ",l.name)
k_best_features_percent_above_1_percent = dict((k, v) for k, v in k_best_features_percent.items() if v >= 0.01/100)
best_features = k_best_features_percent_above_1_percent.keys()
print("features mais importantes ---> ",best_features)
# In[28]:
# melhores features Decision Tree e random forrest regressor
f = features_normalized
l=label_1
feature_list = f.columns.values.tolist()
k_best_features = SelectKBest(mutual_info_regression , k="all")
k_best_features.fit_transform(f,l)
k_best_features_score = k_best_features.scores_
raw_pairs = zip(feature_list[:],k_best_features_score)
ordered_pairs= list(reversed(sorted(raw_pairs, key = lambda x: x[1])))
k_best_features_final =  dict(ordered_pairs[:])
best_features = k_best_features_final.keys()
print("para ---> ",l.name)
k_best_features_percent_above_1_percent = dict((k, v) for k, v in k_best_features_percent.items() if v >= 0.01/100)
best_features = k_best_features_percent_above_1_percent.keys()
print("features mais importantes ---> ",best_features)
# ### Conclusões:
# Em todos os casos, as melhores features, que contribuem com pelo menos 0,1% do peso total, são:
# [['preco_fechamento', 'preco_max', 'preco_min', 'preco_abertura', 'mm_5_dias', 'banda_inferior_5_dias', 'linha_suporte', 'mm_13_dias', 'banda_superior_5_dias', 'linha_resistencia', 'banda_inferior_13_dias', 'banda_superior_13_dias', 'mm_60_dias', 'banda_inferior_60_dias', 'banda_superior_60_dias', 'mm_200_dias', 'banda_superior_200_dias', 'banda_inferior_200_dias', 'volatilidade_entre_bandas_200_dias', 'volatilidade_entre_bandas_60_dias', 'volatilidade_entre_bandas_13_dias', 'volatilidade_entre_bandas_5_dias']]
# # Modelos preditivos
# ## Preparando os dados
# In[29]:
df_enriched_acao = df_enriched
best_features = df_enriched_acao[['preco_fechamento', 'preco_max', 'preco_min', 'preco_abertura', 'mm_5_dias', 
                                  'banda_inferior_5_dias', 'linha_suporte', 'mm_13_dias', 'banda_superior_5_dias',
                                  'linha_resistencia', 'banda_inferior_13_dias', 'banda_superior_13_dias', 'mm_60_dias', 
                                  'banda_inferior_60_dias', 'banda_superior_60_dias', 'mm_200_dias', 
                                  'banda_superior_200_dias', 'banda_inferior_200_dias', 'volatilidade_entre_bandas_200_dias',
                                  'volatilidade_entre_bandas_60_dias', 'volatilidade_entre_bandas_13_dias', 
                                  'volatilidade_entre_bandas_5_dias']].sort_index()
                             
#dimensionamento das features
scaler = MinMaxScaler().fit(best_features)
best_features_normalized = scaler.transform(best_features)
label = df_enriched_acao["preco_1_dias_futuros"].sort_index()
#Criando um dataframe para aglutinar as previsões
df_final_prediction = df_enriched[["data_pregao","sigla_acao","nome_acao","preco_min", "preco_max",
                                   "preco_abertura","preco_fechamento"]].sort_index()
#separando dataset em treino, teste e validação
X_train, X_test, y_train, y_test = train_test_split(best_features_normalized, label, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42) 
print("Tamanho dos datasets de treino, teste e validação:")
print("X_train:",len(X_train),"X_test:", len(X_test),"X_val:", len(X_val),"\ny_train:", len(y_train),"y_test:",        len(y_test),"y_val:", len(y_val))
# ### Regressão linear
# In[30]:
#Regressão linear (treino e teste)
print("Construindo o modelo de regressão linear")
lr = linear_model.LinearRegression()
lr.fit(X_train,y_train)
prediction_test = lr.predict(X_test)
print(f"Coeficiente de determinação do modelo (Teste) R2 Score: {r2_score(y_test,prediction_test)*100:2f}%")
print(f"Mean Absolute Percentage Error - MAPE: {mean_absolute_percentage_error(y_test,prediction_test)*100:2f}%")
print(f"Porcentagem de acerto - (100 - MAPE): {100 - mean_absolute_percentage_error(y_test,prediction_test)*100:2f}%")
#Executando a previsão com regressão linear
final_prediction_array_lr = lr.predict(X_val)
print(f"Coeficiente de determinação do modelo (Validação) R2 Score: {r2_score(y_val,final_prediction_array_lr)*100:2f}%")
print(f"Mean Absolute Percentage Error - MAPE: {mean_absolute_percentage_error(y_val,final_prediction_array_lr)*100:2f}%")
print(f"Porcentagem de acerto - (100 - MAPE): {100 - mean_absolute_percentage_error(y_val,final_prediction_array_lr)*100:2f}%")
# ### Decision Tree
# In[31]:
get_ipython().run_cell_magic('capture', '', '#Decision Tree Regression (parametros)\nparameters_dtr = { \n    \'splitter\': [\'best\', \'random\'],\n    \'max_features\': [\'auto\', \'sqrt\', \'log2\'],\n    \'random_state\' : [42]\n}\n\nprint("Determinando os melhores parâmetros")\ngrid_dtr = GridSearchCV(DecisionTreeRegressor(),parameters_dtr,verbose=1)\ngrid_dtr.fit(X_train,y_train)')
# In[32]:
print("Construindo o modelo Decision Tree Regression com os melhores parâmetros encontrados")
print('best_estimator_', str(grid_dtr.best_estimator_)) 
print('best_score_', str(grid_dtr.best_score_))
dtr=grid_dtr.best_estimator_
dtr.fit(X_train,y_train)
prediction_test_dtr = dtr.predict(X_test)
print(f"Coeficiente de determinação do modelo (Teste) R2 Score: {r2_score(y_test,prediction_test_dtr)*100:2f}%")
print(f"Mean Absolute Percentage Error - MAPE: {mean_absolute_percentage_error(y_test,prediction_test_dtr)*100:2f}%")
print(f"Porcentagem de acerto - (100 - MAPE): {100 - mean_absolute_percentage_error(y_test,prediction_test_dtr)*100:2f}%")
#Executando a previsão com Decision Tree Regression
final_prediction_array_dtr = dtr.predict(X_val)
print(f"Coeficiente de determinação do modelo (Validação) R2 Score: {r2_score(y_val,final_prediction_array_dtr)*100:2f}%")
print(f"Mean Absolute Percentage Error - MAPE: {mean_absolute_percentage_error(y_val,final_prediction_array_dtr)*100:2f}%")
print(f"Porcentagem de acerto - (100 - MAPE): {100 - mean_absolute_percentage_error(y_val,final_prediction_array_dtr)*100:2f}%")
# ### Random Forest Regression
# In[33]:
get_ipython().run_cell_magic('capture', '', '#Random Forrest Regression (parametros)\nparameters_rfr = { \n    \'n_estimators\': [10,20],\n    \'max_features\': [\'sqrt\',\'auto\'],\n    \'max_depth\' : [10,20],\n    \'random_state\' : [42]\n}\n\nprint("Determinando os melhores parâmetros")\ngrid_rfr = GridSearchCV(RandomForestRegressor(),parameters_rfr,verbose=1,scoring=\'r2\')\ngrid_rfr.fit(X_train,y_train)')
# In[34]:
print("Construindo o modelo Random Forrest Regression com os melhores parâmetros encontrados")
print('best_estimator_', str(grid_rfr.best_estimator_)) 
print('best_score_', str(grid_rfr.best_score_))
rfr=grid_rfr.best_estimator_
rfr.fit(X_train,y_train)
prediction_test_rfr = rfr.predict(X_test)
print(f"Coeficiente de determinação do modelo (Teste) R2 Score: {r2_score(y_test,prediction_test_rfr)*100:2f}%")
print(f"Mean Absolute Percentage Error - MAPE: {mean_absolute_percentage_error(y_test,prediction_test_rfr)*100:2f}%")
print(f"Porcentagem de acerto - (100 - MAPE): {100 - mean_absolute_percentage_error(y_test,prediction_test_rfr)*100:2f}%")
#Executando a previsão com Random Forrest Regression
final_prediction_array_rfr = rfr.predict(X_val)
print(f"Coeficiente de determinação do modelo (Validação) R2 Score: {r2_score(y_val,final_prediction_array_rfr)*100:2f}%")
print(f"Mean Absolute Percentage Error - MAPE: {mean_absolute_percentage_error(y_val,final_prediction_array_rfr)*100:2f}%")
print(f"Porcentagem de acerto - (100 - MAPE): {100 - mean_absolute_percentage_error(y_val,final_prediction_array_rfr)*100:2f}%")
# ### Regressão polinomial
# In[35]:
#Regressão polinomial (treino e teste)
print("Construindo o modelo de regressão polinomial (3º grau)")
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = linear_model.LinearRegression()
pol_reg.fit(X_poly, y_train)
prediction_test_poly = pol_reg.predict(poly_reg.fit_transform(X_test))
print(f"Coeficiente de determinação do modelo (Teste) R2 Score: {r2_score(y_test,prediction_test_poly)*100:2f}%")
print(f"Mean Absolute Percentage Error - MAPE: {mean_absolute_percentage_error(y_test,prediction_test_poly)*100:2f}%")
print(f"Porcentagem de acerto - (100 - MAPE): {100 - mean_absolute_percentage_error(y_test,prediction_test_poly)*100:2f}%")
#Executando a previsão com regressão polinomial
final_prediction_array_poly = pol_reg.predict(poly_reg.fit_transform(X_val))
print(f"Coeficiente de determinação do modelo (Validação) R2 Score: {r2_score(y_val,final_prediction_array_poly)*100:2f}%")
print(f"Mean Absolute Percentage Error - MAPE: {mean_absolute_percentage_error(y_val,final_prediction_array_poly)*100:2f}%")
print(f"Porcentagem de acerto - (100 - MAPE): {100 - mean_absolute_percentage_error(y_val,final_prediction_array_poly)*100:2f}%")
# ### Dataframe de validações
# In[36]:
#Montando um dataframe com as previsões 
df_prediction = pd.DataFrame(y_val)
df_prediction[f"{label.name}_previsao_lr"] = final_prediction_array_lr.tolist()
df_prediction[f"{label.name}_previsao_dtr"] = final_prediction_array_dtr.tolist()
df_prediction[f"{label.name}_previsao_rfr"] = final_prediction_array_rfr.tolist()
df_prediction[f"{label.name}_previsao_poly"] = final_prediction_array_poly.tolist()
#Fazendo inner join (merge o pandas) entre a previsão e as datas
df_final_prediction =  pd.merge(df_final_prediction.sort_index(), df_prediction[[f"{label.name}",                     f"{label.name}_previsao_lr",f"{label.name}_previsao_dtr",                     f"{label.name}_previsao_rfr",f"{label.name}_previsao_poly"]]                                .sort_index(), left_index=True, right_index=True) 
df_final_prediction = df_final_prediction.sort_index()
# In[37]:
df_final_prediction[["sigla_acao","preco_1_dias_futuros","preco_1_dias_futuros_previsao_lr",                     "preco_1_dias_futuros_previsao_dtr", "preco_1_dias_futuros_previsao_rfr",                     "preco_1_dias_futuros_previsao_poly"]].sample(10)
# #### Gráficos
# In[38]:
#plotando graficamente
sigla = "ALPA4"
df_final_prediction_acao =  df_final_prediction[df_final_prediction["sigla_acao"]==sigla].tail(10)
f = "preco_1_dias_futuros"
plt.figure(figsize=(20 ,5))
plt.title(f"Peço fechamento previsto vs Preço fechamento real \n {f} {sigla}")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_dtr"],         label="Preço  Previsto D.T.R.", color="yellow",marker="d")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_rfr"],         label="Preço  Previsto R.F.R.", color="red",marker="^")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_poly"],         label="Preço  Previsto R.P.", color="blue",marker="s")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_lr"],         label="Preço  Previsto R.L.", color="green",marker="o")
plt.plot(df_final_prediction_acao["data_pregao"], df_final_prediction_acao[f"{f}"],label="Preço Real",          color="black",marker="o")
#get current axes
ax = plt.gca()
#hide x-axis
ax.get_xaxis().set_visible(False)
plt.legend()
plt.xlabel("Data Pregão")
plt.ylabel("Preço de Fechamento")
# In[39]:
#plotando graficamente
sigla = "PETR3"
df_final_prediction_acao =  df_final_prediction[df_final_prediction["sigla_acao"]==sigla].tail(10)
f = "preco_1_dias_futuros"
plt.figure(figsize=(20 ,5))
plt.title(f"Peço fechamento previsto vs Preço fechamento real \n {f} {sigla}")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_dtr"],         label="Preço  Previsto D.T.R.", color="yellow",marker="d")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_rfr"],         label="Preço  Previsto R.F.R.", color="red",marker="^")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_poly"],         label="Preço  Previsto R.P.", color="blue",marker="s")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_lr"],         label="Preço  Previsto R.L.", color="green",marker="o")
plt.plot(df_final_prediction_acao["data_pregao"], df_final_prediction_acao[f"{f}"],label="Preço Real",          color="black",marker="o")
#get current axes
ax = plt.gca()
#hide x-axis
ax.get_xaxis().set_visible(False)
plt.legend()
plt.xlabel("Data Pregão")
plt.ylabel("Preço de Fechamento")
# In[40]:
#plotando graficamente
sigla = "VALE3"
df_final_prediction_acao =  df_final_prediction[df_final_prediction["sigla_acao"]==sigla].tail(10)
f = "preco_1_dias_futuros"
plt.figure(figsize=(20 ,5))
plt.title(f"Peço fechamento previsto vs Preço fechamento real \n {f} {sigla}")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_dtr"],         label="Preço  Previsto D.T.R.", color="yellow",marker="d")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_rfr"],         label="Preço  Previsto R.F.R.", color="red",marker="^")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_poly"],         label="Preço  Previsto R.P.", color="blue",marker="s")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_lr"],         label="Preço  Previsto R.L.", color="green",marker="o")
plt.plot(df_final_prediction_acao["data_pregao"], df_final_prediction_acao[f"{f}"],label="Preço Real",          color="black",marker="o")
#get current axes
ax = plt.gca()
#hide x-axis
ax.get_xaxis().set_visible(False)
plt.legend()
plt.xlabel("Data Pregão")
plt.ylabel("Preço de Fechamento")
# In[41]:
#plotando graficamente
sigla = "VIVT3"
df_final_prediction_acao =  df_final_prediction[df_final_prediction["sigla_acao"]==sigla].tail(10)
f = "preco_1_dias_futuros"
plt.figure(figsize=(20 ,5))
plt.title(f"Peço fechamento previsto vs Preço fechamento real \n {f} {sigla}")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_dtr"],         label="Preço  Previsto D.T.R.", color="yellow",marker="d")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_rfr"],         label="Preço  Previsto R.F.R.", color="red",marker="^")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_poly"],         label="Preço  Previsto R.P.", color="blue",marker="s")
plt.plot(df_final_prediction_acao["data_pregao"],df_final_prediction_acao[f"{f}_previsao_lr"],         label="Preço  Previsto R.L.", color="green",marker="o")
plt.plot(df_final_prediction_acao["data_pregao"], df_final_prediction_acao[f"{f}"],label="Preço Real",          color="black",marker="o")
#get current axes
ax = plt.gca()
#hide x-axis
ax.get_xaxis().set_visible(False)
plt.legend()
plt.xlabel("Data Pregão")
plt.ylabel("Preço de Fechamento")
# In[ ]:
