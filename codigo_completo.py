
#importando bibliotecas
from math import nan
import pandas as pd
import seaborn as sns
import catboost as cat
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier  
from sklearn.metrics import  classification_report 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#importando base de dados
df_11 = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\RAIS_CTPS_CAGED_2011.csv", sep=";")
df_12 = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\RAIS_CTPS_CAGED_2012.csv", sep=";")
df_13 = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\RAIS_CTPS_CAGED_2013.csv", sep=";")
df_14 = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\RAIS_CTPS_CAGED_2014.csv", sep=";")
df_15 = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\RAIS_CTPS_CAGED_2015.csv", sep=";")
df_16 = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\RAIS_CTPS_CAGED_2016.csv", sep=";")
df_17 = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\RAIS_CTPS_CAGED_2017.csv", sep=";")
df_18 = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\RAIS_CTPS_CAGED_2018.csv", sep=";")
df_19 = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\RAIS_CTPS_CAGED_2019.csv", sep=";")

#importando dicionarios
df_muni = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\municipio_map.csv", sep=";")
df_cbo = pd.read_csv(r"C:\Users\ninja\OneDrive\Documentos\rct2\Awari\Trabalho Final\Brasil\Dados Fonte\CBO_map.csv", sep=";")


# adicionando inflação de cada DF e unificando
df_11 = df_11.assign(Ano=2011)
df_12 = df_12.assign(Ano=2012)
df_13 = df_13.assign(Ano=2013)
df_14 = df_14.assign(Ano=2014)
df_15 = df_15.assign(Ano=2015)
df_16 = df_16.assign(Ano=2016)
df_17 = df_17.assign(Ano=2017)
df_18 = df_18.assign(Ano=2018)
df_19 = df_19.assign(Ano=2019)

#inflação de anual
df_11 = df_11.assign(Inflacao=0.065)
df_12 = df_12.assign(Inflacao=0.0584)
df_13 = df_13.assign(Inflacao=0.0591)
df_14 = df_14.assign(Inflacao=0.0641)
df_15 = df_15.assign(Inflacao=0.1067)
df_16 = df_16.assign(Inflacao=0.0629)
df_17 = df_17.assign(Inflacao=0.0295)
df_18 = df_18.assign(Inflacao=0.0375)
df_19 = df_19.assign(Inflacao=0.0431)

#carregando maps

#Municipio-map
df_muni = df_muni.rename(columns = {'municipo':'municipio'})
df_muni = df_muni[[w.isnumeric() for w in df_muni["municipio"]]].reset_index(drop=True)
df_muni["municipio"] = df_muni["municipio"].astype(int)
muni_map = dict(df_muni.values) 

#CBO_2002-map
df_cbo = df_cbo.rename(columns = {'Classificação Brasileira de Ocupações, criada em 2002 - com atualizações':'classificacao'})
df_cbo = df_cbo[[w.isnumeric() for w in df_cbo["classificacao"]]].reset_index(drop=True)
df_cbo["classificacao"] = df_cbo["classificacao"].astype(int)
cbo_map = dict(df_cbo.values)

#Sexo-map
sexo_map = {1: "homem", 2: "mulher"}

#Continente map
conti_map = { "sia": "Asia", "frica": "Africa", "amrica central e caribe": "america central e caribe", "amrica do sul": "america do sul", "amrica do norte": "america do norte"}

#Raça-map
raca_map = {1:"indigena", 2:"branca", 4:"preta", 6:"amarela", 8:"parda", 9:"nao_ident", -1:"ignorado"}


# agrupando as base de dados
lista_concat = [df_11 , df_12 , df_13 , df_14, df_15 , df_16 , df_17 , df_18 , df_19 ]

concat_df = pd.concat(lista_concat)

# removendo nans 
# antes 1357088 rows
# depois 1184189  row
# Limpando DFs com coisas desnecessarias
def limpar_df(x):   
    x["cbo_2002"] = x["cbo_2002"].map(cbo_map)
    x["municipio"] = x["municipio"].map(muni_map)
    x["raca_cor"] = x["raca_cor"].map(raca_map)
    x["sexo"] = x["sexo"].map(sexo_map)   
    x["continente"] = x["continente"].str.normalize("NFKD").str.lower().str.encode("ascii", errors="ignore").str.decode("utf8")
    x["status_migratorio"] = x["status_migratorio"].str.normalize("NFKD").str.lower().str.encode("ascii", errors="ignore").str.decode("utf8")
    x["pais"] = x["pais"].str.normalize("NFKD").str.lower().str.encode("ascii", errors="ignore").str.decode("utf8")
    x["continente"] = x["continente"].map(conti_map)
    x = x.drop(columns=["cnae_20_subclas"])
    x = x.drop(columns=["movimento"])
    x = x.drop(columns=["cnae_20_classe"])
    x = x.drop(columns=["competencia"])
    x = x.drop(columns=["tipo_mov_desagregado"])
    x = x.drop(columns=["uf"]) 
    x = x.drop(columns=["indtrabintermitente"])
    x = x.drop(columns=["indtrabparcial"])
    x = x
    x = x.dropna()
    return x


# removendo nans 
# antes 1357088 rows
# depois 1184189  rows

#arrumando escrita e substituindo numeros por palavras


#Aplicando mudanças ao DF
concat_df = limpar_df(concat_df) 

df_machine = concat_df

#salvando pasta para Analise Exploratoria
concat_df.to_csv("Analise_Tableau.csv", sep=";")


# Inicio Machine Learning
plt.figure(figsize=(10,10))
sns.heatmap(df_machine.corr(), annot=True)


# Funcao que faz o calculo do IQR e retorna o limite minimo, maximo e a pct de outliers
def iqr(df):
    q1,q3 = df.quantile([.25,.75])
    iqr = q3-q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    pct_outliers = sum(~df.between(lower_bound, upper_bound))/len(~df.isnull())
    return lower_bound, upper_bound, round(pct_outliers,2)

# Funcao que faz o calculo dos percentis e retorna o limite minimo, maximo e a pct de outliers
def percentiles(df, percentile_level=.01):
    lower_bound, upper_bound = df.quantile([percentile_level, 1-percentile_level])
    pct_outliers = sum(~df.between(lower_bound, upper_bound))/len(df)
    return lower_bound, upper_bound, round(pct_outliers,2)


coluna = "salario_mensal"
for cbo in df_machine["cbo_2002"].unique():
    metodo = "iqr"
    lower_bound, upper_bound, pct_outliers = iqr(df_machine.loc[df_machine["cbo_2002"]==cbo, coluna])
    if pct_outliers>.035:
        metodo = "percentile"
        lower_bound, upper_bound, pct_outliers = percentiles(df_machine.loc[df_machine["cbo_2002"]==cbo, coluna])
    print(f"Coluna: {coluna}, serviço: {cbo}, metodo: {metodo}, pct_outliers: {pct_outliers}")

#começo 722624


for cbo in df_machine["cbo_2002"].unique():
    q1,q3 = df_machine.loc[df_machine["cbo_2002"]==cbo, "salario_mensal"].quantile([.25,.75])
    iqr = q3-q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    total_obs = len(df_machine.loc[df_machine["cbo_2002"]==cbo, "salario_mensal"])
    total_outliers = len(df_machine.loc[(df_machine["cbo_2002"]==cbo) & (~df_machine["salario_mensal"].between(lower,upper)), "salario_mensal"])
    df_machine.loc[(df_machine["cbo_2002"]==cbo) & (~df_machine["salario_mensal"].between(lower,upper)), "salario_mensal"] = np.nan
    print(f"cbo: {cbo}, total obs: {total_obs}, total outliers: {total_outliers}, pct_outliers: {total_outliers/total_obs}")



df_machine = df_machine.dropna()

df_machine = df_machine.drop(columns=["status_migratorio"])


#catboost regreção

X1 = df_machine[[w for w in df_machine.columns if w!="salario_mensal"]]
y1 = df_machine["salario_mensal"]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, train_size=.8, random_state=12345)

categorical_features1 = ['pais', 'continente', 'municipio', 'cbo_2002', 'sexo', 
         'raca_cor']

mode = cat.CatBoostRegressor(loss_function="RMSE")


mode.fit(X_train1, y_train1,
        
        cat_features=categorical_features1, # variaveis

        eval_set=(X_test1 , y_test1))

grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}



y_pred1 = mode.predict(X_test1)
rmse = (np.sqrt(mean_squared_error(y_test1, y_pred1)))
r2 = r2_score(y_test1, y_pred1)
print("Testing performance")
print("RMSE: {:.2f}".format(rmse))
print("R2: {:.2f}".format(r2))

def mape(y_test1, y_pred1):
    return np.mean(np.abs(y_test1 - y_pred1)/y_test1)


# verificando modelo

x = mape(y_test1, y_pred1)
print(x)


y = y_test1 - y_pred1

y.sum()
np.mean(y)

sns.displot(y_test1 - y_pred1)

np.mean(y)

np.mean(y)/np.mean(y_test1)


y_test1

y_pred1

y_test1 - y_pred1

np.mean(y_test1 - y_pred1)

np.mean(df_machine["salario_mensal"])

# vendo se esta overthink
y_pred_train = mode.predict(X_train1)

rmse = (np.sqrt(mean_squared_error(y_test1, y_pred1)))
rmse_over = (np.sqrt(mean_squared_error(y_train1, y_pred_train)))
print(rmse - rmse_over)

import catboost
use_best_model = None
catboost.save_model("modelo_regre.cbm")
model = CatBoostRegressor()
model.load_model("modelo_regre.cbm") 

#CatBoost Agrupamento
X = df_machine[[w for w in df_machine.columns if w!="nivel_instrucao"]]
y = df_machine["nivel_instrucao"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=12345)


categorical_features = ['pais', 'continente', 'municipio', 'cbo_2002', 'sexo', 
         'raca_cor', 'status_migratorio',]

cat = CatBoostClassifier(random_state=12345 , cat_features=categorical_features)
cat.fit(X_train, y_train, eval_set=(X_test, y_test))
y_pred = cat.predict(X_test)
print(classification_report(y_test, y_pred))
print(classification_report(y_train, cat.predict(X_train)))
cat.get_feature_importance(prettified=True)

use_best_model = None
cat.save_model("modelo_classif.cbm")
model = CatBoostClassifier()
model.load_model("modelo_classif.cbm") 