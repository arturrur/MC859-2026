import os
import ast
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import euclidean_distances

numeric_cols = ['episodes', 'year', 'score', 'members', 'favorites', 'rank']
other_cols = ['type', 'demographics']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "anime_raw_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "instances", "attributes.graphml")

df = pd.read_csv(INPUT_PATH)

# Padronizar valores faltantes
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df[other_cols] = df[other_cols].fillna('Unknown')


# Multiplos studios e demographics por anime
def simplify_demo(x): # Para demographics resolvi só pegar a primeira mesmo
    try:
        # Se for string de lista '[{...}]', converte e pega o primeiro nome
        if isinstance(x, str) and x.startswith('['):
            data = ast.literal_eval(x)
            return data[0]['name'] if data else 'Unknown'
        return str(x)
    except:
        return 'Unknown'

def get_studio_list(x): # Separa studio em lista
    try:
        # Dado bruto é da forma '[{"name": "Sunrise"}]'
        if isinstance(x, str) and x.startswith('['):
            data = ast.literal_eval(x)
            return [i['name'] for i in data]
    except:
        return ['Unknown']

df['studio_names'] = df['studios'].apply(get_studio_list)
df['demographics'] = df['demographics'].apply(simplify_demo)

df_exploded = df.explode('studio_names') # Replica as linhas, uma com cada valor de studio

#onehot
ct = ColumnTransformer([
    ('onehot', OneHotEncoder(sparse_output=False), other_cols)
], remainder='passthrough')

# Aplicamos a transformação nas colunas de interesse
data_transformed = ct.fit_transform(df_exploded[numeric_cols + other_cols])
column_names = ct.get_feature_names_out()


# Agrupando por studio (usando média)
df_temp = pd.DataFrame(data_transformed, columns=column_names)
df_temp['studio_names'] = df_exploded['studio_names'].values
studio_features = df_temp.groupby('studio_names').mean()


# Normalizando (Z-Score)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(studio_features)

# Calculando Similaridade (Distância Euclidiana)
dist_matrix = euclidean_distances(features_scaled)
sim_matrix = 1 / (1 + dist_matrix) # Inverte distância para similaridade [0, 1]

# Contruindo o grafo
G = nx.Graph()
studios_list = studio_features.index.tolist()


for i in range(len(studios_list)):
    for j in range(i + 1, len(studios_list)):
        weight = sim_matrix[i, j]
        G.add_edge(studios_list[i], studios_list[j], weight=float(weight))


nx.write_graphml(G, OUTPUT_PATH)