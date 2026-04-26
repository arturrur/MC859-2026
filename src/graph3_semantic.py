import pandas as pd
import numpy as np
import networkx as nx
import os
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "anime_raw_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "instances", "semantic.graphml")

df = pd.read_csv(INPUT_PATH)

# Limpeza básica
df = df.dropna(subset=['synopsis'])
df = df[df['synopsis'].str.len() > 20]
df['synopsis'] = df['synopsis'].str.replace(r"\'", "'", regex=False)
df['title'] = df['title'].str.replace(r"\'", "'", regex=False)
df['synopsis'] = df['synopsis'].str.replace("\\", "", regex=False)

# Usado para explodir animes com mais de 1 studio
def extract_studios(x):
    try:
        if isinstance(x, str) and x.startswith('['):
            data = ast.literal_eval(x)
            return [i['name'] for i in data]
        return [s.strip() for s in str(x).split(',')]
    except:
        return ['Unknown']

df['studio_list'] = df['studios'].apply(extract_studios)

df['text'] = df['title'] + ". " + df['synopsis']

df_exploded = df.explode('studio_list')

# 4. Preparação para o SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Agrupamos os conteúdos por estúdio
studio_groups = df_exploded.groupby('studio_list')['text'].apply(list).reset_index()

studio_vectors = []
studios_list = studio_groups['studio_list'].tolist()

print(f"Gerando embeddings...")

for i, contents in enumerate(studio_groups['text']):
    embeddings = model.encode(contents, show_progress_bar=False)
    
    # Mean Pooling: O DNA semântico do estúdio é a média de suas obras
    studio_avg_vector = np.mean(embeddings, axis=0)
    studio_vectors.append(studio_avg_vector)
    
    if i % 50 == 0:
        print(f"Checkpoint: {i+1}/{len(studios_list)} estúdios processados.")

studio_vectors = np.array(studio_vectors)

# Calculando matriz de similaridade
sim_matrix = cosine_similarity(studio_vectors)

# Contruindo o grafo
G = nx.Graph()

for i in range(len(studios_list)):
    for j in range(i + 1, len(studios_list)):
        sim = sim_matrix[i, j]
        G.add_edge(studios_list[i], studios_list[j], weight=float(sim))


nx.write_graphml(G, OUTPUT_PATH)