import pandas as pd
import numpy as np
import networkx as nx
import sys
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EDGE_THRESHOLD = 0.8


# 1. Configurações e Carga de Dados
print("Carregando dataset...")
df = pd.read_csv('../data/anime_raw_data.csv')

df = df.dropna(subset=['synopsis'])
df = df[df['synopsis'].str.len() > 20]

# O segredo é usar regex=False para tratar a barra e a aspa como texto comum
# Remove a barra invertida literal que precede a aspa simples
df['synopsis'] = df['synopsis'].str.replace(r"\'", "'", regex=False)
df['title'] = df['title'].str.replace(r"\'", "'", regex=False)

# Caso a barra esteja duplicada no arquivo (comum em CSVs zoados)
df['synopsis'] = df['synopsis'].str.replace("\\", "", regex=False)

# Usado para explodir animes com mais de 1 studio
def extract_studios(x):
    try:
        # Como você confirmou: é sempre string e começa com '['
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
    print(len(embeddings[0]))
    sys.exit(0)
    
    # Mean Pooling: O DNA semântico do estúdio é a média de suas obras
    studio_avg_vector = np.mean(embeddings, axis=0)
    studio_vectors.append(studio_avg_vector)
    
    if i % 50 == 0:
        print(f"Checkpoint: {i+1}/{len(studios_list)} estúdios processados.")

studio_vectors = np.array(studio_vectors)

# Calculando matriz de similaridade
sim_matrix = cosine_similarity(studio_vectors)

# Contruindo o grafo
G3 = nx.Graph()

for i in range(len(studios_list)):
    for j in range(i + 1, len(studios_list)):
        sim = sim_matrix[i, j]
        if sim > EDGE_THRESHOLD:
            G3.add_edge(studios_list[i], studios_list[j], weight=float(sim))

# Removes vértices isolados
G3.remove_nodes_from(list(nx.isolates(G3)))

# Salva o grafo
output_file = "graph.graphml"
nx.write_graphml(G3, output_file)

print("-" * 30)
print(f"Grafo finalizado: {G3.number_of_nodes()} nós e {G3.number_of_edges()} arestas.")
print(f"Arquivo '{output_file}' gerado com sucesso!")
print("Dica: No Cytoscape, use o layout 'Prefuse Force Directed' para ver os clusters semânticos.")
 