import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import TfidfVectorizer # compare vectors
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


#########################
# 1 - Loading Data and Preprocessing
#########################

EDGE_THRESHOLD = 0.7


df = pd.read_csv('../data/anime_raw_data.csv')

def extract_names(column_data):
    try:
        data_list = ast.literal_eval(column_data)
        return " ".join([item['name'].replace(" ", "_") for item in data_list])
    except:
        return ""
    

df['studio_names'] = df['studios'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else [])
df['tags'] = df['genres'].apply(extract_names) + " " + df['themes'].apply(extract_names)

studios = {}
for _, row in df.iterrows():
    for studio in row['studio_names']:
        if studio not in studios:
            studios[studio] = ""
        studios[studio] += " " + row['tags']
        
#########################
# 2 - Calculating similarity_matrix
#########################

studios_names = list(studios.keys())
studios_tags = list(studios.values())

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(studios_tags)

sim_matrix = cosine_similarity(tfidf_matrix)



#########################
# 3 - Coloring the graph
#########################

studio_main_genre = {}

for studio, tags_string in studios.items():
    # Transformar a string de tags em uma lista (removendo espaços vazios)
    tag_list = [t for t in tags_string.split() if t.strip()]
    
    if tag_list:
        # Counter conta a frequência de cada tag
        most_common_tag = Counter(tag_list).most_common(1)[0][0]
        studio_main_genre[studio] = most_common_tag
    else:
        studio_main_genre[studio] = "Unknown"

unique_genres = sorted(list(set(studio_main_genre.values())))
num_genres = len(unique_genres)

colormap = plt.get_cmap('gist_rainbow') 

genre_color_map = {}
for i, genre in enumerate(unique_genres):
    # Gera uma cor baseada na posição do gênero na lista
    rgba = colormap(i / num_genres) 
    # Converte de RGBA para Hexadecimal (que o Cytoscape entende melhor)
    genre_color_map[genre] = mcolors.rgb2hex(rgba)

#########################
# 4 - Creating the graph
#########################
G = nx.Graph()
for s in studios_names:
    genre = studio_main_genre.get(s, "Unknown")
    G.add_node(s, 
               main_genre=genre, 
               node_color=genre_color_map[genre])


for i in range(len(studios_names)):
    for j in range(i + 1, len(studios_names)):
        similarity = sim_matrix[i][j]
        if similarity > EDGE_THRESHOLD:
            G.add_edge(studios_names[i], studios_names[j], weight=float(similarity))


# 5 - Removing edgeless nodes
isolados = [node for node, degree in dict(G.degree()).items() if degree == 0]

G.remove_nodes_from(isolados)

nx.write_graphml(G, "graph.graphml")