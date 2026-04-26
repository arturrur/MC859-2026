import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
INSTANCES_DIR = os.path.join(BASE_DIR, "instances")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

def filter_and_analyze(graph_filename, threshold):
    os.makedirs(INSTANCES_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    input_path = os.path.join(INSTANCES_DIR, graph_filename)
    if not os.path.exists(input_path):
        print(f"Erro: {graph_filename} não encontrado.")
        return

    G = nx.read_graphml(input_path)
    
    # Filtragem por threshold
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if float(d.get('weight', 1.0)) < threshold]
    G.remove_edges_from(edges_to_remove)
    
    # Remoção de nós isolados (conforme exigência do professor)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # Cálculo de métricas para o terminal
    num_nodes = G.number_of_nodes()
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
    components = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in components]

    print(f"\n" + "="*40)
    print(f"RESULTADOS: {graph_filename}")
    print(f"Threshold: {threshold} | Vértices: {num_nodes} | Arestas: {G.number_of_edges()}")
    print(f"Grau Médio: {avg_degree:.2f} | Componentes: {len(components)}")
    print(f"Maior Comp: {max(comp_sizes) if comp_sizes else 0}")
    print("="*40)

    # Salva instância filtrada para o Cytoscape
    graph_base_name = graph_filename.replace(".graphml", "")
    output_filename = f"{graph_base_name}_filtered_{threshold}.graphml"
    nx.write_graphml(G, os.path.join(INSTANCES_DIR, output_filename))

def generate_unified_dist_plots(instances_list):
    """Gera PDF comparativo da Distribuição de Graus (k vs P(k))"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Comparação da Distribuição de Graus (P(k))', fontsize=16, fontweight='bold')
    colors = ['royalblue', 'seagreen', 'indianred']
    
    for i, (filename, label) in enumerate(instances_list):
        path = os.path.join(INSTANCES_DIR, filename)
        if not os.path.exists(path): continue
        G = nx.read_graphml(path)
        degrees = [d for n, d in G.degree()]
        axes[i].hist(degrees, bins='auto', density=True, color=colors[i], edgecolor='black', alpha=0.7)
        axes[i].set_title(label)
        axes[i].set_xlabel("Grau (k)")
        if i == 0: axes[i].set_ylabel("Densidade P(k)")
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(ANALYSIS_DIR, "comparacao_distribuicoes.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def generate_unified_comp_plots(instances_list):
    """Gera PDF comparativo do Tamanho das Componentes (k vs contagem)"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distribuição de Tamanhos das Componentes', fontsize=16, fontweight='bold')
    colors = ['royalblue', 'seagreen', 'indianred']
    
    for i, (filename, label) in enumerate(instances_list):
        path = os.path.join(INSTANCES_DIR, filename)
        if not os.path.exists(path): continue
        G = nx.read_graphml(path)
        comp_sizes = [len(c) for c in nx.connected_components(G)]
        unique_sizes, counts = np.unique(comp_sizes, return_counts=True)
        
        axes[i].bar(unique_sizes, counts, color=colors[i], edgecolor='black', alpha=0.7)
        axes[i].set_title(label)
        axes[i].set_xlabel("Tamanho (k)")
        if i == 0: axes[i].set_ylabel("Quantidade de Componentes")
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(ANALYSIS_DIR, "comparacao_componentes.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

# --- EXECUÇÃO ---

# 1. Filtra os grafos e gera os arquivos .graphml
filter_and_analyze("theme.graphml", threshold=0.7)
filter_and_analyze("attributes.graphml", threshold=0.7)
filter_and_analyze("semantic.graphml", threshold=0.8)

# 2. Lista das instâncias recém-geradas para os plots unificados
instancias = [
    ("theme_filtered_0.7.graphml", "Grafo 1: Temas"),
    ("attributes_filtered_0.7.graphml", "Grafo 2: Atributos"),
    ("semantic_filtered_0.8.graphml", "Grafo 3: Semântico")
]

# 3. Gera as visualizações unificadas em PDF (Vetorial)
generate_unified_dist_plots(instancias)
generate_unified_comp_plots(instancias)