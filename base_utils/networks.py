import pandas as pd
import networkx as nx
from pathlib import Path
from collections import deque, defaultdict
import pickle as pk
import polars as pl
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from collections import defaultdict, Counter
import sys
import random
from networkx.algorithms.community import greedy_modularity_communities
import scipy.sparse as sp

# Obtains disease genes from a csv file containing disease gene mappings
def isolate_disease_genes(selected_disease, diseases = Path('/work/ccnr/GeneFormer/GeneFormer_repo/PPI/GDA_Filtered_04042022.csv'),
                          column_name = 'NewName', gene_column_name = 'HGNC_Symbol', filter_DisGeNET = False, filter_strong = False):
    '''
    FUNCTION
    Returns a list of genes with a given disease (or other condition) from aa dataframe
    Inputs
    -----------------
        databases : polars dataframe
            Database containing gene-dieases mappings

        selected_disease : string 
            Selected disease to filter dataframe for

        column_name : string
            Name of the column that contains the diseases

        gene_column_name : string
            Name of the column that contains genes

        filter_DisGeNET : bool, default = False
            If True, filters the disease gene list for only DisGeNET-indicated disease genes

    Outputs
    ----------------
        gene_list : list
            Array of filtered genes
        
    '''
    selected_disease = selected_disease.replace("\n", " ")
    diseases = pl.read_csv(diseases)
    
    if filter_DisGeNET == True and selected_disease != 'covid':
        diseases = diseases.filter(diseases['evidence'].str.contains('DisGeNET'))

    if filter_strong == True:
         diseases = diseases.filter(diseases['Strong'] > 0)
        
    filtered_dataframe = diseases.filter(diseases[column_name].str.contains(selected_disease.lower()))
    gene_list = list(filtered_dataframe[gene_column_name])

    return gene_list
    
# Obtains background disease genes against a given set (toggle filter strong on and off for bulk LCC analysis)
def isolate_background_genes(selected_diseases, diseases = Path('/work/ccnr/GeneFormer/GeneFormer_repo/PPI/GDA_Filtered_04042022.csv'), #PPI202207.txt
                          column_name = 'NewName', gene_column_name = 'HGNC_Symbol', filter_DisGeNET = False, filter_strong = False, num_LCCs = 200):
    '''
    FUNCTION
    Returns a list of genes with a given disease (or other condition) from aa dataframe
    Inputs
    -----------------
        databases : polars dataframe
            Database containing gene-dieases mappings

        selected_disease : string 
            Selected disease to filter dataframe for

        column_name : string
            Name of the column that contains the diseases

        gene_column_name : string
            Name of the column that contains genes

        filter_DisGeNET : bool, default = False
            If True, filters the disease gene list for only DisGeNET-indicated disease genes

    Outputs
    ----------------
        gene_list : list
            Array of filtered genes
        
    '''
    
    diseases = pl.read_csv(diseases)
    
    if filter_DisGeNET == True:
        diseases = diseases.filter(diseases['evidence'].str.contains('DisGeNET'))

    if filter_strong == True:
         diseases = diseases.filter(diseases['Strong'] > 0)
        
    all_possible_diseases = list(set(diseases[column_name]) - set(selected_diseases))
    all_possible_diseases = random.sample(all_possible_diseases, num_LCCs)
    
    if 'covid' in all_possible_diseases:
        all_possible_diseases.remove('covid')
        
    return all_possible_diseases
    
# Finds an LCC for a given disease module
def LCC_genes(PPI, disease_genes, subgraph=True, connect_non_LCC=True):
    # Obtains LCC
    sub = PPI.subgraph(disease_genes)
    largest_cc = max(nx.connected_components(sub), key=len)
    LCC_nodes = set(largest_cc)

    if connect_non_LCC:
        for gene in set(disease_genes) - LCC_nodes:
            # Find the shortest path to any node in the LCC
            shortest_path = None
            for target in LCC_nodes:
                try:
                    path = nx.shortest_path(PPI, source=gene, target=target)
                    if shortest_path is None or len(path) < len(shortest_path):
                        shortest_path = path
                except:
                    continue
            
            # Add the nodes and edges from this path to the LCC
            if shortest_path:
                LCC_nodes.update(shortest_path)

    # Return the subgraph or list of nodes depending on the 'subgraph' argument
    if subgraph:
        return PPI.subgraph(LCC_nodes)
    else:
        return list(LCC_nodes)
    
# Maps gene embeddings to PPI
def instantiate_ppi(PPI = Path('/work/ccnr/GeneFormer/GeneFormer_repo/PPI/PPI_2022_2022-04-21.csv'), #Path('PPI/GRN.csv'), #Path('/work/ccnr/GeneFormer/GeneFormer_repo/PPI/PPI_2022_2022-04-21.csv'), 
                    gene_ids = Path("/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/gene_name_id_dict.pkl"),
                    save = False):

    # All possible genes that can be mapped            
    conversion = list(pk.load(open(gene_ids, 'rb')).keys())
    
    # Loads PPI using pandas
    if '.txt' in str(PPI) or '.tsv' in str(PPI):
        PPI = pd.read_table(PPI)
    else:
        try:
            PPI = pd.read_csv(PPI).drop(columns = ['Source'])
            PPI = PPI.rename(columns = {'HGNC_Symbol.1':'source', 'HGNC_Symbol.2':'target'})
        except:
            PPI = pd.read_csv(PPI)
        
        
    # Converts PPI into networkx graph, than trims all nodes that cannot be converted
    PPI = nx.from_pandas_edgelist(PPI)
    PPI = PPI.subgraph(conversion)
    PPI = nx.Graph(PPI)
    self_loops = list(nx.selfloop_edges(PPI))
    PPI.remove_edges_from(self_loops)


    # Saves PPI if applicable
    if save == True:
        with open('PPI.pk', 'wb') as f:
            pk.dump(PPI, f)

    return PPI
    
# Greedy algo for communities
def detect_communities_greedy(G):
    selected_edge = list(G.edges(data = True))[0]
    try:
        weight = selected_edge['weight']
        communities = greedy_modularity_communities(G, weight='weight')
    except:
        communities = greedy_modularity_communities(G, weight='attention')
        
    print('Greedy attention generated!')
    
    return [list(community) for community in communities]
    
# Creates a dictionary of hop distances (keys) and listed nodes
def group_nodes_by_hop_distance(graph, source_node, max_hop_distance = 10):

    nodes_by_hop_distance = defaultdict(list)
    visited = set()
    queue = deque([(source_node, 0)])

    # While there are nodes left in the queue and hop distances not traversed, generates hops and nodes from hops
    while queue:
        current_node, hop_distance = queue.popleft()

        if hop_distance > max_hop_distance:
            break

        nodes_by_hop_distance[hop_distance].append((current_node, graph.nodes[current_node]))
        visited.add(current_node)

        # If maximum hop distance not reached, continues to traverse graph
        if hop_distance < max_hop_distance:
            neighbors = list(graph.neighbors(current_node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, hop_distance + 1))

    return dict(nodes_by_hop_distance)
    
# Finds neighbors of existing nodes in graph 
def neighbor_group(graph, nodes, LCC = None):
    hop_groups = []
    
    # Iterates through all listed nodes 
    if LCC != None:
        if isinstance(nodes, list):
            for node in nodes:
                neighbors = list(graph.neighbors(node))
                hop_groups += [neighbor for neighbor in neighbors if neighbor not in nodes and neighbor not in list(LCC.nodes())]  # Exclude nodes in the original LCC
        else:
             neighbors = list(graph.neighbors(nodes))
             hop_groups += [neighbor for neighbor in neighbors if neighbor not in list(LCC.nodes())]  # Exclude nodes in the original LCC
            
    else:
        if isinstance(nodes, list):
            for node in nodes:
                neighbors = list(graph.neighbors(node))
                hop_groups += [neighbor for neighbor in neighbors if neighbor not in nodes]
        else:
             neighbors = list(graph.neighbors(nodes))
             hop_groups += [neighbor for neighbor in neighbors]  # Exclude nodes in the original LCC
                 
    return hop_groups
    
# Compares two generated graphs
def compare_networks(old_PPI, new_PPI, LCC = None):
   
    def get_log_degree_distribution(graph):
        # Calculate degree for each node and filter out zero degrees
        degrees = [degree for node, degree in nx.degree(graph) if degree > 0]
        # Transform to log-scale; use np.log1p for log(1 + degree) to handle degrees of 1
        log_degrees = np.log1p(degrees)
        return log_degrees
        
    if LCC:
        # Assuming LCC is already the largest connected component
        lcc_nodes = set(LCC.nodes())
        lcc_edges = set(LCC.edges())
    
        # Check node and edge preservation in the new_PPI
        preserved_nodes = lcc_nodes.intersection(set(new_PPI.nodes()))
        preserved_edges = set()
        for edge in lcc_edges:
            if new_PPI.has_edge(*edge) or new_PPI.has_edge(edge[1], edge[0]):
                preserved_edges.add(edge)
        
        # Calculate the preservation percentages
        node_preservation = len(preserved_nodes) / len(lcc_nodes) if lcc_nodes else 0
        edge_preservation = len(preserved_edges) / len(lcc_edges) if lcc_edges else 0
        
        print(f'LCC Node Preservation: {node_preservation:.2%}')
        print(f'LCC Edge Preservation: {edge_preservation:.2%}')
        
        if len(node_preservation) > 0 and len(edge_preservation) > 0:
            # Calculate and print the edges in the new LCC that are not preserved
            new_lcc_edges = set(new_PPI.edges())
            non_preserved_edges = new_lcc_edges.difference(preserved_edges)
            
            # Calculate log-scale degree distributions
            old_ppi_log_degrees = get_log_degree_distribution(LCC)
            new_ppi_log_degrees = get_log_degree_distribution(new_PPI)
            
            # Plot histograms
            plt.hist(old_ppi_log_degrees, bins=50, alpha=0.5, label='Old PPI')
            plt.hist(new_ppi_log_degrees, bins=50, alpha=0.5, label='New PPI')
            plt.xlabel('Log(Degree)')
            plt.ylabel('Frequency')
            plt.xlim(1, None)
            plt.title('Log-Scale Degree Distribution of Old PPI vs. New PPI')
            plt.legend()
            plt.savefig('NewLCColdLCCLogDegreeDisttribution.png')
            
            # Calculates second moment
            def nth_moment_v2(PPI, n):
                degree_np = np.array(list(dict(PPI.degree).values()))
                return (sum(degree_np**n)/len(PPI))
            old_moment = nth_moment_v2(LCC, 2)
            new_moment = nth_moment_v2(new_PPI, 2)
            print(f'LCC second moment: {old_moment}, GF LCC second moment: {new_moment}')
        
    else:
        shared_edges = 0
        total_edges = 0
        
        # Consider bidirectional as unidirectional
        for edge in old_PPI.edges(data=True):
            total_edges += 1
            if new_PPI.has_edge(edge[0], edge[1]) or new_PPI.has_edge(edge[1], edge[0]):
                shared_edges += 1
        
        # Obtains proportion of shared edges
        proportion_shared = shared_edges / total_edges if total_edges > 0 else 0
        print(f'Proportion of shared edges: {proportion_shared}')
        
        # Obtains modularity of both 
        new_PPI = new_PPI.to_undirected()
        partition = community_louvain.best_partition(new_PPI)
        old_partition = community_louvain.best_partition(old_PPI)
        modularity = community_louvain.modularity(partition, new_PPI)
        old_modularity = community_louvain.modularity(old_partition, old_PPI)
        print(f'New PPI modularity: {modularity} Old PPI modularity: {old_modularity}')
        
        # Calculate log-scale degree distributions
        old_ppi_log_degrees = get_log_degree_distribution(old_PPI)
        new_ppi_log_degrees = get_log_degree_distribution(new_PPI)
        
        # Plot histograms
        plt.hist(old_ppi_log_degrees, bins=50, alpha=0.5, label='Old PPI')
        plt.hist(new_ppi_log_degrees, bins=50, alpha=0.5, label='New PPI')
        plt.xlabel('Log(Degree)')
        plt.ylabel('Frequency')
        plt.xlim(1, None)
        plt.title('Log-Scale Degree Distribution of Old PPI vs. New PPI')
        plt.legend()
        plt.savefig('NewPPIoldPPILogDegreeDisttribution.png')
        
        # Calculates second moment
        def nth_moment_v2(PPI, n):
            degree_np = np.array(list(dict(PPI.degree).values()))
            return (sum(degree_np**n)/len(PPI))
        old_moment = nth_moment_v2(old_PPI, 2)
        new_moment = nth_moment_v2(new_PPI, 2)
        print(f'PPI second moment: {old_moment}, GF PPI second moment: {new_moment}')


def compute_all_gamma_ln(N):
    """
    Precomputes all logarithmic gammas.
    """
    return {i: scipy.special.gammaln(i) for i in range(1, N+1)}

def logchoose(n, k, gamma_ln):
    if n-k+1 <= 0:
        return float('inf')
    return gamma_ln[n+1] - (gamma_ln[n-k+1] + gamma_ln[k+1])

def gauss_hypergeom(x, r, b, n, gamma_ln):
    return np.exp(logchoose(r, x, gamma_ln) + logchoose(b, n-x, gamma_ln) - logchoose(r+b, n, gamma_ln))

def pvalue(kb, k, N, s, gamma_ln,):
    p = sum(gauss_hypergeom(n, s, N-s, k, gamma_ln) for n in range(kb, min(k, s)+1))
    return min(p, 1)

def get_neighbors_and_degrees(G):
    neighbors = {node: set(G.neighbors(node)) for node in G.nodes()}
    all_degrees = {node: G.degree(node) for node in G.nodes()}
    return neighbors, all_degrees

def reduce_not_in_cluster_nodes(all_degrees, neighbors, G, not_in_cluster, cluster_nodes, alpha):
    reduced_not_in_cluster = {}
    kb2k = defaultdict(dict)
    for node in not_in_cluster:
        k = all_degrees[node]
        kb = sum(1 for neighbor in neighbors[node] if neighbor in cluster_nodes)
        k += (alpha - 1) * kb
        kb += (alpha - 1) * kb
        kb2k[kb][k] = node

    k2kb = defaultdict(dict)
    for kb, k2node in kb2k.items():
        min_k = min(k2node.keys())
        node = k2node[min_k]
        k2kb[min_k][kb] = node

    for k, kb2node in k2kb.items():
        max_kb = max(kb2node.keys())
        node = kb2node[max_kb]
        reduced_not_in_cluster[node] = (max_kb, k)
    return reduced_not_in_cluster

def diamond_iteration_of_first_X_nodes(G, S, X, alpha):
    N = G.number_of_nodes()
    added_nodes = []
    neighbors, all_degrees = get_neighbors_and_degrees(G)
    cluster_nodes = set(S)
    not_in_cluster = set()
    for node in cluster_nodes:
        not_in_cluster |= neighbors[node]
    not_in_cluster -= cluster_nodes
    s0 = len(cluster_nodes) + (alpha - 1) * len(cluster_nodes)
    N += (alpha - 1) * len(cluster_nodes)
    gamma_ln = compute_all_gamma_ln(N+1)
    all_p = {}

    while len(added_nodes) < X:
        info = {}
        pmin = float('inf')
        next_node = None
        reduced_not_in_cluster = reduce_not_in_cluster_nodes(all_degrees, neighbors, G, not_in_cluster, cluster_nodes, alpha)
        for node, kbk in reduced_not_in_cluster.items():
            kb, k = kbk
            p = all_p.get((k, kb, s0), pvalue(kb, k, N, s0, gamma_ln))
            all_p[(k, kb, s0)] = p
            if p < pmin:
                pmin = p
                next_node = node
            info[node] = (k, kb, p)
        added_nodes.append((next_node, info[next_node][0], info[next_node][1], info[next_node][2]))
        cluster_nodes.add(next_node)
        s0 = len(cluster_nodes)
        not_in_cluster |= (neighbors[next_node] - cluster_nodes)
        not_in_cluster.remove(next_node)

    return added_nodes

def DIAMOnD(G_original, seed_genes, max_number_of_added_nodes, alpha=1):
    """
    Runs the DIAMOnD algorithm

    Parameters:
    - G_original: networkx Graph - The network.
    - seed_genes: set - a set of seed genes.
    - max_number_of_added_nodes: int - number of nodes to add.
    - alpha: int - weight of the seeds, default is 1.

    Returns:
    - added_nodes: list - A list of nodes added by DIAMOnD, each element has:
        * name: str - name of the node
        * k: int - degree of the node
        * kb: int - number of neighbors that are part of the module (at agglomeration)
        * p: float - connectivity p-value at agglomeration
    """
    all_genes_in_network = set(G_original.nodes())
    seed_genes = set(seed_genes)
    disease_genes = seed_genes & all_genes_in_network
    if len(disease_genes) != len(seed_genes):
        print(f"DIAMOnD(): ignoring {len(seed_genes - all_genes_in_network)} of {len(seed_genes)} seed genes that are not in the network")

    added_nodes = diamond_iteration_of_first_X_nodes(G_original, disease_genes, max_number_of_added_nodes, alpha)
    return added_nodes

    
def disparity_filter(g, alpha=0.05):
    if not isinstance(g, nx.DiGraph):
        raise TypeError("Input should be a networkx directed graph.")
    
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        raise ValueError("Input graph is empty or has no edges.")
        
    if any(isinstance(node, tuple) for node in g.nodes()):
        raise ValueError("Graph has tuple nodes. Ensure nodes are consistently labeled with non-tuple data types.")

    # Check for the specific 'attentions' edge data
    if not all('attention' in g.edges[i, j] for i, j in g.edges()):
        raise ValueError("Not all edges have 'attentions' data.")

    try:
        W = nx.adjacency_matrix(g)
    except Exception as e:
        raise ValueError(f"Error in creating adjacency matrix: {e}")

    n = g.number_of_nodes()

    try:
        s = W.sum(axis=1)
        s[s == 0] = 1
        W = W.multiply(sp.coo_matrix(1.0 / s))
    except Exception as e:
        raise ValueError(f"Error in normalizing adjacency matrix: {e}")

    try:
        k = np.array([g.in_degree(node) for node in g.nodes()])
        k = k.reshape(-1, 1)
    except Exception as e:
        raise ValueError(f"Error in obtaining node in-degrees: {e}")

    try:
        log_p = (k - 1) * np.log(1.0 - W.toarray())
    except Exception as e:
        raise ValueError(f"Error in calculating log probabilities: {e}")

    filtered_g = nx.DiGraph()

    try:
        for i, j in zip(*W.nonzero()):
            p_ij = np.exp(log_p[i, j])
            if p_ij < alpha:
                if g.has_edge(i, j):
                    edge_data = g.get_edge_data(i, j)
                    
                    # Add edge to the filtered graph with attributes
                    filtered_g.add_edge(i, j, **edge_data)
    except Exception as e:
        raise ValueError(f"Error in edge filtering and graph construction: {e}")

    # Check if any edge has been added
    if filtered_g.number_of_edges() == 0:
        print("No edges passed the disparity filter. Consider adjusting the alpha value or checking the input graph.")

    return filtered_g
    
# Random Walk with Restart
def random_walk_with_restart(G, start_node, walk_length=100, restart_prob=0.3):
    walk = [start_node]
    for _ in range(walk_length):
        current = walk[-1]
        if random.random() < restart_prob:
            next_node = start_node  # Restarting the walk
        else:
            neighbors = list(G.neighbors(current))
            next_node = random.choice(neighbors) if neighbors else start_node
        walk.append(next_node)
    return walk

# Analyze visitation frequencies
def analyze_visitation_frequencies(walk):
    return Counter(walk)

# Weighted Random Walk
def random_walk_with_restart_weighted(G, start_node, walk_length=100, restart_prob=0.3):
    walk = [start_node]
    for _ in range(walk_length):
        current = walk[-1]
        if random.random() < restart_prob:
            next_node = start_node  # Restarting the walk
        else:
            neighbors = list(G.neighbors(current))
            if neighbors:
                weights = [G[current][neighbor]['attention'] for neighbor in neighbors]
                total_weight = sum(weights)
                
                try:
                    probabilities = [weight / total_weight for weight in weights]
                except:
                    probabilities = [10e-9 for weight in weights]
                    
                next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
            else:
                next_node = start_node
        walk.append(next_node)
    return walk
    
def detect_communities_louvain(G, keyword = 'test'):
    # Compute the best partition
    partition = community_louvain.best_partition(G)
    
    # Draw the communities
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.savefig(f'Louvain_analysis_{keyword}.png')
    
    return partition

# Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0
    
    
def detect_communities(G, visited_nodes):
    """
    Detect communities in the subgraph of visited nodes.
    
    Args:
    G (networkx.Graph): The original graph.
    visited_nodes (list): The nodes visited during the random walk.
    
    Returns:
    dict: A dictionary where keys are node IDs and values are the community IDs they belong to.
    """
    subgraph = G.subgraph(visited_nodes)
    partition = community_louvain.best_partition(subgraph)
    return partition

            
