import pandas as pd
import networkx as nx
from pathlib import Path
from collections import deque, defaultdict
import pickle as pk
import polars as pl
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np

# Obtains disease genes from a csv file containing disease gene mappings
def isolate_disease_genes(selected_disease, diseases = Path('/work/ccnr/GeneFormer/GeneFormer_repo/PPI/GDA_Filtered_04042022.csv'), 
                          column_name = 'NewName', gene_column_name = 'HGNC_Symbol', filter_DisGeNET = True):
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

    filtered_dataframe = diseases.filter(diseases[column_name].str.contains(selected_disease.lower()))
    gene_list = list(filtered_dataframe[gene_column_name])

    return gene_list
    
# Obtains selected disease genes from a larger Protein-Protein Interaction network
def LCC_genes(PPI, disease_genes, subgraph = True):

    # Obtains LCC
    sub = PPI.subgraph(disease_genes)
    largest_cc = max(nx.connected_components(sub), key=len)
    nodes = list(largest_cc)
    
    # Returns a subgraph containing the LCC. Otherwise, returns a list of the nodes 
    if subgraph == True:
        LCC = PPI.subgraph(nodes)    

        return LCC
    else:    
        return nodes
    
# Maps gene embeddings to PPI
def instantiate_ppi(PPI = Path('/work/ccnr/GeneFormer/GeneFormer_repo/PPI/PPI_2022_2022-04-21.csv'), 
                    gene_ids = Path("/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/gene_name_id_dict.pkl"),
                    save = False):

    # All possible genes that can be mapped            
    conversion = list(pk.load(open(gene_ids, 'rb')).keys())
    
    # Loads PPI using pandas
    try:
        PPI = pd.read_csv(PPI).drop(columns = ['Source'])
        PPI = PPI.rename(columns = {'HGNC_Symbol.1':'source', 'HGNC_Symbol.2':'target'})
    except:
        PPI = pd.read_table(PPI)[['source', 'target']]
        
    # Converts PPI into networkx graph, than trims all nodes that cannot be converted
    PPI = nx.from_pandas_edgelist(PPI)
    PPI = PPI.subgraph(conversion)
  
    # Throws out nodes with no embeddings, modules, or edges (disconnected via conversion)
    nodes_to_keep = [node for node, data in PPI.nodes(data=True) if 'embedding' in list(data.keys())]
    
    # Saves PPI if applicable
    if save == True:
        with open('PPI.pk', 'wb') as f:
            pk.dump(PPI, f)

    return PPI
    
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
    
    # Calculates degree distribution
    def get_degree_distribution(graph):
        degrees = [degree for node, degree in nx.degree(graph)]
        return degrees
    
    # Calculate degree distributions
    old_ppi_degrees = get_degree_distribution(old_PPI)
    new_ppi_degrees = get_degree_distribution(new_PPI)
    
    plt.hist(old_ppi_degrees, bins=max(old_ppi_degrees), alpha=0.5, label='Old PPI')
    plt.hist(new_ppi_degrees, bins=max(new_ppi_degrees), alpha=0.5, label='New PPI')
    plt.xlabel('Degree')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution of Old PPI vs. New PPI')
    plt.legend()
    plt.savefig('NewPPIoldPPIDegreeDisttribution.png')
    
    # Calculates second moment
    def nth_moment_v2(PPI, n):
        degree_np = np.array(list(dict(PPI.degree).values()))
        return (sum(degree_np**n)/len(PPI))
    old_moment = nth_moment_v2(old_PPI, 2)
    new_moment = nth_moment_v2(new_PPI, 2)
    print(f'PPI second moment: {old_moment}, GF PPI second moment: {new_moment}')

    if LCC:
        # Identify the LCC in the old PPI
        lcc_nodes = set(max(nx.connected_components(old_PPI), key=len))
        lcc_edges = old_PPI.subgraph(lcc_nodes).edges()

        # Check node and edge preservation in the new PPI
        preserved_nodes = lcc_nodes.intersection(set(new_PPI.nodes()))
        preserved_edges = set()
        for edge in lcc_edges:
            if new_PPI.has_edge(*edge) or new_PPI.has_edge(edge[1], edge[0]):
                preserved_edges.add(edge)

        # Calculate the preservation percentages
        node_preservation = len(preserved_nodes) / len(lcc_nodes) if lcc_nodes else 0
        edge_preservation = len(preserved_edges) / len(lcc_edges) if lcc_edges else 0

        print(f'LCC Node Preservation: {node_preservation}')
        print(f'LCC Edge Preservation: {edge_preservation}')

    

        
            