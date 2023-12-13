# Base imports
import time
import random
import sys
import tqdm
import os
from multiprocessing import Pool, cpu_count
import concurrent
from pathlib import Path
import pickle as pk
from tqdm.contrib.concurrent import process_map
import traceback
import requests
from concurrent.futures import ThreadPoolExecutor

# Data processing/pre-processing/comparison imports
from collections import defaultdict
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, binom_test
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Torch/Geneformer imports
import torch
from geneformer import TranscriptomeTokenizer
from transformers import BertForSequenceClassification, BertForTokenClassification, BertModel
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

# Network imports
import networkx as nx
from .networks import *

# Specifies pytorch device (gpu if available, defaults to cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
PRIMARY FUNCTIONS / CLASSES
--------------------------------------------

GFDataset = Class which can handle Geneformer input data
    input -> dataset (HF dataset)
  
process_batch = Function which processes a single batch of attention matrices (with associated attention heads)
    input -> attentions (attention batch)
    output -> gene_attentions (dictionary of subdictionaries with a source gene key, with target gene keys matches to a list of attention values from each head/sample if in sample)
    
extract_attention = Primary function for extracting attentions. Inputs, outputs, and parameters listed below function definition. REQUIRES GENEFORMER TO BE INSTALLED!
    To install geneformer, go to the repository Geneformer is downloaded in and type "pip install ." 

isolate_disease_genes = Function for extracting a list of genes for a specific condition from a dataframe. All parameters and outputs are defined below function definition

LCC_genes = Function for extracting an LCC from a larger network and a list of nodes

instantiate_PPI = Function which loads a PPI from 
'''

# Primary class for loading in Geneformer input data (child of torch.utils Dataset class)
class GFDataset(Dataset):

    # Max length of 20148 (Geneformer context window)
    def __init__(self, dataset, max_length=2048):
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = item['input_ids']

        # Pad the input_ids with zeros to the specified max_length
        padded_input_ids = input_ids + [0] * (self.max_length - len(input_ids))

        return torch.LongTensor(padded_input_ids)
        
# Processes a batch of attention matrices with associated heads
def process_batch(attentions, context_length = 2048):
    attention_matrices, genes_list = attentions[0], attentions[1]
    num_attention_heads = len(attention_matrices[0])
    gene_attentions = {}

    # Iterates through attention matrices 
    for attention_matrix, genes in zip(attention_matrices, genes_list):
        attention_matrix = np.array(attention_matrix)

        # Normalizes matrix to account for different context lengths
        if attention_matrix.shape[1] < context_length:
            normalization_factor = attention_matrix.shape[1] / context_length
            attention_matrix *= normalization_factor
        
        # Obtains all genes that re not None
        non_none_genes = np.array([gene is not None for gene in genes])
        non_none_indices = np.where(non_none_genes)[0]
        attention_matrix = attention_matrix[:, non_none_indices, :][:, :, non_none_indices]
        source_genes = [genes[i] for i in non_none_indices]
        target_genes = [genes[j] for j in non_none_indices]

        # Iterates through attention heads
        for attention_head_num in range(num_attention_heads):
            attention_head = attention_matrix[attention_head_num]
            attention_weights = attention_head[np.arange(len(source_genes))[:, None], np.arange(len(target_genes))]
            attention_weights = attention_weights - np.diag(np.diag(attention_weights))

            for i, source_gene in enumerate(source_genes):
                for j, target_gene in enumerate(target_genes):
                    gene_attentions.setdefault(source_gene, {}).setdefault(target_gene, []).append(attention_weights[i][j])

    return gene_attentions
    
# Primary runtime function for extracting attention weights
def extract_attention(model_location = '.', data = '/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/genecorpus_30M_2048.dataset',
                      samples = 1000, layer_index = 4, gene_conversion = "/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/gene_name_id_dict.pkl", token_dictionary = "/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/token_dictionary.pkl", 
                      mean = False, top_connections = None, num_processes = os.cpu_count(), filter_genes = None, save = False, data_sample = None, half = False,
                      filter_label = None, save_threshold = False):
    '''
    PARAMETERS
    -----------------------------------------
    samples : int, str
        Number of samples to sample from the dataset. If 'all', the entire dataset will be used.  Defaults to 10,000

    top_connections : int, None
        The number of strongest attention weights that should be returned along with the dictionary of all attention weights. If set to None, none of these weights are collected. Defaults to None

    num_processes : int
        The number of seperate processes that should be used to parallize batch operations. Defaults to 4

    layer_index : int
        What layer attention weights should be extracted from. Defaults to 3

    model_location : str, Path
        Where the model weights are located for loading the model. Can be pretrained or finetuned. Defaults to '/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/genecorpus_30M_2048.dataset'

    data : str, Path
        Location of dataset to be used. Defaults to '/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/genecorpus_30M_2048.dataset'

    gene_conversion : str, Path
        Location of gene conversion dictionary that converts genes to Ensembl. Defaults to "geneformer/gene_name_id_dict.pkl"

    mean : bool
        Whether to use the mean attention (True) or the maximum attention (False) for the gene/gene connection weight. Defaults to True

    normalize : bool
        Whether or not all attention weights should be scaled on the range of [0, 1]. Defaults to False
    
    scale : bool
        Uses standard scaling to scale the attention weights. Defaults to False
        
    filter_genes : None, list
        List of genes that will attempted to be placed into the final sampled

    half : bool
        Whether the model should be initialized with half-precision weights for memory and speed improvements. Defaults to True
        
    filter_label : tuple, None
        Whether the dataset should be filtered for a specific organ. Is in the format (dataset key, dataset value)
        
    OUTPUTS
    -----------------------------------------
    aggregated_gene_attentions : dict
        Dictionary of aggregated gene attentions, processed as either the mean or the max of all relevant attention weights
        
    strong_attentions : list
        List of the top-strength gene connections. Containss n lists of top attention tuples (with the structure (source gene, target gene, attention weight). Is not returned if top_connections == 0 or None
    
    '''
    # Starting time of the function
    start = time.perf_counter()
    
    # Creates conversion from tokens back to genes
    token_dict = TranscriptomeTokenizer().gene_token_dict
    token_dict = {value: key for key, value in token_dict.items()}
    conversion = {value: key for key, value in pk.load(open(gene_conversion, 'rb')).items()}
    conversion = {key: conversion[value] for key, value in token_dict.items() if value in conversion}
    
    # Reads huggingface-formatted dataset
    data = load_from_disk(data)

    # Filters for a specific organ if specified
    if filter_label:
        def label_filter(example):
            return example[filter_label[0]] == filter_label[1]
            
        data = data.filter(label_filter, num_proc=num_processes)
    
    # Generates a subset of data if specified
    if data_sample != None:
        data = data.select(range(data_sample[0], data_sample[1]))
    
    # Filters down to the number of samples specified if no filtration parameters present
    if filter_genes == None:
        if samples != 'all':
            random_indices = random.sample(range(len(data)), samples)
            data = data.select(random_indices)
        else:
            data = data
            
    else: # Greedy sampling to ensure that specified gene IDs are present (if possible) in sampled dataset
        deconversion = {value:key for key, value in conversion.items()}
        gene_size = (samples // len(filter_genes)) + 1
            
        total_indices = list(range(len(data)))
        sampled_indices = []
        filtered_genes = [deconversion[i] for i in filter_genes if i in list(deconversion.keys())]
        
        iter_limit = 10_000_000 / len(filtered_genes)
        
        # Subfunction for identifying a gene token in a subset of input ids
        def identify_indices(index):
            indices = []
            iterations = 0
            while len(indices) < gene_size and iterations < iter_limit:
                random_sample = random.choice(total_indices)
                if index in data[random_sample]['input_ids']:
                    indices.append(random_sample)
                iterations += 1
    
            return indices
        
        # Multiprocessing for identifying indices
        sampled_genes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
            results = list(tqdm.tqdm(executor.map(identify_indices, filtered_genes), total=len(filtered_genes), desc = 'Filtering dataset'))
            
        for indices in results:
            if len(indices) > 0:
                sampled_genes += 1
           
            sampled_indices.extend(indices)
            
        print(f'{sampled_genes} sampled out of {len(filtered_genes)}')
        remaining_indices = samples - len(sampled_indices)
        
        # Create a set of sampled indices for faster lookup
        sampled_set = set(sampled_indices)
        
        # Generate possible indices 
        possible_indices = []
        for i in range(len(data) + 1):
            if i in sampled_set:
                possible_indices.append(i)
        
        # Shuffle the possible indices and take the required number of remaining indices
        print(f'Potential indices identified: {len(possible_indices)}')
        random.shuffle(possible_indices)
        remaining_indices = possible_indices[:remaining_indices]
        
        # Extend the sampled_indices list without creating a new list
        sampled_indices.extend(remaining_indices)
        
        # Filter out None values without creating a new list
        sampled_indices = list(filter(None, sampled_indices))
        print(f'{len(sampled_indices)} balanced samples obtained!')
         
        # Filters and saves data
        data = data.select(sampled_indices)
        total_samples = len(data)
        sample_indices = random.sample(range(total_samples), samples)
        data = data.select(sample_indices)
        data.save_to_disk('Sampled_dataset.dataset')
        
        if len(filter_genes) > 10000:
            filter_genes = None
            
    # Loads model
    model = BertModel.from_pretrained(model_location, num_labels=2, output_attentions=True, output_hidden_states=False).to(device).eval()

    # Creates dataloader for 
    dataloader = DataLoader(GFDataset(data), batch_size=9, num_workers=4)
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Uses half-precsion weights to save time. Only available with gpus 
    if half == True and torch.cuda.is_available():
        model = model.half()  
    
    # Sends model to device and prepares for evaluation
    model = model.to(device).eval()
    
    # Sets up default dictionaries based on mean/max type 
    if mean:
        gene_attentions = defaultdict(lambda: defaultdict(list))
    else:
        gene_attentions = defaultdict(lambda: defaultdict(int))

    # Iterates through each batch of the datalaoder
    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc='Passing data through model'):
        attention_results = []
        inputs = batch.to(device)
        
        # Creates attention mask
        attention_mask = torch.where(inputs != 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
        
        # Passes batch through model and extracts attention matrixes at the given index
        outputs = model(inputs, attention_mask=attention_mask)
        attentions = outputs.attentions[layer_index].cpu().detach().numpy()
        
        # Identifies what genes are present, and converts them to tokens
        genes = []
        for gene_set in batch:
            gene_set = [i for i in gene_set if i != 0]
            genes.append([conversion[token.item()] if token.item() in conversion else None for token in gene_set])
    
        if filter_genes is not None:
            attention_results = (attentions, genes, filter_genes)
        else:
            attention_results = (attentions, genes)
    
        # Empty CUDA cache to avoid memory issues
        torch.cuda.empty_cache()  
    
        # Process the chunked attentions
        if filter_genes is not None:
            #dictionary = targeted_batch(attention_results)
            dictionary = process_batch(attention_results)
        else:
            dictionary = process_batch(attention_results)

        # Adds weights to master dictionary
        for key, value in dictionary.items():
            for sub_key, sub_value in value.items():
                if mean:
                    gene_attentions[key][sub_key].extend(sub_value)
                else:
                    gene_attentions[key][sub_key] = max(gene_attentions[key][sub_key], max(sub_value))
    gene_attentions = dict(gene_attentions)
    
    # Adds weights to master dictionary
    aggregated_gene_attentions = defaultdict(lambda: defaultdict(list))
  
    # Turns the array of attention weights into a single gene/gene attention based on whether the user selects the maximum or the average
    if mean == True:
        # Creates a threshold for saving weights, if enabled
        if save_threshold:
            flattened_genes = [sum(value)/len(value) for _, subdict in gene_attentions.items() for _, value in subdict]
            flattened_mean = sum(flattened_genes)/len(flattened_genes)
            flattened_std = statistics.stdev(flattened_genes)
            threshold = flattened_mean + flattened_std
            
            for source_gene, target_genes in tqdm.tqdm(gene_attentions.items(), total = len(gene_attentions.items()), desc = 'Narrowing results'):
                for target_gene, attentions in target_genes.items():
                    average_attention = np.mean(attentions)
                    if average_attention > threshold:
                        aggregated_gene_attentions[source_gene][target_gene].append((len(attentions), average_attention))
                        
        else:
            for source_gene, target_genes in tqdm.tqdm(gene_attentions.items(), total = len(gene_attentions.items()), desc = 'Narrowing results'):
                for target_gene, attentions in target_genes.items():
                    # Creates new dictionary for source gene if not already existing
    
                    average_attention = np.mean(attentions)
                    aggregated_gene_attentions[source_gene][target_gene].append((len(attentions), average_attention))
    else:

        # Creates a threshold for saving weights, if enabled
        if save_threshold:
            flattened_genes = [value for _, subdict in gene_attentions.items() for _, value in subdict]
            flattened_mean = sum(flattened_genes)/len(flattened_genes)
            flattened_std = statistics.stdev(flattened_genes)
            threshold = flattened_mean + flattened_std
            
            # Removes weights below the threshold
            for key, subdict in gene_attentions.items():
                for subkey, value in subdict.items():
                    if value <= threshold:
                        gene_attentions[key].pop(subkey)
        else:
            aggregated_gene_attentions = gene_attentions
    aggregated_gene_attentions = dict(aggregated_gene_attentions)

    end = time.perf_counter()
    print(f'Total minutes elapsed: {round((end - start)/60, 4)} minutes')
    print(f'{len(aggregated_gene_attentions.keys())} total source genes represented in the final dataset')
    
    if save != False:
        pk.dumps(open(save, 'wb'))
        
    if top_connections and top_connections > 0:
        # Obtains the top N attention pairs from the aggregated gene attention dataset if enabled
        top_attention_pairs = []

        if mean == True:
            for source_gene, target_genes in aggregated_gene_attentions.items():
                for target_gene, avg_attention in target_genes.items():
                    top_attention_pairs.append(((source_gene, target_gene), avg_attention[1]))
        else:
            for source_gene, target_genes in aggregated_gene_attentions.items():
                for target_gene, avg_attention in target_genes.items():
                    top_attention_pairs.append(((source_gene, target_gene), avg_attention))
    
        # Sorts and subsets top pairs
        top_attention_pairs.sort(key=lambda x: x[1], reverse=True)
        top_connections = min(top_connections, len(top_attention_pairs))
        top_connections_dict = {pair[0]: pair[1] for pair in top_attention_pairs[:top_connections]}
        strong_connections = sorted(top_connections_dict.items(), key=lambda x: x[1], reverse=True)

        return aggregated_gene_attentions, strong_connections
        
    else:
        return aggregated_gene_attentions
    
# Compares the CDFs of multiple different distributions
def compare_LCCs(attentions, num_diseases = 10, keyword = None, #guaranteed_LCCs = ['cardiomyopathy hypertrophic', 'heart failure', 'cardiomyopathy dilated', 'dementia', 'arthritis rheumatoid', 'anemia', 'calcinosis', 'parkinsonian disorders'],
                guaranteed_LCCs = ['cardiomyopathy hypertrophic', 'cardiomyopathy dilated', 'carcinoma non small cell lung', 'small cell lung carcinoma', 'heart failure', 'dementia', 'arthritis rheumatoid', 'anemia', 'calcinosis', 'parkinsonian disorders'],
                disease_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/PPI/GDA_Filtered_04042022.csv')):
    network_LCCs = {}

    # Obtains and maps LCCs that are given
    PPI = instantiate_ppi()
    for LCC in tqdm.tqdm(guaranteed_LCCs, total = len(guaranteed_LCCs), desc = 'Obtaining Given LCCs'):
        disease_genes = isolate_disease_genes(LCC.lower())
        LCC_nodes = LCC_genes(PPI, disease_genes, subgraph = True,)
        try:
            LCC_nodes, _, _ = map_attention_attention(LCC_nodes, attentions,)
        except:
            continue
        network_LCCs[LCC] = LCC_nodes
    
    # Samples random LCCs
    diseases = list(set(pd.read_csv(disease_location)['NewName']) - set([i.lower() for i in guaranteed_LCCs]))
    random_LCCs = random.sample(diseases, num_diseases - len(network_LCCs))

    for LCC in tqdm.tqdm(random_LCCs, total = len(random_LCCs), desc = 'Obtaining Random LCCs'):
        disease_genes = isolate_disease_genes(LCC)
        try:
            LCC_nodes = LCC_genes(PPI, disease_genes, subgraph = True,)
        except:
            continue

        network_LCC, _, _ = map_attention_attention(LCC_nodes, attentions)
        network_LCCs[LCC] = network_LCC

    plt.figure()
    plt.xscale('log')

    for LCC, network in network_LCCs.items():
        # Retrieve and adjust attention values
        LCC_attentions = [network[u][v]['attention'] 
                        for u, v in network.edges() 
                        if not isinstance(network[u][v]['attention'], list)]
        sns.ecdfplot(LCC_attentions, label=f'{LCC}') # (n={len(LCC_attentions)})')

    plt.xlabel('Attention Weights')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='lower right', bbox_to_anchor=(0, 0.2), ncol=1, fontsize = 8)
    plt.title('Cumulative Distribution of Attention Weights for Various LCCs')
    plt.tight_layout()

    # Save the plot
    if keyword:
        plt.savefig(f'GroupLCC_{keyword}.svg')
    else:
        plt.savefig('GroupLCC.svg')

# Maps attention scores to PPI
def map_attention_attention(PPI, gene_attentions, save = False):
    # Make a copy of the original graph
    PPI_copy = nx.Graph(PPI)

    # Maps attention weights to PPI and identifies edges without attentions
    edges_to_remove = []
    for u, v in list(PPI_copy.edges()):

        # Selects maximum attention in both directions to account for bidirectionality
        try:
            attention_direction1 = gene_attentions[u][v]
            attention_direction2 = gene_attentions[v][u]
            try:
                attention_ratio = min(attention_direction1, attention_direction2) / max(attention_direction1, attention_direction2)
            except:
                attention_ratio = 1
            selected_attention = max(attention_direction1, attention_direction2)

            if not isinstance(selected_attention, list):
                PPI_copy[u][v]['attention'] = selected_attention
                PPI_copy[u][v]['ratio'] = attention_ratio
            else:
                edges_to_remove.append((u, v))

        except:
            edges_to_remove.append((u, v))
    
    # Remove edges without valid attention mapping
    for u, v in edges_to_remove:
        PPI_copy.remove_edge(u, v)
    
    edge_attentions = [PPI_copy[u][v]['attention'] for u, v in PPI_copy.edges()]

    # Obtains edge attentions for PPI attentions and finds the mean/median
    average_edge_attention = np.mean(edge_attentions)
    
    # Generates total edge attentions
    total_attention = [value for subdict in gene_attentions.values() for value in subdict.values()]
    total_attention = [i for i in total_attention if not isinstance(i, list)]
    average_total_attention = np.mean(total_attention)
    
    # Perform t-test
    t_stat, p_value = ttest_ind(edge_attentions, total_attention)
    
    # Saves if enabled
    if save:
        pk.dump(PPI_copy, open('attention_PPI.pk', 'wb'))

    print(f'Network attention: {average_edge_attention}')
    print(f'Network attention standard deviation: {np.std(edge_attentions)}')
    print('')
    print(f'Total attention: {average_total_attention}')
    print(f'Total attention standard deviation: {np.std(total_attention)}')
    print(f'T-test p-value: {p_value} \n')
    print(f'Average network ratio {np.mean([PPI_copy[u][v]["ratio"] for u, v in PPI_copy.edges()])}')

    return PPI_copy, average_edge_attention, average_total_attention
    
# Function for creating fake edges
def generate_fake_attentions(graph, num_fake_attentions, gene_attentions, node_to_index):

    # Prepares variables and progress bar
    fake_edges = set()
    nodes = list(graph.nodes())
    fake_attentions = []
    pbar = tqdm.tqdm(total=num_fake_attentions, desc="Generating Fake Edges")
    
    # Iterates fake attentions until quota is matched
    while len(fake_attentions) < num_fake_attentions:
    
        # Randomly select two nodes
        node1, node2 = random.sample(nodes, 2)
        
        # Ensure the selected nodes are not connected in the real graph
        if not graph.has_edge(node1, node2) and node1 != node2:

            # Ensure the edge is not already in the fake edges list
            if (node1, node2) not in fake_edges and (node2, node1) not in fake_edges:          
                try:
                    attention = gene_attentions[node1][node2]
                    if not isinstance(attention, list):
                        fake_attentions.append(attention)
                    else:
                        raise ValueError
                except:
                    try:
                        attention = gene_attentions[node2][node1]
                        if not isinstance(attention, list):
                            fake_attentions.append(attention)
                        else:
                            raise ValueError                      
                    except:
                        continue
                
                 # Add the fake edge to the graph and attention 
                graph.add_edge(node1, node2) 
                fake_edges.add((node_to_index[node1], node_to_index[node2]))
                pbar.update(1)
    pbar.close()
    
    return fake_attentions
        
# Calculates F1 scores for PPI
def F1_score_attention(PPI, gene_attention,):

    # Obtains index of all nodes with PPI
    node_to_index = {node: index for index, node in enumerate(PPI.nodes())}
        
    real_attentions = []
    for u, v in PPI.edges():
        try:
            attention = PPI[u][v]['attention']
            if not isinstance(attention, list):
                real_attentions.append(attention)
        except:
            pass
            
    # Generates attention labels
    fake_attentions = generate_fake_attentions(PPI, len(real_attentions), gene_attention, node_to_index)
    fake_labels = [0 for _ in range(len(fake_attentions))]
    real_labels = [1 for _ in range(len(real_attentions))]
    
    labels = fake_labels + real_labels
    attentions = fake_attentions + real_attentions
    
    # Assuming you have y_pred as the predicted labels (0 or 1) and y_prob as predicted probabilities for the positive class
    precision, recall, _ = precision_recall_curve(labels, attentions)
    
    # Compute AUC for precision-recall curve
    pr_auc = auc(recall, precision)
    
    print(f'AUC score: {pr_auc}')

# Calculates F1 scores for PPI and LCC, calculates both their AUC curves and graphs them
def F1_graph_attention(PPI, gene_attentions, LCC = None, show = True, graph = True, LCC_show = True, keyword = None):

    # Dictionary for converting nodes to numerical tokens
    node_to_index = {node: index for index, node in enumerate(PPI.nodes())}
        
    if LCC != None:
        # Obtains LCC attentions
        LCC_attentions = []
        for u, v in LCC.edges():
            try:
                attention = LCC[u][v]['attention']
                if not isinstance(attention, list):

                    # Bulks up size by sampling 3 times
                    LCC_attentions += [attention] * 3
            except:
                pass

        if LCC_show:
            LCC_ranked = []
            for u, v in LCC.edges():
                try:
                    LCC_ranked.append((u, v, LCC[u][v]['attention']))
                except:
                    pass
            LCC_ranked = sorted(LCC_ranked, key = lambda x: x[2], reverse = True)
            print(f'Top ranked: {LCC_ranked[:5]}')
            print(f'Bottom ranked: {LCC_ranked[-5:]}')
        
        # Obtains PPI attentions
        PPI_attentions = []
        for u, v in PPI.edges():
            try:
                attention = PPI[u][v]['attention']
                if not isinstance(attention, list):
                    PPI_attentions.append(attention)
            except:
                pass

        fake_LCC_attentions = generate_fake_attentions(PPI, len(LCC_attentions), gene_attentions, node_to_index)
        fake_PPI_attentions = generate_fake_attentions(PPI, len(PPI_attentions), gene_attentions, node_to_index) 
        PPI_LCC_attentions = random.sample(PPI_attentions, len(LCC_attentions))

        # Generates 'real' and 'fake' labels for attentions
        LCC_fake_labels = [0 for _ in range(len(LCC_attentions))]
        LCC_real_labels = [1 for _ in range(len(LCC_attentions))]
        PPI_fake_labels = [0 for _ in range(len(PPI_attentions))]
        PPI_real_labels = [1 for _ in range(len(PPI_attentions))]
        
        # Generates combined labels
        LCC_labels = LCC_fake_labels + LCC_real_labels
        PPI_labels = PPI_fake_labels + PPI_real_labels

        # Generates LCC-based attentions
        fake_attentions = fake_LCC_attentions + LCC_attentions
        PPI_LCC_attentions = PPI_LCC_attentions + LCC_attentions

        # Generates PPI-based attentions 
        PPI_fake_attentions = fake_PPI_attentions + PPI_attentions
        '''
        # REMOVE - FOR TESTING DEMENTIA LCC
        disease_genes = isolate_disease_genes('dementia')
        dementia_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
        dementia_attentions = []
        for u, v in dementia_LCC.edges():
            try:
                attention = PPI[u][v]['attention']
                if not isinstance(attention, list):
                    # Bulks up size by sampling 3 times
                    dementia_attentions += [attention] * 3
            except:
                pass
        
        if len(dementia_attentions) > len(LCC_attentions):
            dementia_attentions = random.sample(dementia_attentions, len(LCC_attentions))
        else:
            sampled_dementia = random.sample(dementia_attentions * 20, len(LCC_attentions) - len(dementia_attentions))
            dementia_attentions += sampled_dementia

        dem_attentions = fake_LCC_attentions + dementia_attentions
        '''

        # Assuming you have y_pred as the predicted labels (0 or 1) and y_prob as predicted probabilities for the positive class
        fpr1, tpr1, _ = roc_curve(LCC_labels, fake_attentions)
        fpr2, tpr2, _ = roc_curve(LCC_labels, PPI_LCC_attentions)
        fpr3, tpr3, _ = roc_curve(PPI_labels, PPI_fake_attentions)
        #fpr4, tpr4, _ = roc_curve(LCC_labels, dem_attentions)

        # Compute AUC for precision-recall curve
        LCC_auc = auc(fpr2, tpr2)
        fake_auc = auc(fpr1, tpr1)
        PPI_auc = auc(fpr3, tpr3)
        #dem_auc = auc(fpr4, tpr4)

        if show == True:
            print(f'LCC-Background AUC score: {fake_auc}')
            print(f'LCC-PPI AUC score: {LCC_auc}')
            print(f'PPI-Background AUC score: {PPI_auc}')
            #print(f'Dementia LCC-Background AUC score: {dem_auc}')

        # Graphs curves if specified
        if graph == True:
            plt.figure()#figsize = (12, 8))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.xlim([-.05, 1.05])
            plt.ylim([0, 1.05])
            plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--', label = 'Random')
            plt.title('ROC Curve for Finetuned Cardiomyopathy Geneformer Attentions \n (n = 10,000 samples)')
            plt.plot(fpr2, tpr2, color = 'red', label = f'Cardiomyopathy LCC attentions to PPI attentions \n (AUC: {round(LCC_auc, 4)})')
            plt.plot(fpr1, tpr1, color = 'blue', label = f'Cardiomyopathy LCC attentions to Background \n attentions (AUC: {round(fake_auc, 4)})')
            plt.plot(fpr3, tpr3, color = 'green', label = f'PPI attentions to Background \n attentions (AUC: {round(PPI_auc, 4)})')
            #plt.plot(fpr4, tpr4, color = 'orange', label = f'Dementia LCC attentions to Background \n attentions (AUC: {round(dem_auc, 4)})')
            plt.legend(loc='lower right')#, bbox_to_anchor=(0, 0.2), ncol=1, fontsize = 8)
    
            if keyword:
                plt.savefig(f'LCC-PPI-fake_precision_{keyword}.svg')
            else:
                plt.savefig('LCC-PPI-fake_precision.svg')
            
        return fake_auc, LCC_auc
        
    else:
        # Obtains real (network) attentions
        real_attentions = []
        for u, v in PPI.edges():
            try:
                attention = PPI[u][v]['attention']
                if not isinstance(attention, list):
                    real_attentions.append(attention)
            except:
                pass
     
        # Geneates fake attentions
        fake_attentions = generate_fake_attentions(PPI, len(real_attentions), gene_attentions, node_to_index)
        
        # Generates 'real' and 'fake' labels for attentions
        fake_labels = [0 for _ in range(len(fake_attentions))] 
        real_labels = [1 for _ in range(len(real_attentions))]
        
        # Combines labels
        labels = fake_labels + real_labels
        attentions = fake_attentions + real_attentions
        
        # Assuming you have y_pred as the predicted labels (0 or 1) and y_prob as predicted probabilities for the positive class
        fpr, tpr, _ = roc_curve(labels, attentions)
        
        # Compute AUC for precision-recall curve
        pr_auc = auc(fpr, tpr)
        
        if show == True:
            print(f'AUC score: {pr_auc}')
    
        # Plots AUC curve
        if graph == True:
            plt.figure()#figsize = (12, 8))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.xlim([-0.05, 1.0])
            plt.ylim([0, 1.05])
            plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--', label = 'Random')
            plt.title('ROC Curve for Finetuned Cardiomyopathy Geneformer Attentions \n (n = 10,000)')
            plt.plot(fpr, tpr, color = 'green', label = f'PPI attentions to background attentions \n (AUC: {round(pr_auc, 4)})')
            plt.legend(loc='lower right')#, bbox_to_anchor=(0, 0.2), ncol=1)
            plt.tight_layout()
            if keyword:
                plt.savefig(f'PPI_precision_{keyword}.svg')
            else:
                plt.savefig('PPI_precision.svg')
        
        return pr_auc
        
# Processes a list of attention values and a PPI network to obtain a list of mapped and unmapped attentions, cutting off at a specific threshold
def process_edges(PPI, attention_dict, min_threshold = 0.00001):
    real_attentions = []
    total_attentions = []

    # Obtains true edges
    total_edges = list(PPI.edges())
    with tqdm.tqdm(total=len(total_edges), desc='Processing Edges') as pbar:
        for u, v in total_edges:
            try:
                attention = PPI[u][v]['attention']
                if not isinstance(attention, list):
                    real_attentions.append(attention)
            except KeyError:
                pass
            pbar.update()

    # Obtains all edges
    total_attentions = []
    for val in attention_dict.values():
        total_attentions.extend([i for i in list(val.values()) if not isinstance(i, list)])
        
    # Checks all edges by threshold
    real_attentions = [i for i in real_attentions if i > min_threshold]
    total_attentions = [i for i in total_attentions if i > min_threshold]
    
    return real_attentions, total_attentions
        
# Plots attention distributions for background or a PPI/disease if specified
def plot_distributions(attention_dict, disease = None, graph = True, downsample = True, keyword = None):    
        
    if disease == None:
      
        # Instantiates PPI 
        PPI = instantiate_ppi()
        PPI, _, _ = map_attention_attention(PPI, attention_dict)
        
        # Parallel processing to extract attentions
        real_attentions, total_attentions = process_edges(PPI, attention_dict)
        real_attentions = [i for i in real_attentions if not isinstance(i, list)]
        total_attentions = [i for i in total_attentions if not isinstance(i, list)]

        # Plot CDF
        if graph == True:
            plt.figure()
            sns.ecdfplot(total_attentions, color='red', label=f'Background Attentions')# (n={len(total_attentions)})')
            sns.ecdfplot(real_attentions, color='green', label=f'PPI Attentions')# (n={len(real_attentions)})')
            plt.xlabel('Attention Weights')
            plt.ylabel('Cumulative Probability')
            plt.legend(loc='lower right')
            plt.xscale('log')
            plt.xlim(0, 0.1)
            plt.title('Cumulative Distribution of Attention Weights for PPI Network vs Background')
            plt.tight_layout()

            # Save the plot
            if keyword:
                plt.savefig(f'PPIAttentionDist_{keyword}.svg')
            else:
                plt.savefig('PPIAttentionDist.svg')
    else:
        # Instantiates PPI 
        PPI = instantiate_ppi()
        PPI, _, _ = map_attention_attention(PPI, attention_dict)
        
        # Instantiates disease LCC
        disease_genes = isolate_disease_genes(disease)
        disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
        LCC, _, _ = map_attention_attention(disease_LCC, attention_dict)
        
        # Parallel processing to extract attentions
        real_attentions, total_attentions = process_edges(PPI = LCC, attention_dict = attention_dict, min_threshold = 0.00001)

        # Shows top and bottom PPI pairings if enabled
        PPI_attentions, _ = process_edges(PPI = PPI, attention_dict = attention_dict, min_threshold = 0.00001)
        real_attentions = [i for i in real_attentions if not isinstance(i, list)]
        total_attentions = [i for i in total_attentions if not isinstance(i, list)]
        PPI_attentions = [i for i in PPI_attentions if not isinstance(i, list)]

        # REMOVE - DEMENTIA LCC TESTING
        disease_genes = isolate_disease_genes('dementia')
        dementia_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
        dementia_attentions, _ = process_edges(PPI=dementia_LCC, attention_dict = attention_dict, min_threshold = 0.00001)
        dementia_attentions = [i for i in dementia_attentions if not isinstance(i, list)]

        # Plot CDF
        if graph == True:
            plt.figure()
            sns.ecdfplot(total_attentions, color='red', label=f'Background Attentions')# (n={len(total_attentions)})')
            sns.ecdfplot(real_attentions, color='blue', label=f'LCC Attentions')# (n={len(real_attentions)})')
            sns.ecdfplot(PPI_attentions, color='green', label=f'PPI Attentions')# (n={len(PPI_attentions)})')
            sns.ecdfplot(dementia_attentions, color='orange', label=f'Dementia LCC Attentions')
            
            plt.xlabel('Attention Weights')
            plt.ylabel('Cumulative Probability')
            plt.legend(loc='lower right')
            plt.xscale('log')
            plt.title('Cumulative Distribution of Attention Weights for PPI Network vs Background')
            plt.tight_layout()
            # Save the plot
            if keyword:
                plt.savefig(f'LCCPPIAttentionDist_{keyword}.svg')
            else:
                plt.savefig('LCCPPIAttentionDist.svg')


# Performs analysis of attention weights

def analyze_hops(attention_dict,
                diseases = ['Small Cell Lung Carcinoma', 'Cardiomyopathy Dilated', 'Cardiomyopathy Hypertrophic'], top_weights = 10000, shortest_hop = False, keyword = None):
    PPI = instantiate_ppi()
    PPI, _, _ = map_attention_attention(PPI, attention_dict)

    # Calculates correlations
    def calculate_correlation_coefficient(x, y):
        # Calculate means
        mean_x = np.mean(x)
        mean_y = np.mean(y)
    
        # Calculate numerator and denominators
        numerator = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
        denominator_x = sum((x_i - mean_x) ** 2 for x_i in x) ** 0.5
        denominator_y = sum((y_i - mean_y) ** 2 for y_i in y) ** 0.5
    
        # Calculate correlation coefficient
        correlation_coefficient = numerator / (denominator_x * denominator_y) if denominator_x * denominator_y != 0 else 0
    
        # Calculates line of best fit
        x = np.array(x).astype(np.float64)
        y = np.array(y).astype(np.float64)
        coefficients = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coefficients)
        best_fit = polynomial(x)

        return correlation_coefficient, best_fit
    
    if not isinstance(diseases, list):
        diseases = [diseases]

    # Iterates and obtains data for each disease
    disease_dict = {}

    if shortest_hop != True:
        for disease in diseases:
            disease_genes = isolate_disease_genes(disease.lower())
            LCC_gene_list = LCC_genes(PPI, disease_genes)
        
            hop_range = [i for i in range(1, 6)]
            attention_hops = {i:[] for i in hop_range}
            # Subfunction for collecting attention weights from nodes n hops away 
            def identify_hop_attentions(PPI, ref_node, attention_hops, filter_LCC = None):
            
                # If the LCC_gene_list is passed into filter_LCC, no LCC genes will be included
                if filter_LCC == None:
                    filter_LCC = []
                    
                # Sorts all nodes by hop distance in PPI 
                nodes = group_nodes_by_hop_distance(PPI, ref_node)
                for hop_distance in hop_range:
                    
                    hop_attentions = []
                    for node in nodes[hop_distance]:
                        node = node[0]
                        if node not in filter_LCC:
                            try:
                                attention = attention_dict[ref_node][node]
                                hop_attentions.append(attention)
                            except:
                                pass
                    attention_hops[hop_distance].extend(hop_attentions)
                
                return attention_hops
                
            # Iterates through all LCC genes and performs attention hop analysis
            for gene in tqdm.tqdm(LCC_gene_list, total = len(LCC_gene_list), desc = 'Finding hops'):
                attention_hops = identify_hop_attentions(PPI = PPI, ref_node = gene, attention_hops = attention_hops)
            
            average_attentions = []
            attention_lower, attention_upper = [], []
            hop_edges_dict = {}
            flattened_attentions = []
            x = []

            for hop_distance in hop_range:
                hop_edges_attention = [i for i in attention_hops[hop_distance] if not isinstance(i, list)]
                flattened_attentions.extend(hop_edges_attention)
                x.extend([hop_distance]*len(hop_edges_attention))

                # Calculate median and interquartile range
                median = np.median(hop_edges_attention)
                q1 = np.percentile(hop_edges_attention, 25)
                q3 = np.percentile(hop_edges_attention, 75)

                average_attentions.append(median)
                attention_lower.append(median - q1)
                attention_upper.append(q3 - median)

            correlation, _ = calculate_correlation_coefficient(x, flattened_attentions)
            average_correlation, _ = calculate_correlation_coefficient(hop_range, average_attentions)
            print(f'Correlation for {disease}: {correlation}')
            print(f'Average Correlation for {disease}: {average_correlation}')
            disease_dict[disease] = (average_attentions, attention_lower, attention_upper)

        # Plotting
        fig, ax = plt.subplots()

        # Set the width of each bar
        bar_width = 0.2

        # Plot each disease's data
        for i, disease in enumerate(diseases):
            data = disease_dict[disease]
            median_data, q1_data, q3_data = data
            x = np.arange(len(hop_range)) + i * bar_width
            ax.bar(x, median_data, bar_width, label=disease, yerr=[q1_data, q3_data], capsize=4)

        # Set labels, title, and legend
        ax.set_xlabel('Hop Distance')
        ax.set_ylabel('Avg Attention Weight')
        ax.set_title('Average Attention Weight by Hop Distance')
        ax.set_xticks(np.arange(len(hop_range)) + (len(diseases) / 2) * bar_width)
        ax.set_xticklabels(hop_range)
        ax.legend(loc='upper right')

        # Adjust the figure layout
        plt.tight_layout()
        if keyword:
            plt.savefig(f'{keyword}_hopPlot.svg')
        else:
            plt.savefig('old_hopPlot.svg')
        
    
    else:
        # Create a function to calculate the shortest path average for all LCC components to a given node
        def shortest_path_average(PPI, LCC_gene_list, node):
            shortest_paths = [nx.shortest_path_length(PPI, source=gene, target=node) for gene in LCC_gene_list if nx.has_path(PPI, gene, node)]
            return np.mean(shortest_paths) if shortest_paths else None

        # Function to calculate the average attention weight from LCC genes to a given node
        def average_attention_weight(LCC_gene_list, node, attention_dict):
            attention_weights = [attention_dict[gene][node] for gene in LCC_gene_list if gene in attention_dict and node in attention_dict[gene] and not isinstance(attention_dict[gene][node], list)]
            average = np.mean(attention_weights) if attention_weights else None
            stdev = np.std(attention_weights)
            return average, stdev
        
        for disease in disease:
        
            # Lists to store the shortest path averages and average attention weights
            shortest_path_averages = []
            average_attentions = []
            stdev_attentions = []

            # Iterating over every node in the PPI and performing the required calculations
            for node in tqdm.tqdm(PPI.nodes(), desc='Processing nodes'):
                sp_avg = shortest_path_average(PPI, LCC_gene_list, node)
                avg_attention, stdev = average_attention_weight(LCC_gene_list, node, attention_dict)

                if sp_avg is not None and avg_attention is not None:
                    shortest_path_averages.append(sp_avg)
                    average_attentions.append(avg_attention)
                    stdev_attentions.append(stdev)

            correlation, y_fit = calculate_correlation_coefficient(shortest_path_averages, average_attentions)
            print(f'Correlation: {correlation}')
            disease_dict[disease] = (shortest_path_averages, average_attentions)
            
        # Plotting
        plt.figure()
        for disease in diseases:
            data = disease_dict[disease]
            shortest_path_averages, average_attentions = data
            plt.scatter(shortest_path_averages, average_attentions, marker = 'x', label = f'{disease}')

        plt.plot(shortest_path_averages, y_fit, color = 'red', label = 'Linear Fit')
        #plt.scatter(shortest_path_averages, stdev_attentions, color = 'red', marker = 'o', label = 'Standard Deviation')
        plt.xlabel('Avg Shortest Path')
        plt.ylim(0, None)
        plt.ylabel('Avg Attention Weight')
        plt.title(f'Average Attention Weight by LCC Shortest Path \n for {disease} Largest Connected Component')
        plt.legend(loc='upper right', bbox_to_anchor=(0, 0.2), ncol=1)
        plt.tight_layout()
        if keyword:
            plt.savefig(f'{keyword}_hopPlot.svg')
        else:
            plt.savefig('hopPlot.svg')
        
    # Sorts attention weights
    sorted_weights = []
    for key, subdict in attention_dict.items():
        if not isinstance(subdict, list):
            for subkey, value in subdict.items():
                if not isinstance(value, list):
                    sorted_weights.append((key, subkey, value))

    sorted_weights = sorted(
        sorted_weights, key=lambda x: x[2], reverse = True)[:top_weights] + sorted(
        sorted_weights, key=lambda x: x[2], reverse = True)[-top_weights:]
    
    def check_interaction(pair):
        """
        Check if there is an interaction between two genes for human species.
        """
        species_id = 9606  # NCBI Taxonomy ID for human
        string_api_url = "https://string-db.org/api"
        output_format = "json"
        method = "interaction_partners"
        request_url = f"{string_api_url}/{output_format}/{method}"

        params = {
            "identifiers": "%0d".join(pair[:2]),
            "species": species_id,
            "caller_identity": "my_research_project"
        }

        response = requests.post(request_url, data=params)
        interaction_data = response.json()

        # Check if the second gene in the pair is an interaction partner of the first
        try:
            interacts = any(partner['preferredName_B'] == pair[1] for partner in interaction_data)
            return 1 if interacts else 0
        except:
            return 0

    def analyze_interactions(gene_pairs):
        """
        Analyzes the interaction data for a list of human gene pairs.
        """
        # Define the thread pool executor
        with ThreadPoolExecutor(max_workers = 2) as executor:
            # Use tqdm for progress reporting. Wrap the executor.map with tqdm
            interaction_results = list(tqdm.tqdm(executor.map(check_interaction, gene_pairs), total=len(gene_pairs), desc='Analyzing Interactions'))

        cdf, counter = [], 0
        for i in interaction_results:
            counter += i
            cdf.append(counter)

        cdf = [i/counter for i in cdf]
        
        proportion_with_interaction = interaction_results.count(1)/len(interaction_results)

        return proportion_with_interaction, cdf

    def create_cdf(cdf):
        """
        Creates a Cumulative Distribution Function (CDF) plot for the interaction results.
        """
        n = len(cdf)
        background_cdf = [i/n for i in range(n)]
        plt.figure(figsize=(10, 6))
        plt.plot(range(n), cdf, color = 'blue', label='STRING Interaction CDF')
        plt.plot(range(n), background_cdf, color='r', linestyle='--', label='Expected Discovery Rate')
        plt.xlabel('Total Attention Weights by Rank')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of Gene-Gene Interaction Results')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'String_{keyword}_CDF.svg')

    # Analyze interactions
    proportion, interaction_results = analyze_interactions(sorted_weights[:top_weights])
    print(f"Proportion of top {top_weights} gene pairs with interaction: {proportion:.2f}")

    # Create CDF plot
    create_cdf(interaction_results)

# Checks the top attentions of a mapped PPI 
def check_top_attentions(attention_dict, PPI, top_number = None, keyword = None,):

    flattened_dict = [(outer_key, inner_key, value)
                      for outer_key, inner_dict in attention_dict.items()
                      for inner_key, value in inner_dict.items()]
    sorted_tuples = sorted(flattened_dict, key=lambda x: x[2], reverse=True)
    
    # Sets top number of weights to use if not specified
    if top_number == None:
        top_number = int(len(sorted_tuples))

    # Calculates overall ratio of total nodes within PPI 
    background_attention_ratio = len(PPI.edges())/len(flattened_dict)
    print(f'Proportion of total nodes within PPI: {background_attention_ratio}')
    cdf = []
    
    # Calculates proportion of top attentions in PPI for each step
    PPI_nodes = 0
    for i in tqdm.tqdm(range(top_number), total = top_number, desc = 'Calculating CDF'):
        attention_weight = sorted_tuples[i]
        try:
            attention = PPI[attention_weight[0]][attention_weight[1]]   
            PPI_nodes += 1
        except:
            pass
        
        cdf.append(PPI_nodes)  
    
    # Properly adjusts CDF
    total_PPI_edges = cdf[-1]
    cdf = [i/total_PPI_edges for i in cdf]
    
    PPI_ratio = PPI_nodes/top_number  
    background_ratio = [i/top_number for i in range(1, top_number + 1)]
    
    # Calculate CDF
    plt.figure()
    plt.title('Cumulative Density Function of Proportion of PPI Edges \n Present in Top Attention Weights')
    plt.xlabel('Number of top nodes')
    plt.ylabel('Cumulative Proportion in PPI')
    plt.plot([i for i in range(1, top_number + 1)], cdf, linestyle='-', label = 'CDF')
    plt.plot([i for i in range(1, top_number + 1)], background_ratio, label = 'Background Expected', color = 'red')
    plt.legend()
    plt.tight_layout()
    if keyword:
        plt.savefig(f'TopPPI_CDF_{keyword}.svg')
    else:
        plt.savefig('TopPPI_CDF.svg')

    print(f'Proportion of top {top_number} attention weights within the PPI: {PPI_ratio}')

    return PPI_ratio
    
# Merges a list of dictionaries together
def merge_dictionaries(dictionaries, mean = True):
    if mean == True:
        merged_dict =  defaultdict(lambda: defaultdict(list))
    else:
        merged_dict =  defaultdict(lambda: defaultdict(int))
    
    for attention_dict in tqdm.tqdm(dictionaries, total = len(dictionaries), desc = 'Combining dictionaries'):

        try:
            attention_dict = pk.load(open(attention_dict, 'rb'))
        except:
            pass
        
        if mean == True:
            for key in attention_dict.keys():
                subdict = attention_dict[key]
                for sub_key in subdict.keys():
                    merged_dict[key][sub_key].extend(subdict[sub_key])
        else:
            for key in attention_dict.keys():
                subdict = attention_dict[key]
                for sub_key in subdict.keys():
                    merged_dict[key][sub_key] = max(subdict[sub_key], merged_dict[key][sub_key])
    if mean == True:
        for key in merged_dict.keys():
            sub_dict = merged_dict[key]
            for sub_key in sub_dict.keys():
                # Extract the values and counts from the tuples
                counts, values = zip(*sub_dict[sub_key])

                # Create a normalized list by repeating each value based on its count
                normalized_list = np.repeat(values, counts)

                # Calculate the mean of the normalized list
                mean = np.mean(normalized_list)
                sub_dict[sub_key] = np.mean(normalized_list)

    return dict(merged_dict)
    
# Creates a new PPI from the top Geneformer attention weights
def generate_PPI(attention_dict, attention_threshold = None, gene_conversion = Path("/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/gene_name_id_dict.pkl"), save = False,
                  savename = 'Attention_PPI.csv'):
        
    # Creates conversion from tokens back to genes
    token_dict = TranscriptomeTokenizer().gene_token_dict
    token_dict = {value: key for key, value in token_dict.items()}
    conversion = {value: key for key, value in pk.load(open(gene_conversion, 'rb')).items()}
    conversion = {key: conversion[value] for key, value in token_dict.items() if value in conversion}
    
    # Obtains flattened gene-gene attention collection
    flattened_dict = [(outer_key, inner_key, value)
                      for outer_key, inner_dict in attention_dict.items()
                      for inner_key, value in inner_dict.items() if not isinstance(value, list)]
    sorted_tuples = sorted(flattened_dict, key=lambda x: x[2], reverse=True)
    
    # If no specific threshold set, calculates a threshold equivalent to > mean + standard deviation
    if attention_threshold == None:
        mean = sum(flattened_dict)/len(flattened_dict)
        std = statistics.stdev(flattened_dict)
        attention_threshold = mean + std
        
    # Filters connections for attentions above the threshold  
    original_length = len(sorted_tuples)
    sorted_tuples = [i for i in sorted_tuples if i[2] >= attention_threshold]
    new_length = len(sorted_tuples)
    print(f'Original attentions: {original_length} Filtered attentions: {new_length}')
    
    # Creates new unidirectional PPI from attention weights
    PPI = nx.DiGraph()
    for src, tgt, weight in sorted_tuples:
        PPI.add_edge(src, tgt, attention=weight)
  
    # If enabled, saves the PPI as a csv
    if save == True:
        sources, targets, weights = [i[0] for i in sorted_tuples], [i[1] for i in sorted_tuples], [i[2] for i in sorted_tuples]
        PPI_data = pd.DataFrame.from_dict({'source':sources, 'target':targets, 'weights':weights})
        PPI_data.to_csv(savename)
        
    return PPI
    
# Compares the changes between two groups of attention weights
def compare_attentions(base, compare, LCC = None, PPI = None, keyword = None, report_FC = False):

    # Normalizes the attention weights
    comparison = {}
    fold_changes = {}
    for source, values in tqdm.tqdm(base.items(), total = len(base.items()), desc = 'Comparing attentions'):
        try:
            compare_values = compare[source]
        except:
            continue
        comparison[source] = {}
        fold_changes[source] = {}

        for target, attention in values.items():
            try:
                compare_attention = compare_values[target]
            except:
                continue
            if not isinstance(compare_attention, list) and not isinstance(attention, list):
                comparison[source][target] = (attention, compare_attention)
                fold_changes[source][target] = compare_attention / (attention + 0.00001)

    print(f'Comparison source genes: {len(comparison.keys())}')
    print(f'Total comparisons: {sum([len(i) for i in comparison.values()])}')

    # Flattens dictionary
    flattened_dict = [(outer_key, inner_key, value)
                      for outer_key, inner_dict in comparison.items()
                      for inner_key, value in inner_dict.items()]
    sorted_tuples = sorted(flattened_dict, key=lambda x: x[2], reverse=True)

    # Makes LCC comparison
    if LCC != None and PPI != None:
        LCC_edges = [(u, v) for u, v in LCC.edges()]
        PPI_edges = [(u, v) for u, v in PPI.edges()]

        LCC_comparison, PPI_comparison = [], []
        for edge in LCC_edges:
            try:
                compare1 = comparison[edge[0]][edge[1]]
                compare2 = comparison[edge[1]][edge[0]]
                selected_comparison = max(compare1, compare2)
                LCC_comparison.append(selected_comparison)
            except:
                continue

        for edge in PPI_edges:
            try:
                compare1 = comparison[edge[0]][edge[1]]
                compare2 = comparison[edge[1]][edge[0]]
                selected_comparison = max(compare1, compare2)
                PPI_comparison.append(selected_comparison)
            except:
                continue
        
        PPI_edges = random.sample(PPI_edges, len(LCC_comparison) * 200)
        background_edges = random.sample(sorted_tuples, len(LCC_comparison) * 1000) 

        def obtain_attentions(base, compare, edge_list):
            base_attentions, compare_attentions = [], []

            for edge in edge_list:
                attn_found = True

                # Obtains original attention weights
                try:
                    attention = base[edge[0]][edge[1]]
                except:
                    try:
                        attention = base[edge[1]][edge[0]]
                    except:
                        attn_found = False
                            
                # Obtains comparison attention weights
                try:
                    compare_attention = compare[edge[0]][edge[1]]
                except:
                    try:
                        compare_attention = compare[edge[1]][edge[0]]
                    except:
                        attn_found = False
                
                if attn_found == True:
                    if not isinstance(attention, list) and not isinstance(compare_attention, list):
                        base_attentions.append(attention)
                        compare_attentions.append(compare_attention)
 
            return base_attentions, compare_attentions
        
        LCC_base, LCC_compare = obtain_attentions(base, compare, LCC_edges)
        PPI_base, PPI_compare = obtain_attentions(base, compare, PPI_edges)
        Background_base, Background_compare = obtain_attentions(base, compare, background_edges)

        plt.figure(figsize = (10, 10))
        plt.xlabel('Pretrained Weight')
        plt.ylabel('Finetuned Weight')
        plt.scatter(LCC_base, LCC_compare, color = 'blue', label = 'LCC Attention', alpha = .4, s = 8)
        plt.scatter(PPI_base, PPI_compare, color = 'green', label = 'PPI Attention', alpha = .4, s = 8)
        plt.scatter(Background_base, Background_compare, color = 'red', label = 'Background Attention', alpha = .4, s = 8)
        plt.plot([0, max(Background_base)], [0, max(LCC_compare)], color = 'purple', linestyle = '--', label = 'No Change')
        plt.legend(loc = 'upper right')
        plt.savefig(f'{keyword}_change.svg')

        LCC_FC = [LCC_compare[i]/(LCC_base[i] + 0.00001) for i in range(len(LCC_base))]
        PPI_FC = [PPI_compare[i]/(PPI_base[i] + 0.00001) for i in range(len(PPI_base))]
        background_FC =[Background_compare[i]/(Background_base[i] + 0.00001) for i in range(len(Background_base))]

        # Plot each distribution as CDF
        plt.figure()
        print(f'Average LCC FC: {np.mean(LCC_FC)}')
        print(f'Average PPI FC: {np.mean(PPI_FC)}')
        print(f'Average Background FC: {np.mean(background_FC)}')

        sns.ecdfplot(LCC_FC, color='blue', label='LCC Fold Change')
        sns.ecdfplot(PPI_FC, color='green', label='PPI Fold Change')
        sns.ecdfplot(background_FC, color='red', label='Background Fold Change')
        plt.xscale('log')
        plt.axvline(x = 1, color = 'purple', linestyle = '--', label = 'No Change')
        plt.xlabel('Pretrained to Finetuned Fold Change')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function of Distributions')
        plt.legend()
        plt.savefig(f'{keyword}_FC_distribution.svg')

    return sorted_tuples, comparison, fold_changes
