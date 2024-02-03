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
import requests
import copy
import math
import traceback

# Data processing/pre-processing/comparison imports
from collections import defaultdict, Counter
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, binom_test, ks_2samp
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy import interpolate, optimize 
import scipy.stats as stats

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

# STRING DB imports
from .string_db import *

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
def process_batch(attentions, context_length = 2048, selected_heads = None):
    attention_matrices, genes_list = attentions[0], attentions[1]

    num_attention_heads = len(attention_matrices[0])
    gene_attentions = {}

    # List of all attention heads that will be extracted
    if selected_heads == None:
        selected_heads = [i for i in range(num_attention_heads)]
    elif selected_heads == 'Max':
        selected_heads = [num_attention_heads - 1]
    else:
        selected_heads = selected_heads

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
            if attention_head_num in selected_heads:
                attention_head = attention_matrix[attention_head_num]
                attention_weights = attention_head[np.arange(len(source_genes))[:, None], np.arange(len(target_genes))]
                attention_weights = attention_weights - np.diag(np.diag(attention_weights))

                for i, source_gene in enumerate(source_genes):
                    for j, target_gene in enumerate(target_genes):
                        gene_attentions.setdefault(source_gene, {}).setdefault(target_gene, []).append(attention_weights[i][j])

    return gene_attentions
    
# Primary runtime function for extracting attention weights
def extract_attention(model_location = '.', 
                      data = '/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/genecorpus_30M_2048.dataset',
                      samples = 1000, 
                      layer_index = -1, 
                      gene_conversion = "/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/gene_name_id_dict.pkl", 
                      mean = False,  
                      num_processes = os.cpu_count(),
                      filter_genes = None, save = False, 
                      data_sample = None, half = False,
                      filter_label = None, save_threshold = False, 
                      perturb_genes = None, shuffle = False,
                      mean_threshold = 4):
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
    
    perturb_genes : list, None
        Whether the dataset should be perturbed for a specific gene before attention extraction. Defaults to None

    mean_threshold : int
        The amount of times a gene should be present in order to count as a collected gene. Works for mean aggregation only. Defaults to 4

    OUTPUTS
    -----------------------------------------
    aggregated_gene_attentions : dict
        Dictionary of aggregated gene attentions, processed as either the mean or the max of all relevant attention weights

    '''
    
    # Creates conversion from tokens back to genes
    token_dict = TranscriptomeTokenizer().gene_token_dict
    token_dict = {value: key for key, value in token_dict.items()}
    conversion = {value: key for key, value in pk.load(open(gene_conversion, 'rb')).items()}
    conversion = {key: conversion[value] for key, value in token_dict.items() if value in conversion}
    deconversion = {value:key for key, value in conversion.items()}

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
            
    else: 
        # Greedy sampling to ensure that specified gene IDs are present (if possible) in sampled dataset
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
        
        if len(filter_genes) > 10000:
            filter_genes = None
            
    # Loads model
    model = BertModel.from_pretrained(model_location, num_labels=2, output_attentions=True, output_hidden_states=False).to(device).eval()

    # Shuffles order of all tokens in dataset if indicated
    if shuffle:
        def shuffled(seq):
            shuffled_seq = list(seq)  # Create a copy of the original sequence
            random.shuffle(shuffled_seq)
            return shuffled_seq
        
        def shuffle_tokens(dataset):
            shuffled_dataset = dataset.map(lambda example: {"input_ids": shuffled(example["input_ids"])}) 
            return shuffled_dataset
        
        data = shuffle_tokens(data)
    
    # Uses half-precsion weights to save time. Only available with gpus 
    if half == True and torch.cuda.is_available():
        model = model.half()  
    
    # Sends model to device and prepares for evaluation
    model = model.to(device).eval()
    
    # Function for obtaining weights
    def obtain_weights(model, data):

        # Creates dataloader for dataset
        dataloader = DataLoader(GFDataset(data), batch_size=9, num_workers=4)

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
                        if len(attentions) > mean_threshold:
                            average_attention = np.mean(attentions)
                            if average_attention > threshold:
                                aggregated_gene_attentions[source_gene][target_gene].append((len(attentions), average_attention))
                                
            else:
                for source_gene, target_genes in tqdm.tqdm(gene_attentions.items(), total = len(gene_attentions.items()), desc = 'Narrowing results'):
                    for target_gene, attentions in target_genes.items():
                        # Creates new dictionary for source gene if not already existing
                        if len(attentions) > mean_threshold:
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
        print(f'{len(aggregated_gene_attentions.keys())} total source genes represented in the final dataset')
        
        if save != False:
            pk.dumps(open(save, 'wb'))
            
        return aggregated_gene_attentions
    
    # Perturbs dataset if indicated
    if perturb_genes != None:

        # Copies dataset
        perturbed_data = copy.deepcopy(data)
        if type(perturb_genes) != list:
            perturb_genes = [perturb_genes]

        # This will create a new list of input_ids for each example, excluding the specified token ID
        def remove_token(batch, gene_token):
            if gene_token in batch['input_ids']:
                batch['input_ids'] = [i for i in batch['input_ids'] if i != gene_token]
            return batch

        # Perturbs genes
        for gene in perturb_genes:
            gene_token = deconversion[gene]
            try:
                perturbed_data = perturbed_data.map(remove_token, gene_token)
            except:
                try:
                    print(f'Gene {conversion[gene_token]} not a Geneformer-compatible gene! Skipping...')
                except:
                    print(f'Gene token {gene_token} not a Geneformer-compatible gene!')
                    continue
            
        # Obtains regualr and perturbed weights
        regular_weights = obtain_weights(model, data)
        perturbed_weights = obtain_weights(model, perturbed_data)

        return regular_weights, perturbed_weights
    else:
        weights = obtain_weights(model, data)
        return weights

    
# Compares the CDFs of multiple different distributions
def compare_LCCs(attentions, num_diseases = 11, keyword = None, #guaranteed_LCCs = ['cardiomyopathy hypertrophic', 'heart failure', 'cardiomyopathy dilated', 'dementia', 'arthritis rheumatoid', 'anemia', 'calcinosis', 'parkinsonian disorders'],
                guaranteed_LCCs = ['covid','cardiomyopathy hypertrophic', 'cardiomyopathy dilated', 'adenocarcinoma', 'small cell lung carcinoma', 'heart failure', 'dementia', 'arthritis rheumatoid', 'anemia', 'calcinosis', 'parkinsonian disorders'],
                disease_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/PPI/GDA_Filtered_04042022.csv'),
                return_LCCs = False, save = True):
    network_LCCs = {}

    # Obtains and maps LCCs that are given
    PPI = instantiate_ppi()
    for LCC in tqdm.tqdm(guaranteed_LCCs, total = len(guaranteed_LCCs), desc = 'Obtaining Given LCCs'):
        disease_genes = isolate_disease_genes(LCC.lower())
        LCC_nodes = LCC_genes(PPI, disease_genes, subgraph = True,)
        try:
            LCC_nodes, _, _ = map_attention_attention(LCC_nodes, attentions, LCC = True)
        except:
            continue
        network_LCCs[LCC] = LCC_nodes
    
    # Samples random LCCs
    diseases = list(set(pd.read_csv(disease_location)['NewName']) - set([i.lower() for i in guaranteed_LCCs]))
    random_LCCs = random.sample(diseases, num_diseases - len(network_LCCs))

    # Iterates through LCCs and obtains LCC-related attention
    for LCC in tqdm.tqdm(random_LCCs, total = len(random_LCCs), desc = 'Obtaining Random LCCs'):
        disease_genes = isolate_disease_genes(LCC)
        try:
            LCC_nodes = LCC_genes(PPI, disease_genes, subgraph = True,)
        except:
            continue

        network_LCC, _, _ = map_attention_attention(LCC_nodes, attentions)
        network_LCCs[LCC] = network_LCC
    
    if return_LCCs:
        network_LCCs = {LCC: [network[u][v]['attention'] for u, v in network.edges()] for LCC, network in network_LCCs.items()}
        return network_LCCs
    else:
        plt.figure(figsize = (11,8))
        plt.xscale('log')
    
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'black']
        counter = 0

        for LCC, network in network_LCCs.items():
            # Retrieve and adjust attention values
            LCC_attentions = [network[u][v]['attention'] 
                            for u, v in network.edges() 
                            if not isinstance(network[u][v]['attention'], list)]
            sns.ecdfplot(LCC_attentions, label=f'{LCC}', color = colors[counter]) # (n={len(LCC_attentions)})')
            counter += 1

        plt.xlabel('Attention Weights')
        plt.ylabel('Cumulative Probability')
        plt.legend(loc = 'lower right')
        plt.title('Cumulative Distribution of Attention Weights for Various LCCs')
        plt.tight_layout()

        # Save the plot
        if keyword:
            plt.savefig(f'GroupLCC_{keyword}.png')
        else:
            plt.savefig('GroupLCC.png')


# Maps attention scores to PPI
def map_attention_attention(PPI, gene_attentions, 
                            save = True, 
                            LCC = False, 
                            mean = False,
                            show = False):
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
            if mean == False:
                selected_attention = max(attention_direction1, attention_direction2)
            else:
                selected_attention = (attention_direction1 + attention_direction2) / 2

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

    if show == True:
        if LCC == False:
            print(f'PPI attention: {average_edge_attention}')
            print(f'PPI attention standard deviation: {np.std(edge_attentions)}')
        else:
            print(f'LCC attention: {average_edge_attention}')
            print(f'LCC attention standard deviation: {np.std(edge_attentions)}')
            
        print(f'Total attention: {average_total_attention}')
        print(f'Total attention standard deviation: {np.std(total_attention)}')
        print(f'T-test p-value: {p_value} \n')
        print(f'Average network ratio {np.mean([PPI_copy[u][v]["ratio"] for u, v in PPI_copy.edges()])}')

    return PPI_copy, average_edge_attention, average_total_attention
    
# Function for creating fake edges
def generate_fake_attentions(graph, num_fake_attentions, 
                             gene_attentions, 
                             respect_PPI = True, 
                             bidirectional = False,
                             min_threshold = None):

    # Prepares variables and progress bar
    fake_edges = set()
    nodes = list(graph.nodes())
    node_degrees = [graph.degree(node) for node in nodes]
    total_degree = sum(node_degrees)
    node_probabilities = [degree / total_degree for degree in node_degrees]

    pbar = tqdm.tqdm(total=num_fake_attentions, desc="Generating Fake Edges")
    fake_attentions = []

    # Generates attentions following a degree distribution similar to the real graph
    if respect_PPI:
        fake_attentions = []

        # Iterates fake attentions until quota is matched
        while len(fake_attentions) < num_fake_attentions:
            # Randomly select two nodes based on degree distribution
            node1, node2 = np.random.choice(nodes, 2, replace=False, p=node_probabilities)

            # Ensure the selected nodes are not connected in the real graph
            if not graph.has_edge(node1, node2) and node1 != node2:

                # Ensure the edge is not already in the fake edges list
                if (node1, node2) not in fake_edges and (node2, node1) not in fake_edges:
                    try:

                        # Takes the attention weight in one direction or both depending on bidirectioanlity settings
                        if bidirectional:
                            attention1 = gene_attentions[node1][node2]
                            attention2 = gene_attentions[node2][node1]
                            attention = max(attention1, attention2)
                        else:
                            attention = gene_attentions[node1][node2]

                        # Adds the fake edge to the set
                        if not isinstance(attention, list):
                            fake_attentions.append(attention)
                            fake_edges.add((node1, node2)) 
                            pbar.update(1)
                        else:
                            raise ValueError
                    except:
                        continue
        pbar.close()
           
    else:
        # Flattens attention weight dictionary
        flattened_dict = [value
                    for _, inner_dict in gene_attentions.items()
                    for _, value in inner_dict.items()]
        

        while len(fake_attentions) < num_fake_attentions:

            # Randomly selects two attentions
            if bidirectional == True:
                random_attentions = random.sample(flattened_dict, 2)
                attention = max(random_attentions)
            else:
                attention = random.sample(flattened_dict, 1)[0]

            # Adds attention weight to set
            if not isinstance(attention, list):
                fake_attentions.append(attention)
                pbar.update(1)
        pbar.close()

    if min_threshold:
        fake_attentions = [attention for attention in fake_attentions if attention >= min_threshold]

    return fake_attentions  

# Calculates F1 scores for PPI and LCC, calculates both their AUC curves and graphs them
def F1_graph_attention(PPI, gene_attentions, LCC = None, show = True, graph = True, keyword = None,
                       save = True, epsilon = 10e-8):

    if LCC != None:
        # Obtains LCC attentions
        LCC_attentions = []
        for u, v in LCC.edges():
            try:
                attention = LCC[u][v]['attention']

                # Thresholding if enabled
                if min_threshold:
                        if attention < min_threshold:
                            continue

                # Bulks up size by sampling 3 times
                LCC_attentions += [attention] * 3
            except:
                pass
        
        # Obtains PPI attentions
        PPI_attentions = []
        for u, v in PPI.edges():
            try:
                attention = PPI[u][v]['attention']
                if not isinstance(attention, list):
                    PPI_attentions.append(attention)
            except:
                pass

        fake_LCC_attentions = generate_fake_attentions(PPI, len(LCC_attentions), gene_attentions, min_threshold = epsilon)
        fake_PPI_attentions = generate_fake_attentions(PPI, len(PPI_attentions), gene_attentions, min_threshold = epsilon) 
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

        # Assuming you have y_pred as the predicted labels (0 or 1) and y_prob as predicted probabilities for the positive class
        fpr1, tpr1, _ = roc_curve(LCC_labels, fake_attentions)
        fpr2, tpr2, _ = roc_curve(LCC_labels, PPI_LCC_attentions)
        fpr3, tpr3, _ = roc_curve(PPI_labels, PPI_fake_attentions)

        # Saves FPR/TPR data
        if save == True:
            data = {
                'LCC-background FPR': fpr1,
                'LCC-background TPR': tpr1,
                'LCC-PPI FPR': fpr3,
                'LCC-PPI': tpr3,
                'PPI-background FPR': fpr2, 
                'PPI-background TPR': tpr2
            }

            data = pd.DataFrame.from_dict(data, orient = 'index')
            data = data.transpose()
            data.to_csv(f'disease_ROC_{keyword}.csv')

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
            plt.figure(figsize = (10, 7))
            plt.xlabel('False Positive Rate', fontsize = 12)
            plt.ylabel('True Positive Rate', fontsize = 12)
            plt.xlim([-.05, 1.05])
            plt.ylim([0, 1.05])
            plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--', label = 'Random')
            plt.title('ROC Curve for Finetuned Cardiomyopathy Geneformer Attentions \n (n = 10,000 samples)')
            #plt.plot(fpr2, tpr2, color = 'purple', label = f'LCC attentions to PPI attentions \n (AUC: {round(LCC_auc, 4)})')
            plt.plot(fpr1, tpr1, color = 'blue', label = f'LCC-Background Prediction \n (AUC: {round(fake_auc, 3)})')
            plt.plot(fpr3, tpr3, color = 'green', label = f'PPI-Background Prediction \n (AUC: {round(PPI_auc, 3)})')
            plt.legend(loc='lower right', fontsize = 12)#, bbox_to_anchor=(0, 0.2), ncol=1, fontsize = 8)
    
            if keyword:
                plt.savefig(f'LCC-PPI-fake_precision_{keyword}.png')
            else:
                plt.savefig('LCC-PPI-fake_precision.png')
            
        return fake_auc, LCC_auc
        
    else:
        # Obtains real (network) attentions
        real_attentions = []
        for u, v in PPI.edges():
            try:
                attention = PPI[u][v]['attention']
                real_attentions.append(attention)
            except:
                pass

        # Generates fake attentions
        fake_attentions = generate_fake_attentions(PPI, len(real_attentions), gene_attentions, min_threshold = epsilon)
        background_attentions = generate_fake_attentions(PPI, len(real_attentions), gene_attentions, min_threshold = epsilon,
                                                         respect_PPI = False)
        
        print(f'Background mean: {np.mean(background_attentions)}')
        print(f'PPI mean: {np.mean(real_attentions)}')
        print(f'Fake mean: {np.mean(fake_attentions)}')

        # Generates 'real' and 'fake' labels for attentions
        fake_labels = [0 for _ in range(len(fake_attentions))] 
        background_fake_labels = [0 for _ in range(len(background_attentions))]
        real_labels = [1 for _ in range(len(real_attentions))]
        
        # Combines labels
        fake_labels = fake_labels + real_labels
        background_labels = background_fake_labels + real_labels
        fake_attentions = fake_attentions + real_attentions
        background_attentions = background_attentions + real_attentions
        
        # Assuming you have y_pred as the predicted labels (0 or 1) and y_prob as predicted probabilities for the positive class
        fpr1, tpr1, _ = roc_curve(fake_labels, fake_attentions)
        fpr2, tpr2, _ = roc_curve(background_labels, background_attentions)
        
        # Compute AUC for ROC curve
        auc1 = auc(fpr1, tpr1)
        auc2 = auc(fpr2, tpr2)
        
        # Prints AUC scores
        if show == True:
            print(f'Fake-PPI AUC scores: {auc1}')
            print(f'Background-PPI AUC scores: {auc2}')
    
        if save == True:
            data = {
                'PPI-fake FPR': fpr1,
                'PPI-fake TPR': tpr1,
                'PPI-background FPR': fpr2,
                'PPI-background TPR': tpr2
            }

            data = pd.DataFrame.from_dict(data, orient = 'index')
            data = data.transpose()
            data.to_csv(f'AUC_scores_{keyword}.csv', index = False)

        # Plots AUC curve
        if graph == True:
            plt.figure()
            plt.xlabel('False Positive Rate', fontsize = 12)
            plt.ylabel('True Positive Rate', fontsize = 12)
            plt.xlim([-0.05, 1.0])
            plt.ylim([0, 1.05])
            plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--',)
            plt.plot(fpr1, tpr1, color = 'green', label = f'PPI-Fake PPI Attentions \n (AUC: {round(auc1, 4)})')
            #plt.plot(fpr2, tpr2, color = 'blue', label = f'PPI-Background Attentions \n (AUC: {round(auc2, 4)})')
            plt.legend(loc='lower right', fontsize = 12)#, bbox_to_anchor=(0, 0.2), ncol=1)
            plt.tight_layout()
            if keyword:
                plt.savefig(f'PPI_precision_{keyword}.png')
            else:
                plt.savefig('PPI_precision.png')
        
        return auc1, auc2
        
# Processes a list of attention values and a PPI network to obtain a list of mapped and unmapped attentions, cutting off at a specific threshold
def process_edges(PPI, attention_dict, min_threshold = None):
    real_attentions = []
    total_attentions = []

    # Obtains true edges from graph
    total_edges = list(PPI.edges())
    with tqdm.tqdm(total=len(total_edges), desc='Processing Edges') as pbar:
        for u, v in total_edges:
            try:
                attention = PPI[u][v]['attention']
                real_attentions.append(attention)
            except KeyError:
                pass
            pbar.update()

    # Flattens dictionary to a list of values
    total_attentions = [value for _, inner_dict in attention_dict.items()
                      for _, value in inner_dict.items() if not isinstance(value, list)]
        
    # Filters edges if a threshold exists
    if min_threshold:
        real_attentions = [i for i in real_attentions if i > min_threshold]
        total_attentions = [i for i in total_attentions if i > min_threshold]
    
    return real_attentions, total_attentions

# Primary function for community detection
def community_detection(PPI, 
                        attention_dict, 
                        disease  = 'cardiomyopathy hypertrophic', 
                        keyword = None,
                        use_geneformer_ppi=True, 
                        compare_with_lcc=True,
                        max = True, 
                        node_num = 10):


    # Obtain disease genes
    disease_genes = isolate_disease_genes(disease, filter_strong=False)
    LCC_nodes = LCC_genes(PPI, disease_genes, subgraph=True) if compare_with_lcc else None
    strong_genes = isolate_disease_genes(disease, filter_strong=True)
    
    print('Generating Networks...')
    # Define the network for analysis
    if use_geneformer_ppi:
        network_for_analysis = nx.DiGraph()
        for source, targets in tqdm.tqdm(attention_dict.items(), total = len(attention_dict), desc = 'Creating GF PPI'):
            for target, weight in targets.items():
                # Ensure that the edge (u, v) has the maximum weight of attention_dict[u][v] and attention_dict[v][u]
                try:
                    if max == True:                    
                        weight = np.max((weight, attention_dict[target][source]))
                    else:
                        try:
                            weight = np.mean([weight, attention_dict.get(target, {}).get(source, 0)])
                        except:
                            continue
                    if not isinstance(weight, list):
                        network_for_analysis.add_edge(source, target, weight=weight)

                except:
                    pass

        # Perform disparity filtration on PPI
        try:
            network_for_analysis = disparity_filter(network_for_analysis)
        except:
            print('Disparity filtration not working! Passing..')

        # Find all isolated nodes
        isolated_nodes = list(nx.isolates(network_for_analysis))

        # Remove isolated nodes from the graph
        network_for_analysis.remove_nodes_from(isolated_nodes)
        network_nodes = {count + 1:i for count, i in enumerate(network_for_analysis.nodes())}

        # Calculate Degree Centrality and Strength of Nodes
        degree_centrality = dict(network_for_analysis.degree())
        strength_of_nodes = {node: sum([attr['weight'] for _, _, attr in network_for_analysis.edges(node, data=True)]) for node in network_for_analysis.nodes()}

        # Identify Top Nodes by Degree and Strength
        top_nodes_by_degree = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
        top_nodes_by_strength = sorted(strength_of_nodes.items(), key=lambda item: item[1], reverse=True)

        top_degree_nodes = top_nodes_by_degree[:node_num]
        top_strength_nodes = top_nodes_by_strength[:node_num]

        # Analysis of Top Nodes
        print("Top nodes by degree centrality:")
        for node, degree in top_degree_nodes:
            print(f"Node: {node}, Degree: {degree}")

        print("\nTop nodes by strength:")
        for node, strength in top_strength_nodes:
            print(f"Node: {node}, Strength: {strength}")

        # Step 5: Compare with LCC and Disease Genes
        # Assuming LCC_nodes and disease_genes are already defined
        lcc_nodes = set(LCC_nodes)
        disease_genes = set(disease_genes)

        top_strength_genes = set([node for node, _ in top_strength_nodes])
        overlap_with_lcc = top_strength_genes.intersection(lcc_nodes)
        overlap_with_disease_genes = top_strength_genes.intersection(disease_genes)

        print(f"\nOverlap between top strength genes and LCC: {overlap_with_lcc}")
        print(f"Overlap between top strength genes and disease genes: {overlap_with_disease_genes}")

    else:
        network_for_analysis = PPI

    if not strong_genes:
        raise ValueError("No strong disease genes found.")
    start_node = random.choice(list(strong_genes))
    
    bidirectional_network = network_for_analysis.to_undirected()

    # Perform Random Walk with Restart
    walk = random_walk_with_restart(bidirectional_network, start_node)

    # Analyze the visitation frequencies
    visitation_frequencies = analyze_visitation_frequencies(walk)

    # Detect communities
    visited_nodes = list(visitation_frequencies.keys())
    partition = detect_communities(bidirectional_network, visited_nodes)
    
    print('Greedy communities..')
    # Detect communities using Girvan-Newman algorithm
    girvan_newman_communities = detect_communities_greedy(bidirectional_network)

    # Reference set for comparison
    reference_nodes_set = set(LCC_nodes) if compare_with_lcc else set(PPI.nodes())

    print('Comparing communities')
    # Compare identified communities
    community_nodes_set = set(partition.keys())
    girvan_newman_nodes_set = set.union(*map(set, girvan_newman_communities))
    
    # Run DIAMOnD analysis
    diamond_nodes = DIAMOnD(network_for_analysis, seed_genes=strong_genes, max_number_of_added_nodes=len(LCC_nodes), alpha=1)
    diamond_nodes_set = set(diamond_nodes)
    
    # Comparing with reference set (either LCC or original PPI)
    for method, community_set in [("Random Walk", community_nodes_set), ("Girvan-Newman", girvan_newman_nodes_set), ("DIAMOND", diamond_nodes_set)]:
        similarity = jaccard_similarity(reference_nodes_set, community_set)
        added_nodes = community_set - reference_nodes_set
        removed_nodes = reference_nodes_set - community_set

        print(f"{method} Jaccard Similarity: {similarity}")
        print(f"{method} # Nodes added: {len(added_nodes)}")
        print(f"{method} # Nodes removed: {len(removed_nodes)}")

    # Advanced network comparison (for PPI)
    if use_geneformer_ppi == True:
        community_subgraph = bidirectional_network.subgraph(community_nodes_set)
        girvan_newman_subgraph = bidirectional_network.subgraph(girvan_newman_nodes_set)
        diamond_subgraph = bidirectional_network.subgraph(diamond_nodes_set)
        compare_networks(PPI, community_subgraph, LCC_nodes)
        compare_networks(PPI, girvan_newman_subgraph, LCC_nodes)
        compare_networks(PPI, diamond_subgraph, LCC_nodes)

# Arthritis comorbids: ['arthritis\nrheumatoid',  'osteoporosis', 'cardiovascular\ndiseases', 'lupus\nerythematosus\nsystemic',]
# Cardiomyopathy comorbids: diseases = ['cardiomyopathy\nhypertrophic', 'calcinosis', 'cardiomyopathy\ndilated', 'anemia', 'hypertension',]
# Performs cormobidity analysis
def comorbidity_analysis(PPI, attention_dict, keyword=None, bulk_comparison=True,
                         comorbid=['cardiomyopathy\nhypertrophic', 'cardiomyopathy\ndilated', 'anemia', 'atrial\nfibrillation', 'diabetes\nmellitus\ntype\n2', 'hypertension', 'coronary\nartery\ndisease'],
                         save=True, background=['dementia', 'covid', 'parkinsonian\ndisorder', 'dwarfism', 'hepatitis\nb', 'kidney\nfailure\nchronic', 'calcinosis']):
    
    # Function for plottng histograms for comorbid and background weights
    def plot_histogram(comorbid_weights, background_weights, keyword, index):
        plt.figure()
        comorbid_weights, background_weights = np.array(comorbid_weights), np.array(background_weights)
        comorbid_weights, background_weights = comorbid_weights[~np.isnan(comorbid_weights)], background_weights[~np.isnan(background_weights)]
        stack = (comorbid_weights, background_weights)
        bins = np.histogram(np.hstack(stack), bins=30)[1]
        plt.hist(comorbid_weights, bins=bins, color='red', label='Comorbidities', alpha=0.5)
        plt.hist(background_weights, bins=bins, color='blue', label='Background', alpha=0.5)
        plt.ylabel('Frequency')
        plt.xlabel('Attention weights')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'Comorbidity_bulk_{keyword}_{index}.png')

    
    # Combine comorbid and background diseases into a single list
    diseases = comorbid + background
    disease_sets, LCC_sets = {}, {}

    # Function to calculate the average weight along a path
    def avg_weight_of_path(path):
        return np.mean([max(attention_dict[path[i]][path[i+1]], attention_dict[path[i+1]][path[i]]) for i in range(len(path)-1)]) if len(path) > 1 else 0

    # Function to analyze and visualize disease interaction weights
    def analyze_weights(disease_sets, pairwise=True):
        weights = {}
        for disease, genes in disease_sets.items():
            weights[disease] = {}
            for target_disease, target_genes in disease_sets.items():
                if disease != target_disease:
                    # Calculate average weight for each pair of diseases
                    weights[disease][target_disease] = np.mean([max(attention_dict[gene1][gene2], attention_dict[gene2][gene1]) for gene1 in genes for gene2 in target_genes if gene1 in attention_dict and gene2 in attention_dict])

        # Create a matrix and heatmap for the disease interaction weights
        matrix = np.array([[0 if disease == target_disease else weights[disease][target_disease] for target_disease in diseases] for disease in diseases])
        max_value = np.max([i for j in matrix for i in j])
        if save:
            np.savetxt(f"matrix_{keyword}_{'1' if pairwise else '2'}.csv", matrix, delimiter=",")
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=False, fmt=".2f", xticklabels=diseases, yticklabels=diseases, vmax=max_value)
        plt.title("Disease Interaction Average Weights")
        plt.xlabel("Target Disease")
        plt.ylabel("Disease")
        plt.tight_layout()
        plt.savefig(f'comorbidity_heatmap_{1 if pairwise else 2}_{keyword}.png')

    # If bulk_comparison is False, perform pairwise and LCC analysis
    if not bulk_comparison:
        for disease in tqdm.tqdm(diseases, total=len(diseases), desc='Obtaining Given LCCs'):
            disease_genes = isolate_disease_genes(disease)
            LCC_nodes = LCC_genes(PPI, disease_genes, subgraph=True)
            disease_sets[disease] = disease_genes
            LCC_sets[disease] = LCC_nodes

        # Perform pairwise analysis for all diseases
        analyze_weights(disease_sets)
        
        # Calculate and visualize average weights between LCCs
        LCC_weights = {}
        for key, LCC in tqdm.tqdm(LCC_sets.items(), total=len(LCC_sets), desc='Processing shortest paths..'):
            LCC_weights[key] = {}
            for sub_key, sub_LCC in LCC_sets.items():
                if key != sub_key:
                    LCC_weights[key][sub_key] = np.mean([avg_weight_of_path(nx.shortest_path(source=gene1, target=gene2)) for gene1 in LCC for gene2 in sub_LCC])

        # Perform LCC analysis for all LCCs
        analyze_weights(LCC_weights, pairwise=False)
    
    else:  # If bulk_comparison is True, perform bulk analysis
        comorbid = comorbid + ['kidney\nfailure\nchronic', 'calcinosis']
        background_diseases = isolate_background_genes(comorbid, filter_strong=True, num_LCCs=50)
        target_disease = comorbid[0]
        target_disease_genes = isolate_disease_genes(target_disease)
        LCC_nodes = LCC_genes(PPI, target_disease_genes, subgraph=True)

        total_comorbid, total_background = [], []

        # Analysis 1: Calculate and visualize attention weights for comorbid and background diseases
        for disease in comorbid[1:]:
            disease_genes = isolate_disease_genes(disease)
            comorbid_weights = [max(attention_dict[gene][target_gene], attention_dict[target_gene][gene]) for gene in disease_genes for target_gene in target_disease_genes if gene in attention_dict and target_gene in attention_dict]
            total_comorbid.append(np.mean(comorbid_weights))

        for disease in background_diseases:
            disease_genes = isolate_disease_genes(disease)
            background_weights = [max(attention_dict[gene][target_gene], attention_dict[target_gene][gene]) for gene in disease_genes for target_gene in target_disease_genes if gene in attention_dict and target_gene in attention_dict]
            total_background.append(np.mean(background_weights))

        # Plot histograms for comorbid and background weights
        plot_histogram(total_comorbid, total_background, keyword, 1)

        # Analysis 2: Calculate and visualize average weights between comorbidities and backgrounds
        comorbid_weights, background_weights = [], []
        for disease in tqdm.tqdm(comorbid[1:], total=len(comorbid[1:]), desc='Comorbidity paths'):
            disease_genes = isolate_disease_genes(disease)
            disease_LCC = LCC_genes(PPI, disease_genes, subgraph=True)
            comorbid_weights.append(np.mean([avg_weight_of_path(nx.shortest_path(source=gene1, target=gene2)) for gene1 in LCC_nodes for gene2 in disease_LCC]))

        for disease in tqdm.tqdm(background_diseases, total=len(background_diseases), desc='Background paths'):
            disease_genes = isolate_disease_genes(disease)
            disease_LCC = LCC_genes(PPI, disease_genes, subgraph=True)
            background_weights.append(np.mean([avg_weight_of_path(nx.shortest_path(source=gene1, target=gene2)) for gene1 in LCC_nodes for gene2 in disease_LCC]))

        # Plot histograms for comorbid and background weights
        plot_histogram(comorbid_weights, background_weights, keyword, 2)
                

# Plots attention distributions for background or a PPI/disease if specified
def plot_distributions(attention_dict, disease = None, graph = True, keyword = None,
                       epsilon = 10e-6, ratio_comparisons = False,
                       save = True):    
        
    # Plots distributions without a disease
    if disease == None:
      
        # Instantiates PPI 
        PPI = instantiate_ppi()
        PPI, _, _ = map_attention_attention(PPI, attention_dict)

        # Parallel processing to extract attentions
        real_attentions, total_attentions = process_edges(PPI, attention_dict, min_threshold = epsilon)
        PPI_fake = generate_fake_attentions(PPI, len(PPI), attention_dict, respect_PPI = True, min_threshold = epsilon)

        # Saves data
        if save == True:
            with open(f'PPI_attentions_{keyword}.pkl', 'wb') as f:
                pk.dump(random.sample(real_attentions, 1000), f)
            with open(f'total_attentions_{keyword}.pkl', 'wb') as f:
                pk.dump(total_attentions, f)
            with open(f'PPI_fake_attentions_{keyword}.pkl', 'wb') as f:
                pk.dump(random.sample(PPI_fake, 1000), f)

        # Plot CDF
        if graph == True:
            plt.figure()
            sns.ecdfplot(total_attentions, color='red', label=f'Background Attentions')# (n={len(total_attentions)})')
            sns.ecdfplot(real_attentions, color='green', label=f'PPI Attentions')# (n={len(real_attentions)})')
            sns.ecdfplot(PPI_fake, color='black', label=f'PPI Fake Attentions')
            plt.xlabel('Attention Weights')
            plt.ylabel('Cumulative Probability')
            plt.legend(loc='lower right')
            plt.xscale('log')
            plt.xlim(0, 0.1)
            plt.tight_layout()

            # Save the plot
            if keyword:
                plt.savefig(f'PPIAttentionDist_{keyword}.png')
            else:
                plt.savefig('PPIAttentionDist.png')

        
    else:
        # Plots distribution with a disease module
        # Instantiates PPI 
        PPI = instantiate_ppi()
        PPI, _, _ = map_attention_attention(PPI, attention_dict)
        
        # Instantiates disease LCC
        disease_genes = isolate_disease_genes(disease)
        disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
        LCC, _, _ = map_attention_attention(disease_LCC, attention_dict, LCC = True)
        
        # Instantiates fully connected disease LCC
        LCC_connected = nx.complete_graph(disease_genes,)
        LCC_connected, _, _ = map_attention_attention(LCC_connected, attention_dict, LCC = True)

        # Parallel processing to extract attentions
        real_attentions, total_attentions = process_edges(PPI = LCC, attention_dict = attention_dict, min_threshold = epsilon)
        connected_attentions, _ = process_edges(PPI = LCC_connected, attention_dict = attention_dict, min_threshold = epsilon)
        PPI_attentions, _ = process_edges(PPI = PPI, attention_dict = attention_dict, min_threshold = epsilon)
        
        # Filters attention for invalid or empty attentions
        real_attentions = [i for i in real_attentions if not isinstance(i, list)]
        total_attentions = [i for i in total_attentions if not isinstance(i, list)]
        PPI_attentions = [i for i in PPI_attentions if not isinstance(i, list)]
        connected_attentions = [i for i in connected_attentions if not isinstance(i, list)]

        # Saves data
        if save == True:
            with open(f'LCC_attentions_{keyword}.pkl', 'wb') as f:
                pk.dump(real_attentions, f)
            with open(f'total_attentions_{keyword}.pkl', 'wb') as f:
                pk.dump(random.sample(total_attentions, 10000), f)
            with open(f'PPI_attentions_{keyword}.pkl', 'wb') as f:
                pk.dump(random.sample(PPI_attentions, 10000), f)
            with open(f'connected_attentions_{keyword}.pkl', 'wb') as f:
                pk.dump(connected_attentions, f)


        # Performs basic comparisons if enabled
        if ratio_comparisons == True:

            # Identifies proportion of nodes in top section of 
            proportions = [i*10e-4 for i in list(range(1000))]

            remaining_proportions = [[] for _ in range(len(proportions))]
            for proportion_count, proportion in enumerate(proportions):
                background_select = [i for i in total_attentions if i > proportion]
                PPI_select = [i for i in PPI_attentions if i > proportion]
                real_select = [i for i in real_attentions if i > proportion]
                connected_select = [i for i in connected_attentions if i > proportion]

                total_proportion = len(background_select) / len(total_attentions)
                PPI_proportions = len(PPI_select) / len(PPI_attentions)
                real_proportion = len(real_select)/len(real_attentions)
                connected_proportion = len(connected_select) / len(connected_attentions)
                remaining_proportions[proportion_count] = [total_proportion, PPI_proportions, real_proportion, connected_proportion]
            
            print(f'At 10e-3, background {len([i for i in total_attentions if i > 10e-3])/len(total_attentions)}, PPI {len([i for i in PPI_attentions if i > 10e-3])/len(PPI_attentions)}, LCC {len([i for i in real_attentions if i > 10e-3])/len(real_attentions)}')
            print(f'At 10e-2, background {len([i for i in total_attentions if i > 10e-2])/len(total_attentions)}, PPI {len([i for i in PPI_attentions if i > 10e-2])/len(PPI_attentions)}, LCC {len([i for i in real_attentions if i > 10e-2])/len(real_attentions)}')
    
        # Plot CDF
        if graph == True:
            plt.figure()
            sns.ecdfplot(total_attentions, color='red', label=f'Background Attentions')# (n={len(total_attentions)})')
            sns.ecdfplot(real_attentions, color='blue', label=f'LCC Attentions')# (n={len(real_attentions)})')
            sns.ecdfplot(PPI_attentions, color='green', label=f'PPI Attentions')# (n={len(PPI_attentions)})')
            sns.ecdfplot(connected_attentions, color='orange', label=f'GDA Disease \n Attentions')# (n={len(connected_attentions)})')

            plt.xlabel('Attention Weights')
            plt.ylabel('Cumulative Probability')
            plt.legend(loc='lower right')
            plt.xscale('log')
            plt.tight_layout()

            # Save the plot
            if keyword:
                plt.savefig(f'LCCPPIAttentionDist_{keyword}.png')
            else:
                plt.savefig('LCCPPIAttentionDist.png')


# Performs analysis of attention weights
def analyze_hops(attention_dict, CDF = False,
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
                    try:
                        for node in nodes[hop_distance]:
                            node = node[0]
                            if node not in filter_LCC:
                                try:
                                    attention = attention_dict[ref_node][node]
                                    hop_attentions.append(attention)
                                except:
                                    pass
                        attention_hops[hop_distance].extend(hop_attentions)
                    except:
                        continue
                
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

        if CDF != True:
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
            plt.tight_layout()
        else:
            num_diseases = len(diseases)
            fig, axes = plt.subplots(1, num_diseases, figsize=(5 * num_diseases, 6))
            
            # Check if there is only one disease, to avoid indexing issues
            if num_diseases == 1:
                axes = [axes]
            
            for i, disease in enumerate(diseases):
                for hop_distance in hop_range:
                    # Retrieve the attention weights for the current disease and hop distance
                    attention_weights = [i for i in attention_hops[hop_distance] if not isinstance(i, list)]

                    # Calculate CDF
                    sorted_data = np.sort(attention_weights)
                    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

                    # Plot CDF
                    axes[i].plot(sorted_data, yvals, label=f'Hop {hop_distance}')

                axes[i].set_title(f'CDF for {disease}')
                axes[i].set_xlabel('Attention Weight')
                axes[i].set_ylabel('Cumulative Probability')
                axes[i].legend(loc='best')
                axes[i].set_xscale('log')

            plt.tight_layout()
        
        if keyword:
            plt.savefig(f'{keyword}_hopPlot.png')
        else:
            plt.savefig('old_hopPlot.png')
        
    
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
            plt.savefig(f'{keyword}_hopPlot.png')
        else:
            plt.savefig('hopPlot.png')
        
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

    # Analyze interactions
    proportion, interaction_results = analyze_interactions(sorted_weights[:top_weights])
    print(f"Proportion of top {top_weights} gene pairs with interaction: {proportion:.2f}")

    # Create CDF plot
    create_cdf(interaction_results, keyword = keyword)

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
        plt.savefig(f'TopPPI_CDF_{keyword}.png')
    else:
        plt.savefig('TopPPI_CDF.png')

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
def generate_PPI(attention_dict, attention_threshold = None, gene_conversion = Path("/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/gene_name_id_dict.pkl"), save = True,
                  savename = 'Attention_PPI.csv', disparity_filter = True):
        
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

    if disparity_filter:
        # Creates new unidirectional PPI from attention weights
        PPI = nx.DiGraph()
        for src, tgt, weight in sorted_tuples:
            PPI.add_edge(src, tgt, attention=weight)
            
        # Apply disparity filter
        disparity_PPI = nx.DiGraph()
        
        for node in PPI.nodes():
            edges = PPI.edges(node, data=True)
            total_weight = sum([edge_data['attention'] for _, _, edge_data in edges])
            for src, tgt, edge_data in edges:
                weight = edge_data['attention']
                disparity_score = (weight / total_weight)**2
                # Apply some disparity score threshold here, for example:
                if disparity_score > 0.05:  # Threshold example
                    disparity_PPI.add_edge(src, tgt, attention=weight)
        PPI = disparity_PPI
        print(PPI)

    else:
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
def compare_attentions(base, compare, LCC = None, PPI = None, keyword = None, perturb = False,
                       max_bidirectional = True, disease = None, top_proportion = 1000, epsilon = 0.0001,
                       save = True):


    # Obtains the p_value of comparisons
    def t_test_compare(test_background, test_compare, comparison_input = False, KS = False):
            if comparison_input == False:
                flattened_background = [value[1] - value[0]
                        for outer_key, inner_dict in comparison.items()
                        for _, value in comparison[outer_key].items()]
            else:    
                comparison_background, comparison_test = comparison_input
                try:
                    flattened_background = [comparison_test[i] - comparison_background[i] for i in range(len(test_compare))]
                except:
                    flattened_background = [comparison_test[i][2] - comparison_background[i][2] for i in range(len(test_compare))]
            
            try:
                flattened_comparison = [test_compare[i] - test_background[i] for i in range(len(test_compare))]
            except:
                flattened_comparison = [test_compare[i][2] - test_background[i][2] for i in range(len(test_compare))]

            if KS == False:
                t_score, p_value = ttest_ind(flattened_background, flattened_comparison)
                return p_value
            else:
                statistic, p_value = ks_2samp(flattened_background, flattened_comparison)
                return statistic, p_value
    
    def t_test_fold(base, test, KS = False):
        if KS == False:
            try:
                t_score, p_value = ttest_ind([i[2] for i in base], [i[2] for i in test])
            except:
                t_score, p_value = ttest_ind(base, test)
            return p_value
        else:
            try:
                statistic, p_value = ks_2samp([i[2] for i in base], [i[2] for i in test])
            except: 
                statistic, p_value = ks_2samp(base, test)

            return statistic, p_value
    
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
                if compare_attention != attention:  
                    fold_changes[source][target] = compare_attention / (attention + epsilon)
                else:
                    fold_changes[source][target] = 1
    
    print(f'Comparison source genes: {len(comparison.keys())}')
    print(f'Total comparisons: {sum([len(i) for i in comparison.values()])}')

    # Flattens dictionary
    flattened_dict = [(outer_key, inner_key, value)
                      for outer_key, inner_dict in comparison.items()
                      for inner_key, value in inner_dict.items()]
    sorted_tuples = sorted(flattened_dict, key=lambda x: x[2][1] - x[2][0], reverse=True)

    # Finds proportion of largest comparisons with disease genes compared to background
    if disease != None:
        top_values = sorted_tuples[:top_proportion]
        proportion = [1 if i[1] in disease or i[0] in disease else 0 for i in top_values]
        print(f'Proportion of top {top_proportion} with disease genes: {sum(proportion) / len(proportion)}')

        proportion = [1 if i[1] in disease or i[0] in disease else 0 for i in sorted_tuples]
        print(f'Proportion of all comparisons with disease genes: {sum(proportion) / len(proportion)}')
        
        total_proportion, counter = [], 0
        for num, prop in enumerate(proportion):
            counter += prop
            total_proportion.append(counter/(num + 1))

        plt.figure()
        plt.plot(range(len(total_proportion)), total_proportion, color = 'blue')
        plt.xlabel('Ranked Comparisons')
        plt.xscale('log')
        plt.ylabel('Cumulative Ratio of Disease Genes')
        plt.savefig(f'Interactions_{keyword}.png')

    # Makes LCC comparison
    if LCC != None and PPI != None:
        LCC_edges = [(u, v) for u, v in LCC.edges()]
        PPI_edges = [(u, v) for u, v in PPI.edges()]

        background_edges = random.sample([(u, v) for u, v, _ in sorted_tuples if u != v], len(PPI_edges))

        connected_LCC_comparison = []
        for node1 in LCC.nodes():
            for node2 in LCC.nodes():
                if node1 != node2:
                    connected_LCC_comparison.append((node1, node2))

        def obtain_attentions(base, compare, edge_list, omit_edges = False, num_edges = None):
            base_attentions, compare_attentions = [], []
            
            for edge in edge_list:
                attn_found = True

                # Obtains original attention weights
                try:
                    attention_1 = base[edge[0]][edge[1]]
                    attention_2 = base[edge[1]][edge[0]]
                    if max_bidirectional == True:
                        base_attention = max(attention_1, attention_2)
                    else:
                        base_attention = (attention_1 + attention_2) / 2
                except:
                    attn_found = False

                # Obtains comparison attention weights
                try:
                    compare_attention_1 = compare[edge[0]][edge[1]]
                    compare_attention_2 = compare[edge[1]][edge[0]]
                    if max_bidirectional == True:
                        compare_attention = max(compare_attention_1, compare_attention_2)
                    else:
                        compare_attention = (compare_attention_1 + compare_attention_2) / 2
                except:
                        attn_found = False
                
                if attn_found == True:
                    if omit_edges == True:
                        if not isinstance(base_attention, list) and not isinstance(compare_attention, list):
                            base_attentions.append(base_attention)
                            compare_attentions.append(compare_attention)
                    else:
                        if not isinstance(base_attention, list) and not isinstance(compare_attention, list):
                            base_attentions.append((edge[0], edge[1], base_attention))
                            compare_attentions.append((edge[0], edge[1], compare_attention))

            if num_edges != None:
                filtered_base, filtered_compare = [], []
                selected_indices = random.sample(range(len(base_attentions)), num_edges)
                for num, (base, compare) in zip(enumerate(base_attentions, compare_attentions)):
                    if num in selected_indices:
                        filtered_base.append(base)
                        filtered_compare.append(compare)
                return filtered_base, filtered_compare                    
            else:
                return base_attentions, compare_attentions
        
        LCC_base, LCC_compare = obtain_attentions(base, compare, LCC_edges)
        PPI_base, PPI_compare = obtain_attentions(base, compare, PPI_edges)
        Background_base, Background_compare = obtain_attentions(base, compare, background_edges)
        LCC_connected_base, LCC_connected_compare = obtain_attentions(base, compare, connected_LCC_comparison)
        
        # Obtains significance of distributions
        print(f'PPI p-value: {t_test_compare(PPI_base, PPI_compare)}')
        print(f'Connected p-value to LCC: {t_test_compare(LCC_connected_base, LCC_connected_compare, comparison_input = (LCC_connected_base, LCC_connected_compare))}')
        print(f'LCC p-value: {t_test_compare(LCC_base, LCC_compare)}')
        KS_distance = {
            'PPI': t_test_compare(PPI_base, PPI_compare, KS = True)[0],
            'Connected': t_test_compare(LCC_connected_base, LCC_connected_compare, KS = True)[0],
            'LCC': t_test_compare(LCC_base, LCC_compare, KS = True)[0]
        }
        print(KS_distance)

        def obtain_fold_change(base, compare):
            FC = []
            for num, edge_set in enumerate(compare):
                if True: # edge_set[2] != base[num][2]:
                    FC.append((edge_set[0], edge_set[1], edge_set[2]/(base[num][2] + epsilon)))
                else:
                    FC.append((edge_set[0], edge_set[1], 1))
            return FC
        
        LCC_FC = obtain_fold_change(LCC_base, LCC_compare)
        PPI_FC = obtain_fold_change(PPI_base, PPI_compare)
        background_FC = obtain_fold_change(Background_base, Background_compare)
        connected_FC = obtain_fold_change(LCC_connected_base, LCC_connected_compare)

        # Saves data if enabled
        if save == True:
            save_data = {
                "LCC FC": LCC_FC,
                "PPI FC": PPI_FC,
                "Connected FC": connected_FC,
                "Background FC": background_FC
            }
            save_data = pd.DataFrame.from_dict(save_data, orient = 'index')
            save_data = save_data.transpose()
            save_data.to_csv(f'FC_{keyword}.csv')

        print(f"PPI p-value: {t_test_fold(PPI_FC, background_FC)}")
        print(f"Connected p-value: {t_test_fold(connected_FC, background_FC)}")
        print(f"LCC p-value: {t_test_fold(LCC_FC, background_FC)}")

        if perturb == False:

            # Function to calculate CDF
            def calculate_cdf(data):
                sorted_data = np.sort(data)
                yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
                return sorted_data, yvals

            # Function for calculating intersection of two CDF curves
            def find_intersection(LCC_set, connected_set, LCC_data, connected_data):
                try:
                    step = 0.001
                    threshold = None

                    for i in np.arange(0, 1, step):
                        LCC_sample = [j for count, j in enumerate(LCC_data) if LCC_set[count] > i]
                        connected_sample = [j for count, j in enumerate(connected_data) if connected_set[count] > i]
                        if LCC_sample[0] < connected_sample[0]:
                            threshold = i
                            if threshold > 0.6:
                                break

                    print(f'Cut-off at a probability of {threshold}')
                    try:
                        return LCC_sample[0]
                    except:
                        return 1
                except:
                    return 1

            # Finds intersection of connected LCC and LCC edges
            LCC_cdf, LCC_x = calculate_cdf([i[2] for i in LCC_FC])
            connected_cdf, connected_x = calculate_cdf([i[2] for i in connected_FC])
            threshold = find_intersection(LCC_x, connected_x, LCC_cdf, connected_cdf)
            print(f"Intersection threshold: {threshold}")

            # Evaluates results by STRING API
            top_LCC = sorted([LCC_FC[i] for i in range(len(LCC_FC)) if LCC_FC[i][2] >= threshold])
            bottom_LCC = sorted([LCC_FC[i] for i in range(len(LCC_FC)) if LCC_FC[i][2] < 1])

            top_proportion, top_results = analyze_interactions(top_LCC)
            bottom_proportion, bottom_results = analyze_interactions(bottom_LCC)

            print(f"FC threshold: {threshold}")
            print(f"Proportion of thresholded LCC gene pairs with interaction: {top_proportion:.2f}")
            print(f"Proportion of thresholded bottom LCC gene pairs with interaction: {bottom_proportion:.2f}")
            _, proportions = analyze_interactions(top_LCC + bottom_LCC)

            # Create CDF plot
            create_cdf(proportions, keyword = keyword)

        else:
            threshold = None

        # Plot each distribution as CDF
        plt.figure()

        def strip_edges(data):
            return np.array([i[2] for i in data])

        LCC_FC_num = np.array([i[2] for i in LCC_FC])
        PPI_FC_num = np.array([i[2] for i in PPI_FC])
        Connected_FC_num = np.array([i[2] for i in connected_FC])
        background_FC_num = np.array([i[2] for i in background_FC])

        print(f'Average LCC FC: {np.mean(LCC_FC_num)}')
        print(f'Average PPI FC: {np.mean(PPI_FC_num)}')
        print(f'Average Background FC: {np.mean(background_FC_num)}')
        plt.xlim([10e-3, 10e2])
        sns.ecdfplot(LCC_FC_num, color='blue', label='LCC Fold Change')
        sns.ecdfplot(PPI_FC_num, color='green', label='PPI Fold Change')
        sns.ecdfplot(background_FC_num, color='red', label='Background Fold Change')
        sns.ecdfplot(Connected_FC_num, color='orange', label='Disease Gene Connected \n Fold Change')
        if perturb == False:
            plt.xscale('log')
        plt.axvline(x = 1, color = 'purple', linestyle = '--', label = 'No Change')
        plt.xlabel('Pretrained to Finetuned Fold Change')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function of Distributions')
        plt.legend()
        plt.savefig(f'{keyword}_FC_distribution.png')

        # Plots scatter plot of distributions
        plt.figure(figsize = (10, 10))
        plt.xlabel('Pretrained Weight')
        plt.ylabel('Finetuned Weight')

        plt.scatter(strip_edges(Background_base), strip_edges(Background_compare), color = 'red', label = 'Background Attention', alpha = .4, s = 8)
        plt.scatter(strip_edges(PPI_base), strip_edges(PPI_compare), color = 'green', label = 'PPI Attention', alpha = .4, s = 8)
        plt.scatter(strip_edges(LCC_connected_base), strip_edges(LCC_connected_compare), color = 'orange', label = 'LCC Connected Attention', alpha = .4, s = 8)
        plt.scatter(strip_edges(LCC_base), strip_edges(LCC_compare), color = 'blue', label = 'LCC Attention', alpha = .4, s = 8)
        plt.plot([0, max(strip_edges(Background_base))], [0, max(strip_edges(LCC_compare))], color = 'purple', linestyle = '--', label = 'No Change')
        plt.legend(loc = 'upper right')
        plt.savefig(f'{keyword}_change.png')

    # Flattens fold change dictionary
    fold_changes = [(outer_key, inner_key, value)
                      for outer_key, inner_dict in fold_changes.items()
                      for inner_key, value in inner_dict.items()]
    fold_changes = sorted(fold_changes, key=lambda x: x[2], reverse=True)

    return sorted_tuples, comparison, fold_changes, threshold

# Finds intermediaries and direct LCC connections to disease genes and disease LCCs with the highest attention scores
def find_intermediaries(comparison_dict, PPI, LCC, disease_genes, top_pairs = 5, string_map = True,
                        
                        keyword = None):
    
    print('finding intermediaries')
    print('Intermediaries: (Inter, LCC gene, disease gene)')
    print('Direct: (LCC gene, disease gene)')

    def find_one_hop_intermediaries(PPI, LCC, disease_genes):
        intermediaries = {}
        for lcc_gene in LCC:
            if lcc_gene not in PPI:
                continue  # Skip if lcc_gene is not in the graph

            for disease_gene in disease_genes:
                if disease_gene in LCC or lcc_gene == disease_gene:
                    continue

                if disease_gene not in PPI:
                    continue  # Skip if disease_gene is not in the graph

                # Check if there's a one-hop intermediary
                try:
                    lcc_neighbors = set(PPI.neighbors(lcc_gene))
                except KeyError:
                    continue  # Skip if lcc_gene is not in the graph

                try:
                    disease_neighbors = set(PPI.neighbors(disease_gene))
                except KeyError:
                    continue  # Skip if disease_gene is not in the graph

                for inter_gene in lcc_neighbors:
                    if inter_gene in disease_neighbors and inter_gene not in LCC:
                        intermediaries[(inter_gene, lcc_gene, disease_gene)] = 0

        return intermediaries

    def calculate_comparison(attn):
        if isinstance(attn, list) or isinstance(attn, tuple):
            comparison = attn[1] - attn[0]
        else:
            comparison = attn

        return comparison
    
    def calculate_intermediary_scores(intermediaries, comparison_dict):
        for key in list(intermediaries.keys()): 
            inter_gene, lcc_gene, disease_gene = key

            try:
                score1 = max(calculate_comparison(comparison_dict.get(lcc_gene, {}).get(inter_gene, 0)), 
                            calculate_comparison(comparison_dict.get(inter_gene, {}).get(lcc_gene, 0)))
                
                score2 = max(calculate_comparison(comparison_dict.get(disease_gene, {}).get(inter_gene, 0)), 
                            calculate_comparison(comparison_dict.get(inter_gene, {}).get(disease_gene, 0)))
            except:
                continue

            intermediaries[key] = (score1 + score2) / 2

        return intermediaries

    def calculate_direct_scores(LCC, disease_genes, comparison_dict):
        direct_scores = {}
        for lcc_gene in LCC:
            for disease_gene in disease_genes:
                if disease_gene in LCC:
                    continue
                score = max(calculate_comparison(comparison_dict.get(lcc_gene, {}).get(disease_gene, 0)), 
                            calculate_comparison(comparison_dict.get(disease_gene, {}).get(lcc_gene, 0)))
                direct_scores[(lcc_gene, disease_gene)] = score
        return direct_scores

    # Assuming PPI, LCC, disease_genes, and comparison_dict are already defined
    intermediaries = find_one_hop_intermediaries(PPI, LCC, disease_genes)
    intermediary_scores = calculate_intermediary_scores(intermediaries, comparison_dict)
    direct_scores = calculate_direct_scores(LCC, disease_genes, comparison_dict)

    # Combine and rank scores and display the top
    intermediary_scores = sorted(intermediary_scores.items(), key=lambda x: x[1], reverse=True)
    for top_count in range(top_pairs):
        print(f'Intermediary {top_count+1}: {intermediary_scores[top_count]}')

    direct_scores = sorted(direct_scores.items(), key=lambda x: x[1], reverse=True)
    print('====================================================================')
    for top_count in range(top_pairs):
        print(f'Direct {top_count+1}: {direct_scores[top_count]}')

    # If enabled, maps string results of top and bottom tier of attentions
    if string_map == True:

        # Finds cdf for direct interactions
        _, direct_interaction_results = analyze_interactions(direct_scores)

        # Finds cdf for intermediary interactions
        intermediaries = [i for sublist in intermediary_scores for i in ((sublist[0], sublist[1]), (sublist[0], sublist[2]))]
        _, intermediary_interaction_results = analyze_interactions(intermediaries)
        
        plt.figure()
        background_cdf = [i/n for i in range(n)]
        plt.plot(range(n), background_cdf, color='r', linestyle='--', label='Expected Discovery Rate')
        result_names = ['Direct Interactions', 'Intermediary Interactions']
        for count, result in enumerate((direct_interaction_results, intermediary_interaction_results)):
            n = len(result)
            plt.plot(range(n), result, label=f'{result_names[count]} CDF')
        
        plt.xlabel('Total Attention Weights by Rank')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of Gene-Gene Attention Comparison Results')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'String_{keyword}_CDF.png')


    return intermediary_scores, direct_scores

# Compares CDF distances for two difference LCC CDFs
def compare_cdfs(base_LCC, compare_LCC, type = 'KS', LCC_str = None):

    # Preparing data
    base_data = np.array(sorted(base_LCC[LCC_str]))
    compare_data = np.array(sorted(compare_LCC[LCC_str]))

    if type == 'KS':
        # Kolmogorov-Smirnov Statistic
        ks_statistic, ks_pvalue = stats.ks_2samp(base_data, compare_data)
        directed_area = ks_statistic
        if np.mean(base_LCC[LCC_str]) > np.mean(compare_LCC[LCC_str]):
            directionality = 'positive'
        else:
            directionality = 'negative'

    if type == 'EMD':
        # Earth Mover's Distance (Wasserstein distance)
        emd = wasserstein_distance(base_data, compare_data)
        directed_area = emd
        if np.mean(base_LCC[LCC_str]) < np.mean(compare_LCC[LCC_str]):
            directionality = 'positive'
        else:
            directionality = 'negative'
    else:

        # Euclidean Distance between CDFs
        common_x = np.linspace(min(min(base_data), min(compare_data)), max(max(base_data), max(compare_data)), 1000)
        base_cdf = interpolate.interp1d(base_data, np.linspace(0, 1, len(base_data)), bounds_error=False, fill_value="extrapolate")
        compare_cdf = interpolate.interp1d(compare_data, np.linspace(0, 1, len(compare_data)), bounds_error=False, fill_value="extrapolate")
        euclidean_distance = np.sqrt(np.sum((compare_cdf(common_x) - base_cdf(common_x)) ** 2))
        directed_area = euclidean_distance
        if np.mean(base_LCC[LCC_str]) < np.mean(compare_LCC[LCC_str]):
            directionality = 'positive'
        else:
            directionality = 'negative'

    return directed_area, directionality
