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

# Data processing/pre-processing/comparison imports
from collections import defaultdict
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy.stats import ttest_ind
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
def process_batch(attentions):
    attention_matrices, genes_list = attentions[0], attentions[1]
    num_attention_heads = len(attention_matrices[0])
    gene_attentions = {}

    # Iterates through attention matrices 
    for attention_matrix, genes in zip(attention_matrices, genes_list):
        attention_matrix = np.array(attention_matrix)
        if attention_matrix.shape[1] < 2048:
            normalization_factor = attention_matrix.shape[1] / 2048
            attention_matrix *= normalization_factor
        
        non_none_genes = np.array([gene is not None for gene in genes])
        non_none_indices = np.where(non_none_genes)[0]
        attention_matrix = attention_matrix[:, non_none_indices, :][:, :, non_none_indices]

        source_genes = [genes[i] for i in non_none_indices]
        target_genes = [genes[j] for j in non_none_indices]

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
                      samples = 1000, layer_index = 4, gene_conversion = "/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/gene_name_id_dict.pkl", token_dictionary = "/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/token_dictionary.pkl", mean = False, attention_threshold = 0, top_connections = None, normalize = True, num_processes = os.cpu_count(), scale = False, filter_genes = None, save = False, data_sample = None, half = True,
                      organ = None):
    '''
    PARAMETERS
    -----------------------------------------
    samples : int, str
        Number of samples to sample from the dataset. If 'all', the entire dataset will be used.  Defaults to 10,000

    attention_threshold : int
        The number of gene/gene attentions required to be considered a significant attention. Anything below will be thrown out. Defaults to 0

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
        
    organ : str, None
        Whether the dataset should be filtered for a specific organ
        
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
    if organ:
        def organ_filter(example):
            return example["organ_major"] in organ_ids
            
        data = data.filter(organ_filter, num_proc=num_processes)
    
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

        for key, value in dictionary.items():
            for sub_key, sub_value in value.items():
                if mean:
                    gene_attentions[key][sub_key].extend(sub_value)
                else:
                    gene_attentions[key][sub_key] = max(gene_attentions[key][sub_key], max(sub_value))
    gene_attentions = dict(gene_attentions)
    
    # Adds weights to master dictionary
    aggregated_gene_attentions = {}

    # Turns the array of attention weights into a single gene/gene attention based on whether the user selects the maximum or the average
    if mean == True:
        for source_gene, target_genes in tqdm.tqdm(gene_attentions.items(), total = len(gene_attentions.items()), desc = 'Narrowing results'):
            for target_gene, attentions in target_genes.items():
                if len(attentions) > attention_threshold:
                    # Creates new dictionary for source gene if not already existing
                    if source_gene not in aggregated_gene_attentions.keys():
                        aggregated_gene_attentions[source_gene] = {}
    
                    average_attention = sum(attentions) / len(attentions)
                    aggregated_gene_attentions[source_gene][target_gene] = average_attention
    else:
        aggregated_gene_attentions = gene_attentions
    
    # Normalize the entire dictionary to the range [0, 1] if enabled
    if normalize == True:
        min_value = min([min(attentions.values()) for attentions in aggregated_gene_attentions.values() if len(attentions) > 0])
        max_value = max([max(attentions.values()) for attentions in aggregated_gene_attentions.values() if len(attentions) > 0])
        
        minmax = max_value - min_value
        for source_gene, target_genes in aggregated_gene_attentions.items():
            for target_gene in target_genes:

                # Normalize the attention value
                normalized_attention = (aggregated_gene_attentions[source_gene][target_gene] - min_value) / minmax

                # Update the dictionary with the normalized value
                aggregated_gene_attentions[source_gene][target_gene] = normalized_attention
        
    # Scales the entire dictionary to have a mean at 0 if enabled
    if scale == True:
        values = []
        for _, subdict in aggregated_gene_attentions.items():
            values.extend(list(subdict.values()))
        values = [[value] for value in values]

        # Scale the values using StandardScaler
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values)

        # Update aggregated_gene_attentions with scaled values
        index = 0
        for _, subdict in aggregated_gene_attentions.items():
            for target_key in subdict:
                subdict[target_key] = scaled_values[index][0]
                index += 1

    end = time.perf_counter()
    print(f'Total minutes elapsed: {round((end - start)/60, 4)} minutes')
    print(f'{len(aggregated_gene_attentions.keys())} total source genes represented in the final dataset')
    
    if save != False:
        pk.dumps(open(save, 'wb'))
        
    if top_connections and top_connections > 0:
        # Obtains the top N attention pairs from the aggregated gene attention dataset if enabled
        top_attention_pairs = []
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
    
# Maps attention scores to PPI
def map_attention_attention(PPI, gene_attentions, show = True):
    # Make a copy of the original graph
    PPI_copy = nx.Graph(PPI)
 
    # Maps attention weights to PPI and identifies edges without attentions
    edges_to_remove = []
    for u, v in list(PPI_copy.edges()):
        if (u in gene_attentions) and (v in gene_attentions[u]):
            attention = gene_attentions[u][v]
            PPI_copy[u][v]['attention'] = attention
        elif (v in gene_attentions) and (u in gene_attentions[v]):
            attention = gene_attentions[v][u]
            PPI_copy[u][v]['attention'] = attention
        else:
            edges_to_remove.append((u, v))
    
    # Remove edges without valid attention mapping
    for u, v in edges_to_remove:
        PPI_copy.remove_edge(u, v)
    
    edge_attentions = [PPI_copy[u][v]['attention'] for u, v in PPI_copy.edges()]

    # Obtains edge attentions for PPI attentions and finds the mean/median
    average_edge_attention = sum(edge_attentions) / len(edge_attentions)
    
    # Generates edge statistics
    total_attention = [gene_attentions[u][v] for u in gene_attentions for v in gene_attentions[u]]
    average_total_attention = sum(total_attention) / len(total_attention)
    
    
    # Perform t-test
    t_stat, p_value = ttest_ind(edge_attentions, total_attention)
    
    if show == True:
        print(f'Network attention: {average_edge_attention}')
        print(f'Network attention standard deviation: {statistics.median(edge_attentions)}')
        print('')
        print(f'Total attention: {average_total_attention}')
        print(f'Total attention standard deviation: {statistics.median(total_attention)}')
        print(f'T-test p-value: {p_value} \n')
        
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
                    fake_attentions.append(gene_attentions[node1][node2])
                except:
                    try:
                        fake_attentions.append(gene_attentions[node2][node1])                        
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
            real_attentions.append(PPI[u][v]['attention'])
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
def F1_graph_attention(PPI, gene_attentions, LCC = None, show = True, graph = True):

    # Dictionary for converting nodes to numerical tokens
    node_to_index = {node: index for index, node in enumerate(PPI.nodes())}
        
    if LCC != None:
        # Obtains LCC attentions
        LCC_attentions = []
        for u, v in LCC.edges():
            try:
                LCC_attentions.append(LCC[u][v]['attention'])
            except:
                pass
                
        # Obtains PPI attentions
        PPI_attentions = []
        for u, v in PPI.edges():
            try:
                PPI_attentions.append(PPI[u][v]['attention'])
            except:
                pass
        
        # Samples PPI attentions and makes fake attentions that have the same lengths as the LCC attentions
        PPI_attentions = random.sample(PPI_attentions, len(LCC_attentions))
        fake_attentions = generate_fake_attentions(PPI, len(LCC_attentions), gene_attentions, node_to_index)
        
        # Generates 'real' and 'fake' labels for attentions
        fake_labels = [0 for _ in range(len(fake_attentions))]
        real_labels = [1 for _ in range(len(fake_attentions))]
        
        # Generates combined labels
        labels = fake_labels + real_labels
        fake_LCC_attentions = fake_attentions + LCC_attentions
        PPI_LCC_attentions = PPI_attentions + LCC_attentions
        PPI_fake_attentions = fake_attentions + PPI_attentions
        
        # Assuming you have y_pred as the predicted labels (0 or 1) and y_prob as predicted probabilities for the positive class
        fpr1, tpr1, _ = roc_curve(labels, fake_LCC_attentions)
        fpr2, tpr2, _ = roc_curve(labels, PPI_LCC_attentions)
        fpr3, tpr3, _ = roc_curve(labels, PPI_fake_attentions)
        
        # Compute AUC for precision-recall curve
        LCC_auc = auc(fpr2, tpr2)
        fake_auc = auc(fpr1, tpr1)
        PPI_auc = auc(fpr3, tpr3)
        
        if show == True:
            print(f'LCC-Fake AUC score: {fake_auc}')
            print(f'LCC-PPI AUC score: {LCC_auc}')
            print(f'PPI-Fake AUC score: {PPI_auc}')
        
        # Graphs curves if specified
        if graph == True:
            plt.figure(figsize = (12,8))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.xlim([-.05, 1.05])
            plt.ylim([0, 1.05])
            plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--', label = 'Random')
            plt.title('ROC Curve for Finetuned Cardiomyopathy Geneformer Attentions \n (n = 10,000 samples)')
            plt.plot(fpr2, tpr2, color = 'red', label = f'LCC attentions to PPI attentions \n (AUC: {round(LCC_auc, 4)})')
            plt.plot(fpr1, tpr1, color = 'blue', label = f'LCC attentions to Background \n attentions (AUC: {round(fake_auc, 4)})')
            plt.plot(fpr3, tpr3, color = 'green', label = f'PPI attentions to Background \n attentions (AUC: {round(PPI_auc, 4)})')
            plt.legend(loc='upper center', bbox_to_anchor=(0, -0.05), fontsize = 10)
            plt.tight_layout()
            plt.savefig('LCC-PPI-fake_precision.png')
            
        return fake_auc, LCC_auc
        
    else:
        # Obtains real (network) attentions
        real_attentions = []
        for u, v in PPI.edges():
            try:
                real_attentions.append(PPI[u][v]['attention'])
                real_attentions.append(PPI[u][v]['attention'])
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
            plt.figure()
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.xlim([-0.05, 1.0])
            plt.ylim([0, 1.05])
            plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--', label = 'Random')
            plt.title('ROC Curve for Finetuned Cardiomyopathy Geneformer Attentions \n (n = 10,000)')
            plt.plot(fpr, tpr, color = 'red', label = f'PPI attentions to background attentions \n (AUC: {round(pr_auc, 4)})')
            plt.legend(loc='lower left', bbox_to_anchor=(0, 0.2), ncol=1)
            plt.savefig('PPI_precision.png')
        
        return pr_auc
        
def manual_roc_curve(labels, attentions, batch_size=5000):
    # Convert to numpy arrays for efficient computation
    labels = np.array(labels)
    attentions = np.array(attentions)

    # Compute unique thresholds
    thresholds = np.unique(attentions)[::-1]

    # Initialize arrays for FPR and TPR
    fpr_values = [0]
    tpr_values = [0]

    # Compute total positives and negatives
    total_positives = np.sum(labels)
    total_negatives = len(labels) - total_positives

    # Process thresholds in batches
    for start in tqdm.tqdm(range(0, len(thresholds), batch_size), desc='Calculating ROC'):
        end = min(start + batch_size, len(thresholds))
        batch_thresholds = thresholds[start:end]

        # Vectorized computation for the batch
        predictions = attentions[:, None] >= batch_thresholds
        tp_batch = np.sum(predictions & labels[:, None], axis=0)
        fp_batch = np.sum(predictions & ~labels[:, None], axis=0)

        fpr_batch = fp_batch / total_negatives
        tpr_batch = tp_batch / total_positives

        fpr_values.extend(fpr_batch)
        tpr_values.extend(tpr_batch)

    return fpr_values, tpr_values
    
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
                real_attentions.append(attention)
            except KeyError:
                pass
            pbar.update()

    # Obtains all edges
    total_attentions = []
    for val in attention_dict.values():
        total_attentions.extend(list(val.values()))
        
    # Checks all edges by threshold
    real_attentions = [i for i in real_attentions if i > min_threshold]
    total_attentions = [i for i in total_attentions if i > min_threshold]
    
    return real_attentions, total_attentions
        
# Plots attention distributions for background or a PPI/disease if specified
def plot_distributions(attention_dict, disease = None, graph = True, probability = True):    
        
    if disease == None:
      
        # Instantiates PPI 
        PPI = instantiate_ppi()
        PPI, _, _ = map_attention_attention(PPI, attention_dict)
        
        # Parallel processing to extract attentions
        real_attentions, total_attentions = process_edges(PPI, attention_dict)
        real_attentions = [i for i in real_attentions if isinstance(i, float)]
        total_attentions = [i for i in total_attentions if isinstance(i, float)]
        
        # Convert to probability if enabled
        if probability == True:
            total_attentions = np.array(total_attentions) / np.sum(total_attentions) 
            real_attentions = np.array(real_attentions) / np.sum(real_attentions) 
            total_attentions = np.clip(total_attentions, 0, 1)
            real_attentions = np.clip(real_attentions, 0, 1)
            label = 'Probability'
        else:
            label = 'Frequency'
      
        # Plot normalized KDEs
        if graph == True:
            plt.figure(figsize = (12, 8))
            sns.kdeplot(total_attentions, color='red', label=f'Background Attentions (n={len(total_attentions)})')#, bw_adjust=0.5)
            sns.kdeplot(real_attentions, color='green', label=f'PPI Attentions (n={len(real_attentions)})')#, bw_adjust=0.5)
            plt.xlabel(f'Attention Weights')
            plt.ylabel(label)
            #plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
            #plt.xscale('log')
            #plt.yscale('log')
            plt.legend(loc='lower left', bbox_to_anchor=(0, 0.2), ncol=1)
            plt.title('Attention Weight Distribution for PPI Network \n vs Background Attention Weights')
            
            # Save the plot
            plt.savefig('PPIAttentionDist.png')
                    
    else:   
    
        # Instantiates PPI 
        PPI = instantiate_ppi()
        PPI, _, _ = map_attention_attention(PPI, attention_dict)
        
        # Instantiates disease LCC
        disease_genes = isolate_disease_genes(disease)
        LCC_gene_list = LCC_genes(PPI, disease_genes)
        disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
        LCC, _, _ = map_attention_attention(disease_LCC, attention_dict)
        
        # Parallel processing to extract attentions
        real_attentions, total_attentions = process_edges(PPI = LCC, attention_dict = attention_dict, min_threshold = 0.00001)
        PPI_attentions, _ = process_edges(PPI = PPI, attention_dict = attention_dict, min_threshold = 0.00001)
        
        real_attentions = [i for i in real_attentions if isinstance(i, float)]
        total_attentions = [i for i in total_attentions if isinstance(i, float)]
        PPI_attentions = [i for i in PPI_attentions if isinstance(i, float)]
        
        # Calculate percentages
        if probability == True:
            total_attentions = np.array(total_attentions) / np.sum(total_attentions) 
            real_attentions = np.array(real_attentions) / np.sum(real_attentions) 
            PPI_attentions = np.array(PPI_attentions) / np.sum(PPI_attentions) 
            total_attentions = np.clip(total_attentions, 0, 1)
            real_attentions = np.clip(real_attentions, 0, 1)
            PPI_attentions = np.clip(PPI_attentions, 0, 1)
            label = 'Probability'
        else:
            label = 'Frequency'
        
        if graph == True:
            # Plot KDEs
            plt.figure(figsize = (12, 8))
            sns.kdeplot(total_attentions, color='red', label=f'Background Attentions (n={len(total_attentions)})')#, bw_adjust = 0.5)
            sns.kdeplot(real_attentions, color='green', label=f'LCC Attentions (n={len(real_attentions)})')#, bw_adjust = 0.5)
            sns.kdeplot(PPI_attentions, color = 'blue', label = f'PPI Attentions (n={len(PPI_attentions)})')#, bw_adjust = 0.5)
            plt.xlabel('Attention Weights')
            plt.ylabel(label)
            #plt.xscale('log')
            #plt.yscale('log')
            plt.legend(loc='lower left')
            plt.title('Attention Weight Distribution for PPI Network \n vs Background Attention Weights')
            plt.savefig('LCCPPIAttentionDist.png')


# Performs analysis of attention weights
def analyze_hops(attention_dict, disease = 'Cardiomyopathy Dilated', top_weights = 10):
    PPI = instantiate_ppi()
    PPI, _, _ = map_attention_attention(PPI, attention_dict)
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
    attention_errors = []
    
    for hop_distance in hop_range:
        hop_edges_attention = [i for i in attention_hops[hop_distance] if not isinstance(i, list)]
        
        #first_quartile = np.percentile(hop_edges_attention, 25)
        #median = np.percentile(hop_edges_attention, 50)
        #third_quartile = np.percentile(hop_edges_attention, 75)
         
        #average_attention.append(median)
        #attention_errors.append([median - first_quartile, third_quartile - median])
        mean = np.mean(hop_edges_attention)
        stdev = np.std(hop_edges_attention)/2
        average_attentions.append(mean)
        assymetric_bound = [mean - stdev/2 if (mean - stdev/2) > 0 else mean][0]
        attention_errors.append([assymetric_bound, stdev/2])
        
    # Calculates Correlation
    def calculate_mean(values):
        return sum(values) / len(values)
    
    def calculate_correlation_coefficient(x, y_flat, dist = False):
        # Calculates for the entire set of points, not just the means/medians if dist set to True
        if dist == True:
            x_expanded = []
            for x_val in x:
                x_expanded.extend([xval for _ in range(len(y_flat[xval]))])
            x = x_expanded
            y_flat = [item for sublist in y_flat for item in sublist]
            
        # Calculate means
        mean_x = sum(x) / len(x)
        mean_y = sum(y_flat) / len(y_flat)
    
        # Calculate numerator and denominators
        numerator = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y_flat))
        denominator_x = sum((x_i - mean_x) ** 2 for x_i in x) ** 0.5
        denominator_y = sum((y_i - mean_y) ** 2 for y_i in y_flat) ** 0.5
    
        # Calculate correlation coefficient
        correlation_coefficient = numerator / (denominator_x * denominator_y) if denominator_x * denominator_y != 0 else 0
    
        return correlation_coefficient

    x = range(1, 6)
    print(f'Correlation: {calculate_correlation_coefficient(x, average_attentions)}')
    
    # Plotting
    plt.figure()
    plt.errorbar(range(1, 6), average_attentions, yerr=np.array(attention_errors).T, marker='o', linestyle='-', color='b')
    plt.xlabel('Hop Distance')
    plt.ylabel('Avg Attention Weight')
    plt.title(f'Average Attention Weight by Hop Distance \n for {disease} Largest Connected Component')
    plt.tight_layout()
    plt.savefig('hopPlot.png')
    
    sorted_weights = sorted(
      [(key, subkey, value) for key, subdict in attention_dict.items() for subkey, value in subdict.items()],
      key=lambda x: x[2], reverse = True
    )

    for rank, (key, gene, value) in enumerate(sorted_weights[:top_weights], start=1):
        
        print(f"Top {rank} largest connected weight from {key} to {gene}: {value}")
        print(f'Hop distance: {hop_distance}')
        print('-' * 30)

# Checks the top attentions of a mapped PPI 
def check_top_attentions(attention_dict, PPI, top_number = None, make_range = True):

    flattened_dict = [(outer_key, inner_key, value)
                      for outer_key, inner_dict in attention_dict.items()
                      for inner_key, value in inner_dict.items()]
    sorted_tuples = sorted(flattened_dict, key=lambda x: x[2], reverse=True)
    
    if top_number == None:
        top_number = int(len(sorted_tuples))
        
    # Calculates overall ratio of total nodes within PPI 
    background_attention_ratio = len(PPI.edges())/len(flattened_dict)
    print(f'Proportion of total nodes within PPI: {background_attention_ratio}')
    
    if make_range == False:
    
        # Sort the list of tuples by the third element (the values)
        PPI_nodes = 0
        for top_value in top_n_tuples:
            try:
                attention = PPI[top_value[0]][top_value[1]]['attention']
                PPI_nodes += 1
            except:
                pass
        PPI_ratio = PPI_nodes/top_number
        print(f'Proportion of top {top_number} nodes within PPI: {PPI_ratio}')
                
    else:
        num_PPI = []
        cdf = []
        
        # Identifies total number of mappable PPI edges
        PPI_len = len(PPI.edges())
        
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
        #plt.yscale('log')
        #plt.xscale('log')
        plt.plot([i for i in range(1, top_number + 1)], cdf, linestyle='-', label = 'CDF')
        plt.plot([i for i in range(1, top_number + 1)], background_ratio, label = 'Background Expected', color = 'red')
        plt.legend()
        plt.tight_layout()
        plt.savefig('TopPPI_CDF.png')
        
        print(f'Proportion of top {top_number} attention weights within the PPI: {PPI_ratio}')

    return PPI_ratio
    
# Merges a list of dictionaries together
def merge_dictionaries(dictionaries, mean = False):

    if mean == True:
        merged_dict =  defaultdict(lambda: defaultdict(list))
    else:
        merged_dict =  defaultdict(lambda: defaultdict(int))
        
    for attention_dict in tqdm.tqdm(dictionaries, total = len(dictionaries), desc = 'Combining dictionaries'):
        try:
            attention_dict = pk.load(open(attention_dict, 'rb'))
        except:
            #print(traceback.format_exc())
            pass
            
        if mean == True:
            for key in attention_dict.keys():
                subdict = attention_dict[key]
                for sub_key in subdict.keys():
                    merged_dict[key][sub_key].append(subdict[sub_key])
        else:
            for key in attention_dict.keys():
                subdict = attention_dict[key]
                for sub_key in subdict.keys():
                    merged_dict[key][sub_key] = max(subdict[sub_key], merged_dict[key][sub_key])
    
    if mean == True:
        for key in merged_dict.keys():
            sub_dict = merged_dict[key]
            for sub_key in sub_dict.keys():
                sub_dict[sub_key] = sum(sub_dict[sub_key])/len(sub_dict[sub_key])
                
    return dict(merged_dict)
    
# Creates a new PPI from the top Geneformer attention weights
def generate_PPI(attention_dict, attention_threshold = 0.005, gene_conversion = Path("/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/gene_name_id_dict.pkl"), save = False,
                  savename = 'Attention_PPI.csv'):
        
    # Creates conversion from tokens back to genes
    token_dict = TranscriptomeTokenizer().gene_token_dict
    token_dict = {value: key for key, value in token_dict.items()}
    conversion = {value: key for key, value in pk.load(open(gene_conversion, 'rb')).items()}
    conversion = {key: conversion[value] for key, value in token_dict.items() if value in conversion}
    
    # Obtains flattened gene-gene attention collection
    flattened_dict = [(outer_key, inner_key, value)
                      for outer_key, inner_dict in attention_dict.items()
                      for inner_key, value in inner_dict.items()]
    sorted_tuples = sorted(flattened_dict, key=lambda x: x[2], reverse=True)
    
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
    
    
    
    