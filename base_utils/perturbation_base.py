# Base imports
import tqdm
import sys
import random
import pickle as pk
from pathlib import Path
import os
import subprocess
from collections import Counter, defaultdict
from multiprocessing import Pool
import multiprocessing 

# Data processing/pre-processing/comparison imports
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
import seaborn as sns; sns.set()
from sklearn.metrics import auc as precision_auc
import pyarrow as pa
from matplotlib import pyplot as plt
from scipy.stats import ranksums
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Torch/Geneformer imports
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification, EmbExtractor
import torch
import torch.nn.functional as F
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.preprocessing import label_binarize
from datasets import Dataset as HF_Dataset

# Network base imports
import networkx as nx
import fastcluster
from .networks import *

'''
PRIMARY CLASSES / FUNCTIONS

plot_similarity_heatmap = Plots heatmap between different classes/labels
'''
      
# Plots heatmap between different classes/labels
def plot_similarity_heatmap(similarities):
    classes = list(similarities.keys())
    classlen = len(classes)
    arr = np.zeros((classlen, classlen))
    for i, c in enumerate(classes):
        for j, cc in enumerate(classes):
            if cc == c:
                val = 1.0
            else:
                val = similarities[c][cc]
            arr[i][j] = val
        
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap='inferno', vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(classlen), classes, rotation = 45, ha = 'right')
    plt.yticks(np.arange(classlen), classes)
    plt.title("Similarity Heatmap")
    plt.savefig("similarity_heatmap.png")
    
# Function to count token occurrences in a sample
def count_tokens(data):
    return Counter(token for token in data if token in tokens_set)
  
# Function to count token occurrences in a sample    
def count_tokens(sample):
    
    token_counts = {token:0 for token in tokens}
    for token in tokens:
        if token in sample['input_ids']:
            token_counts[token] = 1
            
    return token_counts
    
# Randomly samples down to a set number of unique samples to ensure equality across testing sets 
def equalize_genes(data, tokens_set, num_cells, gene_to_token):    
    # Shuffles data
    data.shuffle()
    
    global tokens
    tokens = list(set([gene_to_token[i] for i in tokens_set]))
    
    # Use multiprocessing to count token occurrences in parallel
    pool = Pool(processes=multiprocessing.cpu_count())
    count_dictionary = {token: [] for token in tokens}
    results = list(tqdm.tqdm(pool.imap(count_tokens, data), total=len(data), desc = 'Pre-processing samples'))
    
    # Populates a dictionary with token counts across dataset
    for count, result in tqdm.tqdm(enumerate(results), total=len(results)):   
        for token in tokens:
            if result[token] == 1:
                count_dictionary[token].append(count)
                
    # Identifies min frequency
    min_frequency = min([len(count_dictionary[key]) for key in list(count_dictionary.keys())])
    
    # Sets minimum frequency to not exceed the number of cells
    if min_frequency > num_cells / (len(tokens_set) + 1):
        min_frequency = num_cells / (len(tokens_set) + 1)
        print(f'Minimum frequency: {min_frequency}')
    else:
        print(f'Minimum frequency of {min_frequency} below requested cell values!')
  
    # Filtering samples containing each token at least min_frequency times (and tries to ensure they are present an equal amount of times)
    filtered_data = []
    for token in tqdm.tqdm(tokens, total = len(tokens), desc = 'Filtering samples'):
        token_counter = 0
        
        # Prepopulates list with samples that already have the token
        if len(filtered_data) > 0:
            for sample in filtered_data:
                if token in sample['input_ids']:
                    token_counter += 1
        
        # Populates tokens up to minimum frequency
        for sample in data:
            if token_counter < min_frequency:
                if token in sample['input_ids']:
                    token_counter += 1
                    filtered_data.append(sample)
            else:
                break
    
    return filtered_data
    
# Primary function for aggregating cosine similarities from embeddings
def aggregate_similarities(true_labels, embs, samples_per_label, label):
    count = 0
    emb_dict = {label:[] for label in list(set(true_labels))}
  
    for num, emb in embs.iterrows():
        key = emb[label]
        if key == 'nf':
             count += 1
        selection = emb.iloc[:256]
        emb = torch.Tensor(selection)
        emb_dict[key].append(emb)

    labels = embs[label].to_list()
    
    cosine_similarities = {label:[] for label in list(set(labels)) if label != 'nf'}
    control_embeddings = [np.expand_dims(emb.numpy(), axis=0) for emb in emb_dict['nf']]

    # Iterates through each label and creates similarities
    for key, embeddings in tqdm.tqdm(emb_dict.items(), total=len(emb_dict.items()), desc='Creating similarities...'):
        if key != 'nf':
            other_embeddings = [np.expand_dims(embedding.numpy(), axis=0) for embedding in embeddings]
            
            similarities = []
            if len(other_embeddings) > len(control_embeddings):
                for i in range(samples_per_label):
                    for emb_num, embedding in enumerate(control_embeddings):
                    
                        # Calculate cosine similarities
                        similar = cosine_similarity(other_embeddings[emb_num + (count * i)], embedding)[0]
                        similarities.append(float(similar))
                    cosine_similarities[key].append(sum(similarities)/len(similarities))
                    #cosine_similarities[key].extend(similarities)
            else:
                for emb_num, embedding in enumerate(control_embeddings):
                    # Calculate cosine similarities
                    similar = cosine_similarity(other_embeddings[emb_num], embedding)[0]
                    similarities.append(float(similar))
                
                # Aggregate the similarities for this key
                cosine_similarities[key] = [sum(similarities) / len(similarities)]
                #cosine_similarities[key] = similarities
            
    return cosine_similarities
            
           
        
            