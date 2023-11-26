import os 
import sys
import time
import tqdm
import pickle as pk
from pathlib import Path
from datasets import Dataset
import polars as pl
from sklearn.utils import shuffle
import seaborn as sns
from geneformer import TranscriptomeTokenizer
import numpy as np
from scipy.stats import nbinom
import itertools
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from scipy.optimize import minimize
import statistics
from Cell_classifier import *
from multiprocessing.pool import ThreadPool as Pool
from torch.nn import Linear, Sequential, ReLU
import torch
from torch.utils.data import Dataset as tDataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as plit
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc as skauc
import copy
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_data(data, polars = False, round_data = False):
    '''
    Hormalizes data for read counts across different samples
    
    Inputs
    --------
    data : polars df, pandas df
        Input dataframe to normalize
    
    polars : bool
        Whether a polars dataframe (True) or pandas dataframe (False) should be used

    round_data : bool
        Whether the output values should be converted to integers or kept as floats

    Outputs
    --------
    data : polars df, pandas df
        Output normalized dataframe
    '''

    data_columns = data.columns
    read_counts = []
    
    if polars == False:
        for _, row in data.iterrows():
            read_counts.append(sum(row[:-1]))
    else:
        for _, row in enumerate(data.iter_rows()):
            read_counts.append(sum(row[:-1]))

    read_mean = sum(read_counts) / len(read_counts)

    
    read_multipliers = [read / read_mean for read in read_counts]
    normal_data = []
    
    if polars == False:
        for num, (_, row) in enumerate(data.iterrows()):
            multiplier = read_multipliers[num]
            normal = [i * multiplier for i in list(row[:-1])] + [row[-1]]
            
            if round_data:
                normal = np.rint(np.array(normal)).astype(int)
            
            normal_data.append(normal)
        data = pd.DataFrame(normal_data, columns=data_columns)
    else:
        for num, row in enumerate(data.iter_rows()):
            multiplier = read_multipliers[num]
            normal = [i * multiplier for i in list(row[:-1])] + [row[-1]]
            
            if round_data:
                normal = np.rint(np.array(normal)).astype(int)
            
            normal_data.append(normal)
        data = pl.DataFrame(normal_data, schema=data_columns,)
            
    return data
    
def add_noise(data, label = 'RA', noise = 0.1, noise_type = None, polars = True):
    '''
    Adds several different forms of noise to a dataframe of data. Ensures the end data is NOT negative (for RNAseq processing)
    
    Input
    --------------
    data: polars df, pandas df
        A data containing the data that should have noise injected
        
    label: str, int
        The label column name containing classes
        
    noise: float, int
        The proportion of noise added (by the proportion of the dataset mean/inddividual sample noise) to the data. Default .1
        
    noise_type: string
        The type of noise. Can either be mean (mean of the dataset noise) or uniform (across a uniform distribution of the maximum/minimum of the column). Defaults to uniform
        
    polars: bool
        Whether polars or pandas should be used. Defaults to True
    '''
    if isinstance(data, np.ndarray):
        X = data
    else:
        if polars == True:
            genes_to_keep = data.columns
            genes_to_keep.remove(label)
            X = data.drop(label).to_numpy()
            label = data.select([label])
        else:
            genes_to_keep = data.columns
            genes_to_keep.remove(label)
            X = data.drop(columns = [label]).to_numpy()
            label = data[label]

    data_length = X.shape[0]
    data_columns = X.shape[1]
    new_X = np.zeros((data_length, data_columns))
    if noise_type == None or noise_type.upper() == 'UNIFORM':
        for col in range(data_columns):
            column = X[:, col]
            max_val, min_val = max(column), min(column)
            for row in range(data_length):
                value = X[row, col]
                value = value + value * np.random.uniform(-1, 1) * noise
                if value < 0:
                    new_X[row, col] = 0
                else:
                    new_X[row, col] = np.rint(value)
                    
    elif noise_type.upper() == 'MEAN':
        mean = 0
        for col in range(X.shape[1]):
            column = X[:, col]
            mean += sum(column) / len(column)
        mean /= data_columns
        new_X = np.zeros((data_length, data_columns))
        for col in range(data_columns):
            column = X[:, col]
            for row in range(data_length):
                value = X[row, col]
                value = value + mean * np.random.uniform(-1, 1) * noise
                if value < 0:
                    new_X[row, col] = 0
                else:
                    new_X[row, col] = np.rint(value)

    if not isinstance(data, np.ndarray):
    
        # Adds labels  back to data
        if polars == True:
            X = pl.DataFrame(new_X.astype(np.int64), schema = genes_to_keep)
            data = pl.concat([X, label], how = 'horizontal')
        else:
            X = pd.DataFrame(new_X, columns = genes_to_keep)
            data = pd.concat([X, label], axis = 1)
    else:
        data = X
    
    return data


def augment_data(data, num_samples, label = 'RA', selected_label = 0, epochs = 20,
                  augment_type = 'nbinom', polars = False, normalize = False, noise = 0):
    '''
    Augments new data samples for RNA-seq analysis

    Inputs
    -------------------------
    data : polars df, pandas df, str
        A dataframe containing the RNA-seq data, or a path to a .csv file of the dataframe

    num_samples : int
        The additional numbers of samples that should be augmented from the data

    label : str
        The label of the df column containing the classification label

    selected_label : str, int
        The selected label that should be amplified. 'all' will amplify all labels to the selected amount

    noise : float, int
        The amount of noise that should be applied to the data. A randomly selected value from the minimum and the maximum
        of the select gene column will be chosen them multiplied by the "noise" variable from -noise to noise, which will 
        then be added to the data.

    augment_type : str
        The type of augmentation that should be performed. A string containing 'nbinom' will sample from negative binomial  
        where applicable, otherwise sampling from a normal distribution, or for genes with no expression in the sample, will 
        just output zeroes. A string containing 'gan' will sample from a generative adversarial network to generate samples.
        Defaults to nbinom

    polars : bool
        Whether a polars (True) or pandas dataframe (False) should be used as the input dataframe. Defaults to False

    normalize_data : bool
        Whether the data should be normalized based on read counts. Defaults to False

    epochs : int
        If a GAN is generated, how many epochs should the model be run for? Defaults to 20

    Outputs
    ---------------
    data : polars df, pandas df
        Output dataframe containing augmented data and old data

    '''
    if polars == True:  
        try:
            data = pl.read_csv(data)
        except:
            pass

        if selected_label != 'all':
            selected_data = data.filter(pl.col(label) == selected_label)
            selected_data = selected_data.drop(label)
            length = len(selected_data)
        else:
            labels = data[label].to_list()
            label_counts = {}
            for l in labels:
                try:
                    label_counts[l] += 1
                except: 
                    label_counts[l] = 1
    else:
        try:
            data = pd.read_csv(data)
        except:
            pass

        if selected_label != 'all':
            selected_data = data[data[label] == selected_label].drop(label)
            length = len(selected_data)
        else:
            labels = list(data[label])
            label_counts = {}
            for l in labels:
                try:
                    label_counts[l] += 1
                except: 
                    label_counts[l] = 1
        
    data_columns = data.columns
    start = time.perf_counter()

    # If enabled, performs normalization of the data
    if normalize == True:
        data = normalize_data(data, round_data = False, polars = polars)
            
    if augment_type.upper() == 'NBINOM':
        def augment_genes(column, data = data, samples = num_samples):
            if column == label:
                return None
            exp = data[column]
            mean = exp.sum() / length
            if mean > 0:
                var = statistics.variance(exp)
                try:
                    k = (mean ** 2) / (var - mean)
                    
                    if k <= 0:
                        std = statistics.stdev(exp)
                        #generated_values = [0 for _ in range(samples)]
                        generated_values = np.rint([i if i > 0 else 0 for i in np.random.normal(mean, std, samples)]).astype(np.int64)
                        #generated_values = np.rint([i if i >= 0 else 0 for i in np.random.uniform(min(exp), max(exp), samples)]).astype(np.int64)
                    else:
                        p = k / (k + mean)
                        #generated_values = np.rint([i if i >= 0 else 0 for i in np.random.uniform(min(exp), max(exp), samples)]).astype(np.int64)
                        #generated_values = np.rint([i if i >= 0 else 0 for i in np.random.normal(mean, std, samples)]).astype(np.int64)
                        generated_values = np.rint([i if i > 0 else 0 for i in nbinom.rvs(n=k, p=p, size=samples)]).astype(np.int64)
                        
                except ZeroDivisionError:
                    std = statistics.stdev(exp)
                    #generated_values = [0 for _ in range(samples)]
                    generated_values = np.rint([i if i > 0 else 0 for i in np.random.normal(mean, std, samples)]).astype(np.int64)
                    #generated_values = np.rint([i if i >= 0 else 0 for i in np.random.uniform(min(exp), max(exp), samples)]).astype(np.int64)
                    
                generated_values = list(generated_values)
            else:
                generated_values = [0 for _ in range(samples)]

            return generated_values

        if selected_label != 'all':
            augmented_data = {}
            for column in tqdm.tqdm(data_columns, total = len(data_columns), 
                                    desc = f'Augmenting {num_samples} samples for label {selected_label}'):
                augmented = augment_genes(column)
                if augmented != None:
                    augmented_data[column] = augmented

            if polars == True:
                augmented_labels = pl.DataFrame({label:[selected_label for _ in range(num_samples)]})
            else:
                augmented_labels = pd.DataFrame.from_dict({label:[selected_label for _ in range(num_samples)]})

        else:
            label_dict = {}
            unlabeled_columns = data_columns
            unlabeled_columns.remove(label)
            augmented_data = {i:[] for i in unlabeled_columns}
            sample_length = int(num_samples / len(list(set(label_counts.keys()))))
            for chosen_label in list(set(label_counts.keys())):
                remaining_samples =  sample_length 
                label_counts[chosen_label] = remaining_samples
                label_dict[chosen_label] = remaining_samples

                if polars == True:
                    selected_data = data.filter(pl.col(label) == chosen_label).drop(label)
                else: 
                    selected_data = data[data[label] == chosen_label]
                    selected_data = selected_data.drop(columns = label)
                length = len(selected_data)
                for column in tqdm.tqdm(unlabeled_columns, total = len(unlabeled_columns),
                                         desc = f'Augmenting {remaining_samples} samples for label {chosen_label}'):
                    augmented = augment_genes(column, data = selected_data, samples = remaining_samples)
                    if augmented != None:
                        augmented_data[column] += augmented
        
            augmented_labels = [key for key, value in label_counts.items() for _ in range(value)]
            
            if polars == True:
                augmented_labels = pl.DataFrame({label:augmented_labels})
            else:
                augmented_labels = pd.DataFrame.from_dict({label:augmented_labels})
        
        augmented_data = {key:np.array(value).astype(np.int64) for key, value in augmented_data.items()}
       
        if polars == True:
            data = pl.concat([pl.DataFrame(augmented_data), augmented_labels], how = "horizontal")

        else:
            data = pd.concat([
                pd.DataFrame(augmented_data), 
                augmented_labels
            ], axis=1, ignore_index=False).astype('int64')
     
    elif augment_type.upper() == 'GAN':
        try:
            from torch import nn
            import torch
            import torch.optim as optim
            from sklearn.preprocessing import StandardScaler, normalize
        except:
            print('GAN functionality requires torch and sklearn! Install them with pip install torch and pip install sklearn')
            sys.exit()

        # Define the Generator and Discriminator classes
        class Generator(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(Generator, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
                )
        
            def forward(self, x):
                return self.model(x)
        
        class Discriminator(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(Discriminator, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_size, hidden_size * 10),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 10, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
                )
        
            def forward(self, x):
                return self.model(x)
            
        if selected_label != 'all':
            X = selected_data.to_numpy()
            input_size = X.shape[1] 
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        else:
            selected_X = {}
            scalers = {}
            sample_numbers = {}
            labels = list(set(data[label]))
            sample_subset = int(num_samples/len(labels))

            if polars != True:
                
                for chosen_label in labels:
                    sample_numbers[chosen_label] = sample_subset - list(data[label]).count(chosen_label)
                    selected_X[chosen_label] =  data[data[label] == chosen_label].drop(label).to_numpy()
                    input_size = selected_X[chosen_label].shape[1] 
                    scaler = StandardScaler()
                    selected_X[chosen_label] = scaler.fit_transform(selected_X[chosen_label])
                    scalers[chosen_label] = scaler
                    
            else:
                for chosen_label in labels:
 
                    sample_numbers[chosen_label] = sample_subset - list(data[label]).count(chosen_label)
                    selected_data = data.filter(pl.col(label) == chosen_label)
                    selected_X[chosen_label] = selected_data.select([c for c in data.columns if c != label]).to_numpy()
                    input_size = selected_X[chosen_label].shape[1] 
                    scaler = StandardScaler()
                    selected_X[chosen_label] = scaler.fit_transform(selected_X[chosen_label])
                    scalers[chosen_label] = scaler

        hidden_size, output_size, batch_size = 512, 1, 64        
        G_lr, D_lr = 0.0002, 0.001               # Learning rate for the discriminator
        num_epochs = epochs          # Number of training epochs
        clip_value = 0.001           # Clip parameter for weight clipping (WGAN-specific)
        
        # Initialize networks
        generator = Generator(input_size, hidden_size, input_size)  # Output size matches input size
        discriminator = Discriminator(input_size, hidden_size, output_size)
        
        # Loss function and optimizers
        optimizer_G = optim.RMSprop(generator.parameters(), lr=G_lr, weight_decay = .001)
        optimizer_D = optim.RMSprop(discriminator.parameters(), lr=D_lr, weight_decay = .001)
        
        def run_model(X, input_scaler, samples = num_samples, chosen_label = selected_label):
            for epoch in tqdm.tqdm(range(num_epochs), total = num_epochs, desc = f'Training GAN to generate {samples} samples of label {chosen_label}'):
                for i in range(0, X.shape[0], batch_size):
                    # Sample real data
                    real_data = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
                    
                    # Sample noise for generator
                    gen_noise = torch.randn(batch_size, input_size)
                    
                    # Generate fake data from noise
                    fake_data = generator(gen_noise)
                    
                    # Discriminator forward and backward pass
                    optimizer_D.zero_grad()
                    
                    # Compute the discriminator scores for real and fake data
                    real_scores = discriminator(real_data)
                    fake_scores = discriminator(fake_data)
                    
                    # Compute the Wasserstein loss
                    loss_D = -torch.mean(real_scores) + torch.mean(fake_scores)
                    loss_D.backward()
                    
                    fake_data, real_scores, fake_scores = fake_data.detach(), real_scores.detach(), fake_scores.detach()
                
                    # Weight clipping (WGAN-specific)
                    for param in discriminator.parameters():
                        param.data.clamp_(-clip_value, clip_value)
                    
                    # Update discriminator
                    optimizer_D.step()
                    
                    # Generator forward and backward pass
                    optimizer_G.zero_grad()
                
                    # Compute the discriminator scores for fake data (detach to avoid backpropagation)
                    fake_scores = discriminator(fake_data)
                
                    # Compute the generator loss
                    loss_G = -torch.mean(fake_scores)
                
                    loss_G.backward()
                
                    # Update generator
                    optimizer_G.step()
                
                # Print progress
                if epoch == num_epochs - 1:
                    print(f"Wasserstein Loss (D): {loss_D.item():.4f}, Wasserstein Loss (G): {loss_G.item():.4f}")
            
            # Generate fake samples
            generated_noise = torch.randn(samples, input_size)
            faked_samples = input_scaler.inverse_transform(generator(generated_noise).detach().numpy())
        
            labels = np.array([chosen_label for _ in range(samples)]).reshape(-1, 1)
    
            fake_samples = np.hstack((np.rint(faked_samples), labels))

            return fake_samples
        
        if selected_label != 'all':
            fake_samples = run_model(X, input_scaler = scaler)
        else:
            for chosen_label in labels:
                scaler = scalers[chosen_label]
                X = selected_X[chosen_label]
                faked = run_model(X, input_scaler = scaler, chosen_label = chosen_label, samples = sample_numbers[chosen_label] )
        
                try:
                    fake_samples = np.vstack((fake_samples, faked))
                except:
                    fake_samples = faked

        if polars == True:
            fake_samples = pl.DataFrame(fake_samples, schema = data_columns)
            data_numpy = np.rint(data.to_numpy())
            data = pl.DataFrame(data_numpy, schema = data_columns)
            data = pl.concat((data, fake_samples), how = "vertical")
        else:
            fake_samples = pd.DataFrame(fake_samples, columns = data_columns)
            data = data.astype('int64')
            data = pd.concat([data, fake_samples], axis = 0)
            
    if noise > 0:
        data = add_noise(data, noise = noise)
        
    end = time.perf_counter()
    print(f'Data augmented to {num_samples} samples in {round(end - start, 4)} seconds')

    return data
    
# Define your custom dataset class
# .save_to_disk() method used to dataset
class CellData(Dataset):
    def __init__(self, test, train, data_split = None, label = "cell_type", ID = 'ENSEMBL'):
        self.label = label
        
        # Uses a train/test split if provided, otherwise creates the test split if only a test value is provided
        try:
            if train:
                pass
            data, ranked_genes = self.convert(test, label, ID, train_test_labels = None)
            
        except:
            train_test_labels = [1 for _ in range(len(train))] + [0 for _ in range(len(test))]
            data = pl.concat((train, test), how = 'vertical')
            data, ranked_genes = self.convert(data, label, ID, train_test_labels)
            
        self.ranked_genes = ranked_genes
        super().__init__(data)
        
    def convert(self, data, label, ID, train_test_labels, count_id = 'expression', gene_id = "genes", GF_limit = 2048):
        tokens = TranscriptomeTokenizer()
        token_dict = tokens.gene_token_dict
        median_dict = tokens.gene_median_dict
        if train_test_labels != None:
            ans_dict = {'input_ids':[], 'length':[], label:[], 'train':train_test_labels}
        else:
            ans_dict = {'input_ids':[], 'length':[], label:[]}
        
        # Normalizes for gene counts in sample
        gene_counts = {}
        for col_name in data.columns:
            if '_' not in col_name:
                count = data[col_name].sum()
                gene_counts[col_name] = count
                    
        genes = list(gene_counts.keys())
        expression = data.select(genes)
        label_index = data.columns.index(label)
        
        for row in expression.iter_rows():
            gexp = {genes[i]:exp for i, exp in enumerate(row[:-1])}
            for num, key in enumerate(list(gexp.keys())):
                try:
                    gexp[key] /= (median_dict[key] * gene_counts[key])
                except:
                    gexp.pop(key)
                
            gexp = sorted(gexp.items(), key = lambda x: x[1], reverse = True)
            ranked_genes = [gexp[i][0] for i in range(len(gexp))][:GF_limit]
            
            input_ids = self.tokenize_dataset(gene_set = ranked_genes, token_dict = token_dict, type = ID)
            ans_dict["input_ids"].append(input_ids)
            ans_dict["length"].append(len(input_ids))
            try:
                ans_dict[label].append(row[label_index])
            except:
                ans_dict[label].append(row[label_index - 1])
            
        data = pa.Table.from_arrays([ans_dict[key] for key in list(ans_dict.keys())], names=list(ans_dict.keys()))
        
        return data, ranked_genes
        
    # Function for tokenizing genes into ranked-value encodings from Geneformer
    def tokenize_dataset(self, gene_set, token_dict, type = None, species = 'human'):
        wrap = True

        if isinstance(gene_set[0], list) == False:
            gene_set = [gene_set]
            wrap = False
            
        pool = Pool()
        converted_set = []

        def process_gene(gene):
             api_url = f"https://rest.ensembl.org/xrefs/symbol/{species}/{gene}?object_type=gene"
             response = requests.get(api_url, headers={"Content-Type": "application/json"})
             try:
                 data = response.json()
                 gene = data[0]['id']
             except:
                 gene = None
             return gene
             
        def process_hgnc(gene):
            for gene in tqdm.tqdm(genes, total = len(genes)):
                api_url = f"https://rest.ensembl.org/xrefs/symbol/{species}/{hgnc_id}?object_type=gene"
                response = requests.get(api_url, headers={"Content-Type": "application/json"})
                try:
                    data = response.json()
                    gene = data[0]['id']
                except:
                    gene = None
                return gene
                        
        def process_go(gene):
             mg = mygene.MyGeneInfo()
             results = mg.query(gene, scopes="go", species=species, fields="ensembl.gene")
    
             ensembl_ids = []
             max_score = 0
             for hit_num, hit in enumerate(results["hits"]):
                 if hit['_score'] > max_score:
                     max_score = hit['_score']
                     chosen_hit = hit
             try:
                 try:
                     gene = chosen_hit["ensembl"]["gene"]
                 except:
                     gene = chosen_hit["ensembl"][0]["gene"]
             except:
                 gene = None
             return gene
             
        if type == None or type.upper() == 'ENSEMBL':
            converted_set = gene_set
        elif type.upper() == 'GENE':
            for genes in gene_set:
                converted_genes = []
                for result in tqdm.tqdm(pool.imap(process_gene, genes), total = len(genes)):
                    converted_genes.append(result)
                converted_set.append(converted_genes)
                
        elif type.upper() == 'GO':
            for genes in gene_set:
                converted_genes = []
                for result in tqdm.tqdm(pool.imap(process_go, genes), total = len(genes)):
                    converted_genes.append(result)
                converted_set.append(converted_genes)
                
        elif type.upper() == 'HGNC':
            for genes in gene_set:
                converted_genes = []
                for result in tqdm.tqdm(pool.imap(process_hgnc, genes), total = len(genes)):
                    converted_genes.append(result)
                converted_set.append(converted_genes)
                
        Chembl = []

        reverse_dict = {}
        for set_num, set in enumerate(converted_set):
            Chembl.append([])
            for gene in set:
                if gene == None:
                    Chembl[set_num].append(None)
                else:
                    try:
                        Chembl[set_num].append(token_dict[gene])
                    
                    except:
                        print(f'{gene} not found in tokenized dataset!')
                        Chembl[set_num].append(None)
    
        if wrap == False:
            Chembl = Chembl[0]
        
        return Chembl    
    
# Primary function for running Geneformer analysis
def format_sci(data, 
               token_dictionary = Path('geneformer/token_dictionary.pkl'), 
               PR = False, 
               augment = False, 
               noise = None, 
               save = 'Genes.dataset', 
               gene_conversion = Path("geneformer/gene_name_id_dict.pkl"), 
               target_label = "RA", 
               GF_samples = 20000, 
               save_file = 'Stats.png', 
               equalize = True,
               finetuned_model_location = 'Geneformer-finetuned'):
    '''
    KEY FUNCTION PARAMETERS
    ------------------------------------------
    data : csv
        CSV file containing expression/labelled data to be loaded into the model
        
    PR : bool, default = False
        Chooses whether to calculate a precision/recall curve.
        
    augment: bool, default = False
        Chooses whether to create augmented data and use it as a training set (with the true dataset as the test set) or not.
        
    noise : None, float, default = None
        If set to a float, noise equivalent to the noise * the original gene mean for each gene will be applied to the dataset.
        
    save : str, path, default = 'Genes.dataset'
        Save name for the GF-compatible saved dataset created when converting to the proper dataset format.
        
    target_label : str, default = 'RA'
        The name of the column in the csv dataset that contains class labels.
        
    GF_samples : int, default = 20000
        The number of samples to augment in total. Each class is represented equally
    
    equalize : bool, default = True
        Equalizes the dataset so that all classes are represented in equal amounts
        
    save_file : bool, None, default = Stats.png
        Save file for the PR/ROC curve generated
    
    finetuned_model_location : str, default = 'Geneformer-finetuned'
        Location where the finetuned model weights are saved
    '''
    
    cols = []
    conversion = {}
    data = pl.read_csv(data)
    
    token_dict = pk.load(open(token_dictionary, 'rb'))
    gene_dict = pk.load(open(gene_conversion, 'rb'))
    
    # Scipher data pre-processing
    if 'das28crp_cat' in data.columns:
    
        for column in data.columns:
            if ':' in column:
                ensembl = column.split(':')[1]
                try:
                    token_dict[ensembl]
                except:
                    continue
            else:
                try:
                    ensembl = gene_dict[column.strip()]
                except:
                    continue
            cols.append(column)
            conversion[column] = ensembl


        labels = data['das28crp_cat']
        labels = [0 if key == 'Remission' else 1 for key in labels]
        data = pl.concat([data, pl.DataFrame({target_label:labels})], how="horizontal")
        keep = cols + [target_label]
        data = data.select(keep)
        data = data.rename(conversion)
 
    # Normalizes individual samples to total read count
    data = normalize_data(data, polars = True, round_data = False)
    data = data.sample(fraction = 1.0, shuffle = True)
    
    if equalize == True:
        data = equalize_data(data)
    labels = [int(i) for i in list(data[target_label])]
    data = data.sample(fraction = 1.0, shuffle = True)
     
    # Calculates dataset bias
    dataset_bias = 1 - (labels.count(0)/labels.count(1))/2
    
    # Augments data
    if augment == True:
        augmented_data = augment_data(data = data, selected_label = 'all', num_samples = GF_samples, polars = True, normalize = False)    
        augmented_data = augmented_data.sample(fraction = 1.0, shuffle = True)
    
    # Adds noise to data if indicated
    if noise:
      data = add_noise(data, noise = noise)
      data = data.sample(fraction = 1.0, shuffle = True)

    # Converts data to GF-applicable format
    try:
        cell_data = CellData(train = augmented_data, test = data, label = target_label)
    except:
        cell_data = CellData(train = None, test = data, label = target_label)
    cell_data.save_to_disk(save)
    
    # If you want to load data instead of creating data for GeneFormer (for replicability), move the 2 lines of save code below to the prior section. To load the dataset, use the bottom line of code
    #with open(save, 'wb') as f:
        #pickle.dump(cell_data, save)
    #cell_data = pickle.load(open(save, 'rb'))
    
    # Selects only genes that are exposed to GeneFormer
    data = data.select(cell_data.ranked_genes + [target_label])
    #augmented_data = augmented_data.select(cell_data.ranked_genes + [target_label])
    data = data.sample(fraction = 1.0, shuffle = True)
    
    if PR == True:
        # Calculates TPR and FPR for GeneFormer
        recall4, precision4, auc4 = finetune_cells(model_location = "/work/ccnr/GeneFormer/GeneFormer_repo", dataset = save, epochs = 30, geneformer_batch_size = 9,
            skip_training = False, label = "RA", inference = False, optimize_hyperparameters = False, emb_dir = 'RA', emb_extract = False, freeze_layers = 1, output_dir = 'GF-finetuned', ROC_curve = False)
        
        # Calculates TPR and FPR for ensemble models
        try:
            recall3, precision3, auc3 = FFN(test_data = data, train_data = augmented_data, total_samples = GF_samples, augment = False, ROC = False)
            recall2,  precision2, auc2 = SVC_model(test_data = data, train_data = augmented_data, total_samples = GF_samples, augment = False, ROC = False)
            recall1, precision1, auc1 = RandomForest(test_data = data, train_data = augmented_data, total_samples = GF_samples, augment = False, ROC = False)
        except:
            recall3, precision3, auc3 = FFN(test_data = data, train_data = None, total_samples = GF_samples, augment = False, ROC = False)
            recall2,  precision2, auc2 = SVC_model(test_data = data, train_data = None, total_samples = GF_samples, augment = False, ROC = False)
            recall1, precision1, auc1 = RandomForest(test_data = data, train_data = None, total_samples = GF_samples, augment = False, ROC = False)

        plt.figure(figsize=(8, 6))
        plt.plot(recall1, precision1, color='darkorange', lw=2, label=f'RF (AUC = {round(auc1, 3)})')
        plt.plot(recall2, precision2, color='green', lw=2, label=f'SVC (AUC = {round(auc2, 3)})')
        plt.plot(recall3, precision3, color = 'blue', lw=3, label = f'Feed-Forward Network (AUC = {round(auc3, 3)})')
        plt.plot(recall4, precision4, color = 'red', lw=3, label = f'GeneFormer (AUC = {round(auc4, 3)})')
        plt.plot([0, 1], [dataset_bias, dataset_bias], color = 'navy', lw = 2, linestyle = '--', label = 'Chance')
        plt.xlim([-.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'RA Dataset Precision/Recall for Geneformer and Ensemble Models \n (n = {len(data)})')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0.2), ncol=1)
        plt.tight_layout()
        
        # Saves image
        if save_img:
            plt.savefig(save_img)
    
    else:
        # Calculates ROC curve for GeneFormer
        fpr4, tpr4, auc4 = finetune_cells(model_location = "/work/ccnr/GeneFormer/GeneFormer_repo", dataset = 'Scipher.dataset', epochs = 50, geneformer_batch_size = 9,
            skip_training = False, label = "RA", inference = False, optimize_hyperparameters = False, emb_dir = 'RA', emb_extract = False, freeze_layers = 0, output_dir = 'GF-finetuned')
            
        try:
            fpr3, tpr3, auc3 = FFN(test_data = data, train_data = augmented_data, total_samples = GF_samples, augment = False)
            fpr2, tpr2, auc2 = SVC_model(test_data = data, train_data = augmented_data, total_samples = GF_samples, augment = False)
            fpr1, tpr1, auc1 = RandomForest(test_data = data, train_data = augmented_data, total_samples = GF_samples, augment = False)
        except:
            fpr3, tpr3, auc3 = FFN(test_data = data, train_data = None, total_samples = GF_samples, augment = False)
            fpr2, tpr2, auc2 = SVC_model(test_data = data, train_data = None, total_samples = GF_samples, augment = False)
            fpr1, tpr1, auc1 = RandomForest(test_data = data, train_data = None, total_samples = GF_samples, augment = False)
            
        plt.figure(figsize=(8, 6))
        plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'RF (AUC = {round(auc1, 3)})')
        plt.plot(fpr2, tpr2, color='green', lw=2, label=f'SVC (AUC = {round(auc2, 3)})')
        plt.plot(fpr3, tpr3, color = 'blue', lw=3, label = f'Feed-Forward Network (AUC = {round(auc3, 3)})')
        plt.plot(tpr4, fpr4, color = 'red', lw=3, label = f'GeneFormer (AUC = {round(auc4, 3)})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-.05, 1.05])
        plt.ylim([0, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        try:
            augmented_data
            plt.title(f'Scipher data Drug Predictions ROC for Ensemble vs Geneformer curve \n (n = {GF_samples}, nbinomial augmenation)')
        except:
            plt.title(f'RA Dataset Drug Predictions ROC for Ensemble vs Geneformer curve \n (n = {len(data)})')
        #plt.title(f'Shuffling before augmentation ROC Curve (n = 20000 samples, mbinomial augmentation)')
        plt.legend(loc='lower left', bbox_to_anchor=(.5, 0.2), ncol=1)
        plt.tight_layout()
        
        # Saves image
        if save_img:
            plt.savefig(save_img)
    
# Function for equalizing labels
def equalize_data(data, label = 'RA'):
    labels = data[label].to_list()
    label_set = list(set(labels))
    freq = {i:labels.count(i) for i in list(set(labels))}
    classes = sorted(freq.items(), key = lambda x: x[1])
    min_class, min_freq = classes[0][0], classes[0][1]
    labels = [i[0] for i in classes]
    labels = [i for i in labels if i != min_class]
    data_columns = data.columns
    
    class_numbers = {key:0 for key in label_set}
    data_rows = []
    for row in data.iter_rows():  
        row_label = row[-1]  
    
        if class_numbers[row_label] < min_freq:
            class_numbers[row_label] += 1
            data_rows.append(row)
    data = pl.DataFrame(data_rows, schema = data_columns)

    return data
    
# Function for shuffling a certain number of labels
def shuffle_labels(data, label = 'RA', fraction = 1):
    data.sample(fraction = 1, shuffle = True)
    labels = data[label].to_list()
    fraction_shuffled = int(len(labels) * fraction)
    shuffled_subset = labels[:fraction_shuffled]
    random.shuffle(shuffled_subset)
    shuffled_labels = shuffled_subset + labels[fraction_shuffled:]
    data = data.drop(label)
    data = pl.concat((data, pl.DataFrame([shuffled_labels], schema = [label])), how = 'horizontal')
    
    return data

# Feed-forward network testing
def FFN(train_data, test_data, epochs = 100, total_samples = 3000, test_size = .2, noise = .25, augment = True, ROC = True):
    
    try:
        if train_data.is_empty():
            pass
            
    except:
        train_data, test_data = train_test_split(test_data, test_size = test_size, random_state=42)
    
    train_labels, test_labels = train_data[:, -1].to_list(), test_data[:, -1].to_list()
    train_size = (1 - test_size) * total_samples
    test_size = total_samples - train_size
    
    if augment == True:
        train_data = augment_data(data = train_data, selected_label = 'all', polars = True,
                                       num_samples = train_size).sample(fraction = 1.0, shuffle = True)

        test_data = augment_data(data = test_data, selected_label = 'all', polars = True,
                                     num_samples = test_size).sample(fraction = 1.0, shuffle = True)
                
    X_train, y_train = train_data[:, :-1].to_numpy(), train_data[:, -1].to_list()
    X_test, y_test = test_data[:, :-1].to_numpy(), test_data[:, -1].to_list()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
                         
    X_train = add_noise(X_train, noise = noise, label = 'RA')
    X_test = add_noise(X_test, noise = noise, label = 'RA')
   
    class SimpleNN(plit.LightningModule):  # Inherit from LightningModule
        def __init__(self, hidden_channels=100, n_classes=len(list(set(y_train)))):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], hidden_channels * 4)
            self.fc2 = nn.Linear(hidden_channels * 4, hidden_channels)
            self.fc3 = nn.Linear(hidden_channels, n_classes)
            self.batch = nn.BatchNorm1d(hidden_channels * 4)
            self.drop = nn.Dropout(0.2)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            x = self.fc1(x)
            x = self.drop(x)
            x = self.batch(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x
    
        def training_step(self, batch, batch_idx):
            inputs, labels = batch
            outputs = self(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            self.log("train_loss", loss)
            return loss
    
        def configure_optimizers(self):
            optimizer = optim.SGD(self.parameters(), lr=0.01)
            return optimizer
    
    # Instantiate the Lightning model
    model = SimpleNN()
    
    # Convert NumPy arrays to PyTorch tensors (ensure X_train, y_train, X_test, y_test are defined)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create TensorDataset and DataLoader for training and test data
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    batch_size = 60  # You can adjust this batch size as needed
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create a Lightning Trainer and fit the model
    trainer = plit.Trainer(max_epochs=epochs)  # Define 'epochs' as needed
    trainer.fit(model, train_loader)
    
    if ROC != True:
    # Test the model
        model.eval()
        with torch.no_grad():
            all_probs = []
            all_labels = []
            for inputs, labels in test_loader:
                test_outputs = model(inputs)
                _, predicted = torch.max(test_outputs, 1)
                probabilities = torch.softmax(test_outputs, dim=1)[:, 1]
                all_probs.extend(probabilities)  # Use the probability of class 1
                all_labels.extend(labels.tolist())
            
        # Assuming you have y_pred as the predicted labels (0 or 1) and y_prob as predicted probabilities for the positive class
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        
        # Compute AUC for precision-recall curve
        pr_auc = skauc(sorted(recall), sorted(precision))
        
        # Print precision and recall
        print('Precision-Recall FFN AUC:', round(pr_auc, 4))
    
        return recall, precision, pr_auc
    else:
        # Test the model
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            all_probs = []
            for inputs, labels in test_loader:
                test_outputs = model(inputs)
                probabilities = torch.softmax(test_outputs, dim=1)[:, 1]
                all_probs.extend(probabilities)
                _, predicted = torch.max(test_outputs, 1)
                all_preds.extend(predicted.tolist())  # Extend the list of predicted labels
                all_labels.extend(labels.tolist())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
       
        # Print accuracy
        print('Accuracy:', round(accuracy, 4))
         
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        
        auc = skauc(fpr, tpr)
     
        print(f'FFN AUC: {auc}')
                     
        return fpr, tpr, auc
        
    
    
def RandomForest(train_data, test_data, total_samples = 3000, num_estimators = 100, test_size = .2, noise = .25, augment = True, ROC = True):

    try:
        if train_data.is_empty():
            pass
        
    except:
        train_data, test_data = train_test_split(test_data, test_size = test_size, random_state=42)
        
    train_labels, test_labels = train_data[:, -1].to_list(), test_data[:, -1].to_list()
    train_size = (1 - test_size) * total_samples
    test_size = total_samples - train_size
    
    if augment == True:
   
        train_data = augment_data(data = train_data, selected_label = 'all', polars = True,
                                       num_samples = train_size).sample(fraction = 1.0, shuffle = True)

        test_data = augment_data(data = test_data, selected_label = 'all', polars = True,
                                     num_samples = test_size).sample(fraction = 1.0, shuffle = True)
                
    X_train, y_train = train_data[:, :-1].to_numpy(), train_data[:, -1].to_list()
    X_test, y_test = test_data[:, :-1].to_numpy(), test_data[:, -1].to_list()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
                         
    X_train = add_noise(X_train, noise = noise, label = 'RA')
    X_test = add_noise(X_test, noise = noise, label = 'RA')
    
    clf = RandomForestClassifier(n_estimators=num_estimators, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    y_prob = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print(f'RF Accuracy: {accuracy}')
        
    if ROC != True:
        # Assuming you have y_pred as the predicted labels (0 or 1) and y_prob as predicted probabilities for the positive class
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        # Compute AUC for precision-recall curve
        pr_auc = skauc(sorted(recall), sorted(precision))
        
        # Print precision and recall
        print('Precision-Recall RF AUC:', round(pr_auc, 4))
    
        
        return recall, precision, pr_auc
    else:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc = skauc(fpr, tpr)
        
        print(f'RF AUC: {auc}')
                     
        return fpr, tpr, auc
    
    
def SVC_model(train_data, test_data , total_samples = 3000, test_size = .2, noise = .25, augment = True, ROC = True):  

    try:
        if train_data.is_empty():
            pass
            
    except:
        train_data, test_data = train_test_split(test_data, test_size = test_size, random_state=42)

    train_labels, test_labels = train_data[:, -1].to_list(), test_data[:, -1].to_list()
    train_size = ( 1 - test_size) * total_samples
    test_size = total_samples - train_size
   
    if augment == True:
   
        train_data = augment_data(data = train_data, selected_label = 'all', polars = True,
                                       num_samples = train_size).sample(fraction = 1.0, shuffle = True)

        test_data = augment_data(data = test_data, selected_label = 'all', polars = True,
                                     num_samples = test_size).sample(fraction = 1.0, shuffle = True)
                
    X_train, y_train = train_data[:, :-1].to_numpy(), train_data[:, -1].to_list()
    X_test, y_test = test_data[:, :-1].to_numpy(), test_data[:, -1].to_list()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
                         
    X_train = add_noise(X_train, noise = noise, label = 'RA')
    X_test = add_noise(X_test, noise = noise, label = 'RA')
    
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_prob = model.decision_function(X_test)[:, ]
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVC Accuracy: {accuracy}')
        
    if ROC != True:
        # Assuming you have y_pred as the predicted labels (0 or 1) and y_prob as predicted probabilities for the positive class
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        # Compute AUC for precision-recall curve
        pr_auc = skauc(sorted(recall), sorted(precision))
        
        # Print precision and recall
        print('Precision-Recall SVC AUC:', round(pr_auc, 4))
        
        return recall, precision, pr_auc
    else:
    
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = skauc(fpr, tpr)
        
        print(f'SVC AUC: {auc}')
                     
        return fpr, tpr, auc
    
    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', dest = 'type', type = str, help = 'Type of run', default = None)
    args = parser.parse_args()
    
    return args
    
if __name__ == '__main__':
    args = get_arguments()
    if 'cancer' not in args.type:
        format_sci(data = Path("GSE97810.csv"), save_file = 'RAROC.svg', save = 'Scipher.dataset', equalize = True) #"Enzo_dataset2.csv" 'enzo/RAmap.csv'
    else:
        format_sci(data = Path("enzo/cancer/lungCancer.csv"), save_file = 'CancerROC.svg', save = 'Scipher.dataset', equalize = True) 

    