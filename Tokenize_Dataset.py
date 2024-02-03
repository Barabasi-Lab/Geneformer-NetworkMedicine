# Package imports
import os 
import sys
import time
import tqdm
import traceback
import pickle as pk
from pathlib import Path
from datasets import Dataset
import polars as pl
from sklearn.utils import shuffle
from geneformer import TranscriptomeTokenizer
import numpy as np
from scipy.stats import nbinom
import itertools
import seaborn as sns
import statistics
from Cell_classifier import *
import GEOparse
from GEOparse import GEOparse
import logging
from geneformer import TranscriptomeTokenizer
import pickle as pk
import gzip
import shutil
from biomart import BiomartServer
import requests
import csv
logging.disable(logging.CRITICAL)

# Function to download RNA-seq data from a specific GEO repository
def download_geo_data(geo_id, clean = False):
    try:
        # Download GEO dataset
        geo = GEOparse.get_GEO(geo=geo_id, destdir=f"./{geo_id}")
        with gzip.open(f"./{geo_id}/{geo_id}_family.soft.gz", 'rb') as f_in:
            with open(f"./{geo_id}/{geo_id}_family.soft", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        print("Downloaded GEO dataset:", geo.metadata['title'][0])
        
        if clean == True:
            # Clean up downloaded files (optional)
            os.remove(f"{geo_id}.soft.gz")
            os.rmdir(geo_id)
        
        print("Data processing and model training code goes here.")
    except Exception as e:
        print("Error:", str(e))

def parse_geo(geo_id = 'GSE97476'):
    gse = GEOparse.get_GEO(filepath=F"./{geo_id}/{geo_id}_family.soft")
    
    # Print basic information about the dataset
    print("Dataset Name:", gse.name)
    
    # Collects samples with RNA counts
    sample_counter, table_counter = -1, 0
    table_dict = {'Positive':[], 'Negative':[]}
    gene_counts = {}
   
    tokens = TranscriptomeTokenizer()
    gene_conversion = pk.load(open(Path("geneformer/gene_name_id_dict.pkl"), 'rb'))
    
    explored_ids = []
    indexer = ".rheum.factor.positive.scr"
    
    # Iterates through each sample in the dataset
    for gsm_name, gsm in tqdm.tqdm(gse.gsms.items(), total = len(gse.gsms.items()), desc = 'Normalizing samples'):
        expression_data = gsm.table
        
        
        # Checks if there is expression data for the patient
        if len(expression_data) > 0:
            meta = gsm.metadata
            desc = meta['characteristics_ch1']
            
            # Filters down the samples to ensure the patient ID is unique
            patient_id = [i for i in desc if 'patient.id' in i][0].split(' ')[-1]
            if patient_id in explored_ids:
                continue
            else:
                explored_ids.append(patient_id)
                

            # Searches for the indexer (for patient condition) in the metadata for the sample
            for num, item in enumerate(desc):
                if indexer in item:
                    
                    check = item.split(':')[1].strip()
                
                    converted_dict = {}
                    table = gsm.table
                    
                
                    # Iterates through each gene in the dataset    
                    for index, row in table.iterrows():
                        gene = row['ID_REF']
                        expression = row['VALUE']
                        
                        try:
                            gene_conv = gene_conversion[gene]
            
                            try:
                                gene_counts[gene_conv] += expression
                            except KeyError:
                                gene_counts[gene_conv] = expression
                    
                            converted_dict[gene_conv] = expression 
                    
                        except KeyError:
                            continue
                    
                    # Assigns label
                    if 'Yes' in check:
                        table_dict['Positive'].append(converted_dict)
                    else:
                        table_dict['Negative'].append(converted_dict)
                    break
    
    total_genes = []
    total_tables = table_dict['Negative'] + table_dict['Positive']
    
    labels = [0 for _ in range(len(table_dict['Negative']))] + [1 for _ in range(len(table_dict['Positive']))]
    for table in tqdm.tqdm(total_tables, total = len(total_tables)):
        table_keys = list(table.keys())
        total_genes = list(set(table_keys + total_genes))
                
    total_genes = total_genes 
    results = {key:[] for key in total_genes}
             
    # Reconstructs dataframe from cell expression data
    GF_conv = {'Positive':pl.DataFrame(), 'Negative':pl.DataFrame()}
    for condition in list(table_dict.keys()):
        for table in tqdm.tqdm(table_dict[condition], total = len(table_dict[condition]), desc = f'Processing {condition} samples'):
            table_keys = list(table.keys())
            for key in list(results.keys()): 
                if key in table_keys:
                    results[key].append(float(table[key]))
                else:
                    results[key].append(0)
    
    # Adds labels to data
    results = pl.concat([pl.DataFrame(results), pl.DataFrame([labels], schema = ['RA'])], how = 'horizontal')
    results.write_csv(f'{geo_id}.csv')
  
# Custom cell dataset class 
class CellData(Dataset):
    def __init__(self, test = None,
                 train = None, 
                 label = "cell_type", 
                 ID = 'ENSEMBL', 
                 dataset_normalization = False):
        '''
        Parameters
        ===================
        train : pl.DataFrame, None
            Polars dataframe containing rows of samples and columns of genes. When set to None, the test dataframe will be used as a train/test split. If specified, this will be the training data for the pipeline

        test : pl.DataFrame, None
            Polars dataframe containing rows of samples and columns of genes. If there is a training dataframe, this will be used as test data. Otherwise this will be split 80/20 samples into a train/test set. 

        label : str, default = "cell_type"
            The column that has labels for the samples. Defaults to "cell_type".

        ID : str, default = "ENSEMBL"
            The ID type of the genes in each column header. If not ENSEMBL, will be converted to Ensembl IDs for Geneformer. Can be GO, GENE, or HGNC. 

        dataset_normalization : bool, default = False
            If set to false, normalizes data just based off the median gene expression in the gene corpus. If true, data will be normalized by the median gene expression and the average gene count in the dataset.
        '''
                   
        # Dataset normalization controls whether data is ranked based on just median normalized data, or if it is ranked based on median and dataset gene count normalization
        self.dataset_normalization = dataset_normalization
                   
        try:
            self.label = label
            
            # Uses a train/test split if provided, otherwise creates the test split if only a test value is provided
            try:
                train_test_labels = [1 for _ in range(len(train))] + [0 for _ in range(len(test))]
                data = pl.concat((train, test), how = 'vertical')
                data, ranked_genes = self.convert(data, label, ID, train_test_labels)
            except:
                data, ranked_genes = self.convert(test, label, ID, train_test_labels = None)
                
            self.ranked_genes = ranked_genes
            super().__init__(data)
        except:
            print(traceback.format_exc())
            pass

    # Converts gene IDs to the correct format
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
            
            # Normalizes each set of genes by median and/or dataset gene count
            for num, key in enumerate(list(gexp.keys())):
                try:
                    if self.dataset_normalization:
                        gexp[key] /= (median_dict[key] * gene_counts[key])
                    else:
                        gexp[key] /= (median_dict[key])
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
        
        # Creates pyarrow tabe out of the data
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

        # Ensembl based searching
        def process_gene(gene):
             api_url = f"https://rest.ensembl.org/xrefs/symbol/{species}/{gene}?object_type=gene"
             response = requests.get(api_url, headers={"Content-Type": "application/json"})
             try:
                 data = response.json()
                 gene = data[0]['id']
             except:
                 gene = None
             return gene
             
        # HGNC ID searching
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
                        
        # GO ID searching
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
             
        # Selects the ID conversion to ensembl
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
                
        # Obtains Cheml ENSEMBL names for each gene if possible
        Chembl = []
        
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
  
# Main runtime 
if __name__ == '__main__':
    download_geo_data(geo_id = 'GSE97810')
    parse_geo(geo_id = 'GSE97810')

    data = pl.read_csv('GSE97810.csv')
    cell_data = CellData(train = None, test = data, label = 'RA')
    cell_data.save_to_disk('GEO_dataset.dataset')
  
