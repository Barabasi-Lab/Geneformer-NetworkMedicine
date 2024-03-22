import os
import torch
from datasets import load_from_disk, Dataset, DataLoader
from transformers import BertForSequenceClassification, BertForTokenClassification, BertModel, Trainer
import geneformer
from gprofiler import GProfiler
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
import GEOparse
import pyarrow as pa
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from transformers.training_args import TrainingArguments
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

from geneformer import DataCollatorForCellClassification

# extra imports needed for hyperparameter tuning
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

########################################################################################################
# Contents
# line 45: tokenize data section
#   line 54: function to parse GEO data
#   line 120: function to tokenize csv data
####### Add 7 to every line reference below this (updated tokenizer, too lazy to retype legend) #######
# line 203: load data and model section
#   line 214: class to handle GF data and return a Dataset object
#   line 307: function to load a model
#   line 336: function to run samples through a model
#   line 405: function to aggregate attentions over multiple samples
# line 516: fine-tuning section
#   line 527: function to pre-process data for a cell classification task
#   line 565: function to optimize hyperparameters
#   line 645: function to fine-tune a model for a cell classification task
########################################################################################################

################################################################################################
# This section of the script contains functions to tokenize expression data from GEO datasets
# The parse_GEO function allows you to parse a GEO soft.gz file and save the expression 
# data and metadata as csv files. The tokenize_csv function takes a dataframe and 
# tokenizes the expression data, saving it as a .dataset folder which can be used by 
# Geneformer. The parse_GEO function outputs csvs with gene IDs as the row index and 
# sample IDs as the column index, which is the required format for the tokenize_csv function.
###############################################################################################

def parse_GEO(accession_id, data_dir, gz_name):
    """
    :param accession_id: string, GEO accession id
    :param data_dir: string, directory to save data
    :param gz_name: string, path to the .soft.gz file

    :return exprs: pandas Dataframe containing the expression data for the samples
    :return metadata: pandas Dataframe containing the metadata for the samples
    """
    gse = GEOparse.get_GEO(filepath=gz_name, silent=True)
    # Get expression data and metadata matrices
    exprs = []
    gsmNames = []
    metadata = {}
    # Iterates through each sample in the dataset
    for gsm_name, gsm in tqdm(gse.gsms.items()):
        # Checks if there is expression data for the patient
        if gsm.metadata['type'][0] == 'RNA':
            # Expression data
            if len(gsm.table) > 0:
                tmp = gsm.table['VALUE']
                tmp.index = gsm.table['ID_REF']
                gsmNames.append(gsm_name)
                if len(exprs) == 0:
                    exprs = tmp.to_frame()
                else:
                    exprs = pd.concat([exprs, tmp.to_frame()], axis=1)

            # Metadata (this section is taken from old code because it formats the metadata nicely)
            for key, value in gsm.metadata.items():
                if (key == 'characteristics_ch1' or key == 'characteristics_ch2') and (
                        len([i for i in value if i != '']) > 1 or value[0].find(': ') != -1):
                    tmpVal = 0
                    for tmp in value:
                        splitUp = [i.strip() for i in tmp.split(':')]
                        if len(splitUp) == 2:
                            if not splitUp[0] in metadata:
                                metadata[splitUp[0]] = {}
                            metadata[splitUp[0]][gsm_name] = splitUp[1]
                        else:
                            if not key in metadata:
                                metadata[key] = {}
                            metadata[key][gsm_name] = splitUp[0]
                else:
                    if not key in metadata:
                        metadata[key] = {}
                    if len(value) == 1:
                        metadata[key][gsm_name] = value[0]
                    else:
                        metadata[key][gsm_name] = value
    # Save expression data and metadata
    directory = os.path.join(data_dir, accession_id)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
        os.mkdir(os.path.join(directory, "exprs"))
        os.mkdir(os.path.join(directory, "metadata"))
        os.mkdir(os.path.join(directory, "annot"))
    # set the column names to the gsm names and save the data
    exprs.columns = gsmNames
    exprs.to_csv(os.path.join(directory, "exprs", accession_id + "_exprs.csv"))
    metadata = pd.DataFrame(metadata)
    metadata.to_csv(os.path.join(directory, "metadata", accession_id + "_metadata.csv"))
    return exprs, metadata

def tokenize_csv(data, gene_ID_type,genes_to_include = None, dataset_normalization = False, path_to_metadata = None, metadata_parser = None):
    """
    :param data: pandas dataframe, the expression data from the GEO dataset. The format should be a matrix with gene names as the row index and patient IDs as the column index
    :param gene_ID_type: str, the type of gene ID used in the dataset. Should be either "ensembl" or the name of your format. Your format should be compatible with the gprofiler package 
    :param genes_to_include: list, a list of genes to include in the dataset. If None, all genes will be included. Defaults to None
        Note: genes to include should be in ENSEMBL format
    :param dataset_normalization: bool, whether to normalize the dataset by median gene expression WITHIN the dataset (in addition to the geneformer normalization). Defaults to False
    :param path_to_metadata: str, the path to the metadata file. Defaults to None, which is fine if no labels are needed
    :param metadata_parser: function, a function that takes the metadata dataframe and the column index as input and returns the label for the sample. Defaults to None, which is fine if no labels are needed

    :return: a HuggingFace Dataset object with the tokenized data
    """
    if path_to_metadata != None:
        metadata = pd.read_csv(path_to_metadata,low_memory = False)
    # get the tokens and the median values for normalization from the geneformer package
    tokens = geneformer.TranscriptomeTokenizer()
    token_dict = tokens.gene_token_dict
    median_dict = tokens.gene_median_dict
    ans_dict = {'input_ids':[], 'length':[], 'label':[],'gene_ids':[]}
    # save the row indices as a list of gene ids
    gene_ids = data.index.tolist()

    if dataset_normalization == True: 
        # for each row in the data, divide the row by its median value
        data = data.div(data.median(axis=1), axis=0)
    # if the genes are not in ensembl format, convert them to ensembl format with gprofiler
    if gene_ID_type != "ensembl":
        gp = GProfiler(return_dataframe=True)
        convert_genes = gp.convert(organism='hsapiens',
                    query=gene_ids,
                    target_namespace='ENSG')
        convert_genes.drop_duplicates(subset = 'incoming', keep = 'first', inplace = True)
        converted_genes = convert_genes['converted'].to_list()
        # Gprofiler returns 'None' for genes that it cannot convert, so we will remove those genes from the dataset
        data['for_filter'] = converted_genes
        data = data[data['for_filter'] != 'None']
        data.index = data['for_filter']
        data = data.drop(columns = 'for_filter')
        new_gene_ids = data.index.tolist()
        print(f"Removed {len(gene_ids)-len(new_gene_ids)} genes from the dataset because they could not be converted to ensembl IDs.")
        # replace the index of the data with the new gene IDs
        data.index = new_gene_ids
    # filter for genes to include. This is useful if you are dealing with bulk data
    if genes_to_include != None:
        data = data.loc[genes_to_include]
    
    # Now we perform the normalization from the geneformer package, which divides each gene's expression by its median value in the pretraining corpus
    # for each row in the data, put the index into median_dict and then divide the row by the median value
    indices_to_drop = []
    counter = 0
    for i in tqdm(range(len(data.index))):
        try:
            median = median_dict[data.index[i]]
            data.iloc[i] = data.iloc[i] / median
        except KeyError:
            indices_to_drop.append(data.index[i])
            counter+=1
    data = data.drop(indices_to_drop)
    
    final_gene_ids = data.index
    print(f"Removed {counter} genes from the dataset because they could not be found in the median dictionary.")
    print(f"Final dataset size: {len(data.index)} genes")

    # Create the rank value encoding for each sample
    for i in tqdm(range(len(data.columns))):
        # make a list of the indices (genes), sorted by descending order of the column (rank value)
        sorted_indices = data.iloc[:,i].sort_values(ascending = False).index.tolist()
        # replace the indices with their token values
        sorted_tokens = [token_dict[i] for i in sorted_indices]
        # cut off the list at 2048, the maximum sequence length
        if len(sorted_indices)>2048:
            sorted_indices = sorted_indices[:2048]
            sorted_tokens = sorted_indices[:2048]
        # add the sorted indices to the input_ids list
        ans_dict['input_ids'].append(sorted_tokens)
        # add the length of the sorted indices to the length list
        ans_dict['length'].append(len(sorted_indices))
        # add the gene_ids
        ans_dict['gene_ids'].append(sorted_indices)

        if path_to_metadata != None:
            # your function should take the metadata frame and the column index as input and return the label for the sample
            your_label = metadata_parser(metadata,i)
            ans_dict['label'].append(your_label)
        else:
            ans_dict['label'].append(None)
    # Create a pyarrow tabel out of the data
    arrow_data = pa.Table.from_arrays([ans_dict[key] for key in list(ans_dict.keys())], names=list(ans_dict.keys()))
    hg_data = Dataset(arrow_data)
    return ans_dict,hg_data

################################################################################################
# This section of the script contains functions to load datasets and models, and to run samples through the model
# It also contains a function to aggregate attentions over multiple samples
# The functions are designed to be run sequentially, and an example can be seen at the bottom of this script
# The example loads a dataset, loads a model, runs the samples through the model, and then outputs the attentions
# The aggregate attentions sample is not included in the example, but it can be run in the same way as the run_samples function
###############################################################################################

# This is a class to handle GF data
# It takes a dataset and pads the samples out to the max length of 2048 so that the lengths of all samples match 
# Optionally, you can use the get_genes function to detokenize data later
class GFDataset(Dataset):

    def __init__(self, dataset, path='/work/ccnr/GeneFormer/Sample.dataset/', n=None, samples=None, tissues=None, max_length=2048):
        """
        :param dataset: should usually be set to None, unless you have a dataset already loaded from a .dataset folder with .arrow files in it. Otherwise, give the path of a .dataset folder to the path variable
        :param path: str, path to the dataset if the dataset is different from the sample dataset (note that different datasets will have different metadata included)
        :param n: int, number of samples to select. Either n or samples should always be set to None to avoid conflicts
        :param samples: specific samples to select, if known OR "all" for all the samples in the dataset
        
        :return: a dataset, padded with zeros out to the maximum length of 2048. The dataset can now be passed to the run_samples function 
        
        Example usage:
        
        data = GFDataset(dataset = None, n = 10) # selects 10 samples from the sample dataset
        
        data_GEO = GFDataset(dataset = None, path = "work/ccnr/GeneFormer/GeneFormer_repo/GEO_dataset.dataset/, n = "all") # selects all samples from a dataset downloaded from GEO
        """
        self.path = path
        self.n = n
        self.samples = samples
        self.tissues = tissues
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = self.load_sample_data()
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # input_ids is the column that contains the rank value encoding for each sample (the actual embedding)
        input_ids = item['input_ids']
        # Pad the input_ids with zeros to the specified max_length. 2048 is the max sample length for any sample, but you could also pad out to the maximum length
        # present in your dataset by using the item['lengths'] column and taking the maximum
        padded_input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        return torch.LongTensor(padded_input_ids)

    def load_sample_data(self):
        data = load_from_disk(self.path)
        # unless "all" or a specific list of samples is supplied, select n random samples
        if self.samples is not None:
            if self.samples == "all":
                subset = data
            else:
                subset = data.select(self.samples)
        else:
            subset = data.select([i for i in range(self.n)])
        return subset

    def get_genes(self, convert_from_ensembl = False):
        """
        :param convert_from_ensembl: bool, if set to True, the function will attempt to convert the ensembl IDs to gene names. If the conversion is not possible, the ensembl ID will be left in the list

        :return: list of genes in the order they appear in the sample

        Example usage:
        genes = data.get_genes(convert_from_ensembl = True) # gets the genes and converts them back to gene names when possible. 
        # This list can be saved and used to interpret embeddings and attentions, since it is in the same order as the original input
        """
        # Get the dictionary to convert genes to indices
        token_dict = geneformer.TranscriptomeTokenizer().gene_token_dict
        token_dict = {value: key for key, value in token_dict.items()}

        ids_list = []
        length_list = []
        # get the ids and lengths for all the ids in current dataset
        for ii in range(len(self.dataset)):
            if convert_from_ensembl == False:
                ids_list.append([token_dict[k] for k in self.dataset.select([ii])["input_ids"][0]])
                length_list.append(self.dataset.select([ii])["length"])
            else:
                token_convert = [token_dict[k] for k in self.dataset.select([ii])["input_ids"][0]]
                new_gene_ids = []
                gp = GProfiler(return_dataframe=True)
                convert_genes = gp.convert(organism='hsapiens',
                            query=token_convert,
                            target_namespace='ENTREZGENE')
                converted_genes = convert_genes['converted'].to_list()
                # Gprofiler returns 'None' for genes that it cannot convert, so we leave those genes in ensembl format
                counter = 0
                for i in range(len(token_convert)):
                    if converted_genes[i] != 'None':
                        new_gene_ids.append(converted_genes[i])
                    else:
                        new_gene_ids.append(token_convert[i])
                        counter+=1
                print(f"{counter} genes from the dataset were not able to be converted from ensembl IDs to gene names. These genes have been left in ensembl format.")

        return ids_list, length_list

# loads the model from the pretrained weights. 
def load_model(task, path = "/work/ccnr/GeneFormer/GeneFormer_repo", n_labels = 2, device= "cpu", attns=False, hiddens=False):
    """
    :param task: str, should be either "general", when using the to look at things like embeddings (as there is no classification layer), "cell" when using geneformer for cell classification, or "gene" when using geneformer for gene classification
    :param path: str, path to model, defaults to pretrained geneformer. If you have a fine tuned model you would like to load, put the path to that model here
    :param n_labels:, int, the number of classes in the data. Only applicable for classification tasks, and currently is 2 for every classification task we have performed
    :param device: torch device, should be set earlier in the script, but can be set as "cuda" or "cpu" if you know which one you're using. In this script, device is set on line 13, under the imports
    :param attns: bool, if set to true, attention heads from any layer can be extracted
    :param hiddens: bool, if set to true, embeddings from any layer can be extracted
    
    :output: a geneformer model, loaded to device
    
    example usage:
    model = load_model(task = "general", attns = True) # loads the general pretrained model to output attention heads
    """
    # load for cell classification (sequence classification)
    if task == "cell":
        model = BertForSequenceClassification.from_pretrained(path, num_labels=n_labels,output_attentions = attns,output_hidden_states = hiddens).to(device)
    # load for gene classification (token classification)
    elif task == "gene":
        model = BertForTokenClassification.from_pretrained(path,num_labels=n_labels,output_attentions = attns,output_hidden_states = hiddens).to(device)
    # load for the general use
    elif task == "general":
        model = BertModel.from_pretrained(path, num_labels=n_labels, output_attentions = attns, output_hidden_states = hiddens).to(device)
    else:
        raise ValueError("Task must be either 'general, which loads BertModel, 'cell', which loads BertForSequenceClassification or 'gene', which loads BertForTokenClassification.")
    
    return model

# runs samples through a given model and outputs the attentions and/or embeddings if requested
def run_samples(data, model, batch_size=20, num_workers = 4, attns = None, hiddens = None, output_format = "csv", save_gene_names = False, heads_to_prune = None):
    """
    :param data: dataset, output by the GFDataset class. It should already be padded but not passed to the dataloader
    :param model: the model loaded by the load_model function
    :param batch_size: int, size of geneformer batch. In general larger batch size  = faster but more memory intensive. If you are only running samples (i.e. to extract embeddings) the batch size can be set in the 100s. Specifically, setting ntasks = 20 
        in an sbatch script with 128 GB of memory allows you to run ~150 samples per minute on the short partition with batch size = 200 and num_workers = 4
    :param num_workers: int, the number of processes to run simultaneously. In experiments with running samples on cpus, this parameter has had no tangible effect on speed or memory, but 4 seems to be a widely used value. The only constraint is that 
        num_workers must be less than or equal to ntasks in the sbatch script
    :param attns: dict, the attention heads to output. The format should be {l:[h1,h2,h3],...} where the keys are layer indices and the values are lists of attention head indices
    :param hiddens: list, the layer indices from which embeddings should be output. format should be [l1,l2,l3...]. Note that for both attns and hiddens, the corresponding variable must be set to True when loading the model, or there will be no matrices available for output.
    :param output_format: str, must be either csv or pkl. Defaults to csv, which outputs a gzipped csv for each attention head or embedding matrix. For large datasets with many samples this can be changed to pkl to save space.
    :param save_gene_names: bool, if set to True, the gene names will be saved in a separate text file. This is useful for interpreting the embeddings and attentions later.
    :param heads_to_prune: dict, the attention heads to prune. The format should be {l:[h1,h2,h3],...} where the keys are layer indices and the values are lists of attention head indices
    
    :output: csvs or pkl files containing embeddings or attentions, whichever you ask for. Either way, you get the raw outputs of the model back to use for later analysis
    Note: the model currently saves embeddings to a subfolder called 'hiddens' in the current directory and attentions to a subfolder called 'attns'. Modifying this function to add an output path would probably be desirable, but I'm not doing that right now, so you can deal with it :)
    """
    gene_ids, lengths = data.get_genes()
    # using the Dataloader function, split into batches. 
    dataloader = DataLoader(data, batch_size = batch_size, num_workers = num_workers)
    
    if heads_to_prune is not None:
        model.prune_heads(heads_to_prune)

    k = 0 # batch number
    for sample in tqdm(dataloader, total=len(dataloader)):
        
        # load sample onto correct device (defined above in model loading section)
        inputs = sample.to(device)
    
        # Get the outputs from the model, currently including the hidden outputs and attention
        attention_mask = torch.where(inputs != 0, torch.tensor(1).to(device), torch.tensor(0).to(device)) # torch complains if no mask because of padding issues
        # run the model on your data!
        outputs = model(inputs, attention_mask=attention_mask)
    
        if hiddens is not None:
            for l in hiddens:
                hiddens = outputs.hidden_states[l].detach().numpy()
                for b in range(hiddens.shape[0]): # b is sample id within batch
                    if output_format == "csv":
                        # file format will be batchnum_emb_samplenum_layernum.csv
                        np.savetxt(f'hiddens/{k}_emb_{b}_{l}.csv.gz', hiddens[b, :lengths[b][0], :lengths[b][0]], delimiter=',', fmt='%.16e') # output the embeddings
                    elif output_format == "pkl":
                        with open(f'hiddens/{k}_emb_{b}_{l}.pkl', "wb") as f:
                            pickle.dump(hiddens[b, :lengths[b][0], :lengths[b][0]], f)
                    else:
                        raise Exception("output_format must be either 'csv' or 'pkl'")

        if attns is not None:
            for l in list(attns.keys()): # l iterates over the layers from the attentions list
                attentions = outputs.attentions[l].detach().numpy() # get the attentions from the relevant layer
                for b in range(attentions.shape[0]): # b is sample id within batch
                    for i in attns[l]: # i iterates over the heads from the relevant layer
                        if output_format == "csv":
                            # file format will be "batchnum_'attn'_samplenum_layernum_headnum.csv"
                            np.savetxt(f'attn_matrix/{k}_attn_{b}_{l}_{i}.csv.gz', attentions[b, i, :lengths[b][0], :lengths[b][0]], delimiter=',', fmt='%.16e') # output the attention head 
                        elif output_format == "pkl":
                            with open(f'attn_matrix/{k}_attn_{b}_{l}_{i}.pkl', "wb") as f:
                                pickle.dump(attentions[b, i, :lengths[b][0], :lengths[b][0]], f)
                        else:
                            raise Exception("output_format must be either 'csv' or 'pkl'")
        if save_gene_names == True:
            for b in range(batch_size):
                idx = k*batch_size + b # we are looking for the bth sample in the kth batch
                with open(f'gene_names/{k}_gene_names_{b}.txt', "w") as f:
                    for gene in gene_ids[idx]:
                        f.write(gene + "\n")
        k+=1
    return outputs

# runs samples through a given model and aggregates attentions over multiple samples
def aggregate_attentions(data, model, batch_size, num_workers = 1, attns = None, aggregation_method = "mean", output_format = "csv", save_gene_names = False, heads_to_prune = None):
    """
    :param data: dataset, output by the GFDataset class. It should already be padded but not passed to the dataloader
    :param model: the model loaded by the load_model function
    :param batch_size: int, size of geneformer batch. In general larger batch size  = faster but more memory intensive. With 1 GPU and 128 GB of memory the batch size should usually be kept under 10, but you should optimize for your application
    :param num_workers: int, the number of processes to run simultaneously. not a parameter we have experimented much with, but it can be used to optimize runtime 
    :param attns: dict, the attention heads to aggregate. The format should be {l:[h1,h2,h3],...} where the keys are layer indices and the values are lists of attention head indices.
        If you would like to aggregate over two separate sets of heads, you can pass a list of dicts, where each dict is a different set of heads to aggregate.
    :param aggregation_method: str, the method to use for aggregating the attention heads. Currently only "mean" and "max" are supported, where max takes the maximum weight between every pair of genes over all available matrices.
    :param output_format: str, must be either csv or pkl. Defaults to csv, which outputs a gzipped csv for each attention head or embedding matrix. For large datasets with many samples this can be changed to pkl to save space.
    :param save_gene_names: bool, if set to True, the gene names will be saved in a separate text file. This is useful for interpreting the embeddings and attentions later.
    :param heads_to_prune: dict, the attention heads to prune. The format should be {l:[h1,h2,h3],...} where the keys are layer indices and the values are lists of attention head indices
    
    :output: csvs or pkl files containing embeddings or attentions, whichever you ask for. Either way, you get the raw outputs of the model back to use for later analysis
    Note: the model currently saves embeddings to a subfolder called 'hiddens' in the current directory and attentions to a subfolder called 'attns'. Modifying this function to add an output path would probably be desirable, but I'm not doing that right now, so you can deal with it :)
    """
    gene_ids, lengths = data.get_genes() # TODO check the format of this
    # using the Dataloader function, split into batches. 
    dataloader = DataLoader(data, batch_size = batch_size, num_workers = num_workers)
    
    if heads_to_prune is not None:
        model.prune_heads(heads_to_prune)

    k = 0 # batch number
    if type(attns) != list:
        attns = [attns] # if attns is not a list, make it a list so we can iterate over it (prevents me having to put the same block of code twice)
    
    matrices_to_aggregate = [[] for i in range(len(attns))] # list of lists of matrices to aggregate, one list for each set of heads in attns
    gene_name_lists = [[] for i in range(len(attns))] # list of lists of gene names, one list for each set of heads in attns

    for sample in tqdm(dataloader, total=len(dataloader)):
        # load sample onto correct device (defined above in model loading section)
        inputs = sample.to(device)
    
        # Get the outputs from the model, currently including the hidden outputs and attention
        attention_mask = torch.where(inputs != 0, torch.tensor(1).to(device), torch.tensor(0).to(device)) # torch complains if no mask because of padding issues
        # run the model on your data!
        outputs = model(inputs, attention_mask=attention_mask)
    
        for a,attn_set in enumerate(attns):
            # count how many total values are in the attention set. This will be useful later for correctly matching gene names with samples
            total_heads = sum([len(attn_set[l]) for l in list(attn_set.keys())])
            for l in list(attn_set.keys()): # iterate over the relevant layers
                attentions = outputs.attentions[l].detach().numpy() # get the attentions from the relevant layer
                for b in range(attentions.shape[0]): # b is sample id within batch
                    for i in attn_set[l]: # i iterates over the heads from the relevant layer
                        # add the attention matrix to the list of matrices to aggregate
                        matrices_to_aggregate[a].append(attentions[b, i, :lengths[b][0], :lengths[b][0]])

            for b in range(batch_size):
                idx = k*batch_size + b # we are looking for the bth sample in the kth batch
                # this is a dumb and hacky way to do this but it will make our lives easier later. We want every matrix index in matrices_to_aggregate to correspond to its correct list of gene names in gene_name_lists
                for t in range(total_heads):
                    gene_name_lists[a].append(gene_ids[idx]) # append the list of gene names to the larger list so we can index with them later
                if save_gene_names == True:
                    with open(f'gene_names/{k}_gene_names_{b}.txt', "w") as f:
                        for gene in gene_ids[idx]:
                            f.write(gene + "\n")
        k+=1
    #first we iterate over sets (s) of attention heads, because I can't rule out that someone will be a pain in the butt and want to do something weird like aggregate heads layer 1 and layer 5 separately (its me I'm someone)
    for s in range(len(matrices_to_aggregate)):
        # make a set of unique gene names from all of the sublists in list s of gene_name_lists
        gene_names = list(set([gene for sublist in gene_name_lists[s] for gene in sublist]))
        # make an NxN matrix of zeros where N is the number of unique genes
        gene_matrix = np.zeros((len(gene_names), len(gene_names)))
        if aggregation_method == "mean":
            # make another matrix to count the number of times each gene is aggregated
            gene_count = np.zeros((len(gene_names), len(gene_names)))
        # make a dictionary of gene names to indices
        gene_dict = {gene:idx for idx,gene in enumerate(gene_names)}
        for gene in gene_names:
            i = gene_dict[gene]
            for m,mat in enumerate(matrices_to_aggregate[s]):
                # if gene is not in the gene name list for a given set of heads, skip it
                if gene not in gene_name_lists[s][m]:
                    continue
                else:
                    # the structure we will follow here is that indices i, j will correspond to gene and target_gene in the overall gene_matrix
                    # k and l will correspond to gene and target_gene in the individual matrices
                    # So first, get the index of gene in mat and assign it to k
                    k = gene_name_lists[s][m].index(gene)
                    # now iterate over the indices of mat, skipping l if it is the same as k
                    for l in range(len(mat)):
                        if l == k:
                            continue
                        else:
                            # get the target gene name
                            target_gene = gene_name_lists[s][m][l]
                            # get the index of the target gene in the overall gene_matrix
                            j = gene_dict[target_gene]
                            # if the aggregation method is mean, add the value to the gene_matrix and increment the count
                            if aggregation_method == "mean":
                                gene_matrix[i, j] += mat[k,l]
                                gene_count[i, j] += 1
                            # if the aggregation method is max, take the max of the current value and the value in the matrix
                            elif aggregation_method == "max":
                                gene_matrix[i, j] = max(gene_matrix[i, j], mat[k,l])
                            else:
                                raise Exception("aggregation_method must be either 'mean' or 'max'")
                            
        if aggregation_method == "mean":
            gene_matrix = gene_matrix/gene_count # if we are using the mean, divide by the count to get the mean
        if output_format == "csv":
            np.savetxt(f'attns/aggregated_attns.csv.gz', gene_matrix, delimiter=',', fmt='%.16e')
        elif output_format == "pkl":
            with open(f'attns/aggregated_attns.pkl', "wb") as f:
                pickle.dump(gene_matrix, f)
        else:
            raise Exception("output_format must be either 'csv' or 'pkl'")
    return gene_matrix

###############################################################################################
# The next section of the script contains functions that are useful for fine tuning geneformer
# Fine tuning is much more task specific than the rest of the guide, so these functions will likely 
# not be useful for your specific application. They are included to give an overview of the process
# and to show some general functions that can be used across different tasks
# Two notable tasks that are not covered here but are included in the geneformer huggingface are cell annotation,
# which sorts cells by tissue before annotating them with a type, and gene classification, which classifies TFs
# as dosage sensitive or insensitive. Because of the added complexity of both of these processes they
# are not covered here. a 
###############################################################################################

def preprocess_for_cell_classification(data, name_of_labels_column = "label"):
    """
    :param data: dataset, output by the GFDataset class. It should already be padded but not passed to the dataloader
    :param name_of_labels_column: str, the name of the column in data that contains the target variable

    :return labeled_train_split, labeled_eval_split: a HuggingFace Dataset object with the tokenized data, split into train and eval sets
    :return target_name_id_dict: dict, a dictionary of label : label id so we can use numerical class ids
    """
    # get the number of unique labels
    num_labels = len(set(list(data[name_of_labels_column])))
    # shuffle datasets and rename columns
    data_shuffled = data.shuffle(seed=42)
    
    # create dictionary of label : label id so we can use numerical class ids
    target_names = list(Counter(data_shuffled[name_of_labels_column]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    
    # change labels to numerical ids
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    labeled_trainset = data_shuffled.map(classes_to_ids, num_proc=16)
    
    # create 80/20 train/eval splits
    labeled_train_split = labeled_trainset.select([i for i in range(0,round(len(labeled_trainset)*0.8))])
    labeled_eval_split = labeled_trainset.select([i for i in range(round(len(labeled_trainset)*0.8),len(labeled_trainset))])
    return labeled_train_split, labeled_eval_split, target_name_id_dict


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }

def optimize_hyperparams(model_init, classifier_trainset, classifier_evalset, output_dir, geneformer_batch_size, epochs, logging_steps = 10):
    """
    This is a function to optimize hyperparameters
    :param model_init: the model loaded by the load_model function
    :param classifier_trainset: the training dataset, from the preprocess_for_cell_classification function
    :param classifier_evalset: the evaluation dataset, from the preprocess_for_cell_classification function (or a different function, if that is not the task you're performing)
    :param output_dir: str, the path to the directory where the model will be saved
    :param geneformer_batch_size: int, the batch size for geneformer. Defaults to 12
    :param epochs: int, the number of epochs to train for. Defaults to 10, which is the number of epochs used by the authors
    :param logging_steps: int, the number of steps to log. Defaults to 10, which is the number of steps used by the authors

    :return best_hyperparameters: dict, the best hyperparameters found by the hyperparameter search
    """

    # set initial training arguments
    training_args = {
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "steps",
        "eval_steps": logging_steps,
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": True,
        "skip_memory_metrics": True, # memory tracker causes errors in raytune
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
    }

    training_args_init = TrainingArguments(**training_args)

    # create the trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=classifier_trainset,
        eval_dataset=classifier_evalset,
        compute_metrics=compute_metrics,
    )

    # specify raytune hyperparameter search space
    ray_config = {
        "num_train_epochs": tune.choice([epochs]),
        "learning_rate": tune.loguniform(1e-6, 1e-3),
        "weight_decay": tune.uniform(0.0, 0.3),
        "lr_scheduler_type": tune.choice(["linear","cosine","polynomial"]),
        "warmup_steps": tune.uniform(100, 2000),
        "seed": tune.uniform(0,100),
        "per_device_train_batch_size": tune.choice([geneformer_batch_size])
    }

    hyperopt_search = HyperOptSearch(
        metric="eval_accuracy", mode="max")

    # optimize hyperparameters
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        resources_per_trial={"cpu":8,"gpu":1},
        hp_space=lambda _: ray_config,
        search_alg=hyperopt_search,
        n_trials=100, # number of trials
        progress_reporter=tune.CLIReporter(max_report_frequency=600,
                                                    sort_by_metric=True,
                                                    max_progress_rows=100,
                                                    mode="max",
                                                    metric="eval_accuracy",
                                                    metric_columns=["loss", "eval_loss", "eval_accuracy"])
    )

    best_hyperparameters = best_trial.hyperparameters

    print("Best Hyperparameters:")
    print(best_hyperparameters)
    return best_hyperparameters

def fine_tune_cell_classifier(trainset,evalset,model,output_dir,optimize_hyper = False, freeze_layers = 0, geneformer_batch_size = 12, lr_schedule_fn = "linear", warmup_steps = 500, epochs = 10, optimizer = "adamw", logging_steps = 10):
    """
    This function fine tunes the geneformer model for cell classification
    :param trainset: training dataset, from the preprocess_for_cell_classification function
    :param evalset: evaluation dataset, from the preprocess_for_cell_classification function
    :param model: the geneformer model loaded by the load_model function. calling load_model(task = "cell") will load the correct model for this function (it will load as a binary classifier)
    :param output_dir: str, the path to the directory where the model will be saved
    :param optimize_hyper: bool, if set to True, the function will optimize hyperparameters using the optimize_hyperparams function. Defaults to False

    The following are hyperparameters for training, which can be optimized by the optimize_hyperparams function
    :param freeze_layers: int, the number of layers to freeze. Defaults to 0, which means no layers are frozen
    :param geneformer_batch_size: int, the batch size for geneformer. Defaults to 12, which is the batch size used by the authors
    :param lr_schedule_fn: str, the learning rate schedule function. Defaults to "linear", which is the schedule used by the authors
    :param warmup_steps: int, the number of warmup steps. Defaults to 500, which is the number of warmup steps used by the authors
    :param epochs: int, the number of epochs to train for. Defaults to 10, which is the number of epochs used by the authors
    :param optimizer: str, the optimizer to use. Defaults to "adamw", which is the optimizer used by the authors
    :param logging_steps: int, the number of steps to log. Defaults to 10, which is the number of steps used by the authors
    """
    max_input_size = 2 ** 11  # 2048
    max_lr = 5e-5
    num_gpus = 1
    
    # define output directory path
    output_dir = f"{output_dir}/{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"
    
    # ensure not overwriting previously saved model
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")

    # make output directory
    subprocess.call(f'mkdir {output_dir}', shell=True)
    
    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
    }
    
    training_args_init = TrainingArguments(**training_args)
    if optimize_hyper == True:
        best_hyperparameters = optimize_hyperparams(model, trainset, evalset, output_dir, geneformer_batch_size, epochs, logging_steps)
        # update training arguments with best hyperparameters
        training_args_init.learning_rate = best_hyperparameters["learning_rate"]
        training_args_init.lr_scheduler_type = best_hyperparameters["lr_scheduler_type"]
        training_args_init.warmup_steps = best_hyperparameters["warmup_steps"]
        training_args_init.num_train_epochs = best_hyperparameters["num_train_epochs"]
        training_args_init.per_device_train_batch_size = best_hyperparameters["per_device_train_batch_size"]
    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=trainset,
        eval_dataset=evalset,
        compute_metrics=compute_metrics
    )
    # train the cell classifier
    trainer.train()
    predictions = trainer.predict(evalset)
    with open(f"{output_dir}/predictions.pickle", "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval",predictions.metrics)
    trainer.save_model(output_dir)


if __name__ == "__main__":

    # The following lines are useful for configuring your environment to use any available GPUs
    GPU_NUMBER = [i for i in range(torch.cuda.device_count())] 
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
    os.environ["NCCL_DEBUG"] = "INFO"
    # sets the device by checking is gpus are available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = GFDataset(dataset = None, n = 10)
    # Load the model
    model = load_model(task = "general", attns = True, hiddens = True)
    # Run the samples
    run_samples(data, model, batch_size = 5, num_workers = 1, attns = {1:[0,1],2:[0]}, hiddens = [1,2], output_format = "csv", save_gene_names = True, heads_to_prune = {3:[0,1],4:[0]})


