import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertForTokenClassification, BertModel
import geneformer
from gprofiler import GProfiler
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
import networkx as nx

def get_disease_genes_all_info(name, gda):
    df = gda.query("NewName==@name & (Strong >=1 or Weak >= 1)")
    return df

def get_disease_gene_names(name,gda):
    df = get_disease_genes_all_info(name,gda)
    return set(df["HGNC_Symbol"])

def load(file_name):
    with open(file_name,'rb') as file:
        obj = pickle.load(file)
    return obj

def save(obj, filename):
    with open(filename,'wb') as file:
        pickle.dump(obj,file)

def glimpse(obj, sample_size=5):
    print(f"Type: {type(obj)}")
    if isinstance(obj, (list, tuple, set)):
        print(f"Sample: {list(obj)[:sample_size]}")
    elif isinstance(obj, dict):
        keys = list(obj.keys())[:sample_size]
        sample = {key: obj[key] for key in keys}
        print(f"Sample: {sample}")
    elif hasattr(obj, 'head'):  # For pandas DataFrame or Series
        print(f"Sample: {obj.head(sample_size)}")
    else:
        print(f"Representation: {repr(obj)}")

class GFDataset(Dataset):

    def __init__(self, dataset, path='/work/ccnr/GeneFormer/Sample.dataset/', n=None, samples=None, tissues=None, max_length=2048):
        """
        A class to load in data for Geneformer and prepare it for use
        
        Parameters:
        ----------
        dataset : None or Dataset
            should usually be set to None, unless you have a dataset already loaded from a .dataset folder with .arrow files in it.
        path : str 
            path to the dataset if the dataset is different from the sample dataset (note that different datasets will have different metadata included)
        n : int
            number of samples to select. Either n or samples should always be set to None to avoid conflicts.
        samples : list or "all" or None
            specific samples to select, if known OR "all" for all the samples in the dataset
        
        Returns:
        -------
        subset : Dataset
            The dataset, padded with zeros out to the maximum length of 2048. It can now be passed to the Dataloader
        
        Example usage:
        
        data = GFDataset(dataset = None, n = 10) # selects 10 random samples from the sample dataset
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
            random_indices = np.random.choice([i for i in range(len(data))],size=self.n)
            subset = data.select(list(random_indices))
        return subset

    def get_genes(self, convert_from_ensembl = False):
        """
        Function to retrieve gene ids and sample lengths
        
        Parameters:
        ----------
        convert_from_ensembl : boolean 
            if set to True, the function will attempt to convert the ensembl IDs to gene names. If the conversion is not possible, the ensembl ID will be left in the list

        Returns:
        ------- 
        ids_list : list of lists
            list of lists where each list corresponds to a sample and contains the genes in the order they appear in the sample
        length_list : list of lists
            list of lists where each list corresponds to a sample and contains the length of that sample (the fact that this is a list of lists is essentially an oversight that I'm too afraid to remove in case it bungles everything else)

        Example:
        ------- 
        genes = data.get_genes(convert_from_ensembl = True) 
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
                ids_list.append(new_gene_ids)
                length_list.append(self.dataset.select([ii])["length"])
        return ids_list, length_list



# loads the model from the pretrained weights. 
def load_model(task, path, n_labels = 2, device= "cpu", attns=False, hiddens=False):
    """
    Function to load a Geneformer model
    Parameters:
    ----------
    task : str
        The task the model is being loaded for. Should be "general" for looking at attentions or embeddings, "gene" for a gene classification task, and "cell" for a cell classification task
    path : str
        path to model
    n_labels : int 
        The number of classes in the data. Should be set to 3 for the cardiomyopathy model (dilated, hypertrophic, or healthy). Can be set to 2 for the pretrained model
    device : str
        "cuda" if using gpus, "cpu" if not 
    attns : bool 
        If set to true, attention heads from any layer can be extracted
    hiddens : bool 
        If set to true, embeddings from any layer can be extracted
    
    Returns:
    -------  
    model : nn.Module
        a geneformer model, loaded to device
    
    Example:
    ------- 
    model = load_model(task = "general",path = "/work/ccnr/GeneFormer/conda_environment", attns = True) # loads the general pretrained model to output attention heads
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
    
def aggregate_attentions(data, model, layers = 4, heads_to_prune = None, batch_size = 20, device = "cuda", num_workers = 1, method = "avg"):
    """
    Aggregates the attention weights from a Geneformer model over a dataset.

    Parameters:
    ----------
    data : GFDataset
        The dataset, output by the GFDataset class. It should already be padded but not passed to the dataloader.
    model : nn.Module
        The transformer model loaded by the load_model function.
    layers : int (default=4)
        The layer to analyze to take weights from. Defaults to 4
    heads_to_prune : dict, optional
        The attention heads to prune. The format should be {l: [h1, h2, h3], ...} where the keys are layer indices and
        the values are lists of attention head indices.
    batch_size : int, optional (default=20)
        Size of geneformer batch. In general, larger batch size = faster but more memory intensive.
        With 1 GPU and 128 GB of memory, the batch size should usually be kept under 10, but you should optimize for your application.
    device : str, optional (default="cuda")
        The torch device to use, either "cuda" or "cpu".
    num_workers : int, optional (default=1)
        The number of processes to run simultaneously. Not a parameter we have experimented much with, but it can be used to optimize runtime.
    method : str, optional (default="avg")
        The method to use for aggregating attention weights. Can be either "avg" for averaging or "max" for taking the maximum.

    Returns:
    -------
    max_mat : numpy.ndarray
        A matrix where element (i, j) contains the aggregated attention weight between gene i and gene j.
    counts_mat : numpy.ndarray
        A matrix where element (i, j) contains the number of times gene i and gene j co-occur in a sample.
    gene_index_dict : dict
        A dictionary mapping unique gene IDs to their corresponding indices in the matrix.

    Example:
    --------
    data = GFDataset(dataset=None, n=10)  # Selects 10 samples from the sample dataset
    model = load_model(...)  # Load your transformer model
    max_mat, gene_index_dict = aggregate_attentions(data, model)
    """

    gene_ids, lengths = data.get_genes(convert_from_ensembl = False) # the format is just one long list of length n_samples but each entry is a one entry list
    # make the list of unique gene ids
    unique_gene_ids = []
    for ls in gene_ids:
        for item in ls:
            if item not in unique_gene_ids:
                unique_gene_ids.append(item)
    gene_index_dict = {unique_gene_ids[i]:i for i in range(len(unique_gene_ids))}
    
    # values_mat = np.zeros((len(unique_gene_ids),len(unique_gene_ids)))
    max_mat = np.zeros((len(unique_gene_ids),len(unique_gene_ids)))
    counts_mat = np.zeros((len(unique_gene_ids),len(unique_gene_ids)))
    
    # using the Dataloader function, split into batches. 
    dataloader = DataLoader(data, batch_size = batch_size, num_workers = num_workers)

    k = 0
    if heads_to_prune is not None:
        model.prune_heads(heads_to_prune)

    for batch in tqdm(dataloader, total=len(dataloader)):
        # load sample onto correct device (defined above in model loading section)
        inputs = batch.to(device)
        # Get the outputs from the model, currently including the hidden outputs and attention
        attention_mask = torch.where(inputs != 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
        # run the model on your data!
        outputs = model(inputs, attention_mask=attention_mask)
        # get the number of samples in the batch
        n_samples = inputs.shape[0]
        for sample in range(n_samples):
            idx = k*batch_size + sample
            genes = gene_ids[idx]
            attention = outputs.attentions[layers][sample].cpu().detach().numpy()
            # Normalize by the length of the sample
            attention = np.array([attn*(lengths[idx][0]/2048) for attn in attention])
            # attention = np.array([(m - np.min(m)) / (np.max(m) - np.min(m)) for m in attention])
            gene_indices = np.array([gene_index_dict[gene] for gene in genes])
            for i in range(len(genes)):
                # we only have to iterate over the upper triangle if we aren't looking at mean aggregation
                ra = range(i,len(genes)) if method=='max' else range(len(genes))
                for j in ra:
                    # Get the indices for genes in the gene_index_dict
                    ind1, ind2 = gene_indices[i], gene_indices[j]
                    
                    # Compute the sum of attention matrices for each pair of genes
                    # values_mat[ind1, ind2] += np.sum(attention[:, i, j])
                    
                    
                    # Update max_mat with the maximum value from attention matrices
                    if method=='max':
                        update = np.max(np.maximum(max_mat[ind1, ind2], attention[:, i, j], attention[:, j, i]))
                        max_mat[ind1, ind2] = update
                        max_mat[ind2, ind1] = update
                        
                        counts_mat[ind1,ind2] += 1.0                            
                        counts_mat[ind2,ind1] += 1.0
                    else:
                        ab = 0
                        ba = 0
                        for h in range(4):
                            ab += attention[h,i,j]
                            ba += attention[h,j,i]
                        ab = ab/4
                        ba = ba/4

                        max_mat[ind1,ind2] += ab
                        max_mat[ind2,ind1] += ba

                        counts_mat[ind1,ind2] += 1.0                            
                        counts_mat[ind2,ind1] += 1.0
        k+=1

        if method!='max':
            with np.errstate(divide='ignore', invalid='ignore'):
                max_mat = np.divide(max_mat, counts_mat)
                max_mat[~np.isfinite(max_mat)] = 0  # replace inf and nan with 0
        else:
            for i in range(len(unique_gene_ids)):
                for j in range(i,len(unique_gene_ids)):
                    m = max(max_mat[i,j], max_mat[j,i])

                    max_mat[i,j] = m
                    max_mat[j,i] = m

    return max_mat, counts_mat, gene_index_dict
    
    
    
def aggregate_embeddings(data, model, out_type, layers = 4, heads_to_prune = None, batch_size = 20, device = "cuda", num_workers = 1, method = "avg"):
    """
    Aggregates the attention weights from a Geneformer model over a dataset.

    Parameters:
    ----------
    data : GFDataset
        The dataset, output by the GFDataset class. It should already be padded but not passed to the dataloader.
    model : nn.Module
        The transformer model loaded by the load_model function.
    layers : int or list of int, optional (default=4)
        The layers to analyze. If an integer is provided, it will analyze the specified layer.
        If a list is provided, it will analyze all specified layers.
    heads_to_prune : dict, optional
        The attention heads to prune. The format should be {l: [h1, h2, h3], ...} where the keys are layer indices and
        the values are lists of attention head indices.
    batch_size : int, optional (default=20)
        Size of geneformer batch. In general, larger batch size = faster but more memory intensive.
        Using 4 cpus and 256 GB of memory, we find that a larger batch size is generally better, with diminishing returns after a batch size of about 50 and a memory limit around 200.
    device : str, optional (default="cuda")
        The torch device to use, either "cuda" or "cpu".
    num_workers : int, optional (default=1)
        The number of processes to run simultaneously. We use 4 with a batch size of 50.
    method : str, optional (default="avg")
        The method to use for aggregating attention weights. Can be either "avg" for averaging or "max" for taking the maximum.

    Returns:
    -------
    max_mat : numpy.ndarray
        A matrix where element (i, j) contains the aggregated attention weight between gene i and gene j.
    gene_index_dict : dict
        A dictionary mapping unique gene IDs to their corresponding indices in the matrix.

    Example:
    --------
    data = GFDataset(dataset=None, n=10)  # Selects 10 samples from the sample dataset
    model = load_model(...)  # Load your transformer model
    max_mat, gene_index_dict = aggregate_attentions(data, model)
    """

    gene_ids, lengths = data.get_genes(convert_from_ensembl = False) # the format is just one long list of length n_samples but each entry is a one entry list
    # make the list of unique gene ids
    unique_gene_ids = []
    for ls in gene_ids:
        for item in ls:
            if item not in unique_gene_ids:
                unique_gene_ids.append(item)
    gene_index_dict = {unique_gene_ids[i]:i for i in range(len(unique_gene_ids))}
    
    # values_mat = np.zeros((len(unique_gene_ids),len(unique_gene_ids)))
    max_mat = np.zeros((len(unique_gene_ids),len(unique_gene_ids)))
    counts_mat = np.zeros((len(unique_gene_ids),len(unique_gene_ids)))
    
    # using the Dataloader function, split into batches. 
    dataloader = DataLoader(data, batch_size = batch_size, num_workers = num_workers)

    k = 0
    if heads_to_prune is not None:
        model.prune_heads(heads_to_prune)
        
    for batch in tqdm(dataloader, total=len(dataloader)):
        # load sample onto correct device (defined above in model loading section)
        inputs = batch.to(device)
        # Get the outputs from the model, currently including the hidden outputs and attention
        attention_mask = torch.where(inputs != 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
        if layer == 'input':
            hiddens = inputs
        else:
            # run the model on your data!
            outputs = model(inputs, attention_mask=attention_mask)
            hiddens = outputs.hidden_states[layers].cpu().detach().numpy()
        # get the number of samples in the batch
        n_samples = inputs.shape[0]
        for sample in range(n_samples):
            idx = k*batch_size + sample
            genes = gene_ids[idx]
            embeds = hiddens[sample, :lengths[idx][0], :lengths[idx][0]]
            gene_indices = np.array([gene_index_dict[gene] for gene in genes])
            for i in range(len(genes)):
                for j in range(i, len(genes)):
                    # Get the indices for genes in the gene_index_dict
                    ind1, ind2 = gene_indices[i], gene_indices[j]
                    print(embeds[i].shape,embeds[j].shape)
                    # get the dot product of the embeddings
                    unnormalized = np.dot(embeds[i], embeds[j])
                    normalization = np.linalg.norm(embeds[i]) * np.linalg.norm(embeds[j])
                    cos_sim = unnormalized / normalization
                    
                    if method=='max':
                        max_mat[ind1, ind2] = max(cos_sim, max_mat[ind1, ind2])
                        max_mat[ind2, ind1] = max(cos_sim, max_mat[ind2, ind1])
                    else:
                        max_mat[ind1, ind2] += cos_sim
                        max_mat[ind2, ind1] += cos_sim
                    
                    counts_mat[ind1, ind2] += 1
                    counts_mat[ind2, ind1] += 1
                                
            k+=1

        if method == 'avg':
            with np.errstate(divide='ignore', invalid='ignore'):
                max_mat = np.divide(max_mat, counts_mat)
                max_mat[~np.isfinite(max_mat)] = 0  # replace inf and nan with 0

    return max_mat, counts_mat, gene_index_dict
    
    
def aggregate_attentions(data, model, layers = 4, heads_to_prune = None, batch_size = 20, device = "cuda", num_workers = 1, method = "avg"):
    """
    Aggregates the attention weights from a Geneformer model over a dataset.

    Parameters:
    ----------
    data : GFDataset
        The dataset, output by the GFDataset class. It should already be padded but not passed to the dataloader.
    model : nn.Module
        The transformer model loaded by the load_model function.
    layers : int (default=4)
        The layer to analyze to take weights from. Defaults to 4
    heads_to_prune : dict, optional
        The attention heads to prune. The format should be {l: [h1, h2, h3], ...} where the keys are layer indices and
        the values are lists of attention head indices.
    batch_size : int, optional (default=20)
        Size of geneformer batch. In general, larger batch size = faster but more memory intensive.
        Using 4 cpus and 256 GB of memory, we find that a larger batch size is generally better, with diminishing returns after a batch size of about 50 and a memory limit around 200.
    device : str, optional (default="cuda")
        The torch device to use, either "cuda" or "cpu".
    num_workers : int, optional (default=1)
        The number of processes to run simultaneously. We use 4.
    method : str, optional (default="avg")
        The method to use for aggregating attention weights. Can be either "avg" for averaging or "max" for taking the maximum.

    Returns:
    -------
    max_mat : numpy.ndarray
        A matrix where element (i, j) contains the aggregated attention weight between gene i and gene j.
    counts_mat : numpy.ndarray
        A matrix where element (i, j) contains the number of times gene i and gene j co-occur in a sample.
    gene_index_dict : dict
        A dictionary mapping unique gene IDs to their corresponding indices in the matrix.

    Example:
    --------
    data = GFDataset(dataset=None, n=10)  # Selects 10 samples from the sample dataset
    model = load_model(...)  # Load your transformer model
    max_mat, gene_index_dict = aggregate_attentions(data, model)
    """

    gene_ids, lengths = data.get_genes(convert_from_ensembl = False) # the format is just one long list of length n_samples but each entry is a one entry list
    # make the list of unique gene ids
    unique_gene_ids = []
    for ls in gene_ids:
        for item in ls:
            if item not in unique_gene_ids:
                unique_gene_ids.append(item)
    gene_index_dict = {unique_gene_ids[i]:i for i in range(len(unique_gene_ids))}
    
    # values_mat = np.zeros((len(unique_gene_ids),len(unique_gene_ids)))
    max_mat = np.zeros((len(unique_gene_ids),len(unique_gene_ids)))
    counts_mat = np.zeros((len(unique_gene_ids),len(unique_gene_ids)))
    
    # using the Dataloader function, split into batches. 
    dataloader = DataLoader(data, batch_size = batch_size, num_workers = num_workers)

    k = 0
    if heads_to_prune is not None:
        model.prune_heads(heads_to_prune)

    for batch in tqdm(dataloader, total=len(dataloader)):
        # load sample onto correct device (defined above in model loading section)
        inputs = batch.to(device)
        # Get the outputs from the model, currently including the hidden outputs and attention
        attention_mask = torch.where(inputs != 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
        # run the model on your data!
        outputs = model(inputs, attention_mask=attention_mask)
        # get the number of samples in the batch
        n_samples = inputs.shape[0]
        for sample in range(n_samples):
            idx = k*batch_size + sample
            genes = gene_ids[idx]
            attention = outputs.attentions[layers][sample].cpu().detach().numpy()
            # Normalize by the length of the sample
            attention = np.array([attn*(lengths[idx][0]/2048) for attn in attention])
            # attention = np.array([(m - np.min(m)) / (np.max(m) - np.min(m)) for m in attention])
            gene_indices = np.array([gene_index_dict[gene] for gene in genes])
            for i in range(len(genes)):
                # we only have to iterate over the upper triangle if we aren't looking at mean aggregation
                ra = range(i,len(genes)) if method=='max' else range(len(genes))
                for j in ra:
                    # Get the indices for genes in the gene_index_dict
                    ind1, ind2 = gene_indices[i], gene_indices[j]
                    
                    # Compute the sum of attention matrices for each pair of genes
                    # values_mat[ind1, ind2] += np.sum(attention[:, i, j])
                    
                    
                    # Update max_mat with the maximum value from attention matrices
                    if method=='max':
                        update = np.max(np.maximum(max_mat[ind1, ind2], attention[:, i, j], attention[:, j, i]))
                        max_mat[ind1, ind2] = update
                        max_mat[ind2, ind1] = update
                        
                        counts_mat[ind1,ind2] += 1.0                            
                        counts_mat[ind2,ind1] += 1.0
                    else:
                        ab = 0
                        ba = 0
                        for h in range(4):
                            ab += attention[h,i,j]
                            ba += attention[h,j,i]
                        ab = ab/4
                        ba = ba/4

                        max_mat[ind1,ind2] += ab
                        max_mat[ind2,ind1] += ba

                        counts_mat[ind1,ind2] += 1.0                            
                        counts_mat[ind2,ind1] += 1.0
        k+=1

        if method!='max':
            with np.errstate(divide='ignore', invalid='ignore'):
                max_mat = np.divide(max_mat, counts_mat)
                max_mat[~np.isfinite(max_mat)] = 0  # replace inf and nan with 0
        else:
            for i in range(len(unique_gene_ids)):
                for j in range(i,len(unique_gene_ids)):
                    m = max(max_mat[i,j], max_mat[j,i])

                    max_mat[i,j] = m
                    max_mat[j,i] = m

    return max_mat, counts_mat, gene_index_dict
    
    
def replace_with_nans(int_matrix, float_matrix):
    """
    Function to replace 0s in an aggregated weight matrix with nans, so that true 0's can be separated from 0s that only occur because gene i and j were never coexpressed
    
    Parameters:
    ----------
    int_matrix : np.ndarray
        A counts matrix saying how many time each combination appeared
    float_matrix : np.ndarray
        An aggregated weight matrix, which must correspond to the count matrix (be indexed the same and have the same shape)
    
    Returns:
    --------
    float_matrix : np.ndarray
        The aggregated weight matrices, where gene indices [i,j] that have not co-occured in the data have their 0 weight replaced by np.nan
    
    """
    # find the indices of the 0 elements in the integer matrix
    indices = np.where(int_matrix == 0)
    # replace the corresponding indices in the float matrix with nans
    float_matrix[indices] = np.nan
    return float_matrix
    
def non_degree_preserving_randomization(graph):
    """
    Perform non-degree preserving randomization on a graph.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph object to be randomized.

    Returns
    -------
    randomized : nx.Graph
        The non-degree-preserving randomized graph.
    """
    nodes = list(graph.nodes())
    randomized = nx.gnm_random_graph(graph.number_of_nodes(), graph.number_of_edges())
    randomized = nx.relabel_nodes(randomized, {i: nodes[i] for i in range(len(nodes))})
    return randomized


def get_background(matrix, indices_used):
    """
    Retrieve background attention values from a matrix, excluding specified indices.

    Parameters
    ----------
    matrix : np.ndarray
        2D numpy array representing the attention values.
    indices_used : list of tuples
        List of index pairs to exclude from the output.

    Returns
    -------
    new_ls : np.ndarray
        Flattened 1D array of remaining attention values after excluding specified indices.
    """
    new_matrix = matrix.copy()
    for ind in indices_used:
        new_matrix[ind[0], ind[1]] = np.nan
    new_ls = new_matrix.flatten()
    new_ls = new_ls[~np.isnan(new_ls)]
    return new_ls


def get_edge_weights(edgelist, matrix, gene_index_dict):
    """
    Extract attention values for edges in a list based on a given matrix.

    Parameters
    ----------
    edgelist : list of tuples
        List of edges (node pairs) to get attention values for.
    matrix : np.ndarray
        2D numpy array containing attention values.
    gene_index_dict : dict
        Dictionary mapping nodes to their respective indices in the matrix.

    Returns
    -------
    edge_attns : list of float
        List of attention values for each edge in the edgelist.
    """
    edge_attns = []
    for edge in edgelist:
        ind1 = gene_index_dict[edge[0]]
        ind2 = gene_index_dict[edge[1]]
        if np.isnan(matrix[ind1, ind2]) == False:
            edge_attns.append(matrix[ind1, ind2])
    return edge_attns


def get_graph_weights(ppi_sub, pretrained_max,gene_index_dict):
    """
    Retrieve attention values for edges in a subgraph and track the indices used.

    Parameters
    ----------
    ppi_sub : nx.Graph
        Subgraph for which attention values are to be retrieved.
    pretrained_max : np.ndarray
        2D numpy array containing maximum attention values.

    Returns
    -------
    ppi_attns : list of float
        List of attention values for each edge in the subgraph.
    indices_used : list of lists
        List of index pairs (i, j) used in retrieving the attention values.
    """
    indices_used = []
    ppi_attns = []
    for edge in list(ppi_sub.edges()):
        ind1 = gene_index_dict[edge[0]]
        ind2 = gene_index_dict[edge[1]]
        if np.isnan(pretrained_max[ind1, ind2]) == False:
            ppi_attns.append(pretrained_max[ind1, ind2])
        indices_used.append([ind1, ind2])
    return ppi_attns, indices_used


def degree_weights(ppi_sub, pretrained_max, gene_index_dict, gene_list=None):
    """
    Calculate attention values and degrees for specified nodes in a graph.

    Parameters
    ----------
    ppi_sub : nx.Graph
        Subgraph in which to calculate attention values and degrees.
    pretrained_max : np.ndarray
        2D numpy array with attention values.
    gene_index_dict : dict
        Dictionary mapping nodes to their indices in the attention matrix.
    gene_list : list of str, optional
        List of nodes for which to compute values. If None, uses all nodes in ppi_sub.

    Returns
    -------
    total_attns : list of float
        Total attention values for each specified node.
    degrees : list of int
        Degree values for each specified node.
    """
    total_attns = []
    degrees = []
    if gene_list == None:
        gene_list = list(ppi_sub.nodes())
    for node in gene_list:
        ind = gene_index_dict[node]
        row = pretrained_max[ind]
        row_for_sum = np.nan_to_num(row)
        total_attn = sum(row_for_sum)
        total_attns.append(total_attn)
        degrees.append(ppi_sub.degree(node))
    
    return total_attns, degrees


def bin_by_degree(total_attns, degrees):
    """
    Group attention values by node degree.

    Parameters
    ----------
    total_attns : list of float
        List of total attention values for nodes.
    degrees : list of int
        List of node degrees corresponding to the attention values.

    Returns
    -------
    deg_dict : dict of {int: list of float}
        Dictionary where keys are unique degrees and values are lists of attention values for nodes of that degree.
    """
    unique_degs = set(degrees)
    deg_dict = {deg: [] for deg in unique_degs}
    for i in range(len(degrees)):
        deg_dict[degrees[i]].append(total_attns[i])
    return deg_dict

def get_some_graph_weights(matrix, ppi_sub, dis_mod_graph, gene_index_dict):
    """
    Retrieve attention values for protein-protein interactions (PPI) in a subgraph, 
    excluding edges between disease genes.

    Parameters
    ----------
    matrix : np.ndarray
        2D numpy array with attention values.
    ppi_sub : nx.Graph
        Subgraph representing protein-protein interactions.
    dis_mod_graph : nx.Graph
        Graph representing the disease module, with nodes being disease genes.
    gene_index_dict : dict
        Dictionary mapping nodes to their respective indices in the matrix.

    Returns
    -------
    ppi_attns : list of float
        List of attention values for PPIs not involving pairs of disease genes.
    """
    ppi_attns = []
    dis_mod_nodes = list(dis_mod_graph.nodes())
    for edge in list(ppi_sub.edges()):
        if edge[0] in dis_mod_nodes and edge[1] in dis_mod_nodes:
            pass
        else:
            ind1 = gene_index_dict[edge[0]]
            ind2 = gene_index_dict[edge[1]]
            if np.isnan(matrix[ind1, ind2]) == False:
                ppi_attns.append(matrix[ind1, ind2])

    return ppi_attns


def dis_mod(matrix, dis_mod_graph, gene_index_dict):
    """
    Retrieve attention values for edges within a disease module.

    Parameters
    ----------
    matrix : np.ndarray
        2D numpy array with attention values.
    dis_mod_graph : nx.Graph
        Graph representing the disease module, where nodes are disease genes.
    gene_index_dict : dict
        Dictionary mapping nodes to their respective indices in the matrix.

    Returns
    -------
    dis_attns : list of float
        List of attention values for edges within the disease module.
    """
    dis_attns = []
    for edge in list(dis_mod_graph.edges()): 
        ind1 = gene_index_dict[edge[0]]
        ind2 = gene_index_dict[edge[1]]
        if np.isnan(matrix[ind1, ind2]) == False:
            dis_attns.append(matrix[ind1, ind2])
    return dis_attns


def fully_connected_dis_mod(matrix, dis_mod_graph, gene_index_dict):
    """
    Retrieve attention values for all pairs of nodes in a fully connected disease module,
    excluding existing edges and self-loops.

    Parameters
    ----------
    matrix : np.ndarray
        2D numpy array with attention values.
    dis_mod_graph : nx.Graph
        Graph representing the disease module, where nodes are disease genes.
    gene_index_dict : dict
        Dictionary mapping nodes to their respective indices in the matrix.

    Returns
    -------
    dis_attns : list of float
        List of attention values for all pairs of nodes in the disease module 
        excluding existing edges and self-loops.
    """
    dis_attns = []
    for node in list(dis_mod_graph.nodes()):
        for node2 in list(dis_mod_graph.nodes()):
            if node == node2 or dis_mod_graph.has_edge(node, node2):
                pass
            else:
                ind1 = gene_index_dict[node]
                ind2 = gene_index_dict[node2]
                if np.isnan(matrix[ind1, ind2]) == False:
                    dis_attns.append(matrix[ind1, ind2])
    return dis_attns

def rwr(network, seed_set, restart):
    """
    Perform a Random Walk with Restart (RWR) on a given network.
    
    Parameters
    ----------
        network : nx.Graph 
            The graph on which to perform RWR. Can be weighted or unweighted.
        seed_set : set 
            A set of seed nodes where the random walk can restart.
        restart : float 
            The restart probability at each step of the walk.
        
    Returns
    -------
        df_pagerank : pd.DataFrame 
            A DataFrame containing nodes and their RWR scores, excluding the seed nodes, sorted by scores in descending order.
    """
    
    # The personalization vector indicates the restart nodes. Whenever the walker restarts, it selects
    # a node in the personalization vector with a probability proportional to the personalization value.
    # In this case, the probability to restart in a seed node is 1/N, where N is the size of the seed, and
    # the probability to restart in any other node in the network is 0.
    personalization = {node: (1 / len(seed_set)) for node in seed_set}

    # The rwr is performed with the algorithm pagerank from google. It iterates over the adjacency matrix
    # to calculate the steady state probability of being located at each node. I can give you literature
    # on how this works if you are interested.
    pagerank_scores = nx.pagerank(network, personalization=personalization, alpha=1-restart,max_iter=500,tol=1e-8)


    # The result of pagerank is a dictionary where keys are the nodes and values are the probabilities in
    # the stationary state. The rest of the function is just to present the result in a pandas dataframe,
    # excluding the original nodes in the seed set.
    df_pagerank = pd.DataFrame(list(pagerank_scores.items()), columns=['Node', 'Score'])

    # Drop the seed nodes from the DataFrame
    df_pagerank = df_pagerank[~df_pagerank['Node'].isin(seed_set)]

    # Sort the DataFrame by the Score column in descending order
    df_pagerank = df_pagerank.sort_values(by='Score', ascending=False).reset_index(drop=True)

    return df_pagerank
    
def weight_network(weights, graph, gene_index_dict, directory):
    """
    Construct a weighted network from a base graph and weights matrix, adjusting weights if embeddings are used.

    Parameters
    ----------
    weights : np.ndarray
        2D numpy array with weight values for graph edges.
    graph : nx.Graph
        NetworkX graph object representing the base network structure.
    gene_index_dict : dict
        Dictionary mapping nodes to their respective indices in the weights matrix.
    directory : str
        Directory path containing the weights data; if it includes 'embeddings', adjusts weights.

    Returns
    -------
    G : nx.Graph
        Weighted NetworkX graph with edges assigned weights from the matrix.
    missing_weights : list of tuple
        List of node index pairs (i, j) for which weights were missing in the weights matrix.
    """
    if 'embeddings' in directory:
        weights += 1
        weights /= 2
    G = nx.Graph()
    nodes = list(graph.nodes())
    missing_weights = []
    for edge in list(graph.edges()):
        try:
            ind1 = gene_index_dict[edge[0]]
            ind2 = gene_index_dict[edge[1]]
            if np.isnan(weights[ind1, ind2]) == False:
                G.add_edge(edge[0], edge[1], weight=weights[ind1, ind2])
            else:
                missing_weights.append((ind1, ind2))
        except KeyError:
            missing_weights.append((ind1, ind2))
    return G, missing_weights


def get_disease_lcc(ppi_sub, disease_genes):
    """
    Retrieve the largest connected component (LCC) of disease-associated genes in a subgraph.

    Parameters
    ----------
    ppi_sub : nx.Graph
        Subgraph of a protein-protein interaction (PPI) network.
    disease_genes : list
        genes (pulled from a GDA table) associated with the disease of interest

    Returns
    -------
    disease_sub : nx.Graph
        Subgraph containing only the largest connected component of disease genes.
    """
    disease_subgraph = nx.subgraph(ppi_sub, disease_genes)
    largest_cc = max(nx.connected_components(disease_subgraph), key=len)
    disease_sub = disease_subgraph.copy()
    disease_sub = disease_sub.subgraph(largest_cc)
    return disease_sub


def get_disease_subgraph(ppi_sub, disease_genes):
    """
    Retrieve a subgraph of disease-associated genes in a PPI network.

    Parameters
    ----------
    ppi_sub : nx.Graph
        Subgraph of a protein-protein interaction (PPI) network.
    disease_genes : list
        genes (pulled from a GDA table) associated with the disease of interest

    Returns
    -------
    disease_subgraph : nx.Graph
        Subgraph containing all nodes associated with the disease.
    """
    disease_subgraph = nx.subgraph(ppi_sub, disease_genes)
    return disease_subgraph
    
def prune_ppi_nodes(ppi, gene_index_dict):
    """
    Prune nodes from a protein-protein interaction (PPI) network, retaining only those present in a gene dictionary.

    Parameters
    ----------
    ppi : nx.Graph
        NetworkX graph representing the protein-protein interaction network.
    gene_index_dict : dict
        Dictionary of genes, where keys are gene identifiers present in the dataset.

    Returns
    -------
    ppi_sub : nx.Graph
        Subgraph of the original PPI network containing only nodes that exist in the gene dictionary.
    """
    nodes_in_data = set(gene_index_dict.keys())
    pruned_nodes = [node for node in ppi.nodes() if node in nodes_in_data]
    ppi_sub = ppi.subgraph(pruned_nodes)
    return ppi_sub
    
def get_counter(result_df):
    """
    Generate a cumulative count of genes that have been found in a ranked gene list.

    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame containing a column 'Node' with ranked genes.

    Returns
    -------
    counter : list of int
        List representing the cumulative count of tossed genes at each position in the ranked list.
        The first element is initialized to 0, and each subsequent element increases by 1 if the gene
        is in `tossed_genes`, otherwise it remains the same.
    """
    ranked_genes = result_df['Node']
    counter = [0]
    for gene in ranked_genes:
        if gene in tossed_genes:
            counter.append(counter[-1] + 1)
        else:
            counter.append(counter[-1])
    return counter
    
def auc_plot(means):
    """
    Calculate the true positive rate (TPR) and false positive rate (FPR) based on a list of mean values for an AUC plot.

    Parameters
    ----------
    means : list of float
        List of mean values representing cumulative counts or similar statistics.

    Returns
    -------
    fpr : list of float
        False positive rate values at each position in the list.
    tpr : list of float
        True positive rate values at each position in the list.
    """
    tpr = [means[i] / means[-1] for i in range(len(means))]
    fpr = [(i - means[i]) / ((i - means[i]) + ((len(means) - i) - (means[-1] - means[i]))) for i in range(len(means))]
    return fpr, tpr


def prc_plot(means):
    """
    Calculate the precision and recall based on a list of mean values for a PRC plot.

    Parameters
    ----------
    means : list of float
        List of mean values representing cumulative counts or similar statistics.

    Returns
    -------
    recall : np.ndarray
        Recall values at each position in the list.
    precision : list of float
        Precision values at each position in the list.
    """
    precision = [means[i] / (i + 1) for i in range(len(means))]
    recall = np.array([means[i] / means[-1] for i in range(len(means))])
    return recall, precision


def auc(x, y):
    """
    Compute the area under the curve (AUC) for given x and y values using the trapezoidal rule.

    Parameters
    ----------
    x : list or np.ndarray
        x-coordinates of the points.
    y : list or np.ndarray
        y-coordinates of the points.

    Returns
    -------
    auc_value : float
        The calculated area under the curve (AUC) based on x and y values.
    """
    return np.trapz(y, x)

