import os
import torch
from datasets import load_from_disk, Dataset, DataLoader
from transformers import BertForSequenceClassification, BertForTokenClassification, BertModel
import geneformer
from gprofiler import GProfiler
from tqdm import tqdm
import numpy as np
import pickle

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
def load_model(task, path = "/work/ccnr/GeneFormer/GeneFormer_repo", n_labels = 2, device=device, attns=False, hiddens=False):
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

def run_samples(data, model, batch_size, num_workers = 1, attns = None, hiddens = None, output_format = "csv", save_gene_names = False, heads_to_prune = None):
    """
    :param data: dataset, output by the GFDataset class. It should already be padded but not passed to the dataloader
    :param model: the model loaded by the load_model function
    :param batch_size: int, size of geneformer batch. In general larger batch size  = faster but more memory intensive. With 1 GPU and 128 GB of memory the batch size should usually be kept under 10, but you should optimize for your application
    :param num_workers: int, the number of processes to run simultaneously. not a parameter we have experimented much with, but it can be used to optimize runtime 
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
