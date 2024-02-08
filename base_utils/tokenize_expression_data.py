import os
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
import geneformer
import GEOparse
from gprofiler import GProfiler
import pyarrow as pa
################################################################################################
# This script contains functions to tokenize expression data from GEO datasets
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

def tokenize_csv(data, gene_ID_type, dataset_normalization = False, path_to_metadata = None, metadata_parser = None):
    """
    :param data: pandas dataframe, the expression data from the GEO dataset. The format should be a matrix with gene names as the row index and patient IDs as the column index
    :param gene_ID_type: str, the type of gene ID used in the dataset. Should be either "ensembl" or the name of your format. Your format should be compatible with the gprofiler package 
    :param dataset_normalization: bool, whether to normalize the dataset by median gene expression WITHIN the dataset (in addition to the geneformer normalization). Defaults to False
    :param path_to_metadata: str, the path to the metadata file. Defaults to None, which is fine if no labels are needed
    :param metadata_parser: function, a function that takes the metadata dataframe and the column index as input and returns the label for the sample. Defaults to None, which is fine if no labels are needed

    :return: a HuggingFace Dataset object with the tokenized data
    """
    if path_to_metadata != None:
        metadata = pd.read_csv(path_to_metadata)
    # get the tokens and the median values for normalization from the geneformer package
    tokens = geneformer.TranscriptomeTokenizer()
    token_dict = tokens.gene_token_dict
    median_dict = tokens.gene_median_dict
    ans_dict = {'input_ids':[], 'length':[], 'label':[]}
    # save the row indices as a list of gene ids
    gene_ids = data.index.tolist()

    if dataset_normalization == True: 
        # for each row in the data, divide the row by its median value
        data = data.div(data.median(axis=1), axis=0)
    # if the genes are not in ensembl format, convert them to ensembl format with gprofiler
    if gene_ID_type != "ensembl":
        new_gene_ids = []
        gp = GProfiler(return_dataframe=True)
        convert_genes = gp.convert(organism='hsapiens',
                    query=gene_ids,
                    target_namespace='ENSG')
        converted_genes = convert_genes['converted'].to_list()

        # Gprofiler returns 'None' for genes that it cannot convert, so we will remove those genes from the dataset
        counter = 0
        for i in range(len(gene_ids)):
            if converted_genes[i] != 'None':
                new_gene_ids.append(converted_genes[i])
            else:
                data = data.drop(gene_ids[i])
                counter+=1
        print(f"Removed {counter} genes from the dataset because they could not be converted to ensembl IDs.")
        # replace the index of the data with the new gene IDs
        data.index = new_gene_ids
    
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
    print(f"Removed {counter} genes from the dataset because they could not be found in the median dictionary.")
    print(f"Final dataset size: {len(data.index)} genes")

    # Create the rank value encoding for each sample
    for i in tqdm(range(len(data.columns))):
        # make a list of the indices (genes), sorted by descending order of the column (rank value)
        sorted_indices = data.iloc[:,i].sort_values(ascending = False).index.tolist()
        # replace the indices with their token values
        sorted_indices = [token_dict[i] for i in sorted_indices]
        # cut off the list at 2048, the maximum sequence length
        if len(sorted_indices)>2048:
            sorted_indices = sorted_indices[:2048]
        # add the sorted indices to the input_ids list
        ans_dict['input_ids'].append(sorted_indices)
        # add the length of the sorted indices to the length list
        ans_dict['length'].append(len(sorted_indices))

        if path_to_metadata != None:
            # your function should take teh metadata frame and the column index as input and return the label for the sample
            your_label = metadata_parser(metadata,i)
        else:
            ans_dict['label'].append(None)
    # Create a pyarrow tabel out of the data
    arrow_data = pa.Table.from_arrays([ans_dict[key] for key in list(ans_dict.keys())], names=list(ans_dict.keys()))
    hg_data = Dataset(arrow_data)
    return hg_data
# Example usage
# if __name__ == "__main__":
#     accession = "GSE97810"
#     dir_path = ""
#     gz_name = ""
#     exprs, metadata = parse_GEO(accession, dir_path, gz_name) 
#     path_to_data = dir_path+"/GSE97810/exprs/GSE97810_exprs.csv"
#     data = pd.read_csv(path_to_data, index_col = 0)
#     tokenized_data = tokenize_csv(data, "gene", True)
#     tokenized_data.save_to_disk('GEO_dataset.dataset')