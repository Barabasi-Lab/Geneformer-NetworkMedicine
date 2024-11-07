import gf_tools as gf
import pandas as pd
import datasets
import pyarrow as pa

path_to_exprs = './datasets/GSE97810.csv'
tokenized_output_path = 'path to save tokenized data'
matrix_output_path = 'path to save aggregated weights'

exprs = pd.read_csv(path_to_exprs)
exprs_T = exprs.transpose()
tokenized = gf.tokenize_csv(exprs_T, 'ensembl')

#### METADATA ####

# add a label column. This step will differ based on the type and format of your metadata
tokenized['label']=exprs.RA.tolist()

##################

# Create a pyarrow tabel out of the data
arrow_data = pa.Table.from_arrays([tokenized[key] for key in list(tokenized.keys())], names=list(tokenized.keys()))
hg_data = datasets.Dataset(arrow_data)
    
hg_data.save_to_disk(tokenized_output_path)

data = gf.GFDataset(dataset = None, path = tokenized_output_path, n=100)

model_path = "/work/ccnr/GeneFormer/conda_environment"
n_labels=2
device = 'cpu'

model = gf.load_model(task = "cell", path = model_path, device= device, attns=True, hiddens=False,n_labels=n_labels)

num_workers = 4
batch_size = 10
method = 'max'
layer_index = 4
    
max_mat, counts_mat, gene_index = gf.aggregate_attentions(data, model, layers = layer_index, 
                                                            heads_to_prune = None, batch_size = batch_size, 
                                                            device = device, num_workers = num_workers,method=method)
gf.save(max_mat,matrix_output_path+'vals.pkl')
gf.save(counts_mat,matrix_output_path+'counts.pkl')
gf.save(gene_index,matrix_output_path+'gene_dict.pkl')
