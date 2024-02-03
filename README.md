# Geneformer-NetworkMedicine
This repository contains aggregated code for applying the internal model states of the BERT model [Geneformer](https://huggingface.co/ctheodoris/Geneformer/discussions/265) to protein-protein interactomes and disease modules, with functionaltiies including Protein-Protein interaction mapping with pre-trained Geneformer, disease module identification with fine-tuned Geneformer, Gene perturbation analysis, comparing attention weight distributions from pretrained to fine=tuned, and the application of Geneformer to Bulk-RNA data. 

# Installation
1. Install the latest version of [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your device
2. Create a new conda environment, specifying python 3.11

   ```conda create --name <environment name> python=3.11```
   
   ```conda activate <environment name>```
    
4. Clone the repository into your working directory
5. Install the requirements with the requirements.txt file

   ```pip install -r requirements.txt```

# Modules
## Geneformer drug prediction
ScipherRA.py contains functionality for analyzing labelled genomic datasets with Geneformer, as well as comparing the predictive power of the trained model against several comparison methods such as a feed-forward network, a random forest classifier, and a support vector machine classifier.

The file contains a custom dataset (CellData) which  is an extension of the huggingface tokenized pyarrow dataset. The dataset initialization takes either a polars test dataset containing rows of samples and columns of genes, or a train and test polars dataset, which are appropriately labelled

The primary function for running analysis is format_sci.

**format_sci Parameters**:

* data : Path, str. The path of the data csv file
* save_file: str, defaut None. The title of the output png/svg file for ROC/TPR curves. If set to None, will not be saved.
* PR : bool, default False. Whether the precision/recall (True) or RoC (False) should be used
* equalize : bool, default True. Whether the labels should be equalized.
* token_dictionary : Path, default Path('geneformer/token_dictionary.pkl'). Path to token conversion pickled dictionary.
* gene_conversion : Path, str, default Path("geneformer/gene_name_id_dict.pkl"). Path to gene conversion pickled dictionary.
* target_label : str, default RA. What label in the data contains the numerical class labels.
* augment : bool, default False. Whether the data should be augmented or not.
* GF_samples : int, default 20000. If augmented data is enabled, how many augmented samples should be created.
* noise : float, default None. If set to a float, noise will be applied to the data.
* augment_combine: bool, default True. If data is augmented and this is set to true, the augmented data will be mixed in with the train and test set. Otherwise, the model will train on augmented data and test on the original dataset.

## Attention Extraction
Perturber.py contains the primary gene perturbation class.

Example perturber workflow:

    pert = Perturber(data_subset = 0.5) # Subsets 50% of dataset 
    pert.create_disease_outside_test(disease = 'Cardiomyopathy Hypertrophic', samples_per_label = 5,) # Calculates LCC, genes one hop away from LCC, then genes one hop away from the 1 hop genes not in the LCC, then random genes. Selects samples_per_label genes. 
    pert.run_perturbation() # Runs perturbation filtration + analysis
    pert.visualize_similarities() # Visualizes box plot of results

Attentions.py contains the primary attention extraction and analysis class

Example attention extraction workflow: 

    new_attention = PPI_attention(layer_index = 4, mean = True)  # Uses either mean or maximum (if set to False) to aggregate weights
    new_attention.scrape_attentions(samples = 1000, disease = None) # Scrapes attentions from 1000 samples. Will focus on genes from a certain disease if a disease is given (ex. Cardiomyopathy Hypertrophic). Case-insensitive
    new_attention.map_PPI_genes() # Main analysis function for distributions, top attentions, ect. 
    #new_attention.map_disease_genes(disease = 'Cardiomyopathy Hypertrophic')
    new_attention.save() # Saves the gene attentions as a pickled dictionary
    new_attention.gen_attention_PPI(attention_threshold = 0.005) # Generates PPI with the attention weights using the given threshold
    new_attention.gen_attention_matrix(save = 'attentionMatrix.csv') # Saves the gene-gene attentions as a matrix with the given name

## Cell and Gene Classification
The two files Cell_classifier.py and Gene_classifier.py contain modularized versions of Geneformer that are appropriate for running inference using Geneformer's sample classifier and gene classifier capabilities. For the Cell_classifier, the file takes a pyarrow-formatted tokenized dataset with specific labels, and fine-tunes the pre-trained Geneformer model against those labels, then running inference. There are options to extract the embeddings from samples and genes as well for downstream analysis and comparison. The Gene_classifier file takes a single csv file containing columns of ENSEMBL gene IDs pertraining to different labels (for example, coding vs non-coding), and the model then pretrains itself on a selected subset of randomly sampled genes from the pre-training corpus and learns the relationships of the gene embeddings to the label. Both these files are highly modular and provide an example implementation of Geneformer when working with variable data. 

