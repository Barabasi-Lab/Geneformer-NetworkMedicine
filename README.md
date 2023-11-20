# Geneformer-NetworkMedicine
This repository contains aggregated code for testing out various functionalities of [Geneformer](https://huggingface.co/ctheodoris/Geneformer/discussions/265) with regards to network medicine, including Protein-Protein interaction mapping, attention weight analysis, gene perturbation analysis, and application of Geneformer to Bulk-RNA data. 

# Modules
## Geneformer drug prediction
ScipherRA.py contains functionality for analyzing labelled genomic datasets with Geneformer, as well as a feed-forward regression network and Random Forest / Support Vector Machine. 

The primary function for running analysis is format_sci.

**Parameters**:

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

## Attention Extraction
Geneformer_classes.py contains examples on how to use the perturber and attention extractor for various purposes, located at the bottom of the file (in the main runtime). 

Example perturber workflow:

    pert = Perturber(data_subset = 0.5) # Subsets 50% of dataset 
    pert.create_disease_outside_test(disease = 'Cardiomyopathy Hypertrophic', samples_per_label = 5,) # Calculates LCC, genes one hop away from LCC, then genes one hop away from the 1 hop genes not in the LCC, then random genes. Selects samples_per_label genes. 
    pert.run_perturbation() # Runs perturbation filtration + analysis
    pert.visualize_similarities() # Visualizes box plot of results
        
Example attention extraction workflow: 

    new_attention = PPI_attention(layer_index = 4, mean = True)  # Uses either mean or maximum (if set to False) to aggregate weights
    new_attention.scrape_attentions(samples = 1000, disease = None) # Scrapes attentions from 1000 samples. Will focus on genes from a certain disease if a disease is given (ex. Cardiomyopathy Hypertrophic). Case-insensitive
    new_attention.map_PPI_genes() # Main analysis function for distributions, top attentions, ect. 
    #new_attention.map_disease_genes(disease = 'Cardiomyopathy Hypertrophic')
    new_attention.save() # Saves the gene attentions as a pickled dictionary
    new_attention.gen_attention_PPI(attention_threshold = 0.005) # Generates PPI with the attention weights using the given threshold
    new_attention.gen_attention_matrix(save = 'attentionMatrix.csv') # Saves the gene-gene attentions as a matrix with the given name

