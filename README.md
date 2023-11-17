# Geneformer-NetworkMedicine
Application of Geneformer for Network Medicine

# Modules
## Geneformer drug prediction
ScipherRA.py contains functionality for analyzing labelled genomic datasets with Geneformer, as well as a feed-forward regression network and Random Forest / Support Vector Machine. 

The primary function for running analysis is format_sci
Parameters:
format_sci(data, save, token_dictionary = Path('geneformer/token_dictionary.pkl'), PR = False, augment = False, noise = None,
               gene_conversion = Path("geneformer/gene_name_id_dict.pkl"), target_label = "RA", GF_samples = 20000, save_file = None):
    
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

Example attention extraction workflow: 

