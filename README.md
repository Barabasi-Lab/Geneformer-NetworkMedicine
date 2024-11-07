# Transformers Enhance the Predictive Power of Network Medicine
*Jonah Spector, Andrés Aldana Gonzalez, Michael Sebek, Joseph Ehlert, Christian DeFrondeville, Susan Dina Ghiassian, and Albert-László Barabási*
## Code and Data Structure
All functions used for analysis are contained in ```gf_tools.py```. The scripts ```figure_X.py``` can be run to generate the figures in the main text, and the scripts ```figure_SX.py``` can be run to generate the supplemental figures. The ```tokenization_example.py``` file shows how to use other expression data with Geneformer. 

The matrices of aggregated weights are stored on [dropbox](https://www.dropbox.com/home/Biology/01_Active_Collaborations/Geneformer/data), and the file structure is specified there. Each branch of the file tree ends in a folder containing 3 files, called ```counts.pkl```, ```vals.pkl```, ```gene_dict.pkl```. The counts matrix is a square matrix of size NxN, where N is the number of unique genes in the sample set. It counts how many times each pair of genes co-occurs in a sample. The vals matrix is the same shape, and contains the aggregated weight between each pair of genes. ```gene_dict.pkl``` is a dictionary that specifies the index of each gene in the matrix.

The ```supplemental_data``` folder contains 4 other files that are needed to generate the figures from the paper. These are the PPI network (```ppi.csv```), the Gene Disease Assosciation (GDA) table (```gda.csv```), DrugBank (```drugbank.csv```), and a dictionary called ```gene_name_id_dict.pkl``` provided by the Geneformer authors that converts gene symbols to Ensembl IDs. 

## How to Install and Run 
The requirements.txt file lists required packages for this project. To avoid compatibility issues, we recommend creating a new virtual environment and installing only the required packages.

The easiest way to set up the data locally is to download this repository, and add a folder called ```data``` that contains the data from [dropbox](https://www.dropbox.com/home/Biology/01_Active_Collaborations/Geneformer/data), using the same file structure. It may be necessary to only take some of the branches of the tree, depending on the availability of storage. If this is the case, and you would like to implement a different file structure, the paths to the matrices can be found at or near the top of each figure script. 

### Running the Figure Scripts
Once the paths to the matrices have been specified, open the figure script you would like to run, and specify your output folder in the ```path_to_figs``` variable. Then the script can be run and the plots will be saved to that folder.

### Using Other Data
The ```gf_tools``` folder also contains the functions needed to create the aggregated matrices stored on [dropbox](https://www.dropbox.com/home/Biology/01_Active_Collaborations/Geneformer/data). This process is laid out in ```tokenization_example.py```. To run the script, fill in your relevant file paths, make sure your expression data is formatted with gene names as the row index, and sample IDs as the column names, and replace the section labeled "METADATA" with code to parse your own metadata and assign labels. For best results, make sure that your data has Ensembl IDs for gene identifiers. 
