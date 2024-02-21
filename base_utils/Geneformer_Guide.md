# Getting Started with Geneformer
Contact: Jonah Spector (spector.jo@northeastern.edu)

### Installation/set up
**Cluster Access**
Geneformer should be run on the discovery cluster, where it is already installed and we can take advantage of GPUs. You will need access to the work/ccnr folders, and for this you need to make sure that you have access to discovery and that Laszlo is your sponsor. 
    
Use a [ServiceNow Access Request form][sponsor-form] if you need to request access to the cluster or update your sponsor

Use a [ServiceNow ticket for partition access][partition-form] if you need to request access to the work/ccnr partition

Finally, you should request access to the netsi (note spelling) partition, which will allow you to use the multi_gpu partition for your jobs. For this you should submit a [Service Now ticket][submit-ticket] (there are no specific form for this because the partition is only accessible to people in network science labs), making sure to say that Laszlo is your PI and that you would like access to the netsi partition. 

In both cases Laszlo will need to approve the request. Letting Rachael know that you're going through this process will help speed it along.
    
### Using Geneformer on Discovery
All of the Geneformer materials are conveniently located in the work/ccnr/Geneformer folder. The conda environment with the necessary packages is called LLM and can be called from `/work/ccnr/GeneFormer/conda_environment/optimus_gene`
If you need to add packages to the environment for your application, open the environment in an interactive job using the following commands

    srun --partition=short --nodes=1 --ntasks=1 --mem=10G --time=00:40:00 --pty /bin/bash
    module load anaconda3/2021.05
    source activate /work/ccnr/GeneFormer/conda_environment/optimus_gene
    pip install <your package>


For most applications, it is preferable to run Geneformer using sbatch commands. An example sbatch script is below. This is sufficient for running small numbers of samples (up to a few hundred) through Geneformer for analysis, but for more intense applications see the relevant section. The following script is also saved at LOCATION.

    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --time=4:00:00
    #SBATCH --job-name=your_job
    #SBATCH --mem=128G
    #SBATCH --partition=short
    #SBATCH -o output_%j.txt       # Standard output file
    #SBATCH -e error_%j.txt        # Standard error file
    
    module load anaconda3/2021.05
    source activate /work/ccnr/GeneFormer/conda_environment/optimus_gene
    python your_script.py 
    
### Using Geneformer
For basic applications of Geneformer, please see the script called `geneformer_basics.py`, which currently is stored in the folder `/work/ccnr/GeneFormer/jjs_adventures` on the discovery cluster. The script walks through useful functions for tasks such as tokenizing an expression dataset, loading the model, running samples, and some basic fine tuning operations. For some notes on these applications that won't be found as easily in the script, continue reading.
**To run samples**
Geneformer can be run without GPUs if you just need to run samples. The `GFDataset` class and the `load_model` function were both written with this purpose in mind. If your goal is to play with Geneformer outputs and get a feel for how the model works, this is a great place to start.
**Extracting attention weights**
If you would like to extract attention matrices, you need to set the `attns = True` when loading the model. For small numbers of samples the attention matrices can be saved as gzipped csv files. For large numbers of samples, pickle files are preferrable for storage.
**Extracting embeddings**
The same process can be followed for extracting embeddings by setting `hiddens = True` when loading the model. Keep in mind that while there are 4 attention matrices for each layer, therer is only one set of embeddings for each layer. 
**Fine Tuning Geneformer**
Fine tuning is difficult to give an overview for as it is so task specific. There are a few primary steps that should be followed, and the rest of the process will be dictated by the task. 
1. Shuffle the data so that classes aren't entered into the model all at once, in chunks.
2. Split the data. This can be done once, with 80% training data and 80% eval data, or it can be done several times using sklearns stratified k fold function to get several measurements for the same fine tuning task.
3. Optimize hyperparameters. Technically optional, but highly recommended to save time and memory, and ensure that you get the best fine tuned model.
4. Initialize and run a trainer object to create a fine tuned model 

Both the [cell classifier][cell-classifier] and the [gene classifier][gene-classifier] from the Geneformer Huggingface repo can be useful resources for more complicated fine tuning. The cell classifier gives an idea of how to fine tune several different models for different tissues, as well as some ideas for handling multiclassification problems. The Gene classifier delves into gene level prediction rather than cell level, which is a powerful application of the model that we do not currently take advantage of.

**Geneformer Datasets**
Geneformer takes as input .dataset folders, containing arrow files. There are several datasets the cluster at locations referenced below, but it may be necessary to make your own from publicly available data. There are two steps to this process, the first being to download your data from GEO and format it as a csv. This step is done in the `parse_GEO` funciton, but it can be tricky depending on how the data is formatted in GEO, and it may be necessary to change some aspects of the code in order to handle your data. The output of this step should be a csv or pandas DataFrame with Gene IDs as row indices, sample IDs as column indices, and cell i,j containing the expression of gene i in sample j. This DataFrame can be passed to the `tokenize_csv` function, which normalizes and tokenizes the data, then saves it in the correct format. 
**Dataset labeling**
In the `tokenize_csv` function, there is an argument called `path_to_metadata` and another called `metdata_parser` that both default to None. These variables can be used to assign labels to a dataset so that it can be used for fine tuning. If you would like to take advantage of these functions you need to write a function that takes as input a metadata file, and a sample index, and pass that into the `tokenize_csv` function. This will add a column to your finished dataset called 'label' that can be used during fine tuning.
**Dataset filtering**
If your data is from bulk RNA sequencing, you may be dealing with more noise than Geneformer was trained to handle. A possible solution to this problem is to filter for relevant genes (i.e. filter for genes usually expressed in blood, if you are dealing with blood samples). This can be done in the `tokenize_csv` function, by passing in a list of genes to inlcude in the data. All other genes will be excluded.

## Locations of datasets
Sample data: `/work/ccnr/GeneFormer/Sample.dataset`
Genecorpus:`/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/genecorpus_30M_2048.dataset`
Cell type annotation (goes with cell classifier fine tuning): `/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset/`
Scipher dataset: `/work/ccnr/GeneFormer/GeneFormer_repo/Scipher.dataset`
RA map dataset from GEO: `/work/ccnr/GeneFormer/GeneFormer_repo/GEO_dataset.dataset`


    
 

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [sponsor-form]: <https://service.northeastern.edu/tech?id=sc_cat_item&sys_id=0ae24596db535fc075892f17d496199c>
   [partition-form]: <https://service.northeastern.edu/tech?id=sc_cat_item&sys_id=0c34d402db0b0010a37cd206ca9619b7>
   [submit-ticket]: <https://service.northeastern.edu/tech?id=sc_cat_item&sys_id=0a0bfc5adb9f1fc075892f17d4961993>
   [cell-classifier]:<https://huggingface.co/ctheodoris/Geneformer/blob/main/examples/cell_classification.ipynb>
   [gene-classifier]: <https://huggingface.co/ctheodoris/Geneformer/blob/main/examples/gene_classification.ipynb>
   
