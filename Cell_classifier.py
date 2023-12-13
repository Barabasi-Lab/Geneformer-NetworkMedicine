# Package Imports
import tqdm
import sys
import polars as pl
import os
from datasets import Dataset
from collections import Counter
import random
import datetime
from pathlib import Path
import subprocess
import seaborn as sns; sns.set()
from datasets import load_from_disk
import fastcluster
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, BertConfig, Trainer
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification, EmbExtractor, TranscriptomeTokenizer
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve,  confusion_matrix
from sklearn.metrics import auc as precision_auc
from sklearn.preprocessing import label_binarize
import pyarrow as pa
import concurrent.futures
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from scipy.stats import ranksums
import ray
import ast
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.search.hyperopt import HyperOptSearch
import numpy as np
import pickle as pk
#random.seed(42)
#torch.manual_seed(42)
#np.random.seed(42)

# Properly sets up NCCV environment
GPU_NUMBER = [i for i in range(torch.cuda.device_count())] 
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function for generating a ROC curve from data
def ROC(prediction, truth, type = 'GeneFormer', label = '', save = False):
   
    fpr, tpr, _ = roc_curve(truth, prediction[:, 1])
    auc = roc_auc_score(truth, prediction[:, 1])
    print(f'{type} AUC: {auc}')
    plt.plot(fpr,tpr, label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(f'{label} ROC Curve')
    plt.legend(loc=4)
    
    if save == True:
        plt.savefig('ROC.png')
    
    return tpr, fpr, auc 
               
# Identifies cosine similarity between two embeddings. 0 is perfectly dissimilar and 1 is perfectly similar
def similarity(tensor1, tensor2, cosine = False):
    
    if cosine == False:
        if tensor1.ndimension() > 1:
            tensor1 = tensor1.view(1, -1)
        if tensor2.ndimension() > 1:
            tensor2 = tensor2.view(1, -1)
        dot_product = torch.matmul(tensor1, tensor2)
        norm_tensor1 = torch.norm(tensor1)
        norm_tensor2 = torch.norm(tensor2)
        epsilon = 1e-8
        similarity = dot_product / (norm_tensor1 * norm_tensor2 + epsilon)
        similarity = (similarity.item() + 1)/2
    else:
        if tensor1.shape != tensor2.shape:
            raise ValueError("Input tensors must have the same shape.")

        # Compute cosine similarity using PyTorch's dot product function
        dot_product = torch.dot(tensor1, tensor2)
        norm_tensor1 = torch.norm(tensor1)
        norm_tensor2 = torch.norm(tensor2)
    
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        similarity = dot_product / (norm_tensor1 * norm_tensor2 + epsilon)
        
    return similarity.item()
    
# Plots heatmap between different classes/labels
def plot_similarity_heatmap(similarities):
    classes = list(similarities.keys())
    classlen = len(classes)
    arr = np.zeros((classlen, classlen))
    for i, c in enumerate(classes):
        for j, cc in enumerate(classes):
            if cc == c:
                val = 1.0
            else:
                val = similarities[c][cc]
            arr[i][j] = val
        
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap='inferno', vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(classlen), classes, rotation = 45, ha = 'right')
    plt.yticks(np.arange(classlen), classes)
    plt.title("Similarity Heatmap")
    plt.savefig("similarity_heatmap.png")
    
# Function for tokenizing genes into ranked-value encodings from Geneformer
def tokenize_dataset(gene_set, type = None, token_set = 'token_dictionary.pkl', species = 'human'):
    token_dataset = open(token_set, 'rb')
    token_dict = pickle.load(token_dataset)
    wrap = True
    
    if isinstance(gene_set[0], list) == False:
        gene_set = [gene_set]
        wrap = False
        
    pool = Pool()
    converted_set = []
    
    def process_gene(gene):
         api_url = f"https://rest.ensembl.org/xrefs/symbol/{species}/{gene}?object_type=gene"
         response = requests.get(api_url, headers={"Content-Type": "application/json"})
         try:
             data = response.json()
             gene = data[0]['id']
         except:
             gene = None
         return gene
         
    def process_hgnc(gene):
        for gene in tqdm.tqdm(genes, total = len(genes)):
            api_url = f"https://rest.ensembl.org/xrefs/symbol/{species}/{hgnc_id}?object_type=gene"
            response = requests.get(api_url, headers={"Content-Type": "application/json"})
            try:
                data = response.json()
                gene = data[0]['id']
            except:
                gene = None
            return gene
                    
    def process_go(gene):
         mg = mygene.MyGeneInfo()
         results = mg.query(gene, scopes="go", species=species, fields="ensembl.gene")

         ensembl_ids = []
         max_score = 0
         for hit_num, hit in enumerate(results["hits"]):
             if hit['_score'] > max_score:
                 max_score = hit['_score']
                 chosen_hit = hit
         try:
             try:
                 gene = chosen_hit["ensembl"]["gene"]
             except:
                 gene = chosen_hit["ensembl"][0]["gene"]
         except:
             gene = None
         return gene
         
    if type == None or type.upper() == 'ENSEMBL':
        converted_set = gene_set
    elif type.upper() == 'GENE':
        for genes in gene_set:
            converted_genes = []
            for result in tqdm.tqdm(pool.imap(process_gene, genes), total = len(genes)):
                converted_genes.append(result)
            converted_set.append(converted_genes)
    elif type.upper() == 'GO':
        for genes in gene_set:
            converted_genes = []
            for result in tqdm.tqdm(pool.imap(process_go, genes), total = len(genes)):
                converted_genes.append(result)
            converted_set.append(converted_genes)
    elif type.upper() == 'HGNC':
        for genes in gene_set:
            converted_genes = []
            for result in tqdm.tqdm(pool.imap(process_hgnc, genes), total = len(genes)):
                converted_genes.append(result)
            converted_set.append(converted_genes)
            
    Chembl = []
    for set_num, set in enumerate(converted_set):
        Chembl.append([])
        for gene in set:
            if gene == None:
                Chembl[set_num].append(None)
            else:
                try:
                    Chembl[set_num].append(token_dict[gene])
                except:
                    print(f'{gene} not found in tokenized dataset!')
                    Chembl[set_num].append(None)

    if wrap == False:
        Chembl = Chembl[0]

    return Chembl    
    

# '/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/'
# '/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset/'
'''
======================================================

PRIMARY CELL - CLASSIFIER AND EMBEDDING EXTRACTOR CLASS
      
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

Runs cell-level classification and embedding extraction with Geneformer

'''

def finetune_cells(token_set = Path('geneformer/token_dictionary.pkl'), median_set = Path('geneformer/gene_median_dictionary.pkl'), pretrained_model = ".",
          dataset = 'Genecorpus-30M/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset/', dataset_split = None, filter_cells = .005, epochs = 1, cpu_cores = os.cpu_count(), geneformer_batch_size = 12, optimizer = 'adamw', max_lr = 5e-5, num_gpus = torch.cuda.device_count(), max_input_size = 2 ** 11, lr_schedule_fn = "linear", warmup_steps = 500, freeze_layers = 0, emb_extract = False, max_cells = 1000, emb_layer = 0, emb_filter = None, emb_dir = 'embeddings', overwrite = True, label = "cell_type", data_filter = None, ROC_curve = True,
          forward_batch = 200, model_location = None, skip_training = False, sample_data = 1, inference = False, optimize_hyperparameters = False, output_dir = None):
    
    '''
    Primary Parameters
    -------------------
    dataset: path
        Path to fine-tuning/testing dataset for training

    model_location: path
        Path to location of existing model to use for inference and embedding extraction
        
    pretrained_model: path
        Path to pretrained GeneFormer 30M model before fine-tuning
        
    inference: bool 
        Chooses whether to perform inference (which causes the function to return the list of similarities). Defaults to False
    
    skip_training: bool 
        Chooses whether to skip training the model. Defaults to False
        
    emb_extract: bool
        Choose whether to extract embeddings and calculate similarities. Defaults to True
        
    optimize_hyperparameters: bool
        Choose whether to optimize model hyperparamters. Defaults to False
        
    
    Customization Parameters
    -------------------
    
    dataset_split: str
        How the dataset should be partitioned (if at all), and what ID should be used for partitioning
        
    data_filter: list
        (For embeddings and inference) Runs analysis subsets of the dataset by the ID defined by dataset_split
        
    label: str
        What feature should be read as a classification label
    
    emb_layer: int
        What layer embeddings should be extracted and compared from.
        
    emb_filter: ['cell1', 'cell2'...]
        Allows user to narrow down range of cells that embeddings will be extracted from.
        
    max_cells: int
        How many embeddings from cells should be extracted. 
        
    freeze_layers: int
        Number of layers should be permanently frozen during fine-tuning (starting from the first layer, 4 brings it up to the pretrained model).
        
    sample_data: float
        What proportion of the HF dataset should be used
        
    '''
          
    dataset_list = []
    evalset_list = []
    split_list = []
    target_dict_list = []
    
    '''
    For loading and pretraining with custom median expressions and/or custom gene conversions
    -------------------------------------------------------------
    
    token set: path
        Path to token conversion dictionary
      
    median set: path
        Path to median gene dictionary (ensembl IDs as the keys)
      
    
    median_data = pickle.load(open(median_set, 'rb'))
    median_data['<pad>'] = None
    median_data['<mask>'] = None
    
    token_set = pickle.load(open(token_set, 'rb'))
    median_dict = {key:median_data[key] for key in list(token_set.keys())}
    '''
    
    train_dataset = load_from_disk(dataset)
    num_samples = int(len(train_dataset) * sample_data)
    random_indices = random.sample(range(len(train_dataset)), num_samples)
    train_dataset = train_dataset.select(random_indices)
   
    sample = int(sample_data  * len(train_dataset))
    sample_indices = random.sample(range(len(train_dataset)), sample)
    train_dataset = train_dataset.select(sample_indices)

    def if_not_rare_celltype(example):
        return example[label] in cells_to_keep
                    
    # change labels to numerical ids
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
                   
    def if_trained_label(example):
        return example["label"] in trained_labels 
        
    if skip_training != True:
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
            
        # Defines custom exceptions for collecting labels (default excluded)
        excep = {"bone_marrow":"immune"}
        
        if dataset_split != None:
            if data_filter != None:
                 split_iter = [data_filter]
            else:
                 split_iter = Counter(train_dataset[dataset_split]).keys()
            for lab in split_iter:
            
                # collect list of tissues for fine-tuning (immune and bone marrow are included together)
                if lab in list(excep.keys()):
                    continue
                elif lab == list(excep.values()):
                    split_ids = [excep.keys(),excep.values()]
                    split_list += [excep.values()]
                else:
                    split_ids = [lab]
                    split_list += [lab]
                
                # filter datasets for given organ
                def if_label(example):
                    return example[dataset_split] == lab
                    
                trainset_label = train_dataset.filter(if_label, num_proc=cpu_cores)
                label_counter = Counter(trainset_label[label])
                total_cells = sum(label_counter.values())
                
                # Throws out cells with a low proportion in the dataset (drop cell types representing <0.5% of cells per deepsort published method)
                cells_to_keep = [k for k,v in label_counter.items() if v>(filter_cells*total_cells)]
                trainset_label_subset = trainset_label.filter(if_not_rare_celltype, num_proc=cpu_cores)
                  
                # shuffle datasets and rename columns
                trainset_label_shuffled = trainset_label_subset.shuffle(seed=42)
                trainset_label_shuffled = trainset_label_shuffled.rename_column(label,"label")
                trainset_label_shuffled = trainset_label_shuffled.remove_columns(dataset_split)
                
                # create dictionary of cell types : label ids
                target_names = list(Counter(trainset_label_shuffled["label"]).keys())
                target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
                target_dict_list += [target_name_id_dict]
                
                labeled_trainset = trainset_label_shuffled.map(classes_to_ids, num_proc=cpu_cores)
                
                # create 80/20 train/eval splits
                labeled_train_split = labeled_trainset.select([i for i in labeled_trainset if i['train'] == 1])
                labeled_eval_split = labeled_trainset.select([i for i in labeled_trainset if i['train'] == 0])
                
                # filter dataset for cell types in corresponding training set
                trained_labels = list(Counter(labeled_train_split["label"]).keys())
                
                labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=cpu_cores)
            
                dataset_list += [labeled_train_split]
                evalset_list += [labeled_eval_split_subset]
                
            trainset_dict = dict(zip(split_list,dataset_list))
            traintargetdict_dict = dict(zip(split_list,target_dict_list))
            evalset_dict = dict(zip(split_list,evalset_list))
                  
            for lab in split_list:
                label_trainset = trainset_dict[lab]
                label_evalset = evalset_dict[lab]
                label_dict = traintargetdict_dict[lab]
                
                # set logging steps
                logging_steps = round(len(label_trainset)/geneformer_batch_size/10)
                if logging_steps == 0:
                    logging_steps = 1
                
                # reload pretrained model
                model = BertForSequenceClassification.from_pretrained("/work/ccnr/GeneFormer/GeneFormer_repo", 
                                                                  num_labels=len(label_dict.keys()),
                                                                  output_attentions = False,
                                                                  output_hidden_states = False).to(device)
                
                # define output directory path
                current_date = datetime.datetime.now()
                datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
                
                if output_dir == None:
                    output_dir = f"{datestamp}_geneformer_CellClassifier_{lab}_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"
                
                # ensure not overwriting previously saved model
                saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
                
                if os.path.isfile(saved_model_test) == True and overwrite == False:
                    raise Exception("Model already saved to this directory.")
            
                # make output directory
                subprocess.call(f'mkdir -p {output_dir}', shell=True)
                
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
                true_labels = label_evalset['label']
              
                
                if optimize_hyperparameters == False:
                    # create the trainer
                    trainer = Trainer(
                    model=model,
                    args=training_args_init,
                    data_collator=DataCollatorForCellClassification(),
                    train_dataset=label_trainset,
                    eval_dataset=label_evalset,
                    compute_metrics=compute_metrics
                    )
        
                    # train the cell type classifier
                    trainer.train()
                    predictions = trainer.predict(label_evalset)
                    print(f'accuracy: {accuracy_score(predictions.argmax(), label_evalset["labels"])}') 
                    
                    tpr, fpr, auc = ROC(predictions.predictions, true_labels)
                    
                    metrics = compute_metrics(predictions)
                    with open(f"{output_dir}predictions.pickle", "wb") as fp:
                        pickle.dump(predictions, fp)
                        
                    trainer.save_metrics("eval",predictions.metrics)
               
                    with open(f'{output_dir}/targets.txt', 'w') as f:
                        if len(target_dict_list) == 1:
                            f.write(str(target_dict_list[0]))
                        else:
                            f.write(str(target_dict_list))
                            
                    trainer.save_model(output_dir)
                else:
                    
                    def model_init():
                        model = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                              num_labels=len(label_dict.keys()),
                                                              output_attentions = False,
                                                              output_hidden_states = False)
                        if freeze_layers is not None:
                            modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
                            for module in modules_to_freeze:
                                for param in module.parameters():
                                    param.requires_grad = False
                        model = model.to(device)   
                        return model
              
                    trainer = Trainer(
                        model_init=model_init,
                        args=training_args_init,
                        data_collator=DataCollatorForCellClassification(),
                        train_dataset=label_trainset,
                        eval_dataset=label_evalset,
                        compute_metrics=compute_metrics
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
                    
                    if torch.device == 'cuda':
                        resources_per_trial={"cpu":8,"gpu":1},
                    else:
                        resources_per_trial={"cpu":8}
                        
                    # optimize hyperparameters
                    best_trial = trainer.hyperparameter_search(
                        direction="maximize",
                        backend="ray",
                        resources_per_trial = resources_per_trial,
                        hp_space=lambda _: ray_config,
                        search_alg=hyperopt_search,
                        n_trials=10, # number of trials
                        progress_reporter=tune.CLIReporter(max_report_frequency=600,
                                                                       sort_by_metric=True,
                                                                       max_progress_rows=100,
                                                                       mode="max",
                                                                       metric="eval_accuracy",
                                                                       metric_columns=["loss", "eval_loss", "eval_accuracy"]))
                    best_hyperparameters = best_trial.hyperparameters
            
                    print("Best Hyperparameters:")
                    print(best_hyperparameters)
                                                                                 
             
                    
        else:
     
             trainset_label = train_dataset
             label_counter = Counter(trainset_label[label])
             total_cells = sum(label_counter.values())
             
             # Throws out cells with a low proportion in the dataset
             cells_to_keep = [k for k,v in label_counter.items() if v>(filter_cells*total_cells)]
             trainset_label_subset = trainset_label.filter(if_not_rare_celltype, num_proc=cpu_cores)
             
             # shuffle datasets and rename columns
             trainset_label_shuffled = trainset_label_subset.shuffle() #seed = 42
             trainset_label_shuffled = trainset_label_shuffled.rename_column(label,"label")
    
             # create dictionary of cell types : label ids
             target_names = list(Counter(trainset_label_shuffled["label"]).keys())
             target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
             target_dict_list = target_name_id_dict
             
             labeled_trainset = trainset_label_shuffled.map(classes_to_ids, num_proc=cpu_cores)
             
             try:
                 def filter_train(example):
                    return example['train'] == 1
                    
                 def filter_test(example):
                    return example['train'] == 0
                    
                 labeled_train_split = labeled_trainset.filter(filter_train)
                 labeled_eval_split = labeled_trainset.filter(filter_test)

             except:
             
                 # create 80/20 train/eval splits
                 labeled_train_split = labeled_trainset.select([i for i in range(0,round(len(labeled_trainset)*0.8))]) 
                 labeled_eval_split = labeled_trainset.select([i for i in range(round(len(labeled_trainset)*0.8),len(labeled_trainset))])
    
             
             # filter dataset for cell types in corresponding training set
             trained_labels = list(Counter(labeled_train_split["label"]).keys())
             labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=cpu_cores) 
            
             # set logging steps
             logging_steps = round(len(trainset_label)/geneformer_batch_size/10)
             
             
             # reload pretrained model
             model = BertForSequenceClassification.from_pretrained(pretrained_model, 
                                                                      num_labels=len(list(set(trained_labels))),
                                                                      output_attentions = False,
                                                                      output_hidden_states = False).to(device).train() 
             
             '''
             START
            
             import pandas as pd
             import networkx as nx
             
             PPI = Path('/work/ccnr/GeneFormer/GeneFormer_repo/PPI/PPI_2022_2022-04-21.csv')
             gene_conversion = Path("/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/gene_name_id_dict.pkl")
             tokens = Path("/work/ccnr/GeneFormer/GeneFormer_repo/geneformer/token_dictionary.pkl")
                  
             # Creates conversion from genes to tokens
             token_dict = TranscriptomeTokenizer().gene_token_dict
             token_dict = {value: key for key, value in token_dict.items()}
             conversion = {value: key for key, value in pk.load(open(gene_conversion, 'rb')).items()}
             conversion = {conversion[value]:key for key, value in token_dict.items() if value in conversion}
        
             # Loads PPI
             try:
                 PPI = pd.read_csv(PPI).drop(columns = ['Source'])
                 PPI = PPI.rename(columns = {'HGNC_Symbol.1':'source', 'HGNC_Symbol.2':'target'})
             except:
                 PPI = pd.read_table(PPI)[['source', 'target']]
             PPI = nx.from_pandas_edgelist(PPI)
             PPI = PPI.subgraph(conversion)
         
             # Get embeddings for each node in the graph
             try:
                 embeddings = pk.load(open('Training_Datasets/embeddings.pk', 'rb'))
             except:
                 # Embeds structure with Node2Vec
                 node2vec = Node2Vec(PPI, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
                 node_model = node2vec.fit(window=5, min_count=1, batch_words=4)
                 embeddings = {node: node_model.wv[node] for node in PPI.nodes()}
                 pk.dump(embeddings, open('Training_Datasets/embeddings.pk', 'wb'))
        

             # Customize Token Embeddings
             bert_embeddings = model.get_input_embeddings()
             for node, embedding in embeddings.items():
                 token_id = conversion[node]
                 if token_id is not None:
                     bert_embeddings.weight.data[token_id] = torch.tensor(embedding, dtype=torch.float)

             END
             '''
                           
             # define output directory path
             current_date = datetime.datetime.now()
             datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
            
             if output_dir == None:
                 output_dir = f"{datestamp}_geneformer_CellClassifier_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"
            
             # ensure not overwriting previously saved model
             saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
             if os.path.isfile(saved_model_test) == True and overwrite == False:
                 raise Exception("Model already saved to this directory.")
        
             # make output directory
             subprocess.call(f'mkdir -p {output_dir}', shell=True)
            
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
               "output_dir": output_dir,}
            
             training_args_init = TrainingArguments(**training_args)
             true_labels = labeled_eval_split_subset['label']
       
             if optimize_hyperparameters == False:
                     
                 # create the trainer
                 trainer = Trainer(
                   model=model,
                   args=training_args_init,
                   data_collator=DataCollatorForCellClassification(),
                   train_dataset=labeled_train_split,
                   eval_dataset=labeled_eval_split_subset,
                   compute_metrics=compute_metrics
                 )
                 
                 # train the cell type classifier
                 trainer.train()
                 predictions = trainer.predict(labeled_eval_split_subset)
             
                 predictions_tensor = torch.Tensor(predictions.predictions)
                 predicted_labels = torch.argmax(predictions_tensor, dim=1)
                 print(f'accuracy: {accuracy_score(labeled_eval_split_subset["label"], predicted_labels)}') 
                 
                 metrics = compute_metrics(predictions)
                 
                 with open(f"{output_dir}predictions.pickle", "wb") as fp:
                     pickle.dump(predictions.predictions.argmax(-1), fp)
                     
                 trainer.save_metrics("eval",predictions.metrics)
                 trainer.save_model(output_dir)
           
                 # Saves label conversion dictionary to output directory
                 with open(f'{output_dir}/targets.txt', 'w') as f:
                     f.write(str(target_dict_list))
                 
                 try:
                     if ROC_curve == True:
                         # Calculates ROC curve
                         fpr, tpr, auc = ROC(predictions.predictions, true_labels) 
                         #print(f'Geneformer AUC: {auc}')
                         
                         return fpr, tpr, auc
                     else:
                     
                         # Generate confusion matrix
                         conf_matrix = confusion_matrix(labeled_eval_split_subset["label"], predicted_labels)
                        
                         # Define class names
                         class_names = ['mtx', 'pred']
                        
                         # Calculate class-wise accuracy
                         class_accuracy = np.diag(conf_matrix) / conf_matrix.sum(axis=1) * 100  # Calculate accuracy in percentage
                        
                         # Create a heatmap with proportions
                         plt.figure(figsize=(8, 6))
                         sns.heatmap(conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100, annot=True, fmt='.1f', cmap='coolwarm', 
                                    xticklabels=class_names, yticklabels=class_names)  # Display percentages with one decimal point
                         plt.xlabel('Predicted')
                         plt.ylabel('Actual')
                         plt.title('Confusion Matrix (Percentages)')
                         
                         plt.savefig('GF_confusion.png')
               
                             
                         precision, recall, _ = precision_recall_curve(labeled_eval_split_subset["label"], predictions.predictions[:, 1])
                         pr_auc = precision_auc(sorted(recall), sorted(precision))
                     
                         print(f'AUC: {pr_auc}')
                         
                         return recall, precision, pr_auc 
                 except:
                     pass    
                  
             else:
                 # Optimizes hyperparameters
                 
                 num_classes = len(list(set(labeled_train_split['label'])))
                 def model_init():
                     model = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                          num_labels=num_classes,
                                                          output_attentions = False,
                                                          output_hidden_states = False)
                                                          
                     if freeze_layers is not None:
                         modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
                         for module in modules_to_freeze:
                             for param in module.parameters():
                                 param.requires_grad = False
                     model = model.to(device)
                     return model
                     
                     
                 # create the trainer
                 trainer = Trainer(
                   model_init=model_init,
                   args=training_args_init,
                   data_collator=DataCollatorForCellClassification(),
                   train_dataset=labeled_train_split,
                   eval_dataset=labeled_eval_split_subset,
                   compute_metrics=compute_metrics
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
                 
                 if torch.device == 'cuda':
                     resources_per_trial={"cpu":8,"gpu":1},
                 else:
                     resources_per_trial={"cpu":8}
                     
                 # optimize hyperparameters
                 best_trial = trainer.hyperparameter_search(
                     direction="maximize",
                     backend="ray",
                     resources_per_trial = resources_per_trial,
                     hp_space=lambda _: ray_config,
                     search_alg=hyperopt_search,
                     n_trials=10, # number of trials
                     progress_reporter=tune.CLIReporter(max_report_frequency=600,
                                                                    sort_by_metric=True,
                                                                    max_progress_rows=100,
                                                                    mode="max",
                                                                    metric="eval_accuracy",
                                                                    metric_columns=["loss", "eval_loss", "eval_accuracy"]))
                 best_hyperparameters = best_trial.hyperparameters
            
                 print("Best Hyperparameters:")
                 print(best_hyperparameters)
                 
             
    # Performs Inference with model
    if inference == True:
        if dataset_split != None and data_filter != None:
            def if_label(example):
                return example[dataset_split] == data_filter 
                
            train_dataset = train_dataset.filter(if_label, num_proc=cpu_cores)
    
        trainset_label_shuffled = train_dataset
        total_cells = len(trainset_label_shuffled)
    
        # loads dictionary of all cell labels model was trained on
        with open(Path(model_location) / 'targets.txt', 'r') as f:
            data = ast.literal_eval(f.read())
        if dataset_split != None and data_filter == None:
            indexer = dataset_split.index(data_filter)
            data = data[indexer]
                
        target_dict_list = {key:value for key, value in enumerate(data)}
        
        # set logging steps
        logging_steps = round(len(trainset_label_shuffled)/geneformer_batch_size/20)
                    
        # reload pretrained model
        input_ids = trainset_label_shuffled["input_ids"]
        inputs = torch.zeros(len(input_ids), max_input_size, dtype=torch.int64)
        attention = torch.zeros(len(input_ids), max_input_size, dtype=torch.int64)
        
        outputs = []
        for i, sentence in enumerate(input_ids):
            sentence_length = len(sentence)
            if sentence_length <= max_input_size:
                inputs[i, :sentence_length] = torch.tensor(sentence)
                attention[i, :sentence_length] = torch.ones(sentence_length)
            else:
                inputs[i, :] = torch.tensor(sentence[:max_input_size])
                attention[i, :] = torch.ones(max_input_size)
        
        # Split inputs and attention_mask tensors into batches
        num_batches = len(inputs) // geneformer_batch_size
        model = BertForSequenceClassification.from_pretrained(model_location, num_labels=len(target_dict_list)).to(device)   
        for i in range(num_batches):
            start_idx = i * geneformer_batch_size
            end_idx = (i + 1) * geneformer_batch_size
            
            inputs_batch = inputs[start_idx:end_idx].to(device)
            attention_mask_batch = attention[start_idx:end_idx].to(device)
            
            # Forward pass for the batched inputs
            model_outputs = model(inputs_batch, attention_mask=attention_mask_batch)["logits"]
            outputs.extend(model_outputs)
          
        
        # Handle the remaining samples (if any) that don't form a complete batch
        if len(inputs) % batch_size != 0:
            start_idx = num_batches * batch_size
            inputs_batch = inputs[start_idx:].to(device)
            attention_mask_batch = attention_mask[start_idx:].to(device)
            
            # Forward pass for the remaining samples
            model_outputs = model(inputs_batch, attention_mask=attention_mask_batch)["logits"]
            outputs.extend(model_outputs)
        
   
        predictions = F.softmax(outputs, dim = -1).argmax(-1)
        predictions = [target_dict_list[int(pred)] for pred in predictions]

        return predictions
    
    # Extracts embeddings from labelled data
    if emb_extract == True:
        if emb_filter == None:
            with open(f'{model_location}/targets.txt', 'r') as f:
                data = ast.literal_eval(f.read())
            if dataset_split != None and data_filter == None:
                indexer = dataset_split.index(data_filter)
                data = data[indexer]
                
            target_dict_list = {key:value for key, value in enumerate(data)}
            total_filter = None
        else:
            total_filter = emb_filter
        
        train_dataset = load_from_disk(dataset)
        if dataset_split != None:
            def if_label(example):
                return example[dataset_split] == data_filter 
                
            train_dataset = train_dataset.filter(if_label, num_proc=cpu_cores)
            
            label_counter = Counter(train_dataset[label])
            total_cells = sum(label_counter.values())
            cells_to_keep = [k for k,v in label_counter.items() if v>(filter_cells*total_cells)]
            
            def if_not_rare(example):
                  return example[label] in cells_to_keep
                  
            train_dataset = train_dataset.filter(if_not_rare, num_proc=cpu_cores)
            
        true_labels = train_dataset[label]
        num_classes = len(list(set(true_labels)))
       
        embex = EmbExtractor(model_type="CellClassifier", num_classes=num_classes,
                             filter_data=total_filter, max_ncells=max_cells, emb_layer=emb_layer,
                             emb_label=[dataset_split,label], labels_to_plot=[label], forward_batch_size=forward_batch, nproc=cpu_cores)
                             
        # example dataset: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
        subprocess.call(f'mkdir -p {emb_dir}', shell = True)
      
        embs = embex.extract_embs(model_directory = model_location, input_data_file = dataset, output_directory = emb_dir, output_prefix = f"{label}_embbeddings")
        true_labels = embex.filtered_input_data[label]

        emb_dict = {label:[] for label in list(set(true_labels))}
        for num, emb in embs.iterrows():
            key = emb[label]
            selection = emb.iloc[:255]
            emb = torch.Tensor(selection)
            emb_dict[key].append(emb)

        for key in list(emb_dict.keys()):
            stack = torch.stack(emb_dict[key], dim = 0)
            emb_dict[key] = torch.mean(stack, dim=0)
        similarities = {key:{} for key in list(emb_dict.keys())}
        
        for key in list(emb_dict.keys()):
            remaining_keys = [k for k in list(emb_dict.keys()) if k != key]
            for k in remaining_keys:
                embedding = emb_dict[k]
                sim = similarity(emb_dict[key], embedding, cosine = True)
                
                similarities[key][k] = sim
       
        plot_similarity_heatmap(similarities)
        
        embex.plot_embs(embs=embs, 
                plot_style="umap",
                output_directory=emb_dir,  
                output_prefix="emb_plot")
        
        embex.plot_embs(embs=embs, 
                plot_style="heatmap",
                output_directory=emb_dir,  
                output_prefix="emb_plot")
        
        return similarities
      
if __name__ == '__main__':
    predictions =  finetune_cells(skip_training = False, dataset_split = None, label = "disease", sample_data = .25, data_filter = 'hcm', epochs = 25, output_dir = 'hcm_model', model_location = 'hcm_model', 
                            emb_extract = False, geneformer_batch_size = 9, inference = True, dataset = "/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/")
    
