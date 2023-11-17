import os
import sys
GPU_NUMBER = [0] # CHANGE WITH MULTIGPU
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# imports
from sklearn.model_selection import train_test_split
import datetime
import subprocess
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from datasets import load_from_disk, Dataset
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import StratifiedKFold
import torch
from transformers import BertForTokenClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
from tqdm.notebook import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from geneformer import DataCollatorForGeneClassification, EmbExtractor
from geneformer.pretrainer import token_dictionary
import ast
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from geneformer import TranscriptomeTokenizer

def vote(logit_pair):    
    a, b = logit_pair
    if a > b:
        return 0
    elif b > a:
        return 1
    elif a == b:
        return "tie"
    
def py_softmax(vector):
	e = np.exp(vector)
	return e / e.sum()
    
 # Identifies cosine similarity between two embeddings. 0 is perfectly dissimilar and 1 is perfectly similar
def similarity(tensor1, tensor2, cosine = True):
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
    
# get cross-validated mean and sd metrics
def get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt):
    wts = [count/sum(all_tpr_wt) for count in all_tpr_wt]
    
    all_weighted_tpr = [a*b for a,b in zip(all_tpr, wts)]
    mean_tpr = np.sum(all_weighted_tpr, axis=0)
    mean_tpr[-1] = 1.0
    all_weighted_roc_auc = [a*b for a,b in zip(all_roc_auc, wts)]
    roc_auc = np.sum(all_weighted_roc_auc)
    roc_auc_sd = math.sqrt(np.average((all_roc_auc-roc_auc)**2, weights=wts))
    return mean_tpr, roc_auc, roc_auc_sd

def validate(data, targets, labels, nsplits, subsample_size, training_args, freeze_layers, output_dir, num_proc, num_labels, pre_model):
    # initiate eval metrics to return
    num_classes = len(set(labels))
    mean_fpr = np.linspace(0, 1, 100)
    
    # create 80/20 train/eval splits
    targets_train, targets_eval, labels_train, labels_eval = train_test_split(targets, labels ,test_size=0.25, shuffle=True)
    label_dict_train = dict(zip(targets_train, labels_train))
    label_dict_eval = dict(zip(targets_eval, labels_eval))
    
    # function to filter by whether contains train or eval labels
    def if_contains_train_label(example):
        a = label_dict_train.keys()
        b = example['input_ids']
        return not set(a).isdisjoint(b)

    def if_contains_eval_label(example):
        a = label_dict_eval.keys()
        b = example['input_ids']
        return not set(a).isdisjoint(b)
        
    # filter dataset for examples containing classes for this split
    print(f"Filtering training data")
    trainset = data.filter(if_contains_train_label, num_proc=num_proc)
    print(f"Filtered {round((1-len(trainset)/len(data))*100)}%; {len(trainset)} remain\n")
    print(f"Filtering evalation data")
    evalset = data.filter(if_contains_eval_label, num_proc=num_proc)
    print(f"Filtered {round((1-len(evalset)/len(data))*100)}%; {len(evalset)} remain\n")
    
    # minimize to smaller training sample
    training_size = min(subsample_size, len(trainset))
    trainset_min = trainset.select([i for i in range(training_size)])
    eval_size = min(training_size, len(evalset))
    half_training_size = round(eval_size/2)
    evalset_train_min = evalset.select([i for i in range(half_training_size)])
    evalset_oos_min = evalset.select([i for i in range(half_training_size, eval_size)])
        
    # label conversion functions
    def generate_train_labels(example):
        example["labels"] = [label_dict_train.get(token_id, -100) for token_id in example["input_ids"]]
        return example

    def generate_eval_labels(example):
        example["labels"] = [label_dict_eval.get(token_id, -100) for token_id in example["input_ids"]]
        return example
    
    # label datasets 
    print(f"Labeling training data")
    trainset_labeled = trainset_min.map(generate_train_labels)
    print(f"Labeling evaluation data")
    evalset_train_labeled = evalset_train_min.map(generate_eval_labels)
    print(f"Labeling evaluation OOS data")
    evalset_oos_labeled = evalset_oos_min.map(generate_eval_labels)
  
    # load model
    model = BertForTokenClassification.from_pretrained(
          pre_model,
          num_labels=num_labels,
          output_attentions = False,
          output_hidden_states = False,
    )
    if freeze_layers is not None:
        modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
            
    model = model.to(device)
    
    # add output directory to training args and initiate
    training_args["output_dir"] = output_dir
    training_args_init = TrainingArguments(**training_args)
    
    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForGeneClassification(),
        train_dataset=trainset_labeled,
        eval_dataset=evalset_train_labeled,
    )
    
    # train the gene classifier
    trainer.train()
    trainer.save_model(output_dir)
    
    fpr, tpr, interp_tpr, conf_mat = classifier_predict(trainer.model, evalset_oos_labeled, 200, mean_fpr)
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, auc_score
    
# cross-validate gene classifier
def cross_validate(data, targets, labels, nsplits, subsample_size, training_args, freeze_layers, output_dir, num_proc, num_labels, pre_model):
    # check if output directory already written to
    # ensure not overwriting previously saved model
    model_dir_test = os.path.join(output_dir, "ksplit0/models/pytorch_model.bin")
    #if os.path.isfile(model_dir_test) == True:
    #    raise Exception("Model already saved to this directory.")
    
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initiate eval metrics to return
    num_classes = len(set(labels))
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    all_roc_auc = []
    all_tpr_wt = []
    label_dicts = []
    confusion = np.zeros((num_classes,num_classes))
    
    # set up cross-validation splits
    skf = StratifiedKFold(n_splits=nsplits, random_state=0, shuffle=True)
    # train and evaluate
    iteration_num = 0
    for train_index, eval_index in tqdm(skf.split(targets, labels)):
        if len(labels) > 500:
            print("early stopping activated due to large # of training examples")
            if iteration_num == 3:
                break
                
        print(f"****** Crossval split: {iteration_num}/{nsplits-1} ******\n")
        
        # generate cross-validation splits
        targets_train, targets_eval = targets[train_index], targets[eval_index]
        labels_train, labels_eval = labels[train_index], labels[eval_index]
        label_dict_train = dict(zip(targets_train, labels_train))
        label_dict_eval = dict(zip(targets_eval, labels_eval))
        label_dicts += (iteration_num, targets_train, targets_eval, labels_train, labels_eval)
        
        # function to filter by whether contains train or eval labels
        def if_contains_train_label(example):
            a = label_dict_train.keys()
            b = example['input_ids']
            
            return not set(a).isdisjoint(b)

        def if_contains_eval_label(example):
            a = label_dict_eval.keys()
            b = example['input_ids']
            
            return not set(a).isdisjoint(b)
        
        # filter dataset for examples containing classes for this split
        print(f"Filtering training data")
        trainset = data.filter(if_contains_train_label, num_proc=num_proc)
        print(f"Filtered {round((1-len(trainset)/len(data))*100)}%; {len(trainset)} remain\n")
        print(f"Filtering evalation data")
        evalset = data.filter(if_contains_eval_label, num_proc=num_proc)
        print(f"Filtered {round((1-len(evalset)/len(data))*100)}%; {len(evalset)} remain\n")

        # minimize to smaller training sample
        training_size = min(subsample_size, len(trainset))
        trainset_min = trainset.select([i for i in range(training_size)])
        eval_size = min(training_size, len(evalset))
        half_training_size = round(eval_size/2)
        evalset_train_min = evalset.select([i for i in range(half_training_size)])
        evalset_oos_min = evalset.select([i for i in range(half_training_size, eval_size)])
        
        # label conversion functions
        def generate_train_labels(example):
            example["labels"] = [label_dict_train.get(token_id, -100) for token_id in example["input_ids"]]
            return example

        def generate_eval_labels(example):
            example["labels"] = [label_dict_eval.get(token_id, -100) for token_id in example["input_ids"]]
            return example
        
        # label datasets 
        print(f"Labeling training data")
        trainset_labeled = trainset_min.map(generate_train_labels)
        print(f"Labeling evaluation data")
        evalset_train_labeled = evalset_train_min.map(generate_eval_labels)
        print(f"Labeling evaluation OOS data")
        evalset_oos_labeled = evalset_oos_min.map(generate_eval_labels)
        
        # create output directories
        ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")
        ksplit_model_dir = os.path.join(ksplit_output_dir, "models/") 
        
        # ensure not overwriting previously saved model
        model_output_file = os.path.join(ksplit_model_dir, "pytorch_model.bin")
        #if os.path.isfile(model_output_file) == True:
        #    raise Exception("Model already saved to this directory.")

        # make training and model output directories
        subprocess.call(f'mkdir -p {ksplit_output_dir}', shell=True)
        subprocess.call(f'mkdir -p {ksplit_model_dir}', shell=True)
        
        # load model
        model = BertForTokenClassification.from_pretrained(
            pre_model,
            num_labels=num_labels,
            output_attentions = False,
            output_hidden_states = False,
        )
        if freeze_layers is not None:
            modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
                
        model = model.to(device)
        
        # add output directory to training args and initiate
        training_args["output_dir"] = ksplit_output_dir
        training_args_init = TrainingArguments(**training_args)
        
        # create the trainer
        trainer = Trainer(
            model=model,
            args=training_args_init,
            data_collator=DataCollatorForGeneClassification(),
            train_dataset=trainset_labeled,
            eval_dataset=evalset_train_labeled
        )

        # train the gene classifier
        trainer.train()
        
        # save model
        trainer.save_model(ksplit_model_dir)
        
        # evaluate model
        fpr, tpr, interp_tpr, conf_mat = classifier_predict(trainer.model, evalset_oos_labeled, 200, mean_fpr)
        
        # append to tpr and roc lists
        confusion = confusion + conf_mat
        all_tpr.append(interp_tpr)
        all_roc_auc.append(auc(fpr, tpr))
        # append number of eval examples by which to weight tpr in averaged graphs
        all_tpr_wt.append(len(tpr))
        
        iteration_num = iteration_num + 1
        
    # get overall metrics for cross-validation
    mean_tpr, roc_auc, roc_auc_sd = get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt)
    return all_roc_auc, roc_auc, roc_auc_sd, mean_fpr, mean_tpr, confusion, label_dicts
    
# Computes metrics
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

# plot ROC curve
def plot_ROC(bundled_data, title):
    plt.figure()
    lw = 2
    for roc_auc, roc_auc_sd, mean_fpr, mean_tpr, sample, color in bundled_data:
        plt.plot(mean_fpr, mean_tpr, color=color,
                 lw=lw, label="{0} (AUC {1:0.2f} $\pm$ {2:0.2f})".format(sample, roc_auc, roc_auc_sd))
                 
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig("ROC.png")
    
    return mean_fpr, mean_tpr, roc_auc
    
# plot confusion matrix
def plot_confusion_matrix(classes_list, conf_mat, title):
    display_labels = []
    i = 0
    for label in classes_list:
        display_labels += ["{0}\nn={1:.0f}".format(label, sum(conf_mat[:,i]))]
        i = i + 1
    display = ConfusionMatrixDisplay(confusion_matrix=preprocessing.normalize(conf_mat, norm="l1"), 
                                     display_labels=display_labels)
    display.plot(cmap="Blues",values_format=".2g")
    plt.title(title)
    plt.savefig("CM.png")
    
# Function to find the largest number smaller
# than or equal to N that is divisible by k
def find_largest_div(N, K):
    rem = N % K
    if(rem == 0):
        return N
    else:
        return N - rem
        
def preprocess_classifier_batch(cell_batch, max_len):
    if max_len == None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])
    def pad_label_example(example):
        example["labels"] = np.pad(example["labels"], 
                                   (0, max_len-len(example["input_ids"])), 
                                   mode='constant', constant_values=-100)
        example["input_ids"] = np.pad(example["input_ids"], 
                                      (0, max_len-len(example["input_ids"])), 
                                      mode='constant', constant_values=token_dictionary.get("<pad>"))
        example["attention_mask"] = (example["input_ids"] != token_dictionary.get("<pad>")).astype(int)
        return example
    padded_batch = cell_batch.map(pad_label_example)
    return padded_batch

# forward batch size is batch size for model inference (e.g. 200)
def classifier_predict(model, evalset, forward_batch_size, mean_fpr):
    predict_logits = []
    predict_labels = []
    model.to('cpu')
    model.eval()
  
    # ensure there is at least 2 examples in each batch to avoid incorrect tensor dims
    evalset_len = len(evalset)
    max_divisible = find_largest_div(evalset_len, forward_batch_size)
    if len(evalset) - max_divisible == 1:
        evalset_len = max_divisible
    
    max_evalset_len = max(evalset.select([i for i in range(evalset_len)])["length"])
    
    for i in range(0, evalset_len, forward_batch_size):
        max_range = min(i+forward_batch_size, evalset_len)
        batch_evalset = evalset.select([i for i in range(i, max_range)])
        padded_batch = preprocess_classifier_batch(batch_evalset, max_evalset_len)
        padded_batch.set_format(type="torch")
        
        input_data_batch = padded_batch["input_ids"]
        attn_msk_batch = padded_batch["attention_mask"]
        label_batch = padded_batch["labels"]
        with torch.no_grad():
            input_ids = input_data_batch
            attn_mask = attn_msk_batch
            labels =  label_batch
            outputs = model(
                
                input_ids = input_ids,
                attention_mask = attn_mask,
                labels = labels
            )
            predict_logits += [torch.squeeze(outputs.logits.to("cpu"))]
            predict_labels += [torch.squeeze(label_batch.to("cpu"))]
            
    logits_by_cell = torch.cat(predict_logits)
    all_logits = logits_by_cell.reshape(-1, logits_by_cell.shape[2])
    labels_by_cell = torch.cat(predict_labels)
    all_labels = torch.flatten(labels_by_cell)
    logit_label_paired = [item for item in list(zip(all_logits.tolist(), all_labels.tolist())) if item[1]!=-100]
    y_pred = [vote(item[0]) for item in logit_label_paired]
    y_true = [item[1] for item in logit_label_paired]
    logits_list = [item[0] for item in logit_label_paired]
    # probability of class 1
    y_score = [py_softmax(item)[1] for item in logits_list]
    conf_mat = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # plot roc_curve for this split
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.show()
    # interpolate to graph
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return fpr, tpr, interp_tpr, conf_mat 
    
def classify_genes(gene_info = "./Genecorpus-30M/example_input_files/gene_info_table.csv", genes = "./Genecorpus-30M/example_input_files/gene_classification/dosage_sensitive_tfs/dosage_sens_tf_labels.csv",
       corpus_30M = "./Genecorpus-30M/genecorpus_30M_2048.dataset/", model = '.',
       max_input_size = 2 ** 11, max_lr = 5e-5, freeze_layers = 4, num_gpus = 1, num_proc = os.cpu_count(), geneformer_batch_size = 9, epochs = 1, filter_dataset = 50_000,
       emb_extract = True, emb_layer = 0, forward_batch = 200, filter_data = None, inference = False, k_validate = True, model_location = "230917_geneformer_GeneClassifier_dosageTF_L2048_B12_LR5e-05_LSlinear_WU500_E1_Oadamw_n10000_F4/", skip_training = False, emb_dir = 'gene_emb', output_dir = None, max_cells = 1000, num_cpus = os.cpu_count()):
      
     
    """"
    Primary Parameters
    -----------
    
    gene_info: path
        Path to gene mappings
    
    corpus_30M: path
        Path to 30M Gene Corpus
        
    model: path
        Path to pretrained GeneFormer model
        
    genes: path
        Path to csv file containing different columns of genes and the column labels
        
    inference: bool
        Whether the model should be used to run inference. If False, model will train with labeled data instead. Defaults to False
        
    k_validate: bool
        Whether the model should run k-fold validation or simply perform regular training/evaluate. Defaults to True
        
    skip_training: bool
        Whether the model should skip the training portion. Defaults to False
        
    emb_extract: bool
        WHether the model should extract embeddings for a given gene (WIP)
        
    
    Customization Parameters
    -----------
    
    freeze_layers: int
        Freezes x number of layers from the model. Default is 4 (2 non-frozen layers)
        
    filter_dataset: int
        Number of cells to filter from 30M dataset. Default is 50_000
        
    emb_layer: int
        What layer embeddings are extracted from. Default is 4
        
    filter_data: str, list
        Filters down embeddings to a single category. Default is None
        
    
    """
          
    # table of corresponding Ensembl IDs, gene names, and gene types (e.g. coding, miRNA, etc.)
    gene_info = pd.read_csv(gene_info, index_col=0)
    labels = gene_info.columns

    # create dictionaries for corresponding attributes
    gene_id_type_dict = dict(zip(gene_info["ensembl_id"],gene_info["gene_type"]))
    gene_name_id_dict = dict(zip(gene_info["gene_name"],gene_info["ensembl_id"]))
    gene_id_name_dict = {v: k for k,v in gene_name_id_dict.items()}

    # function for preparing targets and labels
    def prep_inputs(label_store, id_type):
        target_list = []
        if id_type == "gene_name":
            for key in list(label_store.keys()):
                targets = [gene_name_id_dict[gene] for gene in label_store[key] if gene_name_id_dict.get(gene) in token_dictionary]
                targets_id = [token_dictionary[gene] for gene in targets]
                target_list.append(targets_id)
        elif id_type == "ensembl_id":
            for key in list(label_store.keys()):
                targets = [gene for gene in label_store[key] if gene in token_dictionary]
                targets_id = [token_dictionary[gene] for gene in targets]
                target_list.append(targets_id)
          
        targets, labels = [], []
        for targ in target_list:
            targets = targets + targ
        targets = np.array(targets)
        for num, targ in enumerate(target_list):
            label = [num]*len(targ) 
            labels = labels + label
        labels = np.array(labels)
        unique_labels = num + 1
        
        nsplits = min(5, min([len(targ) for targ in target_list])-1)
        assert nsplits > 2
        
        return targets, labels, nsplits, unique_labels
    
    if skip_training == False:
        # preparing targets and labels for dosage sensitive vs insensitive TFs
        gene_classes = pd.read_csv(genes, header=0)
        if filter_data == None:
            labels = gene_classes.columns
        else:
            if isinstance(filter_data, list):
                labels = filter_data
            else:
                labels = [filter_data]
        label_store = {}
        
        # Dictionary for decoding labels
        decode = {i:labels[i] for i in range(len(labels))}
  
        for label in labels:
            label_store[label] = gene_classes[label].dropna()
    
        targets, labels, nsplits, unique_labels = prep_inputs(label_store, "ensembl_id")
        
        
        
        # load training dataset
        train_dataset=load_from_disk(corpus_30M)
        shuffled_train_dataset = train_dataset.shuffle(seed=42)
        subsampled_train_dataset = shuffled_train_dataset.select([i for i in range(filter_dataset)])
        lr_schedule_fn = "linear"
        warmup_steps = 500
        optimizer = "adamw"
        subsample_size = 10_000
    
        training_args = {
            "learning_rate": max_lr,
            "do_train": True,
            "evaluation_strategy": "no",
            "save_strategy": "epoch",
            "logging_steps": 10,
            "group_by_length": True,
            "length_column_name": "length",
            "disable_tqdm": False,
            "lr_scheduler_type": lr_schedule_fn,
            "warmup_steps": warmup_steps,
            "weight_decay": 0.001,
            "per_device_train_batch_size": geneformer_batch_size,
            "per_device_eval_batch_size": geneformer_batch_size,
            "num_train_epochs": epochs,
        }
        
        # define output directory path
        current_date = datetime.datetime.now()
        datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
        
        if output_dir == None:
            training_output_dir = Path(f"{datestamp}_geneformer_GeneClassifier_dosageTF_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_n{subsample_size}_F{freeze_layers}/")
        else:
            training_output_dir = Path(output_dir)
        
        # make output directory
        subprocess.call(f'mkdir -p {training_output_dir}', shell=True)
        
        # Places number of classes +  in directory
        num_classes = len(set(labels))
        info_list = [num_classes, decode]
        
        with open(training_output_dir / 'classes.txt', 'w') as f:
            f.write(str(info_list))
        
        subsampled_train_dataset.save_to_disk(output_dir / 'dataset')
            
        if k_validate == True:
            ksplit_model ="ksplit0/models"
            ksplit_model_test = os.path.join(training_output_dir, ksplit_model)
            #if os.path.isfile(ksplit_model_test) == True:
            #    raise Exception("Model already saved to this directory.")
                # cross-validate gene classifier
            all_roc_auc, roc_auc, roc_auc_sd, mean_fpr, mean_tpr, confusion, label_dicts = cross_validate(subsampled_train_dataset, targets, labels, nsplits, subsample_size, training_args, freeze_layers, training_output_dir, 1, unique_labels, model)
            
            bundled_data = []
            bundled_data += [(roc_auc, roc_auc_sd, mean_fpr, mean_tpr, "Geneformer", "red")]
            graph_title = " ".join([i + ' vs' if count < len(label_store) - 1 else i for count, i in enumerate(label_store)])
            fpr, tpr, auc = plot_ROC(bundled_data, 'Dosage Sensitive vs Insensitive TFs')
            print(auc)
            # plot confusion matrix
            plot_confusion_matrix(label_store, confusion, "Geneformer")
        else:    
            fpr, tpr, auc  = validate(subsampled_train_dataset, targets, labels, nsplits, subsample_size, training_args, freeze_layers, training_output_dir, 1, unique_labels, model)
            print(auc)
    
    if inference == True:
        # preparing targets and labels for dosage sensitive vs insensitive TFs
        gene_classes = pd.read_csv(genes, header=0)
        targets = []
        for column in gene_classes.columns:  
            targets += list(gene_classes[column])
        tokens = []
        for target in targets:
            try:
                tokens.append(token_dictionary[target])
            except:
                tokens.append(0)
                
        targets = torch.LongTensor([tokens])
        
          
        with open(f'{model_location}classes.txt', 'r') as f:
            info_list = ast.literal_eval(f.read())
        num_classes = info_list[0]
        labels = info_list[1]
        
        model = BertForTokenClassification.from_pretrained(
            model_location,
            num_labels=num_classes,
            output_attentions = False,
            output_hidden_states = False,
            local_files_only = True
        )
        if freeze_layers is not None:
            modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
                
        model = model.to(device)
        
        # evaluate model
        predictions = F.softmax(model(targets.to(device))["logits"], dim = -1).argmax(-1)[0]
        predictions = [labels[int(pred)] for pred in predictions]
        
        return predictions
    
    # Extracts aggregate gene embeddings for each label 
    if emb_extract == True:
        with open(f'{model_location}/classes.txt', 'r') as f:
            data = ast.literal_eval(f.read())
        num_classes = data[0]
        decode = data[1]
        
        gene_classes = pd.read_csv(genes, header=0)
        labels = gene_classes.columns
        tokenize = TranscriptomeTokenizer()
        
        label_dict = {}
        for label in labels:
            genes = gene_classes[label]
            tokenized_genes = []
            for gene in genes:
                try:
                    tokenized_genes.append(tokenize.gene_token_dict[gene])
                except:
                    continue
            label_dict[label] = tokenized_genes
      
        embex = EmbExtractor(model_type="GeneClassifier", num_classes=num_classes, emb_mode = "gene",
                             filter_data=None, max_ncells=max_cells, emb_layer=emb_layer,
                             emb_label=label_dict, labels_to_plot=list(labels), forward_batch_size=forward_batch, nproc=num_cpus)
      

        subprocess.call(f'mkdir -p {emb_dir}', shell = True)
      
        embs = embex.extract_embs(model_directory = model_location, input_data_file = model_location / 'dataset', output_directory = emb_dir, output_prefix = f"{label}_embbeddings")

        emb_dict = {label:[] for label in list(set(labels))}
        similarities = {key:{} for key in list(emb_dict.keys())}
        
        for column in embs.columns:
            remaining_cols = [k for k in embs.columns if k != column]
            for k in remaining_cols:
                embedding = torch.Tensor(embs[k])
                sim = similarity(torch.Tensor(embs[column]), embedding, cosine = True)
                similarities[column][k] = sim
        
        plot_similarity_heatmap(similarities)
        print(similarities)
        
        return similarities
    
if __name__ == '__main__':
    classify_genes(k_validate = False, inference = False, skip_training = False, emb_extract = True, output_dir = Path('gene_emb'), model_location = Path('gene_emb'), epochs = 5, gene_info = "./Genecorpus-30M/example_input_files/gene_info_table.csv", genes = "./Genecorpus-30M/example_input_files/gene_classification/dosage_sensitive_tfs/dosage_sens_tf_labels.csv", corpus_30M = "./Genecorpus-30M/genecorpus_30M_2048.dataset/")

        