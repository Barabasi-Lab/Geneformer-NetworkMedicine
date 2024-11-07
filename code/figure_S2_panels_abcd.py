import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm
import os
import netmedpy

import gf_tools as gf
import sys

path_to_figs = "/work/ccnr/GeneFormer/aggregation_scripts/plotting_scripts/final_figure_scripts/out/"

drugbank = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/drug_repurposing/all_drugbank_drugs.csv')

PPI = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/ppi_with_gf_tokens.csv')
ppi = nx.from_pandas_edgelist(PPI, source = 'ens1', target = 'ens2')
with open("/work/ccnr/GeneFormer/conda_environment/geneformer/gene_name_id_dict.pkl", 'rb') as f:
    symbol_to_ensembl = pickle.load(f)

gda = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/GDA_Filtered_04042022.csv')
disease_genes = list(gda[gda['NewName'].str.contains("cardiomyopathy dilated")]['HGNC_Symbol'])
disease_genes_filtered = [symbol_to_ensembl[gene] for gene in disease_genes if gene in symbol_to_ensembl.keys()]

count_dict = {}
val_dict = {}
gene_dicts = {}
path_dict = {}
for i in range(6):
    fpath = f'/work/ccnr/GeneFormer/aggregated_matrices/aggregated_attentions/cardiomyopathy_failing/dilated/fine_tuned/max/layer_{i}/'
    k = f'attentions layer {i}'
    path_dict[k] = fpath
    counts = gf.load(fpath+'counts.pkl')
    vals = gf.load(fpath+'vals.pkl')
    vals = gf.replace_with_nans(counts,vals)
    np.fill_diagonal(vals,np.nan)
    count_dict[k] = counts
    val_dict[k] = vals
    gene_dicts[k] = gf.load(fpath+'gene_dict.pkl')

for i in range(6):
    fpath = f'/work/ccnr/GeneFormer/aggregated_matrices/aggregated_embeddings/cardiomyopathy_failing/dilated/fine_tuned/max/layer_{i}/'
    k = f'embeddings layer {i}'
    path_dict[k] = fpath
    counts = gf.load(fpath+'counts.pkl')
    vals = gf.load(fpath+'vals.pkl')
    vals = gf.replace_with_nans(counts,vals)
    np.fill_diagonal(vals,np.nan)
    count_dict[k] = counts
    val_dict[k] = vals
    gene_dicts[k] = gf.load(fpath+'gene_dict.pkl')

fpath = f'/work/ccnr/GeneFormer/aggregated_matrices/aggregated_embeddings/cardiomyopathy_failing/dilated/fine_tuned/max/layer_input/'
k = 'input embeddings'
path_dict[k] = fpath
counts = gf.load(fpath+'counts.pkl')
vals = gf.load(fpath+'vals.pkl')
vals = gf.replace_with_nans(counts,vals)
np.fill_diagonal(vals,np.nan)
count_dict[k] = counts
val_dict[k] = vals
gene_dicts[k] = gf.load(fpath+'gene_dict.pkl')

print("Checking if gene index matrices are the same")
for k in list(gene_dicts.keys()):
    for k2 in list(gene_dicts.keys()):
        print(gene_dicts[k]==gene_dicts[k2])
        

gene_dict = list(gene_dicts.values())[0]
ppi_sub_1 = gf.prune_ppi_nodes(ppi,gene_dict)
Gcc = sorted(nx.connected_components(ppi_sub_1), key=len, reverse=True)
ppi_sub_2 = ppi_sub_1.subgraph(Gcc[0])

# make weighted PPIs
ppis_dict = {}

for k in list(val_dict.keys()):
    weight_type = 'attns'
    if 'embed' in k:
        weight_type = 'embeds'
    ppi_weighted, missing_weights = gf.weight_network(val_dict[k],ppi_sub_2,gene_dicts[k],weight_type)
    largest_cc = max(nx.connected_components(ppi_weighted), key=len)
    ppi_weighted = ppi_weighted.subgraph(largest_cc)
    ppis_dict[k] = ppi_weighted

##### Make panels a and b (RWR for disease module discovery) #####

res_keys = list(ppis_dict.keys())
all_results = {k:{'fpr':[],'tpr':[],'auroc':[],'top_hits':[],'full_df':[]} for k in res_keys}

seed = 0

nn = 100
for i in tqdm(range(nn)):
    seed+=1
    print(i)
    np.random.seed(seed)
    
    for k in res_keys:
        pp = ppis_dict[k]
        cardio_sub = gf.get_disease_subgraph(pp, disease_genes_filtered)
        cardio_genes = cardio_sub.nodes()
        seed_num = int(len(cardio_genes)/5)
        tossed_genes = list(np.random.choice(cardio_genes,seed_num,replace=False))
        kept_genes = [gene for gene in cardio_genes if gene not in tossed_genes]
        
        result_df = gf.rwr(pp, set(kept_genes), 0.4)
        ranked_genes = result_df['Node']
        result_df['Positive'] = [1 if node in tossed_genes else 0 for node in ranked_genes]
        fpr, tpr = gf.auc_plot(np.cumsum(np.array(result_df['Positive'])))
        all_results[k]['fpr'].append(fpr)
        all_results[k]['tpr'].append(tpr)
        all_results[k]['auroc'].append(gf.auc(fpr,tpr))
        all_results[k]['top_hits'].append(np.cumsum(result_df['Positive'].tolist()[0:100]))
        all_results[k]['full_df'].append(result_df)
        
sns.set(style='whitegrid',context='talk',font_scale=0.7)
fig, axs = plt.subplots(1,2,figsize=(15,6))
for k in list(all_results.keys()):
    if 'embed' in k:
        linetype = '--'
    else:
        linetype = '-'
    tpr = np.mean(np.array(all_results[k]['tpr']),axis=0)
    fpr = np.mean(np.array(all_results[k]['fpr']),axis=0)
    tpr_std = np.std(np.array(all_results[k]['tpr']),axis=0)/np.sqrt(nn)
    top_hits = np.mean(np.array(all_results[k]['top_hits']),axis=0)
    top_hits_std = np.std(np.array(all_results[k]['top_hits']),axis=0)/np.sqrt(nn)
    truncated_x = [i for i in range(100)]

    axs[0].plot(fpr,tpr,linetype,label=k + ' AUROC = ' + str(round(np.mean(all_results[k]['auroc']),2)))
    axs[0].fill_between(fpr, tpr-tpr_std,tpr+tpr_std,alpha=0.2)
    axs[1].plot(truncated_x, top_hits,linetype, label=k)
    axs[1].fill_between(truncated_x, top_hits-top_hits_std,top_hits+top_hits_std,alpha=0.2)
    
axs[0].legend()
# axs[1].legend()

axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')

axs[1].set_xlabel('Top Candidates')
axs[1].set_ylabel('Cumulative True Positives')

fig.tight_layout()

plt.savefig(path_to_figs+'figure_S2_panels_ab.png',dpi=300,bbox_inches='tight')


##### Make panels c and d (proximity analysis for drug repurposing) ##### 
pos_count = 0
neg_count = 0
pos_indications = ['hypertension','ischemic heart','heart failure' ,'arrhythmia','cardiomyopathy']

filtered_db = drugbank[drugbank.Status.str.contains('approved')]
filtered_db = filtered_db[filtered_db.organism=='Humans']
unique_drugs = list(set(filtered_db.DB_id.to_list()))
print(len(unique_drugs))

all_targets = set()
pos_drugs = []
for drug in unique_drugs:
    possible_pos = False
    
    indication = filtered_db[filtered_db.DB_id==drug].Indication.tolist()[0]
    if type(indication)==str:
        for p in pos_indications:
            if p in indication:
                possible_pos = True
    if possible_pos==True:
        pos_count+=1
        all_targets.update(set(filtered_db[filtered_db.DB_id==drug].Gene_Target.tolist()))
        pos_drugs.append(drug)
        
neg_drugs = []
for drug in unique_drugs:
    possible_neg = False
    if drug not in pos_drugs:

        possible_neg = True
        targets = filtered_db[filtered_db.DB_id==drug].Gene_Target.tolist()
        if len(set(targets).intersection(all_targets))>0:
            possible_neg=False
    if possible_neg==True:
        neg_count+=1
        neg_drugs.append(drug)

########################

distance_matrices = {}
for k in list(ppis_dict.keys()):
    distance_path = path_dict[k] + 'distance_matrix.pkl'
    if os.path.isfile(distance_path) == True:
        D = netmedpy.load_distances(distance_path)
    else:
        D = netmedpy.all_pair_distances(ppis_dict[k], distance="shortest_path", n_processors=20, n_tasks=2000)
        netmedpy.save_distances(D, distance_path)
        D = netmedpy.load_distances(distance_path)
    distance_matrices[k] = D

all_scores = {}
for k in list(ppis_dict.keys()):
    G = ppis_dict[k]
    D = distance_matrices[k]
    cardio_sub = gf.get_disease_subgraph(G,disease_genes_filtered)
    S = list(cardio_sub.nodes())
    disease_dict = {'DCM':S}
    target_dict = {}
    labels_indiv = {}
    for drug in pos_drugs:
        targets = filtered_db[filtered_db.DB_id==drug].Gene_Target.tolist()
        T = [symbol_to_ensembl[gene] for gene in targets if gene in symbol_to_ensembl.keys()]
        target_dict[drug] = T
        labels_indiv[drug] = 1
    for drug in neg_drugs:
        targets = filtered_db[filtered_db.DB_id==drug].Gene_Target.tolist()
        T = [symbol_to_ensembl[gene] for gene in targets if gene in symbol_to_ensembl.keys()]
        target_dict[drug] = T
        labels_indiv[drug] = 0
    screen = netmedpy.screening(target_dict, disease_dict, G, D,
                                score="proximity",
                                properties=["z_score",'p_value_single_tail', 'raw_amspl'],
                                null_model="strength_binning",
                                n_iter=10000,
                                bin_size=100,
                                symmetric=False,
                                n_procs=20)
    score_df = screen['z_score']
    score_df = score_df.replace([np.inf, -np.inf], np.nan)
    score_df = score_df.dropna()
    score_df = score_df.reset_index()
    score_df.columns = ['Name','Score']
    score_df['Positive'] = [labels_indiv[d] for d in score_df.Name.tolist()]
    score_df = score_df.sort_values(by="Score",ascending=True)
    all_scores[k] = score_df
    
plt.clf()
fig,ax = plt.subplots(1,2,figsize=(15,6))

for k in list(all_scores.keys()):
    result_df = all_scores[k]
    fpr, tpr = gf.auc_plot(np.cumsum(np.array(result_df['Positive'])))
    auroc = gf.auc(fpr,tpr)
    top_hits = np.cumsum(result_df['Positive'].tolist()[0:100])
    
    if 'embed' in k:
        linetype='--'
    else:
        linetype='-'

    ax[0].plot(fpr,tpr,linetype)
    ax[1].plot(np.arange(0,100,1),top_hits,linetype,label=f"{k} AUROC = {round(auroc,2)}")
    ax[1].legend(bbox_to_anchor=(1.05,1))
    ax[0].set_xlabel('FPR')
    ax[0].set_ylabel('TPR')
    ax[1].set_xlabel('Top Candidates')
    ax[1].set_ylabel('Cumulative True Positives')
    
fig.tight_layout()
plt.savefig(path_to_figs+'figure_S2_panels_cd.png',dpi=300,bbox_inches='tight')