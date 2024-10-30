import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm

import gf_tools as gf

path_to_figs = "/work/ccnr/GeneFormer/aggregation_scripts/plotting_scripts/final_figure_scripts/out/"

PPI = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/ppi_with_gf_tokens.csv')
ppi = nx.from_pandas_edgelist(PPI, source = 'ens1', target = 'ens2')
with open("/work/ccnr/GeneFormer/conda_environment/geneformer/gene_name_id_dict.pkl", 'rb') as f:
    symbol_to_ensembl = pickle.load(f)

gda = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/GDA_Filtered_04042022.csv')
disease_genes = list(gda[gda['NewName'].str.contains("cardiomyopathy dilated")]['HGNC_Symbol'])
disease_genes_filtered = [symbol_to_ensembl[gene] for gene in disease_genes if gene in symbol_to_ensembl.keys()]

base_path = "/work/ccnr/GeneFormer/aggregated_matrices/"
mat_paths = ["aggregated_attentions/cardiomyopathy_failing/dilated/fine_tuned/max/layer_4/","aggregated_attentions/cardiomyopathy_failing/dilated/pretrained/max/",
"aggregated_embeddings/cardiomyopathy_failing/dilated/fine_tuned/max/layer_4/","aggregated_embeddings/cardiomyopathy_failing/dilated/pretrained/max/"]
labels = ["fine-tuned attentions","pretrained attentions", "fine-tuned embeddings", "pretrained embeddings"]

count_dict = {}
val_dict = {}
gene_dicts = {}
for i in range(len(mat_paths)):
    full_path = base_path+mat_paths[i]
    k = labels[i]
    counts = gf.load(full_path+"counts.pkl")
    vals = gf.load(full_path+"vals.pkl")
    gdict = gf.load(full_path+"gene_dict.pkl")
    vals = gf.replace_with_nans(counts,vals)
    np.fill_diagonal(vals,np.nan)
    count_dict[k] = counts
    val_dict[k] = vals
    gene_dicts[k] = gdict

print("Checking if gene index matrices are the same (if any of the following are False, some sample sets are not the same)")
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
    
ppis_dict['unweighted'] = ppi_sub_2

res_keys = list(ppis_dict.keys())
all_results = {k:{'fpr':[],'tpr':[],'auroc':[],'precision':[],'recall':[],'auprc':[],'top_hits':[],'full_df':[]} for k in res_keys}

seed = 0
n_realizations = 1
n_2 = 100
for i in tqdm(range(n_2)):
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
        recall, precision = gf.prc_plot(np.cumsum(np.array(result_df['Positive'])))
        all_results[k]['precision'].append(precision)
        all_results[k]['recall'].append(recall)
        all_results[k]['auprc'].append(gf.auc(recall,precision))
        all_results[k]['top_hits'].append(np.cumsum(result_df['Positive'].tolist()[0:100]))
        all_results[k]['full_df'].append(result_df)
        
sns.set(style='whitegrid',context='talk',font_scale=0.7)
fig, axs = plt.subplots(2,2,figsize=(15,12))
for k in list(all_results.keys()):
    linetype = '-'
    tpr = np.mean(np.array(all_results[k]['tpr']),axis=0)
    fpr = np.mean(np.array(all_results[k]['fpr']),axis=0)
    tpr_std = np.std(np.array(all_results[k]['tpr']),axis=0)/np.sqrt(20)
    precision = np.mean(np.array(all_results[k]['precision']),axis=0)
    recall = np.mean(np.array(all_results[k]['recall']),axis=0)
    precision_std = np.std(np.array(all_results[k]['precision']),axis=0)
    recall_std = np.std(np.array(all_results[k]['recall']),axis=0)/np.sqrt(20)
    top_hits = np.mean(np.array(all_results[k]['top_hits']),axis=0)
    top_hits_std = np.std(np.array(all_results[k]['top_hits']),axis=0)/np.sqrt(20)
    truncated_x = [i for i in range(100)]

    axs[0,0].plot(fpr,tpr,linetype,label=k + ' AUROC = ' + str(round(np.mean(all_results[k]['auroc']),2)))
    axs[0,0].fill_between(fpr, tpr-tpr_std,tpr+tpr_std,alpha=0.2)
    axs[0,1].plot(recall,precision,linetype,label=k + ' AUPRC = ' + str(round(np.mean(all_results[k]['auprc']),2)))
    axs[0,1].fill_between(recall, precision-precision_std,precision+precision_std,alpha=0.2)
    axs[1,0].plot(truncated_x, precision[0:100],linetype,label=k)
    axs[1,0].fill_between(truncated_x, precision[0:100]-precision_std[0:100],precision[0:100]+precision_std[0:100],alpha=0.2)
    axs[1,1].plot(truncated_x, top_hits,linetype, label=k)
    axs[1,1].fill_between(truncated_x, top_hits-top_hits_std,top_hits+top_hits_std,alpha=0.2)
    
axs[0,0].legend()
axs[1,1].legend()
axs[1,0].legend()
axs[0,1].legend()

axs[0,0].set_xlabel('False Positive Rate')
axs[0,0].set_ylabel('True Positive Rate')

axs[0,1].set_xlabel('Recall')
axs[0,1].set_ylabel('Precision')

axs[1,0].set_xlabel('Top Candidates')
axs[1,0].set_ylabel('Precision')

axs[1,1].set_xlabel('Top Candidates')
axs[1,1].set_ylabel('Cumulative True Positives')

fig.tight_layout()
plt.savefig(f'{path_to_figs}figure_3.png',dpi=300,bbox_inches='tight')
