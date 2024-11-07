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
import operator
import sys

# paths to weight matrices are on line 87 for panels a and b, 198 for c and d, and 343 for e and f

sns.set(style='whitegrid',context='talk',font_scale=0.75)

path_to_figs = "path to figure output folder"

PPI = pd.read_csv('./supplemental_data/ppi.csv')
ppi = nx.from_pandas_edgelist(PPI, source = 'ens1', target = 'ens2')

gda = pd.read_csv('./supplemental_data/gda.csv')

with open("./supplemental_data/gene_name_id_dict.pkl", 'rb') as f:
    symbol_to_ensembl = pickle.load(f)
    
disease_genes = list(gda[gda['NewName'].str.contains("cardiomyopathy dilated")]['HGNC_Symbol'])
disease_genes_filtered = [symbol_to_ensembl[gene] for gene in disease_genes if gene in symbol_to_ensembl.keys()]

def combined_counts(rankings1, rankings2,p=4):
    """
    Combine two ranking DataFrames and calculate various ranking scores based on shared entries.

    Parameters
    ----------
    rankings1 : pd.DataFrame
        DataFrame containing ranked items with columns "Name" and "Score".
    rankings2 : pd.DataFrame
        DataFrame containing ranked items with columns "Name" and "Score".
    p : int, optional
        Exponent for computing the C-rank, by default set to 4.

    Returns
    -------
    final_ranks : pd.DataFrame
        DataFrame containing combined ranks with columns for each ranking method, including:
        "Name", "attn_rank", "embed_rank", "borda_rank", "dowdall_rank", "average_rank", and "crank".
    """
    
    ranking = [i+1 for i in range(rankings1.shape[0])]
    ranking2 = [i+1 for i in range(rankings2.shape[0])]
    rankings1["ranking"]=ranking
    rankings2["ranking"]=ranking2
    rankings2 = rankings2.rename(columns={'ranking': 'ranking2',"Score":"Score2","Positive":"Positive2"})
    rankings = rankings1.merge(rankings2,on="Name")
    max_rank = rankings.shape[0]

    if rankings1.shape[0]!=rankings2.shape[0]:
        print("Dataframes are not the same shape")
        print(f"First Rankings set has {rankings1.shape[0]} voters")
        print(f"Second Rankings set has {rankings2.shape[0]} voters")
        print(f"And {max_rank} of these voters are shared")
    
    nodes = list(rankings["Name"])
    
    borda_tups = []
    for i in range(len(nodes)):
        target_label = nodes[i]
        rank1 = rankings[rankings["Name"]==target_label]["ranking"].values[0]
        rank2 = rankings[rankings["Name"]==target_label]["ranking2"].values[0]
        
        borda_rank = (max_rank-rank1) + (max_rank-rank2)
        dowdall_rank = (1/rank1) + (1/rank2)
        average_rank = (rank1+rank2)/2
        crank = (1/rank1**p) + (1/rank2**p)
        borda_tups.append((target_label,rank1,rank2,borda_rank,dowdall_rank,average_rank,crank))
    
    borda_df = pd.DataFrame(borda_tups,columns=["Name","attn_rank","embed_rank","borda_rank","dowdall_rank","average_rank","crank"])
    final_ranks = rankings.merge(borda_df,on="Name")
    final_ranks = final_ranks.drop(columns = ["ranking","ranking2","Score","Score2","Positive2"])
    
    return final_ranks

####################################################################################################################

attn_path = './data/aggregated_matrices/aggregated_attentions/genecorpus/pretrained/max/'
embed_path = './data/aggregated_matrices/aggregated_embeddings/genecorpus/pretrained/max/'

attn_counts = gf.load(attn_path+'counts.pkl')
attn_vals = gf.load(attn_path+'vals.pkl')
attn_gdict = gf.load(attn_path+'gene_dict.pkl')

embed_counts = gf.load(embed_path+'counts.pkl')
embed_vals = gf.load(embed_path+'vals.pkl')
embed_gdict = gf.load(embed_path+'gene_dict.pkl')

attn_vals = gf.replace_with_nans(attn_counts,attn_vals)
embed_vals = gf.replace_with_nans(embed_counts,embed_vals)

attn_flattened = attn_vals.flatten()
embed_flattened = embed_vals.flatten()

attn_flattened = attn_flattened[~np.isnan(attn_flattened)]
embed_flattened = embed_flattened[~np.isnan(embed_flattened)]

plt.scatter(attn_flattened,embed_flattened,s=1,alpha=0.3)
plt.xlabel('Attention Weight')
plt.ylabel('Cosine Similarity')
plt.savefig(path_to_figs+'figure_S3_panel_a.png',dpi=300,bbox_inches='tight')
plt.clf()

print("Checking if gene index matrices are the same")
print(attn_gdict==embed_gdict)

ls = list(attn_gdict.keys())
ppi_in_data = []
for node in list(ppi.nodes()):
    if node in ls:
        ppi_in_data.append(node)

ppi_sub = ppi.subgraph(ppi_in_data)
ppi_sub.number_of_edges()

def rank_weights(mat,gene_dict,ppi):
    index_dict = {v:k for k,v in gene_dict.items()}
    tups = []
    background_count = 0
    ppi_count = 0
    for i in range(mat.shape[0]):
        for j in range(i,mat.shape[1]):
            if np.isnan(mat[i][j])==False:
                source = index_dict[i]
                target = index_dict[j]
                if ppi.has_edge(source,target):
                    tups.append((mat[i][j],"1_"+str(ppi_count),1))
                    ppi_count+=1
                else:
                    tups.append((mat[i][j],"0_"+str(background_count),0))
                    background_count+=1
                    
    tups = sorted(tups, key=operator.itemgetter(0), reverse=True)
    
    tuplicates = []  # because they're duplicate tups. Get it?
    for i,tup in enumerate(tups):
        tuplicate = (i+1,tup[1],tup[2])
        tuplicates.append(tuplicate)
            
    rankings = pd.DataFrame(tuplicates,columns=["ranking","Name","pos_or_neg"])
    rankings = rankings.sort_values(by="ranking")
    return rankings

attn_ranks = rank_weights(attn_vals,attn_gdict,ppi_sub)
embed_ranks = rank_weights(embed_vals,attn_gdict,ppi_sub)
print("weights ranked")

combined_rankings = combined_counts(attn_ranks, embed_ranks,p=4)

up_cols = ["attn_rank","embed_rank","average_rank"]
up_labels = ["Attention weights","Cosine similarities","Average rank"]
down_cols = ["borda_rank","dowdall_rank","crank"]
down_labels = ["Borda rank","Dowdall rank","C rank"]

counter_dict = {}
for i,col in enumerate(up_cols):
    temp = rankings.sort_values(by=col,ascending=True)
    counter = np.cumsum(np.array(list(temp["pos_or_neg"])))
    counter_dict[up_labels[i]] = counter
    
for i,col in enumerate(down_cols):
    temp = rankings.sort_values(by=col,ascending=False)
    counter = np.cumsum(np.array(list(temp["pos_or_neg"])))
    counter_dict[down_labels[i]] = counter

def auc(means):
    tpr = [means[i]/means[-1] for i in range(len(means))]
    fpr = [(i-means[i])/((i-means[i])+((len(means)-i)-(means[-1]-means[i]))) for i in range(len(means))]
    return np.trapz(tpr,fpr)

def auc_plot(means):
    tpr = [means[i]/means[-1] for i in range(len(means))]
    fpr = [(i-means[i])/((i-means[i])+((len(means)-i)-(means[-1]-means[i]))) for i in range(len(means))]
    return fpr,tpr
    
for k,v in counter_dict.items():
    fpr,tpr = auc_plot(v)
    plt.plot(fpr[::100],tpr[::100],label=f'{k}, AUROC = {round(auc(v),2)}')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.savefig(path_to_figs+'figure_S3_panel_b.svg')

#### Combined rankings for disease module detection (panels a and b) ####

base_path = "./data/aggregated_matrices/"
mat_paths = ["aggregated_attentions/dcm_samples/fine_tuned/max/layer_4/", "aggregated_embeddings/dcm_samples/fine_tuned/max/layer_4/"]
labels = ["fine-tuned attentions", "fine-tuned embeddings"]

count_dict = {}
val_dict = {}
gene_dicts = {}
path_dict = {}
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
    path_dict[k] = full_path
    
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
gdict = list(gene_dicts.values())[0]
for k in list(val_dict.keys()):
    weight_type = 'attns'
    if 'embed' in k:
        weight_type = 'embeds'
    ppi_weighted, missing_weights = gf.weight_network(val_dict[k],ppi_sub_2,gdict,weight_type)
    largest_cc = max(nx.connected_components(ppi_weighted), key=len)
    ppi_weighted = ppi_weighted.subgraph(largest_cc)
    ppis_dict[k] = ppi_weighted

res_keys = list(ppis_dict.keys())
all_results = {k:{'fpr':[],'tpr':[],'auroc':[],'top_hits':[],'full_df':[]} for k in res_keys}

cardio_sub = gf.get_disease_subgraph(ppi_sub_2, disease_genes_filtered)
cardio_genes = cardio_sub.nodes()
seed_num = int(len(cardio_genes)/5)

seed = 0

nn = 100
combined_dfs = []
for i in tqdm(range(nn)):
    seed+=1
    print(i)
    np.random.seed(seed)
    
    tossed_genes = list(np.random.choice(cardio_genes,seed_num,replace=False))
    kept_genes = [gene for gene in cardio_genes if gene not in tossed_genes]
    
    ranking_dfs = []
    all_to_s = []
    for k in res_keys:
        pp = ppis_dict[k]
        result_df = gf.rwr(pp, set(kept_genes), 0.4)
        ranked_genes = result_df['Node']
        result_df.rename(columns={"Node":"Name"},inplace=True)
        result_df['Positive'] = [1 if node in tossed_genes else 0 for node in ranked_genes]
        
        ranking_dfs.append(result_df)
    combined_df = combined_counts(ranking_dfs[0],ranking_dfs[1])
    combined_dfs.append(combined_df)
        
ascend_columns = ["attn_rank","embed_rank","average_rank"]
descend_columns = ["borda_rank", "dowdall_rank","crank"]
all_cols = ["attn_rank","embed_rank","average_rank","borda_rank", "dowdall_rank","crank"]

score_dict = {col:{'tpr':[],'fpr':[],'auroc':[],'top candidates':[]} for col in all_cols}

for col in ascend_columns:
    for i in range(nn):
        combined = combined_dfs[i]
        temp = combined.sort_values(by=col,ascending=True)
        labs = np.array(list(temp.Positive))
        scores = np.cumsum(labs)
        fpr,tpr = gf.auc_plot(scores)
        score_dict[col]['fpr'].append(fpr)
        score_dict[col]['tpr'].append(tpr)
        score_dict[col]['auroc'].append(gf.auc(fpr,tpr))
        score_dict[col]['top candidates'].append(scores[0:100])
        

for col in descend_columns:
    for i in range(nn):
        combined = combined_dfs[i]
        temp = combined.sort_values(by=col,ascending=False)
        labs = np.array(list(temp.Positive))
        scores = np.cumsum(labs)
        fpr,tpr = gf.auc_plot(scores)
        score_dict[col]['fpr'].append(fpr)
        score_dict[col]['tpr'].append(tpr)
        score_dict[col]['auroc'].append(gf.auc(fpr,tpr))
        score_dict[col]['top candidates'].append(scores[0:100])
        
score_dict['attn_rank']['label'] = "Attention Rank"
score_dict['embed_rank']['label'] = "Embedding Rank"
score_dict['average_rank']['label'] = "Average Rank"
score_dict['borda_rank']['label'] = "Borda Rank"
score_dict['dowdall_rank']['label'] = "Dowdall Rank"
score_dict['crank']['label'] = "C Rank"

fig, axs = plt.subplots(1,2,figsize=(15,6))

sns.set(style='whitegrid',context='talk')

for k in list(score_dict.keys()):
    tpr = np.mean(np.array(score_dict[k]['tpr']),axis=0)
    fpr = np.mean(np.array(score_dict[k]['fpr']),axis=0)
    tpr_std = np.std(np.array(score_dict[k]['tpr']),axis=0)/np.sqrt(nn)
    top_hits = np.mean(np.array(score_dict[k]['top candidates']),axis=0)
    top_hits_std = np.std(np.array(score_dict[k]['top candidates']),axis=0)/np.sqrt(nn)
    truncated_x = [i for i in range(100)]

    axs[0].plot(fpr,tpr,label=score_dict[k]['label'] + ' AUROC = ' + str(round(np.mean(score_dict[k]['auroc']),2)))
    axs[0].fill_between(fpr, tpr-tpr_std,tpr+tpr_std,alpha=0.2)
    axs[1].plot(truncated_x, top_hits, label=score_dict[k]['label'])
    axs[1].fill_between(truncated_x, top_hits-top_hits_std,top_hits+top_hits_std,alpha=0.2)
    
axs[0].legend()
# axs[1].legend()

axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')

axs[1].set_xlabel('Top Candidates')
axs[1].set_ylabel('Cumulative True Positives')

fig.tight_layout()
plt.savefig(path_to_figs+'figure_S3_panels_cd.png',dpi=300,bbox_inches='tight')

##### Make panels c and d (drug repurposing) #####
# reload matrices (because the weigths that perform well for drug repurposing are not necessarily the same as for disease module detection)
base_path = "./data/aggregated_matrices/"
mat_paths = ["aggregated_attentions/dcm_samples/fine_tuned/max/layer_5/", "aggregated_embeddings/dcm_samples/fine_tuned/max/layer_0/"]
labels = ["layer 5 attentions", "layer 0 embeddings"]

count_dict = {}
val_dict = {}
gene_dicts = {}
path_dict = {}
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
    path_dict[k] = full_path
    
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
    
### filter drugbank ###########
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
##################################

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

combined = combined_counts(all_scores[0],all_scores[1])

ascend_columns = ["attn_rank","embed_rank","average_rank"]
descend_columns = ["borda_rank", "dowdall_rank","crank"]

score_dict = {}
for col in ascend_columns:
    temp = combined.sort_values(by=col,ascending=True)
    labs = np.array(list(temp.Positive))
    scores = np.cumsum(labs)
    score_dict[col]=scores

for col in descend_columns:
    temp = combined.sort_values(by=col,ascending=False)
    labs = np.array(list(temp.Positive))
    scores = np.cumsum(labs)
    score_dict[col]=scores

plt.clf()
fig,ax = plt.subplots(1,2,figsize=(15,6))    
for k,v in score_dict.items():
    fpr,tpr = gf.auc_plot(v)
    ax[0].plot(fpr,tpr)
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_xlabel('False Positive Rate')

    ax[1].plot(v[0:100],label=f"{k}, AUC={round(gf.auc(fpr,tpr),2)}")
    ax[1].set_ylabel('Cumulative True Positives')
    ax[1].set_xlabel('Top Candidates')

ax[1].legend(bbox_to_anchor=(1.05,1))
fig.tight_layout()
plt.savefig(path_to_figs+'figure_S3_panels_cd.png',dpi=300,bbox_inches='tight')

