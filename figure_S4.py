import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
import networkx as nx

import gf_tools as gf

sns.set(style="whitegrid",context="talk")

path_to_figs = "/work/ccnr/GeneFormer/aggregation_scripts/plotting_scripts/final_figure_scripts/out/"

# paths to genecorpus weight matrices
embed_path = "/work/ccnr/GeneFormer/aggregated_matrices/aggregated_embeddings/genecorpus/pretrained/max/"
attn_path = "/work/ccnr/GeneFormer/aggregated_matrices/aggregated_attentions/genecorpus/pretrained/max/"

# paths to dcm weight matrices
embed_path_dcm = "/work/ccnr/GeneFormer/aggregated_matrices/aggregated_embeddings/cardiomyopathy_failing/dilated/pretrained/max/"
attn_path_dcm = "/work/ccnr/GeneFormer/aggregated_matrices/aggregated_attentions/cardiomyopathy_failing/dilated/pretrained/max/"

PPI = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/ppi_with_gf_tokens.csv')
ppi = nx.from_pandas_edgelist(PPI, source = 'ens1', target = 'ens2')

gda = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/GDA_Filtered_04042022.csv')

with open("/work/ccnr/GeneFormer/conda_environment/geneformer/gene_name_id_dict.pkl", 'rb') as f:
    symbol_to_ensembl = pickle.load(f)

embed_counts = gf.load(embed_path+"counts.pkl")
embed_vals = gf.load(embed_path+"vals.pkl")
embed_dict = gf.load(embed_path+"gene_dict.pkl")

attn_counts = gf.load(attn_path+"counts.pkl")
attn_vals = gf.load(attn_path+"vals.pkl")
attn_dict = gf.load(attn_path+"gene_dict.pkl")

if attn_dict!=embed_dict:
    pint("The gene to index dictionaries are not the same for attention weights and embeddings. Please make sure you are using the same set of samples for both")

embed_vals = gf.replace_with_nans(embed_counts,embed_vals)
attn_vals = gf.replace_with_nans(attn_counts,attn_vals)

ppi_sub = gf.prune_ppi_nodes(ppi,attn_dict)

randomized_ppi = gf.degree_preserving_randomization(ppi_sub, n_swaps = ppi_sub.number_of_edges())
completely_randomized = gf.non_degree_preserving_randomization(ppi_sub)

#### Separation of PPI and background (panels a and b) ####

embed_ppi, indices_used = gf.get_graph_weights(ppi_sub,embed_vals,embed_dict)
attn_ppi, indices_used_attns = gf.get_graph_weights(ppi_sub,attn_vals,attn_dict) # because the samples are the same, the indices used will be the same as well

embed_ppi_dp, indices_used = gf.get_graph_weights(randomized_ppi,embed_vals,embed_dict)
attn_ppi_dp, indices_used_attns = gf.get_graph_weights(randomized_ppi,attn_vals,attn_dict) 

embed_ppi_ndp, indices_used = gf.get_graph_weights(completely_randomized,embed_vals,embed_dict)
attn_ppi_ndp, indices_used_attns = gf.get_graph_weights(completely_randomized,attn_vals,attn_dict) 

embed_background = gf.get_background(embed_vals,indices_used)
attn_background = gf.get_background(attn_vals,indices_used)

# take a random sample of background weights to make the plotting easier
attn_sample = np.random.choice(attn_background, 1_000_000, replace = False)
embed_sample = np.random.choice(embed_background,1_000_000,replace=False)

sns.kdeplot(embed_sample, label = 'Background', log_scale = False,fill=False)
sns.kdeplot(embed_ppi, label = 'PPI',log_scale = False,fill=False)
sns.kdeplot(embed_ppi_dp, label = 'Randomized PPI (DP)',log_scale = False,fill=False)
sns.kdeplot(embed_ppi_ndp, label = 'Randomized PPI (DP)',log_scale = False,fill=False)
plt.legend(bbox_to_anchor= (1.05,1))
plt.xlabel('Cosine Similarity')
plt.savefig(path_to_figs+"figure_S4_panel_a.png",dpi=300,bbox_inches='tight')
plt.clf()

sns.kdeplot(attn_sample, label = 'Background', log_scale = True,fill=False)
sns.kdeplot(attn_ppi, label = 'PPI',log_scale = True,fill=False)
sns.kdeplot(attn_ppi_dp, label = 'Randomized PPI (DP)',log_scale = True,fill=False)
sns.kdeplot(attn_ppi_ndp, label = 'Randomized PPI (DP)',log_scale = True,fill=False)
plt.legend(bbox_to_anchor= (1.05,1))
plt.xlabel('Attention Weight')
plt.savefig(path_to_figs+"figure_S4_panel_b.png",dpi=300,bbox_inches='tight')
plt.clf()

###### Make panels c and d ######

embed_ppi_background = gf.separate_edges(embed_ppi,embed_background)
attn_ppi_background = gf.separate_edges(attn_ppi,attn_background)

sparsified_embed_labs = embed_ppi_background[:-1:100]
sparsified_attn_labs = attn_ppi_background[:-1:100]

fpr_embeds,tpr_embeds = gf.auc_plot(sparsified_embed_labs)
fpr_attns,tpr_attns = gf.auc_plot(sparsified_attn_labs)

attn_auc = gf.auc(fpr_attns,tpr_attns)
embed_auc = gf.auc(fpr_embeds,tpr_embeds)

plt.plot(fpr_attns,tpr_attns,label=f'Ranked attentions, AUC = {round(attn_auc,2)}')
plt.plot(fpr_embeds,tpr_embeds,label=f'Ranked cosine similarities, AUC = {round(embed_auc,2)}')
plt.plot([0,1],[0,1],'--',color='k',label='random expectation')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend(bbox_to_anchor=(1.05,1))

plt.savefig(path_to_figs+"figure_S4_panel_d.png",dpi=300,bbox_inches='tight')
plt.clf()

plt.plot(np.arange(1,len(attn_ppi_background)+1,1)/(len(attn_ppi_background)+1),attn_ppi_background/attn_ppi_background[-1],label=f'Ranked attentions, AUC = {round(attn_auc,2)}')
plt.plot(np.arange(1,len(embed_ppi_background)+1,1)/(len(embed_ppi_background)+1),embed_ppi_background/embed_ppi_background[-1],label=f'Ranked cosine similarities, AUC = {round(embed_auc,2)}')
plt.plot([0,1],[0,1],'--',color='k',label='random expectation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


plt.savefig(path_to_figs+"figure_S4_panel_c.png",dpi=300,bbox_inches='tight')


