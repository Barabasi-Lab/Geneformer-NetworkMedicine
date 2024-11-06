import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import gf_tools as gf
import pickle
from scipy.stats import iqr

sns.set(style="whitegrid",context="talk")

PPI = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/ppi_with_gf_tokens.csv')
ppi = nx.from_pandas_edgelist(PPI, source = 'ens1', target = 'ens2')

gda = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/GDA_Filtered_04042022.csv')

with open("/work/ccnr/GeneFormer/conda_environment/geneformer/gene_name_id_dict.pkl", 'rb') as f:
    symbol_to_ensembl = pickle.load(f)

path_to_figs = "/work/ccnr/GeneFormer/aggregation_scripts/plotting_scripts/final_figure_scripts/out/"

# paths to dcm weight matrices
pt_attn_path = '/work/ccnr/GeneFormer/aggregated_matrices/aggregated_attentions/cardiomyopathy_failing/dilated/pretrained/max/'
ft_attn_path = "/work/ccnr/GeneFormer/aggregated_matrices/aggregated_attentions/cardiomyopathy_failing/dilated/fine_tuned/max/layer_4/"

ft_attn_counts = gf.load(ft_attn_path+"counts.pkl")
ft_attn_vals = gf.load(ft_attn_path+"vals.pkl")
ft_attn_dict = gf.load(ft_attn_path+"gene_dict.pkl")

pt_attn_counts = gf.load(pt_attn_path+"counts.pkl")
pt_attn_vals = gf.load(pt_attn_path+"vals.pkl")
pt_attn_dict = gf.load(pt_attn_path+"gene_dict.pkl")

if ft_attn_dict!=pt_attn_dict:
    print("The gene to index dictionaries are not the same for all four sets of weights. Please make sure you are using the same set of samples for both")
else:
    gdict = ft_attn_dict

ft_attn_vals = gf.replace_with_nans(ft_attn_counts,ft_attn_vals)
pt_attn_vals = gf.replace_with_nans(pt_attn_counts,pt_attn_vals)

ppi_sub = gf.prune_ppi_nodes(ppi,gdict)

disease_genes = list(gda[gda['NewName'].str.contains("cardiomyopathy dilated")]['HGNC_Symbol'])
disease_genes_filtered = [symbol_to_ensembl[gene] for gene in disease_genes if gene in symbol_to_ensembl.keys()]
dcm_dis_mod = gf.get_disease_lcc(ppi_sub, disease_genes_filtered)

ft_attn_ppi, indices_used_attns = gf.get_graph_weights(ppi_sub,ft_attn_vals,gdict)
pt_attn_ppi, indices_used_attns = gf.get_graph_weights(ppi_sub,pt_attn_vals,gdict)

ft_attn_dis_mod = gf.dis_mod(ft_attn_vals, dcm_dis_mod, gdict)
pt_attn_dis_mod = gf.dis_mod(pt_attn_vals, dcm_dis_mod, gdict)

ft_attn_dgs = gf.fully_connected_dis_mod(ft_attn_vals, dcm_dis_mod, gdict)
pt_attn_dgs = gf.fully_connected_dis_mod(pt_attn_vals, dcm_dis_mod, gdict)

ax = sns.kdeplot(pt_attn_ppi,log_scale = True,label="PPI, pretrained",color='orange',fill=False,bw_adjust = 1,zorder = 2)
ax = sns.kdeplot(pt_attn_dis_mod,log_scale = True,label="Disease module, pretrained",fill=False,bw_adjust = 1,zorder = 2,color='purple')
ax = sns.kdeplot(pt_attn_dgs,log_scale = True,label="Disease genes, pretrained",fill=False,bw_adjust = 1,zorder = 2,color='lightblue')

ax = sns.kdeplot(ft_attn_ppi,log_scale = True,label="PPI, pretrained",color='orange',fill=False,bw_adjust = 1,zorder = 2)
ax = sns.kdeplot(ft_attn_dis_mod,log_scale = True,label="Disease module, fine-tuned",fill=False,bw_adjust = 1,zorder = 2,color='purple')
ax = sns.kdeplot(ft_attn_dgs,log_scale = True,label="Disease genes, fine-tuned",fill=False,bw_adjust = 1,zorder = 2,color='lightblue')

x, y = ax.get_lines()[0].get_data()
x2, y2 = ax.get_lines()[1].get_data()
x3, y3 = ax.get_lines()[2].get_data()
x4, y4 = ax.get_lines()[3].get_data()
x5, y5 = ax.get_lines()[4].get_data()
x6, y6 = ax.get_lines()[5].get_data()

plt.clf()

plt.plot(x,-y,color='orange',label='PPI')
plt.plot(x2,-y2,color='purple',label='Disease Module')
plt.plot(x3,-y3,color='lightblue',label='Disease Genes')

plt.fill_between(x4,0,y4,color='orange',alpha=0.3,edgecolor='orange',linewidth=3,label='PPI')
plt.fill_between(x5,0,y5,color='purple',alpha=0.3,edgecolor='purple',linewidth=3,label='Disease Module')
plt.fill_between(x6,0,y6,color='orange',alpha=0.3,edgecolor='lightblue',linewidth=3,label='Disease Genes')

plt.axhline(color='k')

plt.xscale('log')
plt.xlabel('Attention Weight')
# Move x-axis to the top
ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)

frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)

plt.savefig(f'{path_to_figs}figure_S6_panel_a.png',dpi=300,bbox_inches='tight')
plt.clf()

###### Make panel b ######

ft_module_ppi_labs = gf.separate_edges(ft_attn_dis_mod,ft_attn_ppi)
ft_module_dgs_labs = gf.separate_edges(ft_attn_dis_mod,ft_attn_dgs)
pt_module_ppi_labs = gf.separate_edges(pt_attn_dis_mod,pt_attn_ppi)
pt_module_dgs_labs = gf.separate_edges(pt_attn_dis_mod,pt_attn_dgs)

fpr,tpr = gf.auc_plot(ft_module_ppi_labs)
fpr2,tpr2 = gf.auc_plot(ft_module_dgs_labs)
fpr3,tpr3 = gf.auc_plot(pt_module_ppi_labs)
fpr4,tpr4 = gf.auc_plot(pt_module_dgs_labs)

plt.plot(fpr,tpr, color = 'k',label = 'Fine-tuned, background = PPI, AUC = ' + str(round(gf.auc(fpr,tpr),2)))
plt.plot(fpr3,tpr3, color = 'k', linestyle = '--',label = 'Pretrained, background = PPI, AUC = ' + str(round(gf.auc(fpr3,tpr3),2)))

plt.plot(fpr2,tpr2, color = 'r',label = 'Fine-tuned, background = Disease genes, AUC = ' + str(round(gf.auc(fpr2,tpr2),2)))
plt.plot(fpr4,tpr4, color = 'r', linestyle = '--',label = 'Pretrained, background = Disease genes, AUC = ' + str(round(gf.auc(fpr4,tpr4),2)))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(bbox_to_anchor = (1.05,1))
plt.savefig(f'{path_to_figs}figure_S6_panel_b.png',dpi=300,bbox_inches='tight')

###### Make panel d ######

hops = gf.bin_by_hop(ppi_sub,dcm_dis_mod,5)
for ls in hops:
    print(len(ls))
    
hop_attns = [gf.get_specific_weights(pt_attn_vals,hops[0],hops[i],gdict) for i in [0,1,2,3,4]]

for_random_selection = pt_attn_vals.flatten()
for_random_selection = for_random_selection[~np.isnan(for_random_selection)]
true_random = np.random.choice(for_random_selection,100_000,replace=False)
hop_attns.append(true_random)

med = np.median(true_random)
inter = iqr(true_random)
top = np.percentile(true_random, 75)
bottom = np.percentile(true_random, 25)

plt.clf()
sns.violinplot(hop_attns[:-1],log_scale=True,zorder = 2)
plt.hlines(med,-1,5,color = 'k',alpha = 0.8,zorder=1)
plt.fill_between([-1,5],med-bottom, med+top, color='grey',alpha=0.5,zorder=0)
plt.xlim(-1,5)
plt.xticks([0,1,2,3,4],["0","1","2","3","4"])
plt.ylabel("Attention Weight")
plt.xlabel("Hops away from disease module")
plt.tight_layout()
plt.savefig(f'{path_to_figs}figure_S6_panel_d.png',dpi=300,bbox_inches='tight')
