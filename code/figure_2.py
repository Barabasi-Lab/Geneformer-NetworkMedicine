import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
import networkx as nx
from scipy.optimize import curve_fit

import gf_tools as gf

sns.set(style="whitegrid",context="talk")

path_to_figs = "path to figure output folder"

# paths to genecorpus weight matrices
embed_path = "./data/aggregated_matrices/aggregated_embeddings/genecorpus/pretrained/max/"
attn_path = "./data/aggregated_matrices/aggregated_attentions/genecorpus/pretrained/max/"

# paths to dcm weight matrices
embed_path_dcm = "./data/aggregated_matrices/aggregated_embeddings/dcm_samples/pretrained/max/"
attn_path_dcm = "./data/aggregated_matrices/aggregated_attentions/dcm_samples/pretrained/max/"

PPI = pd.read_csv('./supplemental_data/ppi.csv')
ppi = nx.from_pandas_edgelist(PPI, source = 'ens1', target = 'ens2')

gda = pd.read_csv('./supplemental_data/gda.csv')

with open("./supplemental_data/gene_name_id_dict.pkl", 'rb') as f:
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

#### Separation of PPI and background (panels a and b) ####

embed_ppi, indices_used = gf.get_graph_weights(ppi_sub,embed_vals,embed_dict)
attn_ppi, indices_used_attns = gf.get_graph_weights(ppi_sub,attn_vals,attn_dict) # because the samples are the same, the indices used will be the same as well

embed_background = gf.get_background(embed_vals,indices_used)
attn_background = gf.get_background(attn_vals,indices_used)

# take a random sample of background weights to make the plotting easier
attn_sample = np.random.choice(attn_background, 1_000_000, replace = False)
embed_sample = np.random.choice(embed_background,1_000_000,replace=False)

sns.kdeplot(embed_sample, label = 'Background', log_scale = False)
sns.kdeplot(embed_ppi, label = 'PPI',log_scale = False)
plt.legend(bbox_to_anchor= (1.05,1))
plt.xlabel('Cosine Similarity')
plt.savefig(path_to_figs+"figure_1_panel_a.png",dpi=300,bbox_inches='tight')
plt.clf()

sns.kdeplot(attn_sample, label = 'Background', log_scale = True)
sns.kdeplot(attn_ppi, label = 'PPI',log_scale = True)
plt.legend(bbox_to_anchor= (1.05,1))
plt.xlabel('Attention Weight')
plt.savefig(path_to_figs+"figure_1_panel_b.png",dpi=300,bbox_inches='tight')
plt.clf()

#### Dependence on PPI Degree (panels c and d) ####

total_attns, degrees = gf.degree_weights(ppi_sub, attn_vals, attn_dict)
total_embeds, degrees_embeds = gf.degree_weights(ppi_sub, embed_vals, attn_dict)

deg_dict = gf.bin_by_degree(total_attns, degrees)
deg_dict_embeds = gf.bin_by_degree(total_embeds, degrees_embeds)

x = []
y = []
for k in list(deg_dict_embeds.keys()):
    if k==0:
        continue
    else:
        for l in deg_dict_embeds[k]:
            x.append(np.log(k))
            y.append(l)

X = np.array(x)
y = np.array(y)

# Define the model function with parameters b and c
def model_func(X, b, c):
    return b * X + c

# Set initial guesses for b and c
initial_guess = [561, 371]

# Perform curve fitting
params, covariance = curve_fit(model_func, X, y, p0=initial_guess)
b, c = params

# Predict values based on the fitted parameters
y_pred_embeds = model_func(X, b, c)

# Calculate the R-squared value
ss_res = np.sum((y - y_pred_embeds) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Extract standard errors (square roots of diagonal elements in the covariance matrix)
std_errors = np.sqrt(np.diag(covariance))

print("Cosine similarity modeled as A = b*ln(k) + c")
print("R squared = " + str(round(r_squared, 5)))
print("c = " + str(round(c, 3)) + " with an approximate standard error of " + str(round(std_errors[1], 5)))
print("b = " + str(round(b, 3)) + " with an approximate standard error of " + str(round(std_errors[0], 5)))
print("Standard errors are rounded off after 5 decimals")

plt.scatter(degrees_embeds,total_embeds,s=1,zorder=0)
plt.errorbar(list(deg_dict_embeds.keys()),[np.mean(ls) for ls in list(deg_dict_embeds.values())], yerr = [np.std(ls) for ls in list(deg_dict_embeds.values())],capsize=3,fmt='.', color='k',zorder=1)
plt.scatter(list(deg_dict_embeds.keys()),[np.mean(ls) for ls in list(deg_dict_embeds.values())],color='r',s=5,zorder=2)
plt.plot(np.exp(np.array(x)),y_pred_embeds,'--',color='k',zorder=3)
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Degree in PPI')
plt.ylabel('Total cosine similarity')
plt.savefig(path_to_figs+'figure_1_panel_c.png',dpi=300,bbox_inches='tight')
plt.clf()

x = []
y = []
for k in list(deg_dict.keys()):
    if k==0:
        continue
    else:
        for l in deg_dict[k]:
            x.append(np.log(k))
            y.append(l)

X = np.array(x)
y = np.array(y)

# Define the model function with parameters b and c
def model_func(X, b, c):
    return b * X + c

# Set initial guesses for b and c
initial_guess = [6, -1.9]

# Perform curve fitting
params, covariance = curve_fit(model_func, X, y, p0=initial_guess)
b, c = params

# Predict values based on the fitted parameters
y_pred = model_func(X, b, c)

# Calculate the R-squared value
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Extract standard errors (square roots of diagonal elements in the covariance matrix)
std_errors = np.sqrt(np.diag(covariance))

print("Attention modeled as A = b*ln(k) + c")
print("R squared = " + str(round(r_squared, 5)))
print("c = " + str(round(c, 3)) + " with an approximate standard error of " + str(round(std_errors[1], 5)))
print("b = " + str(round(b, 3)) + " with an approximate standard error of " + str(round(std_errors[0], 5)))
print("Standard errors are rounded off after 5 decimals")

plt.scatter(degrees,total_attns,s=1,zorder=0)
plt.errorbar(list(deg_dict.keys()),[np.mean(ls) for ls in list(deg_dict.values())], yerr = [np.std(ls) for ls in list(deg_dict.values())],capsize=3,fmt='.', color='k',zorder=1)
plt.scatter(list(deg_dict.keys()),[np.mean(ls) for ls in list(deg_dict.values())],color='r',s=5,zorder=2)
plt.plot(np.exp(np.array(x)),y_pred,'--',color='k',zorder=3)
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Degree in PPI')
plt.ylabel('Total attention')
plt.savefig(path_to_figs+'figure_1_panel_d.png',dpi=300,bbox_inches='tight')
plt.clf()

#### Separation of DCM disease module from PPI (panels e and f) ####

embed_counts = gf.load(embed_path_dcm+"counts.pkl")
embed_vals = gf.load(embed_path_dcm+"vals.pkl")
embed_dict = gf.load(embed_path_dcm+"gene_dict.pkl")

attn_counts = gf.load(attn_path_dcm+"counts.pkl")
attn_vals = gf.load(attn_path_dcm+"vals.pkl")
attn_dict = gf.load(attn_path_dcm+"gene_dict.pkl")

if attn_dict!=embed_dict:
    pint("The gene to index dictionaries are not the same for attention weights and embeddings. Please make sure you are using the same set of samples for both")

embed_vals = gf.replace_with_nans(embed_counts,embed_vals)
attn_vals = gf.replace_with_nans(attn_counts,attn_vals)

ppi_sub = gf.prune_ppi_nodes(ppi,attn_dict)

embed_ppi, indices_used = gf.get_graph_weights(ppi_sub,embed_vals,attn_dict)
attn_ppi, indices_used_attns = gf.get_graph_weights(ppi_sub,attn_vals,attn_dict)

disease_genes = list(gda[gda['NewName'].str.contains("cardiomyopathy dilated")]['HGNC_Symbol'])
disease_genes_filtered = [symbol_to_ensembl[gene] for gene in disease_genes if gene in symbol_to_ensembl.keys()]

dcm_dis_mod = gf.get_disease_lcc(ppi_sub, disease_genes_filtered)

embed_dis_mod = gf.dis_mod(embed_vals, dcm_dis_mod, attn_dict)
attn_dis_mod = gf.dis_mod(attn_vals, dcm_dis_mod, attn_dict)

sns.kdeplot(embed_ppi,log_scale = False,label="PPI,",color='orange',fill=False)
sns.kdeplot(embed_dis_mod,log_scale = False,label="Disease module",fill=False,color='purple')
plt.xlabel("Cosine Similarity")
plt.legend(bbox_to_anchor = (1.35,1))
plt.savefig(path_to_figs+'figure_1_panel_e.png',dpi=300,bbox_inches='tight')
plt.clf()

sns.kdeplot(attn_ppi,log_scale = True,label="PPI",color='orange',fill=False)
sns.kdeplot(attn_dis_mod,log_scale = True,label="Disease module",fill=False,color='purple')
plt.xlabel("Attention Weight")
plt.legend(bbox_to_anchor = (1.35,1))
plt.savefig(path_to_figs+'figure_1_panel_f.png',dpi=300,bbox_inches='tight')

