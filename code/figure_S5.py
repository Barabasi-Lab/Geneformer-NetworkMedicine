import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import gf_tools as gf
from scipy.stats import wasserstein_distance as emd
import pickle

sns.set(style="whitegrid",context="talk")

PPI = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/ppi_with_gf_tokens.csv')
ppi = nx.from_pandas_edgelist(PPI, source = 'ens1', target = 'ens2')

gda = pd.read_csv('/work/ccnr/GeneFormer/jjs_adventures/figure_3_dis_mods/other_data/GDA_Filtered_04042022.csv')

with open("/work/ccnr/GeneFormer/conda_environment/geneformer/gene_name_id_dict.pkl", 'rb') as f:
    symbol_to_ensembl = pickle.load(f)

path_to_figs = "/work/ccnr/GeneFormer/aggregation_scripts/plotting_scripts/final_figure_scripts/out/"

# paths to dcm weight matrices
pt_attn_path = '/work/ccnr/GeneFormer/aggregated_matrices/aggregated_attentions/cardiomyopathy_failing/dilated/pretrained/max/'
pt_embed_path = '/work/ccnr/GeneFormer/aggregated_matrices/aggregated_embeddings/cardiomyopathy_failing/dilated/pretrained/max/'
ft_embed_path = "/work/ccnr/GeneFormer/aggregated_matrices/aggregated_embeddings/cardiomyopathy_failing/dilated/fine_tuned/max/layer_4/"
ft_attn_path = "/work/ccnr/GeneFormer/aggregated_matrices/aggregated_attentions/cardiomyopathy_failing/dilated/fine_tuned/max/layer_4/"

ft_embed_counts = gf.load(ft_embed_path+"counts.pkl")
ft_embed_vals = gf.load(ft_embed_path+"vals.pkl")
ft_embed_dict = gf.load(ft_embed_path+"gene_dict.pkl")

ft_attn_counts = gf.load(ft_attn_path+"counts.pkl")
ft_attn_vals = gf.load(ft_attn_path+"vals.pkl")
ft_attn_dict = gf.load(ft_attn_path+"gene_dict.pkl")

pt_embed_counts = gf.load(pt_embed_path+"counts.pkl")
pt_embed_vals = gf.load(pt_embed_path+"vals.pkl")
pt_embed_dict = gf.load(pt_embed_path+"gene_dict.pkl")

pt_attn_counts = gf.load(pt_attn_path+"counts.pkl")
pt_attn_vals = gf.load(pt_attn_path+"vals.pkl")
pt_attn_dict = gf.load(pt_attn_path+"gene_dict.pkl")

if (ft_embed_dict==ft_attn_dict==pt_embed_dict==pt_attn_dict)==False:
    print("The gene to index dictionaries are not the same for all four sets of weights. Please make sure you are using the same set of samples for both")
else:
    gdict = ft_embed_dict

ft_embed_vals = gf.replace_with_nans(ft_embed_counts,ft_embed_vals)
ft_attn_vals = gf.replace_with_nans(ft_attn_counts,ft_attn_vals)

pt_embed_vals = gf.replace_with_nans(pt_embed_counts,pt_embed_vals)
pt_attn_vals = gf.replace_with_nans(pt_attn_counts,pt_attn_vals)

ppi_sub = gf.prune_ppi_nodes(ppi,gdict)

ft_embed_ppi, indices_used = gf.get_graph_weights(ppi_sub,ft_embed_vals,gdict)
ft_attn_ppi, indices_used_attns = gf.get_graph_weights(ppi_sub,ft_attn_vals,gdict)
pt_embed_ppi, indices_used = gf.get_graph_weights(ppi_sub,pt_embed_vals,gdict)
pt_attn_ppi, indices_used_attns = gf.get_graph_weights(ppi_sub,pt_attn_vals,gdict)

disease_genes = list(gda[gda['NewName'].str.contains("cardiomyopathy dilated")]['HGNC_Symbol'])
disease_genes_filtered = [symbol_to_ensembl[gene] for gene in disease_genes if gene in symbol_to_ensembl.keys()]

dcm_dis_mod = gf.get_disease_lcc(ppi_sub, disease_genes_filtered)

###### Make panels a and b ######

ft_embed_dis_mod = gf.dis_mod(ft_embed_vals, dcm_dis_mod, gdict)
ft_attn_dis_mod = gf.dis_mod(ft_attn_vals, dcm_dis_mod, gdict)
pt_embed_dis_mod = gf.dis_mod(pt_embed_vals, dcm_dis_mod, gdict)
pt_attn_dis_mod = gf.dis_mod(pt_attn_vals, dcm_dis_mod, gdict)

ax = sns.kdeplot(pt_embed_ppi,log_scale = False,label="PPI, pretrained",color='lightblue',fill=False,bw_adjust = 1,zorder = 2)
ax = sns.kdeplot(pt_embed_dis_mod,log_scale = False,label="Disease module, pretrained",fill=False,bw_adjust = 1,zorder = 2,color='purple')
ax = sns.kdeplot(ft_embed_ppi,log_scale = False,label="PPI, pretrained",color='lightblue',fill=False,bw_adjust = 1,zorder = 2)
ax = sns.kdeplot(ft_embed_dis_mod,log_scale = False,label="Disease module, fine-tuned",fill=False,bw_adjust = 1,zorder = 2,color='purple')

x, y = ax.get_lines()[0].get_data()
x2, y2 = ax.get_lines()[1].get_data()
x3, y3 = ax.get_lines()[2].get_data()
x4, y4 = ax.get_lines()[3].get_data()

plt.clf()

plt.plot(x,-y,color='lightblue',label='PPI')
plt.plot(x2,-y2,color='purple')

# plt.plot(x3,y3,color='lightblue')
# plt.plot(x4,y4,color='purple')

plt.fill_between(x3,0,y3,color='lightblue',alpha=0.3,edgecolor='blue',linewidth=3)
plt.fill_between(x4,0,y4,color='purple',alpha=0.3,edgecolor='purple',linewidth=3)

plt.axhline(color='k')

plt.xlabel('Cosine Similarity')
# Move x-axis to the top
ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)

frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)

plt.savefig(f'{path_to_figs}figure_S5_panel_a.png',dpi=300,bbox_inches='tight')
plt.clf()

ax = sns.kdeplot(pt_attn_ppi,log_scale = True,label="PPI, pretrained",color='lightblue',fill=False,bw_adjust = 1,zorder = 2)
ax = sns.kdeplot(pt_attn_dis_mod,log_scale = True,label="Disease module, pretrained",fill=False,bw_adjust = 1,zorder = 2,color='purple')

ax = sns.kdeplot(ft_attn_ppi,log_scale = True,label="PPI, pretrained",color='lightblue',fill=False,bw_adjust = 1,zorder = 2)
ax = sns.kdeplot(ft_attn_dis_mod,log_scale = True,label="Disease module, fine-tuned",fill=False,bw_adjust = 1,zorder = 2,color='purple')

x, y = ax.get_lines()[0].get_data()
x2, y2 = ax.get_lines()[1].get_data()
x3, y3 = ax.get_lines()[2].get_data()
x4, y4 = ax.get_lines()[3].get_data()

plt.clf()

plt.plot(x,-y,color='lightblue',label='PPI')
plt.plot(x2,-y2,color='purple')

# plt.plot(x3,y3,color='lightblue')
# plt.plot(x4,y4,color='purple')

plt.fill_between(x3,0,y3,color='lightblue',alpha=0.3,edgecolor='blue',linewidth=3)
plt.fill_between(x4,0,y4,color='purple',alpha=0.3,edgecolor='purple',linewidth=3)

plt.axhline(color='k')

plt.xscale('log')
plt.xlabel('Attention Weight')
# Move x-axis to the top
ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)

frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)

plt.savefig(f'{path_to_figs}figure_S5_panel_b.png',dpi=300,bbox_inches='tight')
plt.clf()

###### Make panels c and d ######
diseases = ['cardiomyopathy dilated', 'cardiomyopathy hypertrophic','heart failure', 'colitis ulcerative', 'carcinoma renal cell', 'anemia']
big_df = pd.DataFrame(columns=['Cosine Similarity', 'Disease', 'Model'])
for i,disease in enumerate(diseases):
    # get the disease module 
    dgs = list(gda[gda['NewName'].str.contains(disease)]['HGNC_Symbol'])
    dgs_filtered = [symbol_to_ensembl[gene] for gene in disease_genes if gene in symbol_to_ensembl.keys()]
    disease_sub = gf.get_disease_lcc(ppi_sub,dgs_filtered)
    print(disease, disease_sub.number_of_nodes(), disease_sub.number_of_edges())
    for j in range(2):
        if j == 0:
            dis_mod_attns = gf.dis_mod(pt_embed_vals,disease_sub,gdict)
            models = ['pretrained' for j in range(len(dis_mod_attns))]
        elif j == 1:
            dis_mod_attns = gf.dis_mod(ft_embed_vals,disease_sub,gdict)
            models = ['cardiomyopathy' for j in range(len(dis_mod_attns))]
        dis = [disease for j in range(len(dis_mod_attns))]
        df = pd.DataFrame({'Cosine Similarity': dis_mod_attns, 'Disease': dis, 'Model': models})
        big_df = pd.concat([big_df, df])
        
fig= plt.figure(figsize=(5,20))
sns.violinplot(big_df,x='Cosine Similarity',y='Disease',hue='Model',split='True',hue_order=['cardiomyopathy','pretrained'],palette=['purple','lightblue'],inner='quart',log_scale=False)
plt.savefig(f'{path_to_figs}figure_S5_panel_c.png',dpi=300,bbox_inches='tight')
plt.clf()

diseases = ['cardiomyopathy dilated', 'cardiomyopathy hypertrophic','heart failure', 'colitis ulcerative', 'carcinoma renal cell', 'anemia']
big_df = pd.DataFrame(columns=['Attention weight', 'Disease', 'Model'])
for i,disease in enumerate(diseases):
    # get the disease module 
    dgs = list(gda[gda['NewName'].str.contains(disease)]['HGNC_Symbol'])
    dgs_filtered = [symbol_to_ensembl[gene] for gene in disease_genes if gene in symbol_to_ensembl.keys()]
    disease_sub = gf.get_disease_lcc(ppi_sub,dgs_filtered)
    for j in range(2):
        if j == 0:
            dis_mod_attns = gf.dis_mod(pt_attn_vals,disease_sub,gdict)
            models = ['pretrained' for j in range(len(dis_mod_attns))]
        elif j == 1:
            dis_mod_attns = gf.dis_mod(ft_attn_vals,disease_sub,gdict)
            models = ['cardiomyopathy' for j in range(len(dis_mod_attns))]
        dis = [disease for j in range(len(dis_mod_attns))]
        df = pd.DataFrame({'Attention Weight': dis_mod_attns, 'Disease': dis, 'Model': models})
        big_df = pd.concat([big_df, df])
        
fig= plt.figure(figsize=(5,20))
sns.violinplot(big_df,x='Attention Weight',y='Disease',hue='Model',split='True',hue_order=['cardiomyopathy','pretrained'],palette=['purple','lightblue'],inner='quart',log_scale=True)
plt.savefig(f'{path_to_figs}figure_S5_panel_d.png',dpi=300,bbox_inches='tight')
plt.clf()

###### Make panels e and f ######
cardio_gda = gda[gda['DescriptorName'].str.contains('Cardiovascular')==True]
non_cardio_gda = gda[gda['DescriptorName'].str.contains('Cardiovascular')==False]
cardio_diseases = list(set(list(cardio_gda['NewName'])))
non_cardio_diseases = list(set(list(non_cardio_gda['NewName'])))

cardio_distances = {}

for disease in cardio_diseases:
    try:
        dgs = list(gda[gda['NewName'].str.contains(disease)]['HGNC_Symbol'])
        dgs_filtered = [symbol_to_ensembl[gene] for gene in dgs if gene in symbol_to_ensembl.keys()]
        disease_sub = gf.get_disease_lcc(ppi_sub,dgs_filtered)
        if disease_sub.number_of_nodes()>40:
            pretrained_attns = gf.dis_mod(pt_embed_vals,disease_sub,gdict)
            finetuned_attns = gf.dis_mod(ft_embed_vals,disease_sub,gdict)
        
            diff = emd(pretrained_attns,finetuned_attns)
            direction = np.sign(np.median(finetuned_attns)-np.median(pretrained_attns))
            cardio_distances[disease] = direction*diff
            print(disease,direction*diff)
    except ValueError:
        pass # some of these diseases have empty disease modules that throw a value error
    
non_cardio_distances = {}
for disease in non_cardio_diseases:
    try:
        dgs = list(gda[gda['NewName'].str.contains(disease)]['HGNC_Symbol'])
        dgs_filtered = [symbol_to_ensembl[gene] for gene in dgs if gene in symbol_to_ensembl.keys()]
        disease_sub = gf.get_disease_lcc(ppi_sub,dgs_filtered)
        if disease_sub.number_of_nodes()>40:
            pretrained_attns = gf.dis_mod(pt_embed_vals,disease_sub,gdict)
            finetuned_attns = gf.dis_mod(ft_embed_vals,disease_sub,gdict)
        
            diff = emd(pretrained_attns,finetuned_attns)
            direction = np.sign(np.median(finetuned_attns)-np.median(pretrained_attns))
            non_cardio_distances[disease] = direction*diff
            print(direction*diff)
    except ValueError:
        pass


cardio_ls = ['cardiomyopathy dilated', 'cardiomyopathy hypertrophic']
distances = []
for i,disease in enumerate(cardio_ls):
    dgs = list(gda[gda['NewName'].str.contains(disease)]['HGNC_Symbol'])
    dgs_filtered = [symbol_to_ensembl[gene] for gene in dgs if gene in symbol_to_ensembl.keys()]
    disease_sub = gf.get_disease_lcc(ppi_sub,dgs_filtered)
    
    pretrained_attns = gf.dis_mod(pt_embed_vals,disease_sub,gdict)
    finetuned_attns = gf.dis_mod(ft_embed_vals,disease_sub,gdict)

    diff = emd(pretrained_attns,finetuned_attns)
    direction = np.sign(np.median(finetuned_attns)-np.median(pretrained_attns))
    distances.append(direction*diff)
    print(direction*diff)

fig= plt.figure(figsize=(5,5))
bns = np.linspace(0,0.4,20)
# bns = 20
y,x,_=plt.hist(list(cardio_distances.values()),label='Cardiovascular Diseases',density=True,bins=bns,alpha=0.5)
y2,x2,_=plt.hist(list(non_cardio_distances.values()),label='Non-Cardiovascular Diseases',color='grey',density=True,bins=bns,alpha=0.5)
max_y = max(max(y2),max(y))
plt.axvline(distances[0],color='r',linestyle='--')
plt.axvline(distances[1],color='r',linestyle='--')
plt.text(distances[0],max_y*0.9,'Dilated Cardiomyopathy',fontsize='xx-small')
plt.text(distances[1],max_y*0.8,'Hypertrophic Cardiomyopathy',fontsize='xx-small')
plt.xlabel('Pretrained/Fine-tuned EMD (Cosine Similarity)')
plt.ylabel('Density')
plt.legend(bbox_to_anchor=(1.05,1))   
plt.savefig(f'{path_to_figs}figure_S5_panel_e.png',dpi=300,bbox_inches='tight')
plt.clf()
    
cardio_distances = {}

for disease in cardio_diseases:
    try:
        dgs = list(gda[gda['NewName'].str.contains(disease)]['HGNC_Symbol'])
        dgs_filtered = [symbol_to_ensembl[gene] for gene in dgs if gene in symbol_to_ensembl.keys()]
        disease_sub = gf.get_disease_lcc(ppi_sub,dgs_filtered)
        if disease_sub.number_of_nodes()>40:
            pretrained_attns = gf.dis_mod(pt_attn_vals,disease_sub,gdict)
            finetuned_attns = gf.dis_mod(ft_attn_vals,disease_sub,gdict)
        
            diff = emd(pretrained_attns,finetuned_attns)
            direction = np.sign(np.median(finetuned_attns)-np.median(pretrained_attns))
            cardio_distances[disease] = direction*diff
            print(direction*diff)
    except ValueError:
        pass # some of these diseases have empty disease modules that throw a value error
    
non_cardio_distances = {}
for disease in non_cardio_diseases:
    try:
        dgs = list(gda[gda['NewName'].str.contains(disease)]['HGNC_Symbol'])
        dgs_filtered = [symbol_to_ensembl[gene] for gene in dgs if gene in symbol_to_ensembl.keys()]
        disease_sub = gf.get_disease_lcc(ppi_sub,dgs_filtered)
        if disease_sub.number_of_nodes()>40:
            pretrained_attns = gf.dis_mod(pt_attn_vals,disease_sub,gdict)
            finetuned_attns = gf.dis_mod(ft_attn_vals,disease_sub,gdict)
        
            diff = emd(pretrained_attns,finetuned_attns)
            direction = np.sign(np.median(finetuned_attns)-np.median(pretrained_attns))
            non_cardio_distances[disease] = direction*diff
            print(direction*diff)
    except ValueError:
        pass


cardio_ls = ['cardiomyopathy dilated', 'cardiomyopathy hypertrophic']
distances = []
for i,disease in enumerate(cardio_ls):
    dgs = list(gda[gda['NewName'].str.contains(disease)]['HGNC_Symbol'])
    dgs_filtered = [symbol_to_ensembl[gene] for gene in dgs if gene in symbol_to_ensembl.keys()]
    disease_sub = gf.get_disease_lcc(ppi_sub,dgs_filtered)
    
    pretrained_attns = gf.dis_mod(pt_attn_vals,disease_sub,gdict)
    finetuned_attns = gf.dis_mod(ft_attn_vals,disease_sub,gdict)

    diff = emd(pretrained_attns,finetuned_attns)
    direction = np.sign(np.median(finetuned_attns)-np.median(pretrained_attns))
    distances.append(direction*diff)
    print(direction*diff)
    
fig= plt.figure(figsize=(5,5))
bns = np.linspace(-0.01,0.01,20)
# bns=20
y,x,_=plt.hist(list(cardio_distances.values()),label='Cardiovascular Diseases',density=True,bins=bns,alpha=0.5)
y2,x2,_=plt.hist(list(non_cardio_distances.values()),label='Non-Cardiovascular Diseases',color='grey',density=True,bins=bns,alpha=0.5)
max_y = max(max(y2),max(y))
plt.axvline(distances[0],color='r',linestyle='--')
plt.axvline(distances[1],color='r',linestyle='--')
plt.text(distances[0],max_y*0.9,'Dilated\nCardiomyopathy',fontsize='xx-small',horizontalalignment='center')
plt.text(distances[1],max_y*0.75,'Hypertrophic\nCardiomyopathy',fontsize='xx-small',horizontalalignment='center')
plt.xlabel('Pretrained/Fine-tuned EMD (Attention Weight)')
plt.ylabel('Density')
plt.legend(bbox_to_anchor=(1.05,1))
plt.savefig(f'{path_to_figs}figure_S5_panel_f.png',dpi=300,bbox_inches='tight')
plt.clf()
