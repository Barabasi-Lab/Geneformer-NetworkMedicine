from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid",context="talk")

path_to_figs = "/work/ccnr/GeneFormer/aggregation_scripts/plotting_scripts/final_figure_scripts/out/"

nonfail_path = '/work/ccnr/GeneFormer/datasets/cardiomyopathy_nonfailing.dataset/'
gc_path = '/work/ccnr/GeneFormer/Genecorpus-30M/genecorpus_30M_2048.dataset/'
dcm_path = '/work/ccnr/GeneFormer/datasets/cardiomyopathy_failing_subsets/dcm_samples.dataset/'

path_dict = {'DCM':dcm_path,'nonfailing':nonfail_path,'genecorpus':gc_path}

ns = [10,50,100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000]

res_dict = {}

for k in list(path_dict.keys()):
    data = load_from_disk(path_dict[k])
    n_genes_gc = []
    choices = [i for i in range(len(data))]
    for j in range(10):
        print(j)
        n_genes_i = []
        np.random.seed(j)
        for n in ns:
            random_indices = np.random.choice(choices,size=n)
            subset = data.select(list(random_indices))
            unique_tokens = set()
            for sample in subset:
                unique_tokens.update(set(sample['input_ids']))
            n_genes_i.append(len(unique_tokens))
        n_genes_gc.append(n_genes_i)
    res_dict[k] = np.array(n_genes_gc)

for k in list(res_dict.keys()):
    mean_genes = np.mean(res_dict[k],axis=0)
    std_genes = np.std(res_dict[k],axis=0)
    plt.plot(ns,mean_genes,label=k)
    plt.fill_between(ns,mean_genes-std_genes,mean_genes+std_genes,alpha=0.3)
plt.xlabel('Number of samples')
plt.ylabel('Number of unique genes')
plt.xscale('log')
plt.legend()
plt.savefig(path_to_figs+'figure_S1.png',dpi=300,bbox_inches='tight')
