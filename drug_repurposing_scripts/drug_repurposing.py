# -*- coding: utf-8 -*-
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import netmedpy

from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from time import sleep

from tools import refine_drugbank
from tools import prepare_data
from tools import clean_score_dataframe
from tools import analyze_drug_scores
from tools import invert_score

import sys
import os


def distance_network(net):
    data = nx.to_pandas_edgelist(net)

    refactor = -1*data.weight

    ma = refactor.max()
    mi = refactor.min()

    refactor = 1+(refactor -mi)/(ma-mi)

    data["weight"] = refactor

    res = nx.from_pandas_edgelist(data,source = "source",target = "target",edge_attr=["weight"])
    return res



def drug_repo_screen(ppi_gf,save_path,weighted=True):
    
    ###################Control Variables
    disease = "Familial dilated cardiomyopathy" #Alternatively "Hypertrophic obstructive cardiomyopathy"

    #Proxies for the disease
    diseases_db = ["cardiomyopathy","heart failure","ischemic heart","hypertension","arrhythmia"]


    ###################LOADING TOYS#####################
    #####Load Drug Targets
    drugbank = "../data/drug_repurposing/input/DB_Drug_Targets_2023.csv"
    drugbank = pd.read_csv(drugbank)
    drugbank = refine_drugbank(drugbank)


    #########Load Disease Genes
    gda = "../data/drug_repurposing/input/disgenet_filtered.xlsx"
    gda = pd.read_excel(gda)
    gda = gda.query("`Disease gene score`>=0.3")

    # Load ppi
    ppi_gf = pd.read_csv(ppi_gf)
    
    if weighted:
        binning = "strength_binning"
        ppi_gf = ppi_gf.query("weight != 0")
        ppi_gf= nx.from_pandas_edgelist(ppi_gf,source='hgnc1',target='hgnc2',edge_attr=['weight'])
        ppi_gf = netmedpy.extract_lcc(ppi_gf.nodes, ppi_gf)
        ppi_gf = distance_network(ppi_gf)
    else:
        binning = 'log_binning'
        ppi_gf= nx.from_pandas_edgelist(ppi_gf,source='hgnc1',target='hgnc2')
        ppi_gf = netmedpy.extract_lcc(ppi_gf.nodes, ppi_gf)
        



#     ########## SCREENING FOR PPI-GF
    print("Screening PPI-GF")

    data = prepare_data(diseases_db, disease, ppi_gf, drugbank, gda)

    disease_genes = data["disease_genes"]
    
    drug_targets = data["drug_targets"]
    positives = data["positives"]

    print("Calculating distance matrix")
    matDist = netmedpy.all_pair_distances(ppi_gf, distance="shortest_path",n_processors=20,n_tasks=2000)
    
    sleep(2)

    print("Calculating drug score")
    screen = netmedpy.screening(drug_targets, disease_genes, ppi_gf,
                                matDist,score="proximity",
                                properties=["z_score",'p_value_single_tail', 'raw_amspl'],
                                null_model=binning,
                                n_iter=10000,
                                bin_size=100,
                                symmetric=False,
                                n_procs=30)



    screen['z_score'] = clean_score_dataframe(screen['z_score'])
    screen['raw_amspl'] = clean_score_dataframe(screen['raw_amspl'])
    screen['p_value_single_tail'] = clean_score_dataframe(screen['p_value_single_tail'])
    screen['positives'] = positives


    ################SAVE RESULTS

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + "screen.pkl", 'wb') as file:
        pickle.dump(screen, file)

    metrics = analyze_drug_scores(screen['z_score'], screen['positives'],50, True)



    with open(save_path + "metrics.pkl", 'wb') as file:
        pickle.dump(metrics, file)

    print(metrics)

    print("All done!")


if __name__=='__main__':

    # Control PPI
    ppi_control = "../data/drug_repurposing/input/PPI_2023-05-10.csv"
    out_control = "../data/drug_repurposing/output/unweighted/"

    drug_repo_screen(ppi_control,out_control,weighted=False)

    # Att weights
    for i in range(6):
        ppi = f"../data/drug_repurposing/input/ppis/ppi_att_cardiomyopathy_cardio_failing_l{i}_avg_10000.csv"
        out_dir = f"../data/drug_repurposing/output/attweights/avg/l{i}/"
        
        drug_repo_screen(ppi,out_dir,weighted=True)
        sleep(5)
    
    # Cosine similarity
    layers = ["input"]
    layers = layers + list(range(6))
    for i in layers:
        ppi = f"../data/drug_repurposing/input/ppis/ppi_cs_cardiomyopathy_cardio_failing_l{i}_10000.csv"
        out_dir = f"../data/drug_repurposing/output/embeddings/l{i}/"
        
        drug_repo_screen(ppi,out_dir,weighted=True)
        sleep(5)
        
    print("Done")