# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import ThemeSelector


def refine_drugbank(df):
    df = df[df['Status'].str.contains('approved')]
    df = df.query("organism == 'Humans'")

    return df


def prepare_gda(disease,gda,ppi):
    dg = gda.query("`Disease name`==@disease")
    dg = set(dg["Gene Symbol"]) & set(ppi.nodes)

    return {disease:dg}

def drugbank_to_dictionary(drugbank):

    db_targets = {}

    names = set(drugbank.Name)

    ####Create dictionary of drug targets
    for n in names:
        targets = drugbank.query("Name==@n")
        targets = set(targets.Gene_Target)
        db_targets[n] = targets
    return db_targets


def calculate_positives_negatives(drugbank, diseases):
    regex = "|".join(diseases)

    tp = drugbank[drugbank["Indication"].str.contains(regex,case=False,na=False)]

    tp_names = set(tp.Name)
    tp_genes = set(tp.Gene_Target)
    db_names = set(drugbank.Name)

    tn_names = set()

    for n in db_names:
        if n not in tp_names:
            cgenes = drugbank.query("Name==@n")
            cgenes = set(cgenes.Gene_Target)

            if tp_genes.isdisjoint(cgenes):
                tn_names.add(n)


    df_data = {name: 1 for name in tp_names}
    df_data.update({name:0 for name in tn_names if name not in df_data})

    df = pd.DataFrame(list(df_data.items()),columns=["Name","Positive"])
    return df



def prepare_data(diseases_db, disease_gda, ppi, drugbank, gda):
    #########Prepare gda
    ad_gda = gda[gda['Gene Symbol'].isin(ppi.nodes)]
    disease_genes = prepare_gda(disease_gda, ad_gda, ppi)


    ####Prepare drug bank stuff
    db = drugbank[drugbank["Gene_Target"].isin(ppi.nodes)]

    positives_negatives = calculate_positives_negatives(db, diseases_db)
    db_cases = db[db["Name"].isin(set(positives_negatives.Name))]

    db = drugbank_to_dictionary(db_cases)

    return {"disease_genes":disease_genes,"drug_targets":db, "positives":positives_negatives}



def analyze_drug_scores(scores, positives, N, invert):
    merged = pd.merge(scores, positives, on='Name')

    # Ensure the Score is numeric and drop any rows with NaN or infinite values
    merged['Score'] = pd.to_numeric(merged['Score'], errors='coerce')

    if invert:
        merged['Score']= -1*merged['Score']

    merged = merged.sort_values(by='Score', ascending=False)

    if N > len(merged):
        N = len(merged)  # Adjust N if it's greater than the number of available candidates
    top_n_hits = merged.head(N)['Positive'].cumsum()

    top_n_hits_df = pd.DataFrame({
        'Top N Candidates': range(1, N+1),
        'Cumulative True Positives': top_n_hits.values
    })


    fpr, tpr, roc_thresholds = roc_curve(merged['Positive'], merged['Score'])
    roc_auc = auc(fpr, tpr)

    precision, recall, pr_thresholds = precision_recall_curve(merged['Positive'], merged['Score'])
    pr_auc = auc(recall, precision)

    uninverted = merged
    if invert:
        uninverted['Score'] = -1*uninverted['Score']

    score_range = np.linspace(uninverted['Score'].min(), uninverted['Score'].max(), 300)
    kde_pos = gaussian_kde(uninverted[uninverted['Positive'] == 1]['Score'])(score_range)
    kde_neg = gaussian_kde(uninverted[uninverted['Positive'] == 0]['Score'])(score_range)

    kde_df = pd.DataFrame({
        'Score': score_range,
        'Positives': kde_pos,
        'Negatives': kde_neg
    })

    # Collecting the dataframes
    dataframes = {
        'roc_curve': pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': roc_thresholds}),
        'roc_auc': roc_auc,
        'prec_rec': pd.DataFrame({'Precision': precision[:-1], 'Recall': recall[:-1], 'Threshold': pr_thresholds}),
        'prec_vs_rec': pd.DataFrame({'Precision': precision, 'Recall': recall}),
        'kde_dist': kde_df,
        'hits': top_n_hits_df,
        'Drug Scores': merged  # Including the merged DataFrame


    }

    return dataframes



def clean_score_dataframe(dataframe):
    df = dataframe.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    df = df.reset_index()
    df.columns = ['Name','Score']
    return df

def invert_score(dataframe, column):
    df= dataframe
    df[column] = -1*df[column]
    return df
