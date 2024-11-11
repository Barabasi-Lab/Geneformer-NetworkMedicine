import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


def precision_N(df, score_col='Score', positive_col='Positive'):    
    sorted_df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    precision_list = []
    N_list = []

    for N in range(1, len(sorted_df) + 1):
        top_N = sorted_df.head(N)
        precision = top_N[positive_col].sum() / N


        N_list.append(N)
        precision_list.append(precision)


    precision_df = pd.DataFrame({
        'Top': N_list,
        'Precision': precision_list
    })

    return precision_df


def recall_N(df, score_col='Score', positive_col='Positive'):
    sorted_df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    total_positives = sorted_df[positive_col].sum()

    recall_list = []
    N_list = []

    for N in range(1, len(sorted_df) + 1):
        top_N = sorted_df.head(N)

        recall = top_N[positive_col].sum() / total_positives if total_positives > 0 else 0

        N_list.append(N)
        recall_list.append(recall)

    recall_df = pd.DataFrame({
        'Top': N_list,
        'Recall': recall_list
    })

    return recall_df


def calculate_ml_metrics(df, score_col='Total_Borda', positive_col='Positive'):
    scores = df[score_col]
    positives = df[positive_col]

    fpr, tpr, roc_thresholds = roc_curve(positives, scores)
    roc_auc = auc(fpr, tpr)
    precN = precision_N(df,score_col=score_col,positive_col=positive_col)
    recallN = recall_N(df,score_col=score_col,positive_col=positive_col)


    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Threshold': roc_thresholds
    })


    dataframes = {
        'roc_curve': roc_data,
        'roc_auc': roc_auc,
        'precision': precN,
        'recall': recallN,
        'Drug Scores': df  # Including the merged DataFrame
    }


    return dataframes




def borda_count(df1, df2, score_col, drug_col):
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    merged_df = pd.merge(df1_copy,
                         df2_copy,
                         on=drug_col,
                         suffixes=('_df1', '_df2'))

    merged_df['Rank_df1'] = merged_df[f"{score_col}_df1"].rank(ascending=False)
    merged_df['Rank_df2'] = merged_df[f"{score_col}_df2"].rank(ascending=False)

    N = len(merged_df.index)

    merged_df['Borda_df1'] = N - merged_df['Rank_df1']
    merged_df['Borda_df2'] = N - merged_df['Rank_df2']


    # Sum the Borda points
    merged_df['Total_Borda'] = merged_df['Borda_df1'] + merged_df['Borda_df2']


    return merged_df


def dawdall_count(df1, df2, score_col, drug_col):
    # Work with copies to avoid modifying the original dataframes
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # Merge the two dataframes on the Drug column
    merged_df = pd.merge(df1_copy,
                         df2_copy,
                         on=drug_col,
                         suffixes=('_df1', '_df2'))

    # Rank drugs by score in each dataframe
    merged_df['Rank_df1'] = merged_df[f"{score_col}_df1"].rank(ascending=False, method='min')
    merged_df['Rank_df2'] = merged_df[f"{score_col}_df2"].rank(ascending=False, method='min')

    # Calculate Dawdall points: 1 for rank 1, 1/2 for rank 2, 1/3 for rank 3, etc.
    merged_df['Dawdall_df1'] = 1 / merged_df['Rank_df1']
    merged_df['Dawdall_df2'] = 1 / merged_df['Rank_df2']

    # Sum the Dawdall points
    merged_df['Total_Dawdall'] = merged_df['Dawdall_df1'] + merged_df['Dawdall_df2']

    return merged_df



def crank_count(df1, df2, score_col, drug_col, p=1):
    # Work with copies to avoid modifying the original dataframes
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # Merge the two dataframes on the Drug column
    merged_df = pd.merge(df1_copy,
                         df2_copy,
                         on=drug_col,
                         suffixes=('_df1', '_df2'))

    # Rank drugs by score in each dataframe
    merged_df['Rank_df1'] = merged_df[f"{score_col}_df1"].rank(ascending=False, method='min')
    merged_df['Rank_df2'] = merged_df[f"{score_col}_df2"].rank(ascending=False, method='min')

    # Calculate CRank points using 1 / Rank^p
    merged_df['CRank_df1'] = 1 / (merged_df['Rank_df1'] ** p)
    merged_df['CRank_df2'] = 1 / (merged_df['Rank_df2'] ** p)

    # Sum the CRank points
    merged_df['Total_CRank'] = merged_df['CRank_df1'] + merged_df['CRank_df2']

    return merged_df


def logistic_regression_score(df1, df2, score_col, drug_col, positive_col):
    # Merge df1 and df2 on the drug column
    merged_df = pd.merge(df1, df2, on=drug_col, suffixes=('_df1', '_df2'))

    # Define features (scores from df1 and df2) and the target (Positive)
    X = merged_df[[f'{score_col}_df1', f'{score_col}_df2']]
    y = merged_df[positive_col + '_df1']  # Same column in both df1 and df2

    # Fit the logistic regression model with balanced class weights
    model = LogisticRegression(class_weight='balanced')
    model.fit(X, y)

    # Predict the probability of being positive (class 1)
    merged_df['Reg_Score'] = model.predict_proba(X)[:, 1]

    return merged_df

def linear_model_score(df1, df2, score_col, drug_col, positive_col):
    # Merge df1 and df2 on the drug column
    merged_df = pd.merge(df1, df2, on=drug_col, suffixes=('_df1', '_df2'))

    # Create interaction term: S1 * S2
    merged_df['S1_S2'] = merged_df[f'{score_col}_df1'] * merged_df[f'{score_col}_df2']

    # Define features (S1, S2, S1 * S2) and the target (Positive)
    X = merged_df[[f'{score_col}_df1', f'{score_col}_df2', 'S1_S2']]
    y = merged_df[positive_col + '_df1']  # Same column in both df1 and df2

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Extract the coefficients a1, a2, a3
    a1, a2, a3 = model.coef_

    # Predict the final score S using the model: S = a1*S1 + a2*S2 + a3*S1*S2
    merged_df['Reg_Score'] = model.predict(X)

    return merged_df, a1, a2, a3

