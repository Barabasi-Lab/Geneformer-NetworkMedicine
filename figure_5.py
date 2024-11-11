"""
This script generates the plots presented in Figure 5 of the paper 'Transformers Enhance the Predictive Power of Network Medicine.'

### Instructions:

1. **Prepare Data for Plotting:**
   - First, execute the data preparation script `drug_repurposing.py` to generate the necessary data for plotting.
   - Run the following commands:
     ```bash
     cd YOUR_PATH_TO_GENEFORMER-NETWORKMEDICINE/drug_repurposing_scripts/
     export PYTHONPATH=YOUR_PATH_TO_GENEFORMER-NETWORKMEDICINE/drug_repurposing_scripts/
     python3 drug_repurposing.py
     ```
   - **Note:** This script may take some time to complete, so please be patient.

2. **Generate Plots:**
   - Once the data preparation is complete, execute this plotting script by running:
     ```bash
     cd YOUR_PATH_TO_GENEFORMER-NETWORKMEDICINE/
     export PYTHONPATH=YOUR_PATH_TO_GENEFORMER-NETWORKMEDICINE/
     python3 figure_5.py
     ```

3. **Output:**
   - The resulting plots will be saved in the `plots` folder.
"""


import pickle
import pandas as pd
import numpy as np

from tools import combine_scores as combiner
from tools import n_plotter as plotter

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

def plot_ml_group(metrics,colors,labels,saveDir):
        ##### COMBINED ROC
        plotter.plot_roc(metrics = metrics,
                  colors = colors,
                  labels = labels,
                  save_file= saveDir+"roc.pdf" if saveDir != None else None)



        ##### COMBINED HITS
        # plotter.plot_hits(metrics = metrics,
        #           colors = colors,
        #           labels = labels,
        #           save_file=saveDir+"hits.pdf" if saveDir != None else None)

        ##### COMBINED PRECISION
        plotter.plot_precision(metrics,
                  colors = colors,
                  labels = labels,
                  k=100,
                  name_top='Top',
                  name_prec='Precision',
                  save_file=saveDir+"precision.pdf" if saveDir != None else None)

        ##### COMBINED Recall
        plotter.plot_recall(metrics,
                  colors = colors,
                  labels = labels,
                  name_top='Top',
                  name_rec='Recall',
                  k=100,
                  save_file=saveDir+"recall.pdf" if saveDir != None else None)


def score_correlation(df, file):
    plt.figure()
    sns.scatterplot(data=df, x='Score_at', y='Score_cs', hue='Positive_at', palette=['blue', 'red'], marker='o')

    # Linear regression
    X = df['Score_at'].values.reshape(-1, 1)
    y = df['Score_cs'].values

    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    # Plot the regression line
    plt.plot(df['Score_at'], y_pred, color='black')

    # Calculate Pearson correlation coefficient and R-squared
    pearson_corr, _ = pearsonr(df['Score_at'], df['Score_cs'])
    r_squared = r2_score(y, y_pred)

    # Adding the legend with Pearson correlation and R-squared on separate lines
    plt.legend(title=f'Pearson: {pearson_corr:.2f}\nR²: {r_squared:.2f}', loc='best')


    legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], color='w', label=f'Pearson: {pearson_corr:.2f}', linestyle='None'),
    Line2D([0], [0], color='w', label=f'R²: {r_squared:.2f}', linestyle='None')
    ]

    # Adding the custom legend to the plot
    plt.legend(handles=legend_elements, loc='best',frameon=True)


    plt.xlabel("Attention Weight Score")
    plt.ylabel("Cosine Similarity Score")
    if file != None:
        plt.savefig(file, dpi=600)
    plt.show()




if __name__=='__main__':

    #The basic three
    metrics_unweighted = "data/drug_repurposing/output/unweighted/metrics.pkl"
    with open(metrics_unweighted, 'rb') as file:
        metrics_unweighted = pickle.load(file)
    s_un = metrics_unweighted['Drug Scores']
    s_un['Score'] = -1*s_un['Score']


    path = "data/drug_repurposing/output/attweights/avg/l5/metrics.pkl"
    with open(path, 'rb') as file:
        metrics_atw = pickle.load(file)
    s_at=metrics_atw['Drug Scores']
    s_at['Score']=-1*s_at['Score']

    path = "data/drug_repurposing/output/embeddings/l0/metrics.pkl"

    with open(path, 'rb') as file:
        metrics_cs = pickle.load(file)
    s_cs=metrics_cs['Drug Scores']
    s_cs['Score']=-1*s_cs['Score']


    # Then combinations
    scores_borda = combiner.borda_count(s_at,
                                        s_cs,
                                        'Score',
                                        'Name')
    s_borda = scores_borda[['Name','Positive_df1','Total_Borda']]
    s_borda.columns = ['Name','Positive','Score']


    scores_dawdall = combiner.dawdall_count(s_at,
                                            s_cs,
                                            'Score',
                                            'Name')
    s_dawdall = scores_dawdall[['Name','Positive_df1','Total_Dawdall']]
    s_dawdall.columns = ['Name','Positive','Score']

    scores_crank = combiner.crank_count(s_at,
                                        s_cs,
                                        'Score',
                                        'Name',
                                        p=4)
    s_crank = scores_crank[['Name','Positive_df1','Total_CRank']]
    s_crank.columns = ['Name','Positive','Score']

    scores = [s_un,s_at,s_cs,s_borda,s_dawdall,s_crank]
    metrics = [combiner.calculate_ml_metrics(df,'Score','Positive') for df in scores]
    colors = ['black','firebrick','salmon','dodgerblue','slateblue','violet']
    labels = ['Control','AW L5','CS L0','Borda','Dawdall','CRank']

    plot_ml_group(metrics, colors, labels, "plots/figure_5/combined_scores_")

    # Score correlation
    merged_df = pd.merge(s_at,
                         s_cs,
                         on='Name',
                         suffixes=('_at', '_cs'))
    score_correlation(merged_df,"plots/figure_5/correlation.pdf")

