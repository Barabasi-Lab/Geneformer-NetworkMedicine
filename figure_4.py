"""
This script generates the plots presented in Figure 4 of the paper 'Transformers Enhance the Predictive Power of Network Medicine.'

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
     python3 figure_4.py
     ```

3. **Output:**
   - The resulting plots will be saved in the `plots` folder.
"""

import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


import pandas as pd
import numpy as np

from tools import ThemeSelector
from tools import ml_plotter as plotter

# Set global parameters for automatic scaling
plt.rcParams.update({
    'figure.figsize': (10, 6),  # Default figure size
    'axes.titlesize': 25,  # Title size
    'axes.labelsize': 25,  # Axes label size
    'xtick.labelsize': 20,  # X-axis tick label size
    'ytick.labelsize': 20,  # Y-axis tick label size
    'legend.fontsize': 20,  # Legend font size
    'lines.linewidth': 2,  # Line width
    'lines.markersize': 6  # Marker size
})


lfontsize = 15

def plot_ml_group(metrics,colors,labels, widths = [], saveDir= None, xlims= None):
        ##### COMBINED ROC
        plotter.plot_roc(metrics = metrics,
                  colors = colors,
                  labels = labels,
                  widths= widths,
                  xlims= xlims,
                  save_file= saveDir+"roc.pdf" if saveDir != None else None)



        ##### COMBINED HITS
        plotter.plot_hits(metrics = metrics,
                  colors = colors,
                  labels = labels,
                  widths = widths,
                  save_file=saveDir+"hits.pdf" if saveDir != None else None)

        ##### COMBINED PRECISION
        plotter.plot_precision(metrics = metrics,
                  colors = colors,
                  labels = labels,
                  k=100,
                  widths = widths,
                  save_file=saveDir+"precision.pdf" if saveDir != None else None)

        ##### COMBINED Recall
        plotter.plot_recall(metrics = metrics,
                  colors = colors,
                  labels = labels,
                  k=100,
                  widths = widths,
                  save_file=saveDir+"recall.pdf" if saveDir != None else None)


def summary_plot(cs_data, aw_data, control_data,ylim,ylabel, outFile= None):

    ran = list(range(len(aw_data)-1))
    labels = ["Input"] + [f"L{r}" for r in ran]

    # Number of labels
    x = np.arange(len(labels))

    # Width of the bars
    width = 0.35

    # Create the bar plot
    fig, ax = plt.subplots()

    cmap = plt.get_cmap('Blues')
    blues = [to_hex(cmap(i)) for i in np.linspace(0, 1, 4)]
    colors = [blues[2],blues[-1]]


    # Bar plot for the first list
    bars1 = ax.bar(x - width/2, cs_data, width, label='CS',color=colors[0])

    # Bar plot for the second list
    bars2 = ax.bar(x + width/2, aw_data, width, label='ATW',color=colors[1])

    plt.axhline(y=control_data,color='gray',linestyle='--')


    # Add some labels and title
    plt.ylim(ylim[0],ylim[1])
    ax.set_xlabel('Layers')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best',frameon=False)
    if outFile != None:
        plt.savefig(outFile, format="pdf", dpi=300)

    # Display the plot
    plt.show()



if __name__=='__main__':

    
    

    ## All layers attention weights
    metrics_unweighted = "data/drug_repurposing/output/unweighted/metrics.pkl"
    with open(metrics_unweighted, 'rb') as file:
        metrics_unweighted = pickle.load(file)

    metrics_at = [metrics_unweighted]
    colors = ['black']
    labels = ['Control']

    size = 6

    for i in range(size):
        path = f"data/drug_repurposing/output/attweights/avg/l{i}/metrics.pkl"

        with open(path, 'rb') as file:
            m = pickle.load(file)

        metrics_at.append(m)
        labels.append(f'AW L{i}')


    colors = colors + ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#d62728']
    widths = [2,1,1,1,1,1,2]

    plot_ml_group(metrics_at, colors, labels, widths, "plots/figure_4/att_weights_", xlims = (0,1.1))



    ## All layers embeddings
    metrics_cs = [metrics_unweighted]
    colors = ['black']
    labels = ['Control']

    layers = list(range(6))
    layers = ["input"] + layers

    for layer in layers:
        path = f"data/drug_repurposing/output/embeddings/l{layer}/metrics.pkl"

        with open(path, 'rb') as file:
            m = pickle.load(file)

        metrics_cs.append(m)
        labels.append(f'CS L{layer}')

    labels[1] = "CS L-in"

    #cmap = plt.get_cmap('tab10')
    #viridis = [to_hex(cmap(i)) for i in np.linspace(0, 1, size+3)]
    #viridis = viridis[::-1]
    #colors = colors + viridis
    colors = colors + ['#17becf','#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#d62728']
    widths = [2,1,2,1,1,1,1,1]


    plot_ml_group(metrics_cs, colors, labels,widths, "plots/figure_4/cosine_sim_",xlims=(0,1.1))

    #Summary plots
    metrics_at = metrics_at[1:]
    metrics_cs = metrics_cs[1:]

    # ROC Summary
    cs_data = [metrics['roc_auc'] for metrics in metrics_cs]
    aw_data = [metrics['roc_auc'] for metrics in metrics_at]
    aw_data = [0] + aw_data
    control_data = metrics_unweighted['roc_auc']

    summary_plot(cs_data,
                 aw_data,
                 control_data,
                 (0.5,0.8),
                 ylabel = "AUROC",
                 outFile="plots/figure_4/summary_roc.pdf")


    # Precision Summary
    cs_data = []
    for metrics in metrics_cs:
        aupc_cs = plotter.calculate_area_under_curve(metrics["prec_rec"]["Threshold"], metrics["prec_rec"]["Precision"])
        cs_data.append(aupc_cs)

    aw_data = []
    for metrics in metrics_at:
        aupc_aw = plotter.calculate_area_under_curve(metrics["prec_rec"]["Threshold"], metrics["prec_rec"]["Precision"])
        aw_data.append(aupc_aw)
    aw_data = [0] + aw_data

    control_data = plotter.calculate_area_under_curve(metrics_unweighted["prec_rec"]["Threshold"], metrics_unweighted["prec_rec"]["Precision"])

    summary_plot(cs_data,
                 aw_data,
                 control_data,
                 (0.3,0.6),
                 ylabel="AUPC",
                 outFile="plots/figure_4/summary_prec.pdf")


    # Recall Summary
    cs_data = []
    for metrics in metrics_cs:
        auc_cs = plotter.calculate_area_under_curve(metrics["prec_rec"]["Threshold"], metrics["prec_rec"]["Recall"])
        cs_data.append(auc_cs)

    aw_data = []
    for metrics in metrics_at:
        auc_aw = plotter.calculate_area_under_curve(metrics["prec_rec"]["Threshold"], metrics["prec_rec"]["Recall"])
        aw_data.append(auc_aw)
    aw_data = [0] + aw_data

    control_data = plotter.calculate_area_under_curve(metrics_unweighted["prec_rec"]["Threshold"], metrics_unweighted["prec_rec"]["Recall"])

    summary_plot(cs_data,
                 aw_data,
                 control_data,
                 (0.3,0.6),
                 ylabel="AURC",
                 outFile="plots/figure_4/summary_rec.pdf")
