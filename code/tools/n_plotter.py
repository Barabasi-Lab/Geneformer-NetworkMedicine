# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt

import numpy as np

from tools import ThemeSelector



# Set global parameters for automatic scaling
plt.rcParams.update({
    'figure.figsize': (10, 6),  # Default figure size
    'axes.titlesize': 25,  # Title size
    'axes.labelsize': 25,  # Axes label size
    'xtick.labelsize': 20,  # X-axis tick label size
    'ytick.labelsize': 20,  # Y-axis tick label size
    'legend.fontsize': 22,  # Legend font size
    'lines.linewidth': 2,  # Line width
    'lines.markersize': 6  # Marker size
})


lfontsize = 17


def plot_roc(metrics, colors, labels, save_file=None):

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--',color='gray')

    for i in range(len(metrics)):
        m = metrics[i]

        roc = m['roc_curve']
        auc = m['roc_auc']

        plt.plot(roc.FPR, roc.TPR, label=f'{labels[i]} (AUROC = {auc:.2f})', color= colors[i],alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right',fontsize=lfontsize)

    if save_file != None:
        plt.savefig(save_file, dpi=600)
    plt.show()


def plot_hits(metrics, colors, labels, save_file = None):
    scores = metrics[0]['Drug Scores']
    total_pos = scores.Positive.sum()

    m = total_pos/len(scores.index)
    xl = len(metrics[0]['hits'].index)

    plt.figure()
    plt.plot([0, xl], [0, m*xl], linestyle = '--',color='gray')
    for i in range(len(metrics)):
        hits = metrics[i]['hits']
        plt.plot(hits['Top N Candidates'], hits['Cumulative True Positives'], label=labels[i],color = colors[i],alpha=1)

    plt.xlabel('Top Candidates')
    plt.ylabel('Cummulative True Positives')
    plt.legend(loc='best',fontsize=lfontsize)
    if save_file != None:
        plt.savefig(save_file, dpi=600)
    plt.show()


def plot_precision(prec_data, colors, labels,k,name_top='Top',name_prec='Precision',save_file = None):
    plt.figure()

    for i in range(len(prec_data)):
        data = prec_data[i]['precision']
        data = data.head(k)

        plt.plot(data[name_top],data[name_prec],color=colors[i],label=labels[i])

    plt.ylabel("Precision")
    plt.xlabel("Top Candidates")
    plt.legend(loc='lower right',fontsize=lfontsize)
    if save_file != None:
        plt.savefig(save_file,dpi=600)
    plt.show()

def plot_recall(recall_data, colors, labels,k,name_top='Top',name_rec='Recall',save_file = None):
    plt.figure()

    for i in range(len(recall_data)):
        data = recall_data[i]['recall']
        data = data.head(k)
        plt.plot(data[name_top],data[name_rec],color=colors[i],label=labels[i])

    plt.ylabel("Recall")
    plt.xlabel("Top Candidates")
    plt.legend(loc='lower right',fontsize=lfontsize)
    if save_file != None:
        plt.savefig(save_file,dpi=600)
    plt.show()
