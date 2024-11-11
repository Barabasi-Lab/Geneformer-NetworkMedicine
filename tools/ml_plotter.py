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


def plot_roc(metrics, colors, labels, widths, xlims= None, save_file=None):

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--',color='gray')

    for i in range(len(metrics)):
        m = metrics[i]

        roc = m['roc_curve']
        auc = m['roc_auc']

        if widths == None:
            w = 1
        else:
            w = widths[i]

        plt.plot(roc.FPR, roc.TPR, label=f'{labels[i]} (AUROC={auc:.2f})', color= colors[i],alpha=1, linewidth=w)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if xlims != None:
        plt.xlim(*xlims)
    plt.legend(loc='lower right',fontsize=lfontsize, ncol=1)

    if save_file != None:
        plt.savefig(save_file, dpi=600)
    plt.show()


def plot_hits(metrics, colors, labels, widths, save_file = None):
    scores = metrics[0]['Drug Scores']
    total_pos = scores.Positive.sum()

    m = total_pos/len(scores.index)
    xl = len(metrics[0]['hits'].index)

    plt.figure()
    plt.plot([0, xl], [0, m*xl], linestyle = '--',color='gray')


    for i in range(len(metrics)):
        hits = metrics[i]['hits']

        if widths == None:
            w = 1
        else:
            w = widths[i]

        plt.plot(hits['Top N Candidates'], hits['Cumulative True Positives'], label=labels[i],color = colors[i],alpha=1, linewidth=w)

    plt.xlabel('Top Candidates')
    plt.ylabel('Cummulative True Positives')
    plt.legend(loc='best',fontsize=lfontsize)
    if save_file != None:
        plt.savefig(save_file, dpi=600)
    plt.show()


def prec_rec(s):
    scores = s.sort_values(by='Score',ascending=True)
    scores = scores.reset_index(drop=True)

    pos = scores['Positive']
    total_pos = pos.sum()

    tops = np.array(range(len(scores)))

    p = []
    r = []

    p_ac = 0

    for t in tops:
        if scores.loc[t,'Positive'] == 1:
            p_ac = p_ac+1

        p.append(p_ac / (t+1))
        r.append(p_ac / total_pos)

    tops = tops+1

    p_c= total_pos / len(scores.index)
    r_c = total_pos*(tops/(len(scores.index)*len(scores.index)))

    return (tops,p,r,p_c,r_c)

def plot_precision(metrics, colors, labels,k, widths, save_file = None):
    plt.figure()
    tu,pu,ru,pcu,rcu = prec_rec(metrics[0]['Drug Scores'])
    plt.axhline(y=pcu,color='gray',linestyle='--')

    for i in range(len(metrics)):
        t,p,r,pc,rc = prec_rec(metrics[i]['Drug Scores'])

        if k != 'all':
            t = t[0:k]
            p = p[0:k]
            r = r[0:k]
            rc = rc[0:k]

        if widths == None:
            w = 1
        else:
            w = widths[i]

        plt.plot(t,p,color=colors[i],label=f"{labels[i]}",linewidth=w)

    plt.ylabel("Precision")
    plt.xlabel("Top Candidates")
    plt.ylim(bottom=-0.15)
    plt.legend(loc='lower right',fontsize=lfontsize,ncol=2)
    if save_file != None:
        plt.savefig(save_file,dpi=600)
    plt.show()

def plot_recall(metrics, colors, labels,k, widths, save_file = None):
    plt.figure()
    tu,pu,ru,pcu,rcu = prec_rec(metrics[0]['Drug Scores'])
    if k != 'all':
        tu = tu[0:k]
        pu = pu[0:k]
        ru = ru[0:k]
        rcu = rcu[0:k]
    plt.plot(tu,rcu,color='gray',linestyle='--')

    for i in range(len(metrics)):
        t,p,r,pc,rc = prec_rec(metrics[i]['Drug Scores'])

        if k != 'all':
            t = t[0:k]
            p = p[0:k]
            r = r[0:k]
            rc = rc[0:k]

        if widths == None:
            w = 1
        else:
            w = widths[i]

        plt.plot(t,r,color=colors[i],label=labels[i], linewidth=w)

    plt.ylabel("Recall")
    plt.xlabel("Top Candidates")

    plt.legend(loc='best',fontsize=lfontsize)
    if save_file != None:
        plt.savefig(save_file,dpi=600)
    plt.show()


def calculate_area_under_curve(x, y):
    
    if len(x) != len(y):
        raise ValueError("Input vectors x and precision must have the same length.")

    # Sort x and precision based on x (in case x is not sorted)
    sorted_indices = np.argsort(x)
    x_sorted = np.array(x)[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]

    # Calculate the area using the trapezoidal rule
    area_trapz = np.trapz(y_sorted, x_sorted)

    # Normalize the area so that the maximum possible area is 1
    max_area = x_sorted[-1] - x_sorted[0]
    area_trapz_normalized = area_trapz / max_area

    return area_trapz_normalized
