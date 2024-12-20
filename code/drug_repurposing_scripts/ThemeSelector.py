# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:23:59 2024

@author: aalda
"""
import matplotlib.pyplot as plt

# print(plt.style.available)
plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams["figure.figsize"] = [9, 6]
plt.rcParams["figure.autolayout"] = True

plt.rcParams['font.size'] = 15
plt.rcParams.update({'font.size':15})
# Set the axes labels font size
plt.rc('axes', labelsize=15)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=15)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=15)
