import requests
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
import tqdm
import matplotlib.pyplot as plt
import numpy as np

def check_interaction(pair):
    """
    Check if there is an interaction between two genes for human species.
    """
    species_id = 9606  # NCBI Taxonomy ID for human
    string_api_url = "https://string-db.org/api"
    output_format = "json"
    method = "interaction_partners"
    request_url = f"{string_api_url}/{output_format}/{method}"

    params = {
        "identifiers": "%0d".join(pair[:2]),
        "species": species_id,
        "caller_identity": "my_research_project"
    }

    response = requests.post(request_url, data=params)
    interaction_data = response.json()

    # Check if the second gene in the pair is an interaction partner of the first
    try:
        interacts = any(partner['preferredName_B'] == pair[1] for partner in interaction_data)
        return 1 if interacts else 0
    except:
        return 0

def analyze_interactions(gene_pairs, CDF = False):
    """
    Analyzes the interaction data for a list of human gene pairs.
    """
    # Define the thread pool executor
    with ThreadPoolExecutor(max_workers = 2) as executor:
        # Use tqdm for progress reporting. Wrap the executor.map with tqdm
        interaction_results = list(tqdm.tqdm(executor.map(check_interaction, gene_pairs), total=len(gene_pairs), desc='Analyzing Interactions'))

    cdf, counter = [], 0
    for i in interaction_results:
        counter += i
        cdf.append(counter)
    
    if CDF == True:
        cdf = [i/counter for i in cdf]
        
        proportion_with_interaction = interaction_results.count(1)/len(interaction_results)
    
        return proportion_with_interaction, cdf
    else:
        ratio = []
        for counter, i in enumerate(cdf):
            ratio.append(i/(counter + 1))
    
        return ratio[-1], ratio
        
def create_cdf(cdf, keyword, ratio_distribution = True):
    """
    Creates a Cumulative Distribution Function (CDF) plot for the interaction results.
    """
    n = len(cdf)
    plt.figure(figsize=(10, 6))
    if ratio_distribution != True:
        background_cdf = [i/n for i in range(n)]
        plt.plot(range(n), cdf, color = 'blue', label='STRING Interaction CDF')
        plt.plot(range(n), background_cdf, color='r', linestyle='--', label='Expected Discovery Rate')
        plt.xlabel('Total Attention Weights by Rank')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of Gene-Gene Interaction Results')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'String_{keyword}_CDF.png')
    else:
        background = [i for i in range(len(cdf))]
        plt.plot(background, cdf, color = 'blue', label = 'STRING Interaction \n Ratio')
        plt.plot([0, len(cdf)], [np.mean(cdf), np.mean(cdf)], label = 'Expected Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'String_{keyword}_Ratio.png')
        
        