# File imports
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import xml.etree.ElementTree as ET
from torch.optim import Adam, lr_scheduler, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch_geometric
import scipy.interpolate as interpolate
from Bio.KEGG.KGML import KGML_parser

# Imports attention_base functions, packages, and utilities
from base_utils.attention_base import *

# Primary class for attention extraction
class PPI_attention:
    def __init__(self, mean = True, 
                  dataset_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/genecorpus_30M_2048.dataset"),
                  model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo"),
                  attention_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Max_attentions"),
                  layer_index = 4):
                  
        # Initializes model variables
        self.gene_attentions = {}
        self.mean = mean
        self.model_location = model_location
        self.dataset_location = dataset_location
        self.layer_index = layer_index
        
        # Creates attention location if it does not exist
        try:
            os.mkdir(attention_location)
        except:
            pass
        self.attention_location = attention_location
      
    # Saves the model
    def save(self, filename = 'Attention_Extraction_obj.pk'):
        with open(filename, 'wb') as file:
            pk.dump(self, file)
        
    # Randomly samples from given dataset to extract attentions
    def scrape_attentions(self, samples = 100, disease = None, filter_label = None,
                          save_threshold = False):
        # Note - save_threshold set to True only saves weights that are above the threshold mean + standard deviation, to save on compute
        
        
        if disease == None:
            # Obtains attentions
            gene_attentions = extract_attention(model_location = self.model_location,
                                                data = self.dataset_location,
                                                mean = self.mean,
                                                layer_index = self.layer_index,
                                                data_sample = None,
                                                samples = samples,
                                                filter_label = filter_label,
                                                save_threshold = save_threshold)
    
        else:
             PPI = instantiate_ppi()
            
             # Identifies disease genes
             disease_genes = isolate_disease_genes(disease)
             
             # Identifies LCC 
             disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
        
             # Obtains attentions while filtering for samples with relevant genes
             gene_attentions = extract_attention(model_location = self.model_location, 
                                                filter_label = filter_label,
                                                data = self.dataset_location,
                                                mean = self.mean,
                                                layer_index = self.layer_index,
                                                data_sample = None,
                                                samples = samples,
                                                filter_genes = list(disease_LCC.nodes()),
                                                save_threshold = save_threshold)
                                  
       
        # Merges attentions with base attention
        combine_dictionaries = [gene_attentions, self.gene_attentions]
        self.gene_attentions = merge_dictionaries(combine_dictionaries, mean = self.mean)                                         
        
     # Randomly samples from given dataset to extract attentions
    def scrape_perturbed_attentions(self, samples = 100, disease = None, filter_label = None,
                          save_threshold = False, perturbation = None, random_perturb_num = 30):
    
        # Identifies disease genes
        disease_genes = isolate_disease_genes(disease)

        # Loads interactome
        PPI = instantiate_ppi()

        # Identifies perturbed gene
        if perturbation == None:
            LCC_gene = LCC_genes(PPI, disease_genes, subgraph = False)
            perturbation = random.sample(LCC_gene, random_perturb_num)
            print(perturbation)

        if disease == None:
            # Obtains attentions
            gene_attentions, perturbed_attentions = extract_attention(model_location = self.model_location,
                                                data = self.dataset_location,
                                                mean = self.mean,
                                                layer_index = self.layer_index,
                                                data_sample = None,
                                                samples = samples,
                                                filter_label = filter_label,
                                                save_threshold = save_threshold,
                                                perturb_genes = perturbation)
    
        else:
        
            # Obtains attentions while filtering for samples with relevant genes
            gene_attentions, perturbed_attentions = extract_attention(model_location = self.model_location, 
                                            filter_label = filter_label,
                                            data = self.dataset_location,
                                            mean = self.mean,
                                            layer_index = self.layer_index,
                                            data_sample = None,
                                            samples = samples,
                                            filter_genes = perturbation,
                                            save_threshold = save_threshold,
                                            perturb_genes = perturbation)
                                
    
        return gene_attentions, perturbed_attentions, perturbation

    # Scrapes a smaller subset of the total data
    def scrape_subset(self, instance = 1, total_jobs = 30_000, total_samples = 30_000_000, samples = None, disease = None, filter_label = None,
                      save_threshold = False):
    
        # Dilineates subset parameters
        chunk_size = total_samples // total_jobs
        bottom_index = (instance - 1) * chunk_size
        top_index = args.instance * chunk_size
        dataset_sample = (bottom_index, top_index)
        if samples == None:
            sample_num = dataset_sample[1] - dataset_sample[0]
        else:
            sample_num = samples
        
        if disease == None:
            # Obtains attentions
            gene_attentions = extract_attention(model_location = self.model_location,
                                                data = self.dataset_location,
                                                mean = self.mean,
                                                layer_index = self.layer_index,
                                                samples = sample_num,
                                                data_sample = dataset_sample,
                                                filter_label = filter_label,
                                                save_threshold = save_threshold)
        
        else:
            PPI = instantiate_ppi()
            
            # Identifies disease genes
            disease_genes = isolate_disease_genes(disease)    
            disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
      
            # Obtains attentions while filtering for samples with relevant genes
            gene_attentions = extract_attention(model_location = self.model_location,
                                                data = self.dataset_location,
                                                mean = self.mean,
                                                layer_index = self.layer_index,
                                                data_sample = None,
                                                samples = sample_num,
                                                filter_genes = list(disease_LCC.nodes()),
                                                save_threshold = save_threshold)
        
        # Saves attention to attention dictionary location
        with open(f"{self.attention_location}/{instance}.pk", 'wb') as f:
            pk.dump(gene_attentions, f)
        print(f"{self.attention_location}/{instance}.pk")
        
            
    # Combines all dictionaries from location with saved attentions
    def merge_attentions(self, show = False, scale = False, normalize = False, save = True,
                         limit = None, attention_location = None, mean = None):
        
        if attention_location == None:
            attention_location = self.attention_location
            
        # List attention files in the specified directory
        attention_files = list(attention_location.glob('*.pk'))
        
        if limit != None:
            attention_files = random.sample(attention_files, limit)
        
        # Merge attention dictionaries efficiently 
        if mean == None:
            gene_attentions = merge_dictionaries(attention_files, self.mean)
        else:
            gene_attentions = merge_dictionaries(attention_files, mean)
        
        # Normalizes aggregated dictionary if specified
        if normalize == True:
            min_value = min([min(attentions.values()) for attentions in gene_attentions.values() if len(attentions) > 0])
            max_value = max([max(attentions.values()) for attentions in gene_attentions.values() if len(attentions) > 0])
            
            minmax = max_value - min_value
            for source_gene, target_genes in tqdm.tqdm(gene_attentions.items(), total = len(gene_attentions), desc = 'Normalizing combined dict'):
                for target_gene in target_genes:
    
                    # Normalize the attention value
                    normalized_attention = (gene_attentions[source_gene][target_gene] - min_value) / minmax
    
                    # Update the dictionary with the normalized value
                    gene_attentions[source_gene][target_gene] = normalized_attention
            
        # Scales aggregated dictionary if specified
        if scale == True:
            values = []
            for _, subdict in gene_attentions.items():
                values.extend(list(subdict.values()))
            values = [[value] for value in values]
    
            # Scale the values using StandardScaler
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(values)
    
            # Update aggregated_gene_attentions with scaled values
            index = 0
            for _, subdict in self.gene_attentions.items():
                for target_key in subdict:
                    subdict[target_key] = scaled_values[index][0]
                    index += 1
        
        if show == True:
            print(f'{len(gene_attentions)} source genes identified')
            total_pairs = sum(len(attentions) for attentions in gene_attentions.values())
            print(f'{total_pairs} total gene-gene attention pairs!')
        
        if save == True:
            self.gene_attentions = gene_attentions
        else:
            return gene_attentions 
    
    # Performs mapping and analysis of a certain disease LCC
    def map_disease_genes(self, disease = 'Cardiomyopathy Hypertrophic', keyword = None):
        PPI = instantiate_ppi()
    
        # Identifies disease genes
        disease_genes = isolate_disease_genes(disease)
        
        # Identifies LCC 
        disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
            
        # Maps gene attentions to the PPI and the LCC 
        self.PPI, _, _ = map_attention_attention(PPI, self.gene_attentions, save = True,)
        LCC, _, _ = map_attention_attention(disease_LCC, self.gene_attentions,)
    
        # Calculates AUC predictions   
        F1_graph_attention(LCC = LCC, PPI = self.PPI, gene_attentions = self.gene_attentions, keyword = keyword)
        
        # Performs comorbidity analysis
        #comorbidity_analysis(PPI = self.PPI, attention_dict = self.gene_attentions, keyword = keyword)

        # Plots distributions of attention with regards to disease genes
        plot_distributions(self.gene_attentions, disease = disease, keyword = keyword)

        # Performs community analysis
        community_detection(PPI, attention_dict = self.gene_attentions, disease = 'cardiomyopathy hypertrophic', keyword = keyword)
        
        # Analyze hops
        #analyze_hops(self.gene_attentions, keyword = keyword)
    
        # Analyzes top attentions
        #check_top_attentions(attention_dict = self.gene_attentions, PPI = disease_LCC, keyword = keyword)

        # Analyzes multiple LCCs
        #compare_LCCs(attentions = self.gene_attentions, keyword = keyword)
         
    # Performs mapping and analysis of the PPI
    def map_PPI_genes(self, keyword = None):
     
        PPI = instantiate_ppi()
        
        # Maps attention weights to PPI
        self.PPI, _, _ = map_attention_attention(PPI, self.gene_attentions, save = True)
        
        # Scores PPI
        F1_graph_attention(PPI = self.PPI, gene_attentions = self.gene_attentions, keyword = keyword)

        # Performs comorbidity analysis
        #comorbidity_analysis(PPI = self.PPI, attention_dict = self.gene_attentions, keyword = keyword)

        # Plots distribution of weights 
        plot_distributions(self.gene_attentions, disease = None, keyword = keyword)

        # Performs community analysis
        community_detection(PPI, attention_dict = self.gene_attentions, disease = 'cardiomyopathy hypertrophic', keyword = keyword)
        
        # Analyze hops
        #analyze_hops(self.gene_attentions, keyword = keyword)
        
        # Analyzes top attentions
        #check_top_attentions(attention_dict = self.gene_attentions, PPI = PPI, keyword = keyword)
        
        # Analyzes multiple LCCs
        #compare_LCCs(attentions = self.gene_attentions, keyword = keyword)


    # Uses attention directionality to map interaction direction
    def map_direction(self, comparison = 'median', interactions = Path('PPI/GRN.csv'), #Path('enzo/Directional.csv'), 
                      direction = 'source', direct_comparison = False,
                      median_direction = 'target'): 
         
        if direct_comparison == True:
            # Obtains connections
            interactions = pd.read_csv(interactions)
            GRN = nx.from_pandas_edgelist(interactions, create_using = nx.DiGraph())

            if comparison == 'median':
                # Obtains the median attention given to a gene
                median_attention = {}
                for gene, attentions in self.gene_attentions.items():
                    for sub_gene, att in attentions.items():
                        if sub_gene not in median_attention:
                            median_attention[sub_gene] = []
                        if median_direction == 'target':
                            median_attention[sub_gene].append(att)
                        else: 
                            median_attention[gene].append(att)

                median_attentions = {gene: np.median(attentions) for gene, attentions in median_attention.items()}
                prediction_accuracy = [] 
                for u, v in GRN.edges:
                    # Tries to map connection
                    try:
                        if direction == 'forwards':
                            directed_attention = self.gene_attentions[u][v]
                            backwards_attention = self.gene_attentions[v][u]
                        else:
                            directed_attention = self.gene_attentions[v][u]
                            backwards_attention = self.gene_attentions[u][v]

                    except:
                        continue

                    scaled_directed_attention = directed_attention / median_attentions[u]
                    scaled_backwards_attention = backwards_attention / median_attentions[v]

                    if (scaled_directed_attention == 0 and scaled_backwards_attention == 0) or (isinstance(scaled_directed_attention, list) or isinstance(scaled_backwards_attention, list)):
                        continue

                    if scaled_directed_attention >= scaled_backwards_attention:
                        prediction = 1
                    elif scaled_directed_attention < scaled_backwards_attention:
                        prediction = 0
                
                    prediction_accuracy.append(prediction)

            elif comparison == 'mean':
                # Obtains the median attention given to a gene
                median_attention = {}
                for gene, attentions in self.gene_attentions.items():
                    for sub_gene, att in attentions.items():
                        if sub_gene not in median_attention:
                            median_attention[sub_gene] = []
                        if median_direction == 'target':
                            median_attention[sub_gene].append(att)
                        else: 
                            median_attention[gene].append(att)
                
                median_attentions = {gene: np.mean(attentions) for gene, attentions in median_attention.items()}

                prediction_accuracy = [] 
                for u, v in GRN.edges:
                    # Tries to map connection
                    try:
                        directed_attention = self.gene_attentions[u][v]
                        backwards_attention = self.gene_attentions[v][u]
                    except:
                        continue
                    if direction == 'forwards':
                        scaled_directed_attention = self.gene_attentions[u][v]
                        scaled_backwards_attention = self.gene_attentions[v][u]
                    else:
                        scaled_directed_attention = self.gene_attentions[v][u]
                        scaled_backwards_attention = self.gene_attentions[u][v]

                    if (scaled_directed_attention == 0 and scaled_backwards_attention == 0) or (isinstance(scaled_directed_attention, list) or isinstance(scaled_backwards_attention, list)):
                        continue

                    if scaled_directed_attention > scaled_backwards_attention:
                        prediction = 1
                    elif scaled_directed_attention < scaled_backwards_attention:
                        prediction = 0
                    prediction_accuracy.append(prediction)
            else:
                prediction_accuracy = []
                for u, v in GRN.edges:
                    # Tries to map connection
                    try:
                        if direction == 'forwards':
                            scaled_directed_attention = self.gene_attentions[u][v]
                            scaled_backwards_attention = self.gene_attentions[v][u]
                        else:
                            scaled_directed_attention = self.gene_attentions[v][u]
                            scaled_backwards_attention = self.gene_attentions[u][v]                        
                    except:
                        continue
                    
                    if (scaled_directed_attention == 0 and scaled_backwards_attention == 0) or (isinstance(scaled_directed_attention, list) or isinstance(scaled_backwards_attention, list)):
                        continue

                    if scaled_directed_attention > scaled_backwards_attention:
                        prediction = 1
                    else:
                        prediction = 0

                    prediction_accuracy.append(prediction)

            print(f'Out of {len(prediction_accuracy)} connections, {sum(prediction_accuracy)} were correctly predicted.')
            print(f'Proportion of 0 edges: {(len(GRN.edges()) - len(prediction_accuracy))/len(GRN.edges())}')
            print(f'Hits: {sum(prediction_accuracy) / len(prediction_accuracy) * 100} %')
        else:

            def convert_gene_symbols_to_uniprot_ids(gene_symbols):
                uniprot_ids = []
                for gene in tqdm.tqdm(gene_symbols, total = len(gene_symbols), desc = 'Converting to Uniprot'):
                    url = f'https://rest.uniprot.org/uniprotkb/search?query={gene}%20human'
                    Data = requests.get(url).text
                    firstResultIndex = Data.index('primaryAccession')
                    uniprot_accession = Data[firstResultIndex + 19:firstResultIndex + 25] 

                    uniprot_ids.append(uniprot_accession)
                return uniprot_ids

            def fetch_kinetic_data_from_sabio(uniProt_ids):
                base_url = "http://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/sbml"  # Adjusted to the endpoint for searching kinetic laws
                kinetic_data = {}

                for uniProt_id in tqdm.tqdm(uniProt_ids, desc='Fetching kinetic data from SABIO-RK..'):
                    response = requests.get(f"{base_url}?q=UniProtKB_AC:{uniProt_id}")  # Query by UniProtKB AC
                    if response.status_code == 200:
                        data = None
                        try:
                            data = response.text
                        except Exception as e:
                            pass

                        if data and "No results found for query" not in data:
                            extract_interaction_targets(data)
                    else:
                        pass

                return kinetic_data
            
            def extract_interaction_targets(sbml_data):
                root = ET.fromstring(sbml_data)
                ns = {
                    'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
                    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
                }
                
                species_info = {}
                for species in root.findall('.//sbml:species', ns):
                    species_id = species.attrib['id']
                    species_name = species.attrib.get('name', species_id)
                    species_info[species_id] = species_name

                gene_interactions = {}
                for reaction in root.findall('.//sbml:reaction', ns):
                    reactants = [ref.attrib['species'] for ref in reaction.findall('.//sbml:listOfReactants/sbml:speciesReference', ns)]
                    products = [ref.attrib['species'] for ref in reaction.findall('.//sbml:listOfProducts/sbml:speciesReference', ns)]

                    # Extract kinetic law and parameters
                    kinetic_law = reaction.find('.//sbml:kineticLaw', ns)
                    parameters = {}
                    if kinetic_law is not None:
                        for param in kinetic_law.findall('.//sbml:listOfLocalParameters/sbml:localParameter', ns):
                            try:
                                param_id = param.attrib['id']
                                param_value = param.attrib['value']
                                parameters[param_id] = param_value
                            except:
                                continue
                    
                    for reactant in reactants:
                        reactant_name = species_info[reactant]
                        product_names = [species_info[product] for product in products]
                        
                        if reactant_name not in gene_interactions:
                            gene_interactions[reactant_name] = {}
                        
                        for product_name in product_names:
                            # Here we store the target gene and associated kinetic parameters
                            gene_interactions[reactant_name][product_name] = parameters

                    # Base URL for SABIO-RK API
                    base_url = "https://sabiork.h-its.org/sabioRestWebServices/"

                    # Endpoint for searching reactions by ligand
                    search_reactions_endpoint = "searchReactions"

                    target_proteins = []
                    for ligand in gene_interactions.keys():   
                        # Parameters for the GET request
                        params = {
                            "substratename": ligand,
                            "format": "json"
                        }
                        
                        # Make the GET request to the SABIO-RK API
                        response = requests.get(base_url + search_reactions_endpoint, params=params)
                        
                        print(response)
                        # Check if the response was successful
                        if response.status_code == 200:
                            # Parse the response JSON

                            reactions_data = response.json()
                            
                            # Extract protein information (usually enzymes) from the reactions
                            proteins = set()
                            for reaction in reactions_data:
                                enzyme_info = reaction.get("enzyme", [])
                                for enzyme in enzyme_info:
                                    proteins.add(enzyme["name"])
                            
                            print(proteins)
                            target_proteins.append(proteins)

                return target_proteins

            def fetch_pathway_data_from_sabio(uniProt_ids):

                base_url = "http://sabiork.h-its.org/sabioRestWebServices/"
                pathway_data = {}

                for uniProt_id in tqdm.tqdm(uniProt_ids, desc='Fetching pathway data from SABIO-RK..'):
                    # Adjust the endpoint for searching pathways by UniProtKB AC
                    response = requests.get(f"{base_url}reactions/reactionIDs?q=UniProtKB_AC:{uniProt_id}")
                    
                    if response.status_code == 200:
                        data = None
                        try:
                            data = response.text
                        except Exception as e:
                            print(f"Error processing response: {e}")
                            continue

                        if data and "No results found for query" not in data and len(data) > 0:
                            data = extract_protein_pathways(data)
                            pathway_data[uniProt_id] = data
                    else:
                        print(f"Failed to fetch data for {uniProt_id}: {response.status_code}")

                return pathway_data

            def extract_protein_pathways(sbml_data):
                root = ET.fromstring(sbml_data)
                pathway_ids = [reaction.find('SabioReactionID').text for reaction in root.findall('.//SabioReaction')]
                
                base_url = "http://sabiork.h-its.org/sabioRestWebServices/kineticLaws/"

                reactions = {}
                for pathway_id in pathway_ids:
                    response = requests.get(f"{base_url}{pathway_id}")
                    if response.status_code == 200:
                        data = response.text

                        # Parse the SBML data
                        root = ET.fromstring(data)
                        
                        # Namespace map
                        ns = {'sbml': 'http://www.sbml.org/sbml/level3/version1/core'}
                        
                        # Find all reaction elements
                        reactions = root.findall('.//sbml:reaction', ns)
                        
                        # List to store reaction details
                        reaction_list = []
                        
                        # Extract details for each reaction
                        for reaction in reactions:
                            # Basic reaction details
                            reaction_id = reaction.get('id')
                            reaction_name = reaction.get('name')
                            
                            # Reactants and Products
                            reactants = [(reactant.get('species'), reactant.get('stoichiometry'))
                                        for reactant in reaction.findall('./sbml:listOfReactants/sbml:speciesReference', ns)]
                            products =  [(product.get('species'), product.get('stoichiometry')) 
                                        for product in reaction.findall('./sbml:listOfProducts/sbml:speciesReference', ns)]
                            
                            # Modifiers
                            modifiers = [modifier.get('species') 
                                        for modifier in reaction.findall('./sbml:listOfModifiers/sbml:modifierSpeciesReference', ns)]
                            
                            # Define the URL for the UniProt API
                            uniprot_url = "https://rest.uniprot.org/uniprotkb/stream?query="

                            gene_reactants, gene_products = [], []
                            for spc_id, score in reactants:  
                                full_url = uniprot_url + spc_id
                                
                                # Send a GET request to the UniProt API
                                response = requests.get(full_url)

                                # Check if the response was successful
                                if response.status_code == 200:
                                    data = response.json()
                                    try:
                                        results = data['results'][0]
                                    except:
                                        results = data['results']

                                    gene = results['genes']['geneName']['value']
                                    gene_reactants.append(gene)

                            for spc_id, score in products:  
                                full_url = uniprot_url + spc_id
                                
                                # Send a GET request to the UniProt API
                                response = requests.get(full_url)

                                # Check if the response was successful
                                if response.status_code == 200:
                                    data = response.json()
                                    try:
                                        results = data['results'][0]
                                    except:
                                        results = data['results']
                                    gene = results['genes']['geneName']['value']
                                    gene_products.append(gene)
                            print(gene_reactants)
                            print(gene_products)
                            sys.exit()
                                   

            # Creates PPI
            PPI = instantiate_ppi()
            PPI, _, _ = map_attention_attention(PPI, self.gene_attentions, save = True)

            # Obtains gene and pathwaydata from KEGG
            genes_to_map = random.sample(list(PPI.nodes()), 100)
            uniprot_ids = convert_gene_symbols_to_uniprot_ids(genes_to_map)
            pathways = fetch_pathway_data_from_sabio(uniprot_ids)
            #pathways = fetch_kinetic_data_from_sabio(uniprot_ids)
            print(pathways)

    # Maps training from pretrained to finetuned model
    def map_attention_changes(self, base_location = Path('Max_attentions'),
                            comparison_location = Path('attentions'),
                            disease = None,
                            limit = 1, keyword = 'covid',
                            comparison_type = 'merge', intermediaries = False,
                            LCC_compare = True,
                            samples = 100, layer_index = -1, 
                            save = True):
        
        # Obtains disease LCC
        PPI = instantiate_ppi()
        disease_genes = isolate_disease_genes(disease)

        LCC = LCC_genes(PPI, disease_genes, subgraph = True)

        # Obtain merged attention dictionary for both conditions   
        if comparison_type == 'merge': 
            base = self.merge_attentions(attention_location = base_location, limit = limit, save = False)
            compare = self.merge_attentions(attention_location = comparison_location, limit = limit, save = False)
            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword,)

        elif comparison_type == 'cardiomyopathy': # Compares cardiomyopathy hypertrophic
        
            new_attention = PPI_attention(layer_index = layer_index, mean = self.mean, dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            new_attention.scrape_attentions(samples = samples, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = layer_index, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                         mean = self.mean, dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            
            new_attention.scrape_attentions(samples = samples, disease = disease, filter_label = ('disease', 'hcm'))
            compare = new_attention.gene_attentions

            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)
        elif comparison_type == 'dilated': # Compares cardiomyopathy dilated
        
            new_attention = PPI_attention(layer_index = layer_index, mean = self.mean,)# dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            new_attention.scrape_attentions(samples = samples, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = layer_index, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                         mean = self.mean, dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            
            new_attention.scrape_attentions(samples = samples, disease = disease, filter_label = ('disease', 'dcm'))
            compare = new_attention.gene_attentions

            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)
            
        elif comparison_type == 'arthritis': # Compares arthritis 
            
            new_attention = PPI_attention(layer_index = layer_index, mean = False, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/"))
            new_attention.scrape_attentions(samples = samples, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = layer_index, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/Arthritis_small"),
                         mean = False) 
            new_attention.scrape_attentions(samples = samples, disease = disease,)
            compare = new_attention.gene_attentions
            
            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)


        elif comparison_type == 'carcinoma': # Compares adenocarcinoma
            
            new_attention = PPI_attention(layer_index = layer_index, mean = self.mean, 
                                          model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/"))
            new_attention.scrape_attentions(samples = samples, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = layer_index, 
                        model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/carcinoma_model"), mean = self.mean) 
            new_attention.scrape_attentions(samples = samples, disease = disease,)
            compare = new_attention.gene_attentions
            
            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)


        elif comparison_type == 'covid': # Compares covid
            new_attention = PPI_attention(layer_index = layer_index, mean = self.mean, 
                                          model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo"))
            new_attention.scrape_attentions(samples = samples, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = layer_index, 
                        model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/COVID_model"), mean = self.mean) 
            new_attention.scrape_attentions(samples = samples, disease = disease,)
            compare = new_attention.gene_attentions
            
            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)

        elif comparison_type == 'combined': # Obtains attentions from a large variety of pretrained models, and compares one disease model to all the others
            
            disease_collection = ['covid', 'arthritis rheumatoid', 'adenocarcinoma', 'cardiomyopathy hypertrophic']
            model_loc = [Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/COVID_model"),
                Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/Arthritis"), 
                Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/carcinoma_model"),
                Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            ]
            background_dictionaries = []

            # Iterates through each disease
            for count, coll in enumerate(disease_collection):
                new_attention = PPI_attention(layer_index = layer_index, mean = self.mean, 
                                          model_location = model_loc[count])
                new_attention.scrape_attentions(samples = samples, disease = coll,)
                if coll == disease:
                    compare = new_attention.gene_attentions
                else:
                    background_dictionaries.append(new_attention.gene_attentions)
            base = merge_dictionaries(background_dictionaries, mean = self.mean)
            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)

        elif comparison_type == 'perturb': # Compares the ebfore/after of a gene perturbation

            new_attention = PPI_attention(layer_index = layer_index, mean = self.mean, 
                                           model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),)
            base, compare, perturbed_genes = new_attention.scrape_perturbed_attentions(samples = samples,
                                                                       disease = disease,)
            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword,
                                                                    perturb = True)
              
            # Performs perturbation analysis
            LCC_changes = []     
            for u, v in LCC.edges():
                try:
                    weight = comparison_dict[u][v][1] - comparison_dict[u][v][0]
                    #weight = comparison_dict[u][v][1] / (comparison_dict[u][v][0] + 0.00001)
                    LCC_changes.append(weight)
                except:
                    continue

            # Finds edges in LCC directly connected to perturbed gene 
            direct_edges = [] 
            for gene in perturbed_genes:
                for u, v in LCC.edges():
                    if u == gene or v == gene:
                        try:
                            weight = comparison_dict[u][v][1] - comparison_dict[u][v][0]
                            #weight = comparison_dict[u][v][1] / (comparison_dict[u][v][0] + 0.00001)
                            direct_edges.append(weight)
                        except:
                            continue

            # Finds neighbors of LCC, and samples to the number of samples
            disease_set = set(disease_genes)
            primary_neighbors = set(neighbor_group(graph = PPI, nodes = disease_genes, LCC = LCC))
            primary_changes = []
            for u, v in PPI.edges():
                if (u in disease_set and v in primary_neighbors) or (v in disease_set or v in primary_neighbors):
                    try:
                        weight = comparison_dict[u][v][1] - comparison_dict[u][v][0]
                        #weight = comparison_dict[u][v][1] / (comparison_dict[u][v][0] + 0.00001)
                        primary_changes.append(weight)
                    except:
                        continue

            # Finds secondary neighbors of each selected node, and samples to the number of samples
            secondary_changes = []
            for node in primary_neighbors:
                seconds = neighbor_group(graph = PPI, nodes = node, LCC = LCC)
                for second in seconds:
                    try:
                        weight = comparison_dict[node][second][1] - comparison_dict[node][second][0]
                        #weight = comparison_dict[u][v][1] / (comparison_dict[u][v][0] + 0.00001)
                        secondary_changes.append(weight)
                    except:
                        continue
            
            random_genes = random.sample([i for i in PPI.nodes() if i not in LCC.nodes()], len(disease_genes))
            random_changes = []
            for gene in random_genes:
                for gene_2 in random_genes:
                    if gene != gene_2:
                        try:
                            weight = comparison_dict[gene][gene_2][1] - comparison_dict[gene][gene_2][0]
                            #weight = comparison_dict[u][v][1] / (comparison_dict[u][v][0] + 0.00001)
                            random_changes.append(weight)
                        except:
                            continue
            
            # Extracts attention comparison
            print(f'Average direct LCC edge change: {np.mean(direct_edges)}')
            print(f'Average LCC change: {np.mean(LCC_changes)}')
            print(f'Average primary change: {np.mean(primary_changes)}')
            print(f'Average secondary change: {np.mean(secondary_changes)}')
            print(f'Average random change: {np.mean(random_changes)}')
            means = [
                np.mean(direct_edges),
                np.mean(LCC_changes),
                np.mean(primary_changes),
                np.mean(secondary_changes),
                np.mean(random_changes)
            ]

            # Create bar plot
            plt.figure(figsize=(10, 6))
            # Create bar plot with log scale

            plt.yscale('log')  # Log scale for y-axis
            plt.ylim([1e-4, max(means) * 10])  # Setting y-axis limits
            plt.bar(
                ('Direct \n Changes', 'LCC \n Changes', 
                 'Primary \n Changes', 'Secondary \n Changes', 'Random \n Changes'),
                means,
                color=('purple','blue', 'green', 'orange', 'red'),
                label=('Direct Changes', 'LCC Changes', 'Primary Changes', 'Secondary Changes', 'Random Changes')
            )
            plt.ylabel('Attention Weight Change from Non-Perturbed to Perturbed')
            plt.title('Distribution of Changes Across Different Categories')
            plt.savefig(f'perturb_analysis_{keyword}.png')

        # If threshold is none, evaluates proportion of thresholded results within PPI 
        if threshold != None:
            top_fold_changes = [i for i in fold_changes if i[2] > threshold]
            top_disease = [i for i in top_fold_changes if i[1] in disease_genes or i[0] in disease_genes]
            print(f'Top FC proportion in disease genes: {len(top_disease) / len(top_fold_changes)}')

            bottom_fold_changes = [i for i in fold_changes if i[2] < 1]
            bottom_disease = [i for i in bottom_fold_changes if i[1] in disease_genes or i[0] in disease_genes]
            print(f'Bottom FC proportion in disease genes: {len(bottom_disease) / len(bottom_fold_changes)}')

        # Compares LCC weight distributions between base and compare
        if LCC_compare == True:

            # Obtains base and comparison attentions 
            base_LCC = compare_LCCs(attentions = base, return_LCCs = True)
            compare_LCC = compare_LCCs(attentions = compare, return_LCCs = True)
            
            directed_area, directionality = {}, {}

            # Obtains CDF distances for a range of LCCs 
            LCCs = ['covid','cardiomyopathy hypertrophic', 'cardiomyopathy dilated', 
                    'adenocarcinoma', 'small cell lung carcinoma', 'heart failure', 'dementia', 
                    'arthritis rheumatoid', 'anemia', 'calcinosis', 'parkinsonian disorders']
            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'yellow', 'black',]
            color_assignments = {}
            for num, LCC_str in enumerate(LCCs):
                area, direction = compare_cdfs(base_LCC, compare_LCC, type = 'EMD', LCC_str = LCC_str)
                directed_area[LCC_str], directionality[LCC_str] = area, direction
                color_assignments[LCC_str] = colors[num]
            for key, value in directed_area.items():
                if directionality[key] != 'positive':
                    directed_area[key] = value * -1

            # Saves data
            if save == True:
                directed_df = pd.DataFrame.from_dict(directed_area, orient = "index")
                directed_df.to_csv(f'LCC_distance_{keyword}.csv')

            
            # Plots distance comparison
            directed_area = sorted(directed_area.items(), key=lambda x: x[1],)
            plt.figure()
            keys = [k for k, _ in directed_area]
            values = [v for _, v in directed_area]
            plt.bar(keys, values, color='blue')
            plt.xticks(keys, rotation=45, ha='right')
            plt.legend().remove()
            plt.xlabel('Diseases')
            plt.ylabel('Directed Area Value')
            plt.title('Directed Area Comparison')
            plt.tight_layout()  
            plt.savefig(f'LCC_distance_{keyword}.png')
        
        # Extract comparison weights for edges within LCC
        LCC_comparison_weights = {}
        for u, v in LCC.edges():
            if u in comparison_dict and v in comparison_dict[u]:
                LCC_comparison_weights[(u, v)] = comparison_dict[u][v]
            elif v in comparison_dict and u in comparison_dict[v]:
                LCC_comparison_weights[(v, u)] = comparison_dict[v][u]

        # Sort edges by comparison weights and print the bottom 5
        bottom_5_LCC_edges = sorted([(edge, weight) for edge, weight in LCC_comparison_weights.items() if edge[1] != edge[0]], key=lambda x: x[1])[:5]
        print("Bottom 5 LCC edges by comparison weight:")
        for edge, weight in bottom_5_LCC_edges:
            print(f"Edge: {edge}, Weight: {weight}")

        # If enabled, finds intermediaries
        if intermediaries == True:
            intermediary, direct = find_intermediaries(comparison_dict = fold_changes, 
                                                       PPI = PPI, LCC = LCC, 
                                                       disease_genes = disease_genes, 
                                                       top_pairs = 5, keyword = keyword)
                
# Function for obtaining user-input arguments
def get_arguments():
    parser = argparse.ArgumentParser()

    # Type of job
    parser.add_argument('-c', '--class', dest = 'chosen_class', type = str, help = 'Class to use', default = 'perturber')

    # Instance and total number of jobs (if applicable)
    parser.add_argument('-i', '--instance', dest = 'instance', type = int, help = 'Instance', default = None)
    parser.add_argument('-t', '--total_jobs', dest = 'total_jobs', type = int, help = 'Total jobs', default = None)
    args = parser.parse_args()
    
    return args


# Main Function Runtime
if __name__ == '__main__':
    args = get_arguments()

    # Attempts to map PPI directionality
    if args.chosen_class == 'direction':
        new_attention = PPI_attention(mean = False) 
        new_attention.scrape_attentions(disease = 'Cardiomyopathy Hypertrophic', samples = 9)
        #new_attention.scrape_attentions(samples = 100, )
        #new_attention.merge_attentions(limit = 20)
        new_attention.map_direction( comparison = 'default')
    

    elif args.chosen_class.lower() == 'compare':
        new_attention = PPI_attention(mean = False)

        # COVID
        '''
        new_attention.map_attention_changes(samples = 100, disease = 'covid', 
                                           keyword = 'arthritis_combined_mean',
                                          comparison_type = 'covid_combined', layer_index = -1)
        '''

        # Cardiomyopathy Hypertrophic
        new_attention.map_attention_changes(samples =100, disease = 'cardiomyopathy hypertrophic', 
                                           keyword = 'hcm_max',
                                         comparison_type = 'cardiomyopathy', layer_index = 4)
        
        # Cardiomyopathy Dilated
        '''
        new_attention.map_attention_changes(samples = 100, disease = 'cardiomyopathy dilated', 
                                           keyword = 'dcm_max',
                                          comparison_type = 'cardiomyopathy', layer_index = -1)
        '''

        # Adenocarcinoma
        '''
        new_attention.map_attention_changes(samples = 250, disease = 'adenocarcinoma', keyword = 'carcinoma',
                                           comparison_type = 'carcinoma', layer_index = -1)
        
        '''

        # Arthritis
        '''
        new_attention.map_attention_changes(samples = 400, disease = 'arthritis rheumatoid', keyword = 'rheumatoid_max', layer_index = 4,
                                           comparison_type = 'arthritis')
        '''
        
    # Routine for coralling attention weights into artificial LCCs
    elif args.chosen_class.lower() == 'partition':
        new_attention = PPI_attention(mean = True) 
        new_attention.merge_attentions(limit = 1, normalize = False, 
                                       attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Mean_attentions'))
        new_attention.save()
        new_attention.create_partition()

    # Routine for merging attention dictionaries of attention weights
    elif args.chosen_class == 'merge':
        new_attention = PPI_attention(mean = False) 
        new_attention.merge_attentions(limit = 10, normalize = False, scale = False, attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Max_attentions'))
        new_attention.map_PPI_genes(keyword = 'merge_PPI',)

    # Routine for calculating finetuned model on a specific disease
    elif args.chosen_class == 'disease':
        if args.instance == None:

            # Cardiomyopathy Hypertrophic
            new_attention = PPI_attention(layer_index = 4, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            mean = False, dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            new_attention.scrape_attentions(samples = 300, disease = 'cardiomyopathy hypertrophic', filter_label = ('disease', 'hcm'))
            new_attention.map_disease_genes(keyword = 'finetuned_hcm_alter_max', disease = 'cardiomyopathy hypertrophic')

            # Cardiomyopathy Dilated
            '''
            new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"), mean = False)
            new_attention.scrape_attentions(samples = 400, disease = 'Cardiomyopathy Dilated', filter_label = ('disease', 'dcm'))
            new_attention.map_disease_genes(disease = 'Cardiomyopathy Dilated', keyword = 'disease_dcm',)
            '''

            # Arthritis
            new_attention = PPI_attention(layer_index = -1, mean = False, 
                                          model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/Arthritis_small"),)
            new_attention.scrape_attentions(samples = 500, disease = 'arthritis rheumatoid',)
            new_attention.map_disease_genes(disease = 'arthritis rheumatoid', keyword = 'mean_arthritis')
            new_attention.save()

            # COVID 
            '''
            new_attention = PPI_attention(mean = True, layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/COVID_model"),)
            new_attention.scrape_attentions(samples = 100, disease = 'covid',)
            new_attention.map_disease_genes(disease = 'covid', keyword = 'max_covid_1000')
            #new_attention.map_PPI_genes()
            '''

            # Adenocarcinoma
            '''
            new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/carcinoma_model"), 
                                          mean = True)
            new_attention.scrape_attentions(samples = 200, disease = 'adenocarcinoma',)
            new_attention.map_disease_genes(disease = 'adenocarcinoma', keyword = 'mean_adenocarcinoma')
            #new_attention.map_PPI_genes()
            '''

        else:
            new_attention = PPI_attention(mean = False, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                                        dataset_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            new_attention.scrape_subset(filter_label = ('disease', 'hcm'), total_jobs = args.total_jobs, instance = args.instance, disease = 'Cardiomyopathy Hypertrophic',)
    
    # Routine for calculating general attentions from pretrained model
    else:

        # Single-shot PPI attention
        if args.instance == None:
            new_attention = PPI_attention(layer_index = 4, mean = True,
                dataset_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/")) # attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Mean_attentions'))
            new_attention.scrape_attentions(samples = 100, disease = None)
            new_attention.map_PPI_genes(keyword = 'PPI_mean',)
            #new_attention.map_disease_genes(keyword = 'PPI_disease', disease = 'calcinosis', samples = 200)

        else:
            # Obtains a batch of PPI attentions
            new_attention = PPI_attention(mean = True,)
            new_attention.scrape_subset(filter_label = ('disease', 'hcm'), total_jobs = args.total_jobs, instance = args.instance, disease = 'Cardiomyopathy Hypertrophic',
                                        model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                                        dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
    
    
        
