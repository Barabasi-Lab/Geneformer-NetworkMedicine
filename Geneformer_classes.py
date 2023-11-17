from base_utils.perturbation_base import *
from base_utils.attention_base import *
import argparse
import combine_dicts

# Primary Perturber class 
class Perturber:
    def __init__(self, dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"),
                  model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                  num_cells = 5000, tokens_list = None, data_subset = None, label = 'disease', dataset_destination = 'Filtered_embdataset'):
             
        # Sets up initial model params
        self.base_condition = 'nf'
        self.num_cells = num_cells
        self.model_location = model_location
        self.dataset_location = dataset_location
        self.label = label
        self.dataset_destination = dataset_destination
        
        # Loads required dictionaries
        token_dict = pk.load(open(Path('geneformer/token_dictionary.pkl'), 'rb'))
        gene_dict = pk.load(open(Path("geneformer/gene_name_id_dict.pkl"), 'rb'))
        gene_dict = {value: key for key, value in gene_dict.items()}
        token_dict = {value: key for key, value in token_dict.items()}
    
        # Create a dictionary to map tokens to genes
        token_to_gene = {}
        for _id, _ in token_dict.items():
            try:
                ensembl_name = token_dict[_id]
                token_to_gene[_id] = gene_dict[ensembl_name]
            except:
                continue
        self.gene_to_token = {value:key for key, value in token_to_gene.items()}

        # Loads dataset
        dataset = load_from_disk(dataset_location)
        
        # Subsets data if applicable
        if data_subset != None:
            data_sub = random.sample([i for i in range(len(dataset))], int(len(dataset) * data_subset))
            dataset = dataset.select(data_sub)
        
        self.tokens_list = tokens_list 
        self.labels = tokens_list
            
        try:
            # Filters data for relevant conditions and relevant genes
            def filter_sample(data):
                return data[self.base_condition] == self.label
            
            self.dataset = dataset.filter(filter_sample)
        except:
            self.dataset = dataset
    
        
    # Method for setting up a comparison based on nodes adjacent to disease LCC and their neighbors    
    # Primarily applicable if tokens_list = None
    def create_disease_outside_test(self, disease = 'cardiomyopathy dilated', samples_per_label = 5):
        self.samples_per_label = samples_per_label
        
        # Obtains disease LCC
        PPI = instantiate_ppi()
        selected_genes = isolate_disease_genes(disease.lower())
        LCC = LCC_genes(PPI, selected_genes, subgraph=True)
        nodes = list(LCC.nodes())
               
        # Finds largest connected component of LCC and samples from LCC
        largest_component = max(nx.connected_components(LCC), key=len)
        ref_node = max(largest_component, key=LCC.degree)
        LCC_nodes = random.sample(nodes, samples_per_label)
        
        # Finds neighbors of LCC, and samples to the number of samples
        primary_neighbors = neighbor_group(graph = PPI, nodes = nodes, LCC = LCC)
        primary_neighbors = random.sample(primary_neighbors, samples_per_label)
        
        # Finds secondary neighbors of each selected node, and samples to the number of samples
        secondary_neighbors = []
        for node in primary_neighbors:
            seconds = neighbor_group(graph = PPI, nodes = node, LCC = LCC)
            secondary_neighbors += seconds
            
        secondary_neighbors = random.sample(secondary_neighbors, samples_per_label)
        random_genes = random.sample([i for i in PPI.nodes() if i not in LCC.nodes()], samples_per_label)
        
        self.labels = ['LCC'] * samples_per_label + ['Hop 1'] * samples_per_label + ['Hop 2'] * samples_per_label + ['Random'] * samples_per_label
        self.tokens_list = LCC_nodes + primary_neighbors + secondary_neighbors + random_genes
        
        self.dataset = equalize_genes(self.dataset, tokens_set = set(self.tokens_list), num_cells = self.num_cells, gene_to_token = self.gene_to_token)
        print(f'Subsampled test dataset length: {len(self.dataset)}')
    
    # Method for setting up a comparison based on a set of given genes  
    # Primarily applicable if tokens_list = None
    def create_outside_test(self, tokens_list = None, samples_per_label = 5):
        if tokens_list == None:
            tokens_list = self.tokens_list
            
        self.dataset = equalize_genes(self.dataset, tokens_set = set(tokens_list), num_cells = self.num_cells, gene_to_token = self.gene_to_token)
        print(f'Subsampled test dataset length: {len(self.dataset)}')
        
    # Runs perturbation with the model 
    def run_perturbation(self, num_labels = 3):
        dataset_dict = {
        "input_ids": [sample["input_ids"] for sample in self.dataset],
        self.label: ['nf' for _ in range(len(self.dataset))],
        "length": [len(sample["input_ids"]) for sample in self.dataset]}    
        self.dataset = HF_Dataset.from_dict(dataset_dict)
    
        # Amplifies the original dataset with sub-datasets with each gene token removed
        for token_num, token in tqdm.tqdm(enumerate(self.tokens_list), total = len(self.tokens_list), desc = 'Amplifying tokens'):
            token = self.gene_to_token[token]
            
            copied_dataset = self.dataset.map(lambda example: {
                "input_ids": [token_id for token_id in example["input_ids"] if token_id != token],
                "attention_mask": [1 if count < len(example["input_ids"]) - 1 else 0 for count in range(2048)],
                "length":len(example["input_ids"]) - 1,
                self.label: self.labels[token_num],})
                
            if token_num == 0:
                dataset = concatenate_datasets([self.dataset, copied_dataset])
            else:
                dataset = concatenate_datasets([dataset, copied_dataset])
       
        self.dataset = dataset   
        self.dataset.save_to_disk(self.dataset_destination)
        self.labels.insert(0, 'nf')
        
        # Runs embedding extraction
        embex = EmbExtractor(model_type = "CellClassifier", num_classes = num_labels, 
                            filter_data = {self.label:self.labels}, max_ncells = len(self.dataset), emb_layer = 0,
                            emb_label = [None,self.label], labels_to_plot = [self.label], forward_batch_size = 9, nproc=os.cpu_count())
    
        embs = embex.extract_embs(model_directory = self.model_location, input_data_file = self.dataset_destination, 
                              output_directory = 'embeddings', output_prefix = f"disease_embeddings")
               
        # Properly parses embeddings               
        true_labels = embex.filtered_input_data[self.label]
        self.cosine_similarities = aggregate_similarities(true_labels = true_labels, samples_per_label = self.samples_per_label, embs = embs, label = self.label)
        
        print(self.cosine_similarities)
        
    # Creates a box plot visualization of the simlilarity data
    def visualize_similarities(self, title = 'SimBoxPlot.svg'):
        plt.figure()
        labels = ['LCC', 'Hop 1', 'Hop 2', 'Random']
        #plt.boxplot(list(self.cosine_similarities.values()), labels = list(self.cosine_similarities.keys()))
        plt.boxplot([self.cosine_similarities[label] for label in labels], labels = labels)
        plt.title(f'Boxplot for Embedding Cosine Similarities to Controls for Genes with \n n Hop Distances from LCC')
        plt.ylabel('Cosine similarity to control')
        plt.savefig(title)
        
# Primary class for attention extraction
class PPI_attention:
    def __init__(self, mean = False, 
                  dataset_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/genecorpus_30M_2048.dataset"),
                  model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo"),
                  attention_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/attentions"),
                  layer_index = 4):
                  
        # Initializes model variables
        self.gene_attentions = {}
        self.mean = False
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
    def scrape_attentions(self, attention_threshold = 0, samples = 100, disease = None, organ = None, normalize = False, scale = False):
        
        if disease == None:
            # Obtains attentions
            gene_attentions = extract_attention(model_location = self.model_location,
                                                data = self.dataset_location,
                                                mean = self.mean,
                                                layer_index = self.layer_index,
                                                attention_threshold = attention_threshold,
                                                data_sample = None,
                                                normalize = normalize,
                                                scale = scale,
                                                samples = samples,
                                                organ = organ)
        else:
             PPI = instantiate_ppi()
            
             # Identifies disease genes
             disease_genes = isolate_disease_genes(disease)
             
             # Identifies LCC 
             LCC_gene_list = LCC_genes(PPI, disease_genes)
             
             disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
             
             # Obtains attentions while filtering for samples with relevant genes
             gene_attentions = extract_attention(model_location = self.model_location, 
                                                normalize = normalize,
                                                scale = scale,
                                                organ = organ,
                                                data = self.dataset_location,
                                                mean = self.mean,
                                                layer_index = self.layer_index,
                                                attention_threshold = attention_threshold,
                                                data_sample = None,
                                                samples = samples,
                                                filter_genes = list(disease_LCC.nodes()))
                                                
                         
        # Merges attentions with base attention
        combine_dictionaries = [gene_attentions, self.gene_attentions]
        self.gene_attentions = merge_dictionaries(combine_dictionaries, mean = self.mean)                                         
              
    # Scrapes a smaller subset of the total data
    def scrape_subset(self, instance = 1, total_jobs = 30_000, total_samples = 30_000_000, samples = None, disease = None, organ = None,
                        attention_threshold = 0, normalize = False, scale = False):
    
        # Dilineates subset parameters
        chunk_size = total_samples // total_jobs
        bottom_index = (instance - 1) * chunk_size
        top_index = args.instance * chunk_size
        dataset_size = top_index - bottom_index
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
                                                attention_threshold = attention_threshold,
                                                samples = sample_num,
                                                data_sample = dataset_sample,
                                                normalize = normalize,
                                                scale = scale,
                                                organ = organ)
        else:
            PPI = instantiate_ppi()
            
            # Identifies disease genes
            disease_genes = isolate_disease_genes(disease)
            
            # Identifies LCC 
            LCC_gene_list = LCC_genes(PPI, disease_genes)
            
            disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
      
            # Obtains attentions while filtering for samples with relevant genes
            gene_attentions = extract_attention(model_location = self.model_location,
                                                data = self.dataset_location,
                                                mean = self.mean,
                                                layer_index = self.layer_index,
                                                attention_threshold = attention_threshold,
                                                data_sample = None,
                                                samples = sample_num,
                                                filter_genes = list(disease_LCC.nodes()))
        
        # Saves attention to attention dictionary location
        with open(f"{self.attention_location}/{instance}.pk", 'wb') as f:
            pk.dump(gene_attentions, f)
            
    # Combines all dictionaries from location with saved attentions
    def merge_attentions(self, show = True, scale = False, normalize = True, limit = None, attention_location = None):
        if attention_location == None:
            attention_location = self.attention_location
            
        # List attention files in the specified directory
        attention_files = list(attention_location.glob('*.pk'))
        
        if limit != None:
            attention_files = random.sample(attention_files, limit)
        
        # Merge attention dictionaries efficiently 
        self.gene_attentions = merge_dictionaries(attention_files)
        
        # Normalizes aggregated dictionary if specified
        if normalize == True:
            min_value = min([min(attentions.values()) for attentions in self.gene_attentions.values() if len(attentions) > 0])
            max_value = max([max(attentions.values()) for attentions in self.gene_attentions.values() if len(attentions) > 0])
            
            minmax = max_value - min_value
            for source_gene, target_genes in tqdm.tqdm(self.gene_attentions.items(), total = len(self.gene_attentions), desc = 'Normalizing combined dict'):
                for target_gene in target_genes:
    
                    # Normalize the attention value
                    normalized_attention = (self.gene_attentions[source_gene][target_gene] - min_value) / minmax
    
                    # Update the dictionary with the normalized value
                    self.gene_attentions[source_gene][target_gene] = normalized_attention
            
        # Scales aggregated dictionary if specified
        if scale = True:
            values = []
            for _, subdict in self.gene_attentions.items():
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
            print(f'{len(self.gene_attentions)} source genes identified')
            total_pairs = sum(len(attentions) for attentions in self.gene_attentions.values())
            print(f'{total_pairs} total gene-gene attention pairs!')
    
    # Performs mapping and analysis of a certain disease LCC
    def map_disease_genes(self, disease = None):
         PPI = instantiate_ppi()
        
         # Identifies disease genes
         disease_genes = isolate_disease_genes(disease)
         
         # Identifies LCC 
         LCC_gene_list = LCC_genes(PPI, disease_genes)
         disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
                
         # Maps gene attentions to the PPI and the LCC 
         self.PPI, _, _ = map_attention_attention(PPI, self.gene_attentions)
         LCC, _, _ = map_attention_attention(disease_LCC, self.gene_attentions)
      
         # Calculates AUC predictions   
         F1_graph_attention(LCC = LCC, PPI = self.PPI, gene_attentions = self.gene_attentions)
            
         # Plots distributions of attention with regards to disease genes
         plot_distributions(self.gene_attentions, disease = disease)
         
         # Analyze hops
         analyze_hops(self.gene_attentions,)
        
         # Analyzes top attentions
         check_top_attentions(attention_dict = self.gene_attentions,)
         
    # Performs mapping and analysis of the PPI
    def map_PPI_genes(self):
        PPI = instantiate_ppi()
        
        # Maps attention weights to PPI
        self.PPI, _, _ = map_attention_attention(PPI, self.gene_attentions)
        
        # Scores PPI
        F1_graph_attention(PPI = self.PPI, gene_attentions = self.gene_attentions)
        
        # Plots distribution of weights 
        plot_distributions(self.gene_attentions, disease = None)
        
        # Analyze hops
        analyze_hops(self.gene_attentions,)
        
        # Analyzes top attentions
        check_top_attentions(attention_dict = self.gene_attentions,)
       
    # Creates a new PPI from existing attention weight mappings
    def gen_attention_PPI(self, attention_threshold = 0.005, save = False):
        
        # Creates mapping
        GF_PPI = generate_PPI(attention_dict = self.gene_attentions, attention_threshold = attention_threshold, save = False)
        
        print(f'{len(self.PPI.edges())} connections in original PPI, {len(GF_PPI.edges())} connections in GF PPI')
        
        # Performs basic comparison of shared GF PPI/PPI edges
        compare_networks(GF_PPI, self.PPI)
        
        
    # Creates a matrix for all attentions
    def create_attention_matrix(self, save = 'attentionMatrix.csv'):
        columns = list(self.gene_attentions.keys())
        
        # Initializes matrix
        data = np.empty((len(columns), len(columns)))

        # Efficiently iterate through the dictionary
        for i, (key, sub_dict) in enumerate([self.gene_attentions[col] for col in columns]):
            data[i] = [sub_dict[col] for col in columns]
      
        # Convert to a Pandas DataFrame
        gene_attentions = pd.DataFrame(data, columns=columns, index=columns)
        
        # Saves as csv
        gene_attentions.to_csv(save)
        
        
# Function for obtaining local arguments
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--class', dest = 'chosen_class', type = str, help = 'Class to use', default = 'perturber')
    parser.add_argument('-i', '--instance', dest = 'instance', type = int, help = 'Instance', default = None)
    parser.add_argument('-t', '--total_jobs', dest = 'total_jobs', type = int, help = 'Total jobs', default = None)
    args = parser.parse_args()
    
    return args
     
if __name__ == '__main__':
    # Obtains user arguments
    args = get_arguments()
    total_samples = 30_000_000
    
    # Routine for gene perturber
    if args.chosen_class == 'perturber':
        pert = Perturber(data_subset = 0.5)
        pert.create_disease_outside_test(disease = 'Cardiomyopathy Hypertrophic', samples_per_label = 5,)
        pert.run_perturbation()
        pert.visualize_similarities()
    
    # Routine for merging attention dictionaries
    elif args.chosen_class == 'merge':
        new_attention = PPI_attention()
        new_attention.merge_attentions(limit = 20, attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Max_attentions'))
        new_attention.map_PPI_genes()
        new_attention.save()
        new_attention.gen_attention_PPI()
        
    # Routine for calculating finetuned model on a specific disease
    elif args.chosen_class == 'disease':
        if args.instance == None:
            new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"), mean = True)
            new_attention.scrape_attentions(samples = 500, disease = 'Cardiomyopathy Hypertrophic')
            new_attention.map_disease_genes(disease = 'Cardiomyopathy Hypertrophic')
            new_attention.save()
            new_attention.gen_attention_PPI()
            
        else:
            new_attention = PPI_attention()
            new_attention.scrape_subset(total_jobs = args.total_jobs, instance = args.instance, disease = None)
            
    # Routine for calculating general attentions from pretrained model
    else:
        if args.instance == None:
            new_attention = PPI_attention(layer_index = 4, mean = True)
            new_attention.scrape_attentions(samples = 1000, disease = None)
            #new_attention.map_PPI_genes()
            new_attention.map_disease_genes(disease = 'Cardiomyopathy Hypertrophic')
            new_attention.save()
            new_attention.gen_attention_PPI()
        else:
            new_attention = PPI_attention(mean = True)
            new_attention.scrape_subset(total_jobs = args.total_jobs, instance = args.instance, disease = None)
    
    
        