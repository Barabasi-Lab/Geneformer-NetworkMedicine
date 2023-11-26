from base_utils.perturbation_base import *
import argparse

# Primary Perturber class 
class Perturber:
    def __init__(self, dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"),
                  model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                  num_cells = 5000, tokens_list = None, data_subset = None, label = 'disease', dataset_destination = 'Filtered_embdataset', condition = 'nf'):
             
        # Sets up initial model params
        self.condition = condition
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
                return data[self.label] == self.condition
            
            self.dataset = dataset.filter(filter_sample)
        except:
            self.dataset = dataset
    
    # Method for obtaining embeddings from all possible tokens and mapping them to PPI nodes
    def map_embeddings(self, save_PPI = 'embedded_PPI.pk', layer_index = 4, save_embeddings = 'embeddings.pk'):

        # Obtains list of tokens
        tokens = list([token for _, token in self.gene_to_token.items()])
        genes = list([gene for gene, _ in self.gene_to_token.items()])
        embedding_dict = {}

        # Loads model for token classification
        model = BertForTokenClassification.from_pretrained(self.model_location,
                                                 output_hidden_states=True)

        # Loads PPI 
        PPI = instantiate_ppi()

        # Pass tokens through the model to obtain embeddings
        with torch.no_grad():
            for count, token in tqdm.tqdm(enumerate(tokens), total = len(tokens), desc = 'Embedding tokens'):
                token = torch.LongTensor([[token]])
                output = model(token)
                embedding = output.hidden_states[layer_index]
                embedding_dict[genes[count]] = embedding
        
        # Adds embeddings to PPI
        for u, v in PPI.edges:
            try:
                source_gene = genes[u]
                target_gene = genes[v]
                similarity = cosine_similarity(embedding_dict[source_gene], embedding_dict[target_gene])
                PPI[u][v]['similarity'] = similarity
            except:
                pass
    
        # Saves PPI
        pk.dump(PPI, open(save_PPI, 'wb'))

        # Saves general embeddings
        pk.dump(embedding_dict, open(save_embeddings, 'wb'))

    # Method for setting up a comparison based on nodes adjacent to disease LCC and their neighbors    
    # Primarily applicable if tokens_list = None
    def create_disease_outside_test(self, disease = 'cardiomyopathy dilated', samples_per_label = 5, equalize = True):
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
        
        # If enabled, equalizes dataset to have the same frequency of genes
        if equalize == True:
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
           
    # Creates a box plot visualization of the simlilarity data
    def visualize_similarities(self, title = 'SimBoxPlot.svg'):
        plt.figure()
        labels = ['LCC', 'Hop 1', 'Hop 2', 'Random']
        #plt.boxplot(list(self.cosine_similarities.values()), labels = list(self.cosine_similarities.keys()))
        plt.boxplot([self.cosine_similarities[label] for label in labels], labels = labels)
        plt.title(f'Boxplot for Embedding Cosine Similarities to Controls for Genes with \n n Hop Distances from LCC')
        plt.ylabel('Cosine similarity to control')
        plt.tight_layout()
        plt.savefig(title)
        print('Saved similarities!')

# Function for obtaining local arguments
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--class', dest = 'chosen_class', type = str, help = 'Class to use', default = 'perturber')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    # Obtains user arguments
    args = get_arguments()
    
    # Routine for gene perturber
    if args.chosen_class == 'perturber':
        pert = Perturber(data_subset = .1, condition = 'hcm')
        pert.create_disease_outside_test(disease = 'Cardiomyopathy Hypertrophic', samples_per_label = 30, equalize = False)
        pert.run_perturbation()
        pert.visualize_similarities()

    elif args.chosen_class == 'embeddings':
        pert = Perturber(data_subset = .05, condition = 'hcm')
        pert.map_embeddings()
        