from base_utils.attention_base import *
import combine_dicts
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import argparse
        
# Primary class for attention extraction
class PPI_attention:
    def __init__(self, mean = True, 
                  dataset_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/genecorpus_30M_2048.dataset"),
                  model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo"),
                  attention_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/attentions"),
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
    def scrape_attentions(self, samples = 100, disease = None, organ = None,
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
                                                organ = organ,
                                                save_threshold = save_threshold)
        else:
             PPI = instantiate_ppi()
            
             # Identifies disease genes
             disease_genes = isolate_disease_genes(disease)
             
             # Identifies LCC 
             LCC_gene_list = LCC_genes(PPI, disease_genes)
             
             disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
        
             # Obtains attentions while filtering for samples with relevant genes
             gene_attentions = extract_attention(model_location = self.model_location, 
                                                organ = organ,
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
              
    # Scrapes a smaller subset of the total data
    def scrape_subset(self, instance = 1, total_jobs = 30_000, total_samples = 30_000_000, samples = None, disease = None, organ = None,
                      save_threshold = False):
    
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
                                                samples = sample_num,
                                                data_sample = dataset_sample,
                                                organ = organ,
                                                save_threshold = save_threshold)
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
                                                data_sample = None,
                                                samples = sample_num,
                                                filter_genes = list(disease_LCC.nodes()),
                                                save_threshold = save_threshold)
        
        # Saves attention to attention dictionary location
        with open(f"{self.attention_location}/{instance}.pk", 'wb') as f:
            pk.dump(gene_attentions, f)
        print(f"{self.attention_location}/{instance}.pk")
        
            
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
        if scale == True:
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
    def map_disease_genes(self, disease = 'Cardiomyopathy Hypertrophic'):
         PPI = instantiate_ppi()
        
         # Identifies disease genes
         disease_genes = isolate_disease_genes(disease)
         
         # Identifies LCC 
         LCC_gene_list = LCC_genes(PPI, disease_genes)
         disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
                
         # Maps gene attentions to the PPI and the LCC 
         self.PPI, _, _ = map_attention_attention(PPI, self.gene_attentions, save = True)
         LCC, _, _ = map_attention_attention(disease_LCC, self.gene_attentions)
      
         # Calculates AUC predictions   
         F1_graph_attention(LCC = LCC, PPI = self.PPI, gene_attentions = self.gene_attentions)
            
         # Plots distributions of attention with regards to disease genes
         plot_distributions(self.gene_attentions, disease = disease)
         
         # Analyze hops
         analyze_hops(self.gene_attentions,)
        
         # Analyzes top attentions
         check_top_attentions(attention_dict = self.gene_attentions, PPI = PPI)
         
    # Performs mapping and analysis of the PPI
    def map_PPI_genes(self):
        PPI = instantiate_ppi()
        
        # Maps attention weights to PPI
        self.PPI, _, _ = map_attention_attention(PPI, self.gene_attentions, save = True)
        
        # Scores PPI
        F1_graph_attention(PPI = self.PPI, gene_attentions = self.gene_attentions)
        
        # Plots distribution of weights 
        plot_distributions(self.gene_attentions, disease = None)
        
        # Analyze hops
        analyze_hops(self.gene_attentions,)
        
        # Analyzes top attentions
        check_top_attentions(attention_dict = self.gene_attentions, PPI = PPI)
       
    # Creates a new PPI from existing attention weight mappings
    def gen_attention_PPI(self, attention_threshold = 0.005, save = False, disease = None):
        
        if disease != None:
            # Identifies disease genes
            disease_genes = isolate_disease_genes(disease)
             
            # Identifies LCC 
            LCC_gene_list = LCC_genes(self.PPI, disease_genes)
            disease_LCC = LCC_genes(self.PPI, disease_genes, subgraph = True)
            GF_PPI = generate_PPI(attention_dict = self.gene_attentions, attention_threshold = attention_threshold, save = False,)
        else:
            # Creates mapping
            GF_PPI = generate_PPI(attention_dict = self.gene_attentions, attention_threshold = attention_threshold, save = False,)
            disease_LCC = None
        
        print(f'{len(self.PPI.edges())} connections in original PPI, {len(GF_PPI.edges())} connections in GF PPI')
        
        # Performs basic comparison of shared GF PPI/PPI edges
        compare_networks(self.PPI, GF_PPI, LCC = disease_LCC)
        
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
        
    # Performs GNN analysis of PPI using embedding and attention features
    def predict_PPI(self, embedding_PPI_location = 'embedded_PPI.pk', attention_PPI_location = 'attention_PPI.pk',
                    embeddings_location = 'embeddings.pk'):
        
        # Obtains short list of background attentions
        attentions = self.merge_attentions(limit = 3)

        # Loads embeddings and attention PPI
        embedding_PPI = pk.load(open(embedding_location, 'rb'))
        attention_PPI = pk.load(open(attention_PPI_location, 'rb'))
        embeddings = pk.load(open(embeddings_location, 'rb'))
        
        # Obtains network of all embeddings and attentions
        edges_to_remove = []
        for u, v in tqdm.tqdm(attention_PPI.edges(), total = len(attention_PPI.edges()), desc = 'Sampling real values'):
            try:
                attention_PPI[u][v]['embeddings'] = embedding_PPI[u][v]
            except:
                edges_to_remove.append((u, v))
    
        # Remove edges without valid embedding mapping
        for u, v in edges_to_remove:
            attention_PPI.remove_edge(u, v)

        # Labes real edges
        for u, v in attention_PPI.edges():
            attention_PPI[u][v]['label'] = 1

        # Generate fake edges and label them as '0'
        all_nodes = list(attention_PPI.nodes())
        for _ in tqdm.tqdm(range(len(attention_PPI.edges())), desc='Adding fake edges'):
            u, v = random.sample(all_nodes, 2)
            while attention_PPI.has_edge(u, v):
                u, v = random.sample(all_nodes, 2)
            fake_attention = attentions[u][v] if (u, v) in attentions else default_attention_value
            fake_embedding = cosine_similarity([embeddings[u]], [embeddings[v]])[0][0]
            attention_PPI.add_edge(u, v, embeddings=fake_embedding, label=0)

        # Convert to PyTorch Geometric format
        edge_index = torch.tensor([list(pair) for pair in attention_PPI.edges()], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([[
            attention_PPI[u][v]['embeddings'], 
            attention_PPI[u][v]['label']] for u, v in attention_PPI.edges()], dtype=torch.float)

        data = Data(edge_index=edge_index, edge_attr=edge_attr)
        class EdgePredictorGNN(nn.Module):
            def __init__(self, node_feature_dim, edge_feature_dim):
                super(EdgePredictorGNN, self).__init__()
                self.conv1 = GCNConv(node_feature_dim, 128)
                self.conv2 = GCNConv(128, 64)
                self.fc1 = nn.Linear(64 + edge_feature_dim, 32)  # Combining edge features
                self.fc2 = nn.Linear(32, 2)  # Binary classification

            def forward(self, data):
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

                # Apply GCN layers
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)

                # Use global mean pooling
                x = torch_geometric.nn.global_mean_pool(x, data.batch)

                # Concatenate with edge features
                x = torch.cat([x, edge_attr], dim=1)

                # Fully connected layers
                x = F.relu(self.fc1(x))
                x = self.fc2(x)

                return F.log_softmax(x, dim=1)

        # Assuming node_feature_dim and edge_feature_dim are defined
        model = EdgePredictorGNN(node_feature_dim, edge_feature_dim)

        def accuracy(output, labels):
            preds = output.max(1)[1].type_as(labels)  # Get the index of the max log-probability
            correct = preds.eq(labels).double()
            correct = correct.sum()
            return correct / len(labels)

        def evaluate(model, loader):
            model.eval()
            correct = 0
            for data in loader:
                out = model(data)
                correct += accuracy(out, data.y).item()  # Assuming data.y is your label tensor
            return correct / len(loader)

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            train_acc = accuracy(out, data.y).item()
            val_acc = evaluate(model, val_loader)  # Assuming you have a DataLoader for validation data

            if epoch % 10 == 0:
                print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')


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

    if args.chosen_class == 'GNN':
        new_attention = PPI_attention(mean = True) 
        new_attention.predict_PPI()

    # Routine for merging attention dictionaries
    elif args.chosen_class == 'merge':
        new_attention = PPI_attention(mean = True) 
        new_attention.merge_attentions(limit = 10, normalize = True,)# attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Max_attentions'))
        new_attention.map_disease_genes()
        new_attention.save()
        new_attention.gen_attention_PPI()
        
    # Routine for calculating finetuned model on a specific disease
    elif args.chosen_class == 'disease':
        if args.instance == None:
            new_attention = PPI_attention(layer_index = 4, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"), mean = False)
            new_attention.scrape_attentions(samples = 100, disease = 'Cardiomyopathy Hypertrophic',)
            new_attention.map_disease_genes(disease = 'Cardiomyopathy Hypertrophic')
            #new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/GF-finetuned"), mean = True)
            #new_attention.scrape_attentions(samples = 200, disease = 'small cell lung carcinoma',)
            #new_attention.map_disease_genes(disease = 'small cell lung carcinoma')
            #new_attention.map_PPI_genes()
            new_attention.save()
            new_attention.gen_attention_PPI()
            
        else:
            new_attention = PPI_attention()
            new_attention.scrape_subset(total_jobs = args.total_jobs, instance = args.instance, disease = None)
            
    # Routine for calculating general attentions from pretrained model
    else:
        if args.instance == None:
            new_attention = PPI_attention(layer_index = 4, mean = True,)
            new_attention.scrape_attentions(samples = 1000, disease = None)
            new_attention.map_PPI_genes()
            new_attention.map_disease_genes(disease = 'Cardiomyopathy Hypertrophic')
            new_attention.save()
            new_attention.gen_attention_PPI()
        else:
            new_attention = PPI_attention(mean =True,)
            new_attention.scrape_subset(total_jobs = args.total_jobs, instance = args.instance, disease = None,)
    
    
        