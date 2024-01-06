from base_utils.attention_base import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GINConv, MLP
from torch.optim import Adam, lr_scheduler, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch_geometric
import scipy.interpolate as interpolate
import scipy.stats as stats
from scipy.stats import wasserstein_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
                          save_threshold = False, perturbation = None, random_perturb_num = 5):
    
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
        #F1_graph_attention(LCC = LCC, PPI = self.PPI, gene_attentions = self.gene_attentions, keyword = keyword)
        
        # Plots distributions of attention with regards to disease genes
        plot_distributions(self.gene_attentions, disease = disease, keyword = keyword)
        
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
        #F1_graph_attention(PPI = self.PPI, gene_attentions = self.gene_attentions, keyword = keyword)
        
        # Plots distribution of weights 
        plot_distributions(self.gene_attentions, disease = None, keyword = keyword)
        
        # Analyze hops
        analyze_hops(self.gene_attentions, keyword = keyword)
        
        # Analyzes top attentions
        #check_top_attentions(attention_dict = self.gene_attentions, PPI = PPI, keyword = keyword)
        
        # Analyzes multiple LCCs
        compare_LCCs(attentions = self.gene_attentions, keyword = keyword)


    # Uses attention directionality to map interaction direction
    def map_direction(self, comparison = 'median', interactions = Path('enzo/Directional.csv'), 
                      direction = 'forwards', #Path('PPI/GRN.csv')
                      median_direction = 'target'): 
        
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

                if scaled_directed_attention > scaled_backwards_attention:
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

        
    # Creates a new LCC graph partition from the top attention weights, then compares the partition the original LCC
    def create_partition(self, initial_threshold=1e-3, max_threshold=1e-1, 
                         step=1e-3, disease='Cardiomyopathy Hypertrophic',
                         static = False, method='structural', alpha=0.05):
        
        # Function for backbone extraction
        def backbone_extraction(subgraph):
            if method == 'statistical':
                p_values = {edge: binom_test(subgraph.edges[edge]['attention'], n=1000, p=0.05) for edge in subgraph.edges}
                return [edge for edge, p in p_values.items() if p < alpha]
            elif method == 'structural':
                return [edge for edge in subgraph.edges if subgraph.edges[edge]['attention'] > np.median([subgraph.edges[e]['attention'] for e in subgraph.edges])]
            else:
                raise ValueError("Invalid method. Choose 'statistical' or 'structural'.")

        PPI = instantiate_ppi()
        PPI, _, _ = map_attention_attention(PPI, self.gene_attentions, save = True)
        disease_genes = isolate_disease_genes(disease)
        disease_LCC = LCC_genes(PPI, disease_genes, subgraph = True)
        backbone_edges = backbone_extraction(disease_LCC)
        disease_LCC = disease_LCC.edge_subgraph(backbone_edges).copy()
        
        if static == True:
            community_subgraph = generate_PPI(attention_dict = self.gene_attentions, attention_threshold = initial_threshold, save = False,)
        else:
            # Precompute weights within the disease LCC
            precomputed_weights = {
                (gene, target_gene): weight 
                for gene, attentions in self.gene_attentions.items() 
                for target_gene, weight in attentions.items() 
                if gene in disease_LCC and target_gene in disease_LCC
            }

            # Modularity optimization
            best_modularity = -1
            best_threshold = initial_threshold
            best_subgraph = None

            for threshold in tqdm.tqdm(np.arange(initial_threshold, max_threshold, step), desc='Evaluating thresholds'):
                subgraph = nx.Graph()
                subgraph.add_edges_from(
                    ((gene, target_gene, {'attention': weight}) 
                    for (gene, target_gene), weight in precomputed_weights.items() 
                    if weight > threshold)
                )

                if not subgraph.nodes:
                    continue

                modularity = nx.algorithms.community.modularity(subgraph, [set(subgraph.nodes)])
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_threshold = threshold
                    best_subgraph = subgraph

            print(f'Best modularity: {best_modularity} at threshold: {best_threshold}')
            community_subgraph = best_subgraph if best_subgraph else nx.Graph()

        # Calculate properties of the new community
        num_edges = community_subgraph.number_of_edges()
        avg_attention = np.mean([attr['attention'] for _, _, attr in community_subgraph.edges(data=True)])
        print(f'New LCC edges: {num_edges}, new LCC avg. attention: {avg_attention}')

        # Compare networks
        compare_networks(PPI, community_subgraph, LCC=disease_LCC)


    # Performs GNN analysis of PPI using embedding and attention features
    def predict_PPI(self, embedding_PPI_location = 'embedded_PPI.pk', attention_PPI_location = 'attention_PPI.pk',
                    embeddings_location = 'embeddings.pk',
                    epochs = 3000, new_attentions = False,
                    save_weights = False):
        
        def generate_false_edges(graph, num_edges, degrees):
            all_nodes = list(graph.nodes())
            false_edges = set()

            # Precompute degree distribution probabilities
            total_degree = sum(degrees.values())
            probabilities = [degrees[node] / total_degree for node in all_nodes]

            # Create a set of existing edges for quick lookup
            existing_edges = set(graph.edges())

            # Using tqdm for the progress bar
            pbar = tqdm.tqdm(total=num_edges, desc="Generating false edges")
            while len(false_edges) < num_edges:
                u = np.random.choice(all_nodes, p=probabilities)
                v = np.random.choice(all_nodes, p=probabilities)

                # Check if edge is valid and does not already exist
                if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
                    false_edges.add((u, v))
                    pbar.update(1)

            pbar.close()
            return false_edges
        
        # Obtains short list of background attentions if not already loaded
        if new_attentions == True:
            try:
                attentions = self.gene_attentions
            except:
                attentions = self.merge_attentions(limit = 3, mean = self.mean)
            PPI = instantiate_ppi()
            PPI, _, _ = map_attention_attention(PPI, attentions, save = True)
        else:
            attention_PPI = pk.load(open(attention_PPI_location, 'rb'))

        # Loads embeddings and attention PPI
        embedding_PPI = pk.load(open(embedding_PPI_location, 'rb'))
        embeddings = pk.load(open(embeddings_location, 'rb'))

        # Obtains network of all embeddings and attentions
        edges_to_remove = []  # replace with your edges to remove
        for u, v in edges_to_remove:
            attention_PPI.remove_edge(u, v)

        for u, v in attention_PPI.edges():
            attention_PPI[u][v]['label'] = 1

        # Generating false edges
        degrees = dict(attention_PPI.degree())
        num_false_edges = int(len(attention_PPI.edges()) * 1)
        false_edges = generate_false_edges(attention_PPI, num_false_edges, degrees)

        for u, v in false_edges:
            attention_PPI.add_edge(u, v, label=0)

        filtered_nodes = {node: embeddings[node] for node in attention_PPI.nodes() if node in embeddings}
        attention_PPI = attention_PPI.subgraph(filtered_nodes.keys())

        node_features_tensor = torch.stack([torch.FloatTensor(embeddings[node]) for node in attention_PPI.nodes()])
        node_to_index = {node: idx for idx, node in enumerate(attention_PPI.nodes())}

        edge_index_list = []
        edge_attributes_list = []

        for u, v, attr in attention_PPI.edges(data=True):
            if u in node_to_index and v in node_to_index:
                try:
                    similarity = float(cosine_similarity(embeddings[u].reshape(1, -1), embeddings[v].reshape(1, -1))[0])
                    attention = max(self.gene_attentions[u][v], self.gene_attentions[v][u])
                    label = attr.get('label', 0)
                    edge_index_list.append((node_to_index[u], node_to_index[v]))
                    edge_attributes_list.append((attention, similarity, label))
                except KeyError:
                    continue

        class EdgeDataset(Dataset):
            def __init__(self, node_features, edge_index, edge_attr, labels):
                self.node_features = node_features
                self.edge_index = edge_index
                self.edge_attr = edge_attr
                self.labels = labels

            def __len__(self):
                return self.edge_index.size(1)

            def __getitem__(self, idx):
                node_indices = self.edge_index[:, idx]
                node_feature = torch.cat((self.node_features[node_indices[0]],
                                           self.node_features[node_indices[1]]), dim=0).squeeze()
                edge_feature = self.edge_attr[idx].clone().detach().unsqueeze(1)
                feature = torch.cat((node_feature, edge_feature), dim=1).flatten()
                
                label = self.labels[idx].clone().detach().long() 
                    
                return feature, label

        # Primary model
        class MLPClassifier(nn.Module):
            def __init__(self, input_dim, hidden_dim=256):  # Increase the hidden layer size
                super(MLPClassifier, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.bn1 = nn.BatchNorm1d(hidden_dim)  # Add batch normalization
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.bn2 = nn.BatchNorm1d(hidden_dim // 2)  # Add batch normalization
                self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
                self.bn3 = nn.BatchNorm1d(hidden_dim // 4)  # Add batch normalization
                self.fc4 = nn.Linear(hidden_dim // 4, 2)  # Additional layer
                self.dropout1 = nn.Dropout(0.3)  
                self.dropout2 = nn.Dropout(0.2)  

            def forward(self, x):
                x = x.view(x.size(0), -1)  # Reshape x to [batch_size, feature_size]
                x = F.relu(self.bn1(self.fc1(x)))
                x = self.dropout1(x)
                x = F.relu(self.bn2(self.fc2(x)))
                x = self.dropout2(x)
                x = F.relu(self.bn3(self.fc3(x)))
                x = self.fc4(x)

                return F.log_softmax(x, dim=1)

        # Training function
        def train(model, loader, optimizer, criterion):
            model.train()
            total_loss = 0
            for features, labels in loader:
                optimizer.zero_grad()
                features, labels = features.to(device), labels.to(device)
                out = model(features)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        # Evaluation function
        def evaluate(model, loader):
            model.eval()
            predictions, labels = [], []
            with torch.no_grad():
                for features, label in loader:
                    features, label = features.to(device), label.to(device)
                    out = model(features)
                    preds = out.max(1)[1]
                    predictions.extend(preds.cpu().numpy())
                    labels.extend(label.cpu().numpy())

            correct = [1 if i == labels[count] else 0 for count, i in enumerate(predictions)]
            accuracy = sum(correct) / len(correct)
            fpr, tpr, _ = roc_curve(labels, predictions)
            auc_score = auc(fpr, tpr)
            return accuracy, auc_score

        # Initialize the model and DataLoader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = 514 # Node features * 2 + edge features (excluding label)
        model = MLPClassifier(input_dim=input_dim).to(device)

        # Data splitting and DataLoader creation
        edge_index = torch.LongTensor(edge_index_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attributes_list)
        labels = edge_attr[:, -1].long()
        edge_attr = edge_attr[:, :-1]

        dataset = EdgeDataset(node_features_tensor, edge_index, edge_attr, labels)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # Optimizer and loss function
        optimizer = torch.optim.SDG(model.parameters(), lr=0.001)#  AdamW(model.parameters(), lr=0.001)
        criterion = torch.nn.NLLLoss()

        # Training and evaluation loops
        for epoch in tqdm.tqdm(range(epochs), total=epochs, desc='Training...'):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_accuracy, auc_score = evaluate(model, val_loader)
            if epoch % 10 == 0:
                print(f'Epoch {epoch} - Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, AUC: {auc_score:.4f}')

        def predict_edges(test_nodes, node_features = node_features_tensor, attention_PPI = attention_PPI):
            node_to_index = {node: idx for idx, node in enumerate(attention_PPI.nodes())}

            # Prepares features
            test_nodes = [(node1, node2) for (node1, node2) in test_nodes if node1 in node_to_index and node2 in node_to_index]
            print(f'Filtered Length: {len(test_nodes)}')
            edge_index_list = [(node_to_index[node1], node_to_index[node2]) for (node1, node2) in test_nodes]
            embeddings =  [(node_features_tensor[node1], node_features_tensor[node2]) for (node1, node2) in edge_index_list]
            similarities = [cosine_similarity(embeddings[node1].reshape(1, -1), embeddings[node2].reshape(1, -1))[0] for (node1, node2) in indexed_nodes]
            attentions = [max(self.gene_attentions[node1][node2], self.gene_attentions[node2][node1]) for (node1, node2) in indexed_nodes]
            edge_attributes_list = [(attentions[node1], similarities[node1], 1) for (node1, node2) in indexed_nodes]

            edge_index = torch.LongTensor(edge_index_list).t().contiguous()
            edge_attr = torch.FloatTensor(edge_attributes_list)
            labels = edge_attr[:, -1].long()
            edge_attr = edge_attr[:, :-1]
            test_dataset = EdgeDataset(node_features_tensor, edge_index, edge_attr, labels)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

            # Passes through model
            predictions = []
            model.eval()
            with torch.no_grad():
                for features, _ in test_loader:
                    features = features.to(device)
                    out = model(features)
                    preds = out.max(1)[1]
                    predictions.extend(preds.cpu().numpy())

            return predictions

        if save_weights:
            torch.save(cnn_edge_classifier.state_dict(), 'cnn_model_weights.pth')
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

    # Maps training from pretrained to finetuned model
    def map_attention_changes(self, base_location = Path('Max_attentions'),
                            comparison_location = Path('attentions'),
                            disease = None,
                            limit = 1, keyword = 'covid',
                            comparison_type = 'merge', intermediaries = False,
                            LCC_compare = True,
                            samples = 100, layer_index = -1):
        
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

        elif comparison_type == 'cardiomyopathy':
        
            new_attention = PPI_attention(layer_index = layer_index, mean = self.mean,
                                          )#dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            new_attention.scrape_attentions(samples = samples, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = layer_index, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                         mean = self.mean, dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            
            new_attention.scrape_attentions(samples = samples, disease = disease, filter_label = ('disease', 'hcm'))
            compare = new_attention.gene_attentions

            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)

        elif comparison_type == 'arthritis':
            
            new_attention = PPI_attention(layer_index = layer_index, mean = False, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/"))
            new_attention.scrape_attentions(samples = samples, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = layer_index, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Arthritis"),
                         mean = False) #dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/")
            
            new_attention.scrape_attentions(samples = samples, disease = disease,)# filter_label = ('disease', 'hcm'))
            compare = new_attention.gene_attentions
            
            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)


        elif comparison_type == 'carcinoma':
            
            new_attention = PPI_attention(layer_index = layer_index, mean = self.mean, 
                                          model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/"))
            new_attention.scrape_attentions(samples = samples, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = layer_index, 
                        model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/carcinoma_model"), mean = self.mean) 
            new_attention.scrape_attentions(samples = samples, disease = disease,)
            compare = new_attention.gene_attentions
            
            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)


        elif comparison_type == 'covid': 
            new_attention = PPI_attention(layer_index = layer_index, mean = self.mean, 
                                          model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo"))
            new_attention.scrape_attentions(samples = samples, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = layer_index, 
                        model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/GF-finetuned"), mean = self.mean) 
            new_attention.scrape_attentions(samples = samples, disease = disease,)
            compare = new_attention.gene_attentions
            
            results, comparison_dict, fold_changes, threshold = compare_attentions(base = base, compare = compare, 
                                                                    PPI = PPI, LCC = LCC, keyword = keyword, disease = disease_genes)

        elif comparison_type == 'perturb':

            # Perturbation comparison
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
            std = {
                "Direct Changes": np.std(direct_edges)/4,
                "LCC Changes": np.std(LCC_changes)/4,
                "Primary Changes": np.std(primary_changes)/4,
                "Secondary Changes": np.std(secondary_changes)/4,
                "Random Changes": np.std(random_changes)/4
            }

            # Create bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(
                ('Direct \n Changes', 'LCC \n Changes', 
                 'Primary \n Changes', 'Secondary \n Changes', 'Random \n Changes'),
                means,
                color=('purple','blue', 'green', 'orange', 'red'),
                yerr=list(std.values()),
                capsize=5,  
                label=('Direct Changes', 'LCC Changes', 'Primary Changes', 'Secondary Changes', 'Random Changes')
            )
            plt.ylabel('Attention Weight Change from Non-Perturbed to Perturbed')
            plt.title('Distribution of Changes Across Different Categories')
            plt.savefig(f'perturb_analysis_{keyword}.png')
            sys.exit()

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
            base_LCC = compare_LCCs(attentions = base, return_LCCs = True)
            compare_LCC = compare_LCCs(attentions = compare, return_LCCs = True)

            # Iterates through each LCC 
            def compare_cdfs(base_LCC, compare_LCC, type = 'KS'):
                # Preparing data
                base_data = np.array(sorted(base_LCC[LCC_str]))
                compare_data = np.array(sorted(compare_LCC[LCC_str]))

                if type == 'KS':
                    # Kolmogorov-Smirnov Statistic
                    ks_statistic, ks_pvalue = stats.ks_2samp(base_data, compare_data)
                    directed_area[LCC_str] = ks_statistic
                    if np.mean(base_LCC[LCC_str]) > np.mean(compare_LCC[LCC_str]):
                        directionality[LCC_str] = 'positive'
                    else:
                        directionality[LCC_str] = 'negative'

                if type == 'EMD':
                    # Earth Mover's Distance (Wasserstein distance)
                    emd = wasserstein_distance(base_data, compare_data)
                    directed_area[LCC_str] = emd
                    if np.mean(base_LCC[LCC_str]) < np.mean(compare_LCC[LCC_str]):
                        directionality[LCC_str] = 'positive'
                    else:
                        directionality[LCC_str] = 'negative'
                else:
                    # Euclidean Distance between CDFs (Your original method)
                    common_x = np.linspace(min(min(base_data), min(compare_data)), max(max(base_data), max(compare_data)), 1000)
                    base_cdf = interpolate.interp1d(base_data, np.linspace(0, 1, len(base_data)), bounds_error=False, fill_value="extrapolate")
                    compare_cdf = interpolate.interp1d(compare_data, np.linspace(0, 1, len(compare_data)), bounds_error=False, fill_value="extrapolate")
                    euclidean_distance = np.sqrt(np.sum((compare_cdf(common_x) - base_cdf(common_x)) ** 2))
                    directed_area[LCC_str] = euclidean_distance
                    if np.mean(base_LCC[LCC_str]) < np.mean(compare_LCC[LCC_str]):
                        directionality[LCC_str] = 'positive'
                    else:
                        directionality[LCC_str] = 'negative'

                return directed_area, directionality

            directed_area, directionality = compare_cdfs(base_LCC, compare_LCC, type = 'EMD')

            # Get a list of unique colors
            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'yellow', 'black']
            plt.figure(figsize=(11, 8))
            for i, (key, value) in enumerate(directed_area.items()):                          
                if directionality[key] == 'positive':
                    plt.bar(key, value, color=colors[i], label=key)
                else:
                    plt.bar(key, -value, color=colors[i], label=key)

            plt.legend()
            plt.xlabel('LCC')
            plt.ylabel('KS Score')

            # Set log scale with limit
            plt.yscale('symlog', linthresh=0.01)
            plt.ylim(bottom=-np.max(list(directed_area.values())) * 10, top=np.max(list(directed_area.values())) * 10)

            # Removing x-axis labels
            plt.xticks([])
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

    if args.chosen_class == 'direction':
        new_attention = PPI_attention(mean = False) 
        #new_attention.scrape_attentions(disease = 'Cardiomyopathy Hypertrophic', samples = 1000)
        new_attention.scrape_attentions(samples = 1000)
        #new_attention.merge_attentions(limit = 250)
        new_attention.map_direction()
        print('pretrained_median_target')

    elif args.chosen_class == 'GNN':
        new_attention = PPI_attention(mean = False) 
        new_attention.merge_attentions(limit = 1)
        new_attention.predict_PPI()

    elif args.chosen_class.lower() == 'compare':
        new_attention = PPI_attention(mean = False)
        #new_attention.map_attention_changes(samples = 400, disease = 'cardiomyopathy hypertrophic', 
        #                                   keyword = 'hcm_max_KS',
        #                                  comparison_type = 'cardiomyopathy', layer_index = -1)
        #new_attention.map_attention_changes(samples = 250, disease = 'adenocarcinoma', keyword = 'carcinoma',
        #                                   comparison_type = 'carcinoma', layer_index = -1)
        #new_attention.map_attention_changes(samples = 600, disease = 'covid', keyword = 'covid_max', layer_index = -1,
        #                                   comparison_type = 'covid')
        #new_attention.map_attention_changes(samples = 100, disease = 'adenocarcinoma', keyword = 'adenocarcinoma',
        #                                   comparison_type = 'carcinoma', layer_index = -1)
        new_attention.map_attention_changes(disease = 'cardiomyopathy hypertrophic', keyword = 'max_perturb',
                                           comparison_type = 'perturb', samples = 500, layer_index = -1)
    

    elif args.chosen_class.lower() == 'partition':
        new_attention = PPI_attention(mean = True) 
        new_attention.merge_attentions(limit = 1, normalize = False, 
                                       attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Mean_attentions'))
        new_attention.save()
        new_attention.create_partition()

    # Routine for merging attention dictionaries
    elif args.chosen_class == 'merge':
        new_attention = PPI_attention(mean = False) 
        new_attention.merge_attentions(limit = 10, normalize = False, scale = False, attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Max_attentions'))
        #new_attention.map_disease_genes(keyword = 'PPI_merge', disease = 'Cardiomyopathy Hypertrophic')
        new_attention.map_PPI_genes(keyword = 'merge_PPI',)
        #new_attention.save()
        #new_attention.gen_attention_PPI()
        new_attention.predict_PPI()

    # Routine for calculating finetuned model on a specific disease
    elif args.chosen_class == 'disease':
        if args.instance == None:
            '''
            new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            mean = False) # dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/")
            new_attention.scrape_attentions(samples = 100, disease = 'Cardiomyopathy Hypertrophic',)# filter_label = ('disease', 'hcm'))
            new_attention.map_disease_genes(keyword = 'disease_shuffled', disease = 'Cardiomyopathy Hypertrophic')
            #new_attention.map_PPI_genes(keyword = 'merge_PPI')
            '''
            '''
            new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"), mean = False)
            new_attention.scrape_attentions(samples = 400, disease = 'Cardiomyopathy Dilated', filter_label = ('disease', 'dcm'))
            new_attention.map_disease_genes(disease = 'Cardiomyopathy Dilated', keyword = 'disease_dcm',)
            '''
            '''
            new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/GF-finetuned"), mean = True)
            new_attention.scrape_attentions(samples = 100, disease = 'arthritis rheumatoid',)
            new_attention.map_disease_genes(disease = 'arthritis rheumatoid', keyword = 'max_arthritis')
            new_attention.save()
            '''
            new_attention = PPI_attention(mean = True, layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/GF-finetuned"),)
            new_attention.scrape_attentions(samples = 100, disease = 'covid',)
            new_attention.map_disease_genes(disease = 'covid', keyword = 'max_covid_1000')
            #new_attention.map_PPI_genes()
            '''
            new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/GF-finetuned"), 
                                          mean = True)
            new_attention.scrape_attentions(samples = 250, disease = 'adenocarcinoma',)
            new_attention.map_disease_genes(disease = 'adenocarcinoma', keyword = 'adenocarcinoma')
            #new_attention.map_PPI_genes()
            '''
        else:
            new_attention = PPI_attention(mean = False, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                                        dataset_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            new_attention.scrape_subset(filter_label = ('disease', 'hcm'), total_jobs = args.total_jobs, instance = args.instance, disease = 'Cardiomyopathy Hypertrophic',)
    
    # Routine for calculating general attentions from pretrained model
    else:
        if args.instance == None:
            new_attention = PPI_attention(layer_index = 4, mean = False,) # attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Mean_attentions'))
            new_attention.scrape_attentions(samples = 1000, disease = None)
            new_attention.map_PPI_genes(keyword = 'PPI_hops')
            #new_attention.map_disease_genes(disease = 'Cardiomyopathy Hypertrophic', keyword = 'min_PPI')
            '''
            new_attention = PPI_attention(layer_index = 4, mean = True, attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Mean_attentions'))
            new_attention.scrape_attentions(samples = 200, disease = None)
            new_attention.map_disease_genes(disease = 'Arthritis Rheumatoid', keyword = 'min_PPI')
            new_attention.save()
            '''
            #new_attention.gen_attention_PPI()
            #new_attention.predict_PPI()

        else:
            new_attention = PPI_attention(mean = False,)
            new_attention.scrape_subset(filter_label = ('disease', 'hcm'), total_jobs = args.total_jobs, instance = args.instance, disease = 'Cardiomyopathy Hypertrophic',
                                        model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                                        dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
    
    
        