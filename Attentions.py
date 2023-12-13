from base_utils.attention_base import *
import combine_dicts
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch.utils.data import random_split
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GINConv, MLP
from torch.optim import Adam, lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader, TensorDataset

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
        #plot_distributions(self.gene_attentions, disease = disease, keyword = keyword)
        
        # Analyze hops
        #analyze_hops(self.gene_attentions, keyword = keyword)
    
        # Analyzes top attentions
        #check_top_attentions(attention_dict = self.gene_attentions, PPI = disease_LCC, keyword = keyword)

        # Analyzes multiple LCCs
        compare_LCCs(attentions = self.gene_attentions, keyword = keyword)
         
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
        
    # Creates a graph partition from the top attention weights
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

            print(community_subgraph)
            print(PPI)
            print(disease_LCC)

        # Calculate properties of the new community
        num_edges = community_subgraph.number_of_edges()
        avg_attention = np.mean([attr['attention'] for _, _, attr in community_subgraph.edges(data=True)])
        print(f'New LCC edges: {num_edges}, new LCC avg. attention: {avg_attention}')

        # Compare networks
        compare_networks(PPI, community_subgraph, LCC=disease_LCC)


    # Performs GNN analysis of PPI using embedding and attention features
    def predict_PPI(self, embedding_PPI_location = 'embedded_PPI.pk', attention_PPI_location = 'attention_PPI.pk',
                    embeddings_location = 'embeddings.pk',
                    epochs = 200, new_attentions = False,
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
        edges_to_remove = []
        for u, v in tqdm.tqdm(attention_PPI.edges(), total = len(attention_PPI.edges()), desc = 'Sampling real values'):
            try:
                attention_PPI[u][v]['embeddings'] = embedding_PPI[u][v]
            except:
                edges_to_remove.append((u, v))

        # Remove edges without valid embedding mapping
        for u, v in edges_to_remove:
            attention_PPI.remove_edge(u, v)

        # Labels real edges
        for u, v in attention_PPI.edges():
            attention_PPI[u][v]['label'] = 1
        
        # Generates fake edges
        degrees = dict(attention_PPI.degree())
        num_false_edges = len(attention_PPI.edges())
        false_edges = generate_false_edges(attention_PPI, num_false_edges, degrees)

        # Adds fake edges to graph
        for u, v in false_edges:
            attention_PPI.add_edge(u, v, label = 0)

        # Filters graph and prepares nodes
        filtered_nodes = {node: embeddings[node] for node in attention_PPI.nodes() if node in embeddings}
        attention_PPI = attention_PPI.subgraph(filtered_nodes.keys())
        scaler = StandardScaler()
        node_features_tensor = torch.stack([torch.FloatTensor(embeddings[node]) for node in attention_PPI.nodes()])
        node_features_tensor = torch.FloatTensor(scaler.fit_transform(node_features_tensor.squeeze()))

        # Create a Mapping from Node to Index
        node_to_index = {node: idx for idx, node in enumerate(attention_PPI.nodes())}

        # Prepare Edge Indices and Attributes
        edge_index_list = []
        edge_attributes_list = []
        for u, v, attr in attention_PPI.edges(data=True):
            if u in node_to_index and v in node_to_index:
                edge_index_list.append((node_to_index[u], node_to_index[v]))

                # 'attr.get' is used to handle cases where 'attention' or 'embeddings' might not exist for false edges
                attention = attr.get('attention', 0)  # Default to 0 if not present
                similarity = attr.get('embeddings', {}).get('similarity', 0)  # Default to 0 if not present
                label = attr['label']
                edge_attributes_list.append((attention, similarity, label))

        class EdgeDataset(Dataset):
            def __init__(self, node_features, edge_index, edge_attr, labels):
                self.node_features = node_features
                self.edge_index = edge_index
                self.edge_attr = edge_attr
                self.labels = labels

            def __len__(self):
                return len(self.edge_index)

            def __getitem__(self, idx):
                node_index = self.edge_index[idx]
                node_feature = torch.cat((self.node_features[node_index[0]], self.node_features[node_index[1]]), dim=0)
                edge_feature = self.edge_attr[idx]
                feature = torch.cat((node_feature, edge_feature), dim=0)
                label = self.labels[idx]
                return feature, label

        def train(model, loader, optimizer, criterion, model_type='gnn'):
            model.train()
            total_loss = 0
            for edge_features, labels in loader:
                optimizer.zero_grad()
                if model_type == 'gnn':
                    out = model(data.x.view(-1, 256).to(device), data.edge_index.to(device), data.edge_attr.to(device))
                    loss = criterion(out, data.y.to(device))
                else:
                    edge_features, labels = edge_features.to(device), labels.to(device)
                    out = model(edge_features)
                    loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        def evaluate(model, loader, model_type='gnn',):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for edge_features, labels in loader:
                    if model_type == 'gnn':
                        out = model(data.x.view(-1, 256).to(device), data.edge_index.to(device), data.edge_attr.to(device))
                        preds = out.max(1)[1].cpu()
                        correct += preds.eq(data.y).sum().item()
                        total += data.y.size(0)
                    else:
                        edge_features, labels = edge_features.to(device), labels.to(device)
                        out = model(edge_features)
                        preds = out.max(1)[1].cpu()
                        correct += preds.eq(labels).sum().item()
                        total += labels.size(0)
            accuracy = correct / total
         
            return accuracy
                        
        class EdgePredictorGNN(nn.Module):
            def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64):
                super(EdgePredictorGNN, self).__init__()

                # GCN layer to capture local structure
                self.gcn1 = GCNConv(node_feature_dim, hidden_dim * 2)

                # GAT layer to focus on relevant parts
                self.gat1 = GATConv(hidden_dim * 2, hidden_dim, heads=4, concat=True, dropout=0.5)

                # GIN layer for complex pattern recognition
                mlp = MLP([hidden_dim * 4, hidden_dim, hidden_dim], dropout=0.5)
                self.gin1 = GINConv(mlp)

                # Edge attribute processing
                self.edge_attr_fc = nn.Linear(edge_feature_dim, hidden_dim)

                # Fully connected layers for edge classification
                self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, 2)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x, edge_index, edge_attr):
                # Process node features through each layer
                x = F.relu(self.gcn1(x, edge_index))
                x = F.relu(self.gat1(x, edge_index))
                x = F.relu(self.gin1(x, edge_index))

                # Process edge attributes
                edge_attr_processed = F.relu(self.edge_attr_fc(edge_attr))
                # Extract features of nodes at both ends of each edge
                row, col = edge_index
                x_row = x[row]
                x_col = x[col]
                # Concatenate the features of both nodes and the processed edge attributes
                edge_features = torch.cat([x_row, x_col, edge_attr_processed], dim=1)
                # Pass through fully connected layers
                x = self.dropout(self.fc1(edge_features))
                x = self.fc2(x)

                return F.log_softmax(x, dim=1)
        
        class CNNEdgeClassifier(nn.Module):
            def __init__(self, input_features, hidden_dim=32):
                self.hidden_dim = hidden_dim
                super(CNNEdgeClassifier, self).__init__()
                self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1)

                # Update the input size of the linear layer
                self.fc1 = nn.Linear(hidden_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, 2)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                # x should be of shape (batch_size, num_features)
                x = x.unsqueeze(1)  # Add channel dimension
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                
                # Flatten the tensor for the fully connected layers
                x = x.view(x.size(0), -1)
                x = self.dropout(self.fc1(x))
                x = self.fc2(x)

                return torch.log_softmax(x, dim=1)
                            
        class SimpleEdgeClassifier(nn.Module):
            def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64):
                super(SimpleEdgeClassifier, self).__init__()
                self.edge_feature_dim = edge_feature_dim
                self.node_feature_dim = node_feature_dim
                
                self.fc1 = nn.Linear(edge_feature_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, 2)
                self.dropout = nn.Dropout(0.5)
               
            def forward(self, x):
                # Pass through fully connected layers
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)

                return torch.log_softmax(x, dim=1)
    
        edge_index = torch.LongTensor(edge_index_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attributes_list)
        edge_attr[:, :2] = torch.FloatTensor(scaler.fit_transform(edge_attr[:, :2]))  # Scale only the features, not labels

        # Create PyTorch Geometric Data Object
        data = Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_attr[:,:2])
        data.y = edge_attr[:,2].long()
        print(list(edge_attr[:, 2]).count(0))

        # Data splitting
        transform = RandomLinkSplit(is_undirected=False)
        train_data, val_data, test_data = transform(data)

        # Creates non-GNN datasets
        dataset = EdgeDataset(node_features_tensor, edge_index, edge_attr, labels)

        # Data splitting and DataLoader creation (example split ratios)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # Model Initialization
        node_feature_dim = 256 
        edge_feature_dim = train_data.edge_attr.size(1) 
    
        # Instantiates model
        model = EdgePredictorGNN(node_feature_dim, edge_feature_dim).to(device)
        cnn_edge_classifier = CNNEdgeClassifier(input_features=train_data.edge_attr.size(1)).to(device)
        cnn_edge_optimizer = Adam(cnn_edge_classifier.parameters(), lr=0.001)
        simple_edge_classifier = SimpleEdgeClassifier(node_feature_dim, edge_feature_dim).to(device)

        # Optimizer loss definition
        simple_edge_optimizer = Adam(simple_edge_classifier.parameters(), lr=0.001)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.NLLLoss()
        
        # Training and Evaluation Loops
        model_types = {'gnn': model, 'cnn': cnn_edge_classifier, 'simple': simple_edge_classifier}
        optimizers = {'gnn': optimizer, 'cnn': cnn_edge_optimizer, 'simple': simple_edge_optimizer}
        accuracies = {key: [] for key in model_types.keys()}

        # Training and Evaluation Loops
        for epoch in tqdm.tqdm(range(epochs), total=epochs, desc='Training...'):
            for model_type in model_types.keys():
                # Choosing the appropriate DataLoader
                if model_type in ['cnn', 'simple']:
                    train_loader_current = train_loader
                    val_loader_current = val_loader
                else: # For GNN model, we use the entire graph
                    train_loader_current = train_data
                    val_loader_current = val_data

                # Training
                train_loss = train(model_types[model_type], train_loader_current, optimizers[model_type], criterion, model_type)

                # Validation
                val_accuracy = evaluate(model_types[model_type], val_loader_current, model_type)
                accuracies[model_type].append(val_accuracy)

                if epoch % 10 == 0:
                    print(f'{model_type.capitalize()} Model Accuracy: {val_accuracy:.4f} | Loss: {train_loss:.4f}')

        true_labels, pred_probs = evaluate(model, test_data, probs = True)

        # Calculate ROC
        fpr, tpr, _ = roc_curve(true_labels, pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        print(f'ROC AUC: {roc_auc}')

        # Plots accuracy and epoch statistics
        plt.figure(figsize=(10, 10))
        plt.plot(range(len(cnn_comparison_accuracy)), cnn_comparison_accuracy, label=f'CNN Model \n Accuracy: {cnn_comparison_acc:.4f}')
        plt.plot(range(len(main_accuracy)), main_accuracy, label=f'GNN Model \n Accuracy: {acc:.4f}') 
        plt.plot(range(len(ffn_comparison_accuracy)), ffn_comparison_accuracy, label=f'FFN Model \n Accuracy: {comparison_acc:.4f}')    
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower right")
        plt.savefig('Accuracy_curve.svg')

        if save_weights:
            torch.save(cnn_edge_classifier.state_dict(), 'cnn_model_weights.pth')
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

    # Maps training from pretrained to finetuned model
    def map_attention_changes(self, base_location = Path('Max_attentions'),
                            comparison_location = Path('attentions'),
                            disease = 'arthritis rheumatoid',
                            limit = 5, keyword = 'compare_sampled',
                            merge = False, find_intermediaries = True):
        
        # Obtains disease LCC
        PPI = instantiate_ppi()
        disease_genes = isolate_disease_genes(disease)

        LCC = LCC_genes(PPI, disease_genes, subgraph = True)
        edges = [(u, v) for u, v in LCC.edges()]

        # Obtain merged attention dictionary for both conditions   
        if merge == True: 
            base = self.merge_attentions(attention_location = base_location, limit = limit, save = False)
            compare = self.merge_attentions(attention_location = comparison_location, limit = limit, save = False)
        else:
            '''
            new_attention = PPI_attention(layer_index = 4, mean = False, attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Mean_attentions'))
            new_attention.scrape_attentions(samples = 100, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = 4, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                         mean = False) #dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/")
            
            new_attention.scrape_attentions(samples = 100, disease = 'Cardiomyopathy Hypertrophic',)# filter_label = ('disease', 'hcm'))
            compare = new_attention.gene_attentions
            '''
            new_attention = PPI_attention(layer_index = -1, mean = False, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/GF-finetuned"))
            new_attention.scrape_attentions(samples = 100, disease = None)
            base = new_attention.gene_attentions
            new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/GF-finetuned"),
                         mean = False) #dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/")
            
            new_attention.scrape_attentions(samples = 100, disease = disease,)# filter_label = ('disease', 'hcm'))
            compare = new_attention.gene_attentions

        # Compares results
        results, comparison_dict, fold_changes = compare_attentions(base = base, compare = compare, PPI = PPI, LCC = LCC, keyword = keyword)
        results = [i for i in results if not isinstance(i[2], list)]

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
        if find_intermediaries == True:
            print('finding intermediaries')
            print('Intermediaries: (Inter, LCC gene, disease gene)')
            print('Direct: (LCC gene, disease gene)')

            def find_one_hop_intermediaries(PPI, LCC, disease_genes):
                intermediaries = {}
                for lcc_gene in LCC:
                    if lcc_gene not in PPI:
                        continue  # Skip if lcc_gene is not in the graph

                    for disease_gene in disease_genes:
                        if disease_gene in LCC or lcc_gene == disease_gene:
                            continue
                        if disease_gene not in PPI:
                            continue  # Skip if disease_gene is not in the graph

                        # Check if there's a one-hop intermediary
                        try:
                            lcc_neighbors = set(PPI.neighbors(lcc_gene))
                        except KeyError:
                            continue  # Skip if lcc_gene is not in the graph

                        try:
                            disease_neighbors = set(PPI.neighbors(disease_gene))
                        except KeyError:
                            continue  # Skip if disease_gene is not in the graph

                        for inter_gene in lcc_neighbors:
                            if inter_gene in disease_neighbors and inter_gene not in LCC:
                                intermediaries[(inter_gene, lcc_gene, disease_gene)] = 0

                return intermediaries

            def calculate_comparison(attn):
                if isinstance(attn, list) or isinstance(attn, tuple):
                    comparison = attn[1] - attn[0]
                else:
                    comparison = attn

                return comparison
            
            def calculate_intermediary_scores(intermediaries, comparison_dict):
                for key in list(intermediaries.keys()): 
                    inter_gene, lcc_gene, disease_gene = key

                    try:
                        score1 = max(calculate_comparison(comparison_dict.get(lcc_gene, {}).get(inter_gene, 0)), 
                                    calculate_comparison(comparison_dict.get(inter_gene, {}).get(lcc_gene, 0)))
                        
                        score2 = max(calculate_comparison(comparison_dict.get(disease_gene, {}).get(inter_gene, 0)), 
                                    calculate_comparison(comparison_dict.get(inter_gene, {}).get(disease_gene, 0)))
                    except:
                        continue

                    intermediaries[key] = (score1 + score2) / 2

                return intermediaries

            def calculate_direct_scores(LCC, disease_genes, comparison_dict):
                direct_scores = {}
                for lcc_gene in LCC:
                    for disease_gene in disease_genes:
                        if disease_gene in LCC:
                            continue
                        score = max(calculate_comparison(comparison_dict.get(lcc_gene, {}).get(disease_gene, 0)), 
                                    calculate_comparison(comparison_dict.get(disease_gene, {}).get(lcc_gene, 0)))
                        direct_scores[(lcc_gene, disease_gene)] = score
                return direct_scores

            # Assuming PPI, LCC, disease_genes, and comparison_dict are already defined
            intermediaries = find_one_hop_intermediaries(PPI, LCC, disease_genes)
            intermediary_scores = calculate_intermediary_scores(intermediaries, comparison_dict)
            direct_scores = calculate_direct_scores(LCC, disease_genes, comparison_dict)

            # Combine and rank scores and display the top
            intermediary_scores = sorted(intermediary_scores.items(), key=lambda x: x[1], reverse=True)
            for top_count in range(20):
                print(f'Intermediary {top_count+1}: {intermediary_scores[top_count]}')

            direct_scores = sorted(direct_scores.items(), key=lambda x: x[1], reverse=True)
            print('====================================================================')
            for top_count in range(20):
                print(f'Direct {top_count+1}: {direct_scores[top_count]}')
                
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

    elif args.chosen_class.lower() == 'compare':
        new_attention = PPI_attention(mean = False)
        new_attention.map_attention_changes(limit = 8)

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

    # Routine for calculating finetuned model on a specific disease
    elif args.chosen_class == 'disease':
        if args.instance == None:
            new_attention = PPI_attention(layer_index = -1, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
            mean = False) # dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/")
            new_attention.scrape_attentions(samples = 500, disease = 'Cardiomyopathy Hypertrophic',)# filter_label = ('disease', 'hcm'))
            new_attention.map_disease_genes(keyword = 'disease')#disease = 'Cardiomyopathy Hypertrophic'
            #new_attention.map_PPI_genes(keyword = 'merge_PPI')
        else:
            new_attention = PPI_attention(mean = False, model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                                        dataset_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
            new_attention.scrape_subset(filter_label = ('disease', 'hcm'), total_jobs = args.total_jobs, instance = args.instance, disease = 'Cardiomyopathy Hypertrophic',)
    
    # Routine for calculating general attentions from pretrained model
    else:
        if args.instance == None:
            new_attention = PPI_attention(layer_index = 4, mean = False,) # attention_location = Path('/work/ccnr/GeneFormer/GeneFormer_repo/Mean_attentions'))
            new_attention.scrape_attentions(samples = 500, disease = None)
            new_attention.map_disease_genes(keyword = 'PPI_hops')
            #new_attention.map_disease_genes(disease = 'Cardiomyopathy Dilated', keyword = 'min_PPI')
       

        else:
            new_attention = PPI_attention(mean = False,)
            new_attention.scrape_subset(filter_label = ('disease', 'hcm'), total_jobs = args.total_jobs, instance = args.instance, disease = 'Cardiomyopathy Hypertrophic',
                                        model_location = Path("/work/ccnr/GeneFormer/GeneFormer_repo/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"),
                                        dataset_location = Path("Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset/"))
    
    
        
