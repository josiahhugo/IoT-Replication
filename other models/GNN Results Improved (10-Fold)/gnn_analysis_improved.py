'''
Improved Graph Neural Network (GNN) for IoT Malware Detection
Enhanced with class imbalance handling, focal loss, and better training strategies
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import networkx as nx
import os
from collections import defaultdict

# Check if PyTorch Geometric is available
try:
    import torch_geometric
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.nn import BatchNorm
    TORCH_GEOMETRIC_AVAILABLE = True
    print("✅ PyTorch Geometric available")
except ImportError as e:
    TORCH_GEOMETRIC_AVAILABLE = False
    print(f"❌ PyTorch Geometric import error: {e}")
except Exception as e:
    TORCH_GEOMETRIC_AVAILABLE = False
    print(f"❌ PyTorch Geometric error: {e}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
RESULTS_DIR = "GNN Results Improved (10-Fold)"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}/")

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            ce_loss = F.cross_entropy(inputs, targets, reduce=False)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduce=False)
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

def load_opcode_graphs():
    """Load the original opcode transition graphs"""
    try:
        # Try optimized graphs first (adjacency matrices)
        print("🔄 Loading optimized opcode graphs...")
        with open('opcode_graphs_optimized.pkl', 'rb') as f:
            adj_matrices = pickle.load(f)
        
        # Load labels separately
        with open('improved_cig_output.pkl', 'rb') as f:
            data = pickle.load(f)
            labels = data['labels']
        
        print(f"✅ Loaded {len(adj_matrices)} optimized adjacency matrices")
        print(f"   Matrix shape: {adj_matrices[0].shape}")
        print(f"   Labels: {len(labels)} ({np.sum(labels)} malware, {len(labels) - np.sum(labels)} benign)")
        
        return adj_matrices, labels, 'adjacency'
        
    except Exception as e:
        print(f"❌ Failed to load optimized graphs: {e}")
        try:
            print("🔄 Trying original NetworkX graphs...")
            with open('opcode_graphs.pkl', 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'graphs' in data and 'labels' in data:
                graphs = data['graphs']
                labels = data['labels']
            else:
                graphs, labels = data
            
            print(f"✅ Loaded {len(graphs)} original graphs")
            return graphs, labels, 'networkx'
        except Exception as e2:
            print(f"❌ Failed to load original graphs: {e2}")
            return None, None, None

def create_graph_dataset(graphs_data, labels, data_type='adjacency'):
    """Convert graph data to PyTorch Geometric format"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None, None
    
    graph_list = []
    valid_labels = []
    
    print(f"\n🔄 Converting {data_type} graphs to PyTorch Geometric format...")
    
    if data_type == 'adjacency':
        # Handle adjacency matrices
        for i, (adj_matrix, label) in enumerate(zip(graphs_data, labels)):
            try:
                # Skip empty or invalid matrices
                if adj_matrix is None or adj_matrix.size == 0:
                    continue
                
                adj_matrix = np.array(adj_matrix)
                n_nodes = adj_matrix.shape[0]
                
                if n_nodes == 0:
                    continue
                
                # Convert adjacency matrix to edge index and edge weights
                edge_indices = np.nonzero(adj_matrix)
                edge_weights = adj_matrix[edge_indices]
                
                if len(edge_indices[0]) == 0:
                    continue
                
                edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
                edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
                
                # Create node features based on graph properties
                # Feature 1: Node degree (sum of weights)
                node_degrees = np.sum(adj_matrix, axis=1)
                # Feature 2: In-degree (sum of incoming weights)  
                in_degrees = np.sum(adj_matrix, axis=0)
                # Feature 3: Node centrality (normalized degree)
                centrality = node_degrees / (np.max(node_degrees) + 1e-9)
                
                # Combine features
                x = torch.stack([
                    torch.tensor(node_degrees, dtype=torch.float),
                    torch.tensor(in_degrees, dtype=torch.float), 
                    torch.tensor(centrality, dtype=torch.float)
                ], dim=1)
                
                # Create PyG data object
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor(label, dtype=torch.long)
                )
                
                graph_list.append(data)
                valid_labels.append(label)
                
            except Exception as e:
                print(f"⚠️ Skipping adjacency matrix {i}: {e}")
                continue
    
    else:
        # Handle NetworkX graphs (original code)
        for i, (graph, label) in enumerate(zip(graphs_data, labels)):
            try:
                if isinstance(graph, dict):
                    # Convert dict to NetworkX graph
                    G = nx.DiGraph()
                    
                    # Add nodes
                    if 'nodes' in graph:
                        for node_id, node_data in graph['nodes'].items():
                            G.add_node(node_id, **node_data)
                    
                    # Add edges
                    if 'edges' in graph:
                        for edge_data in graph['edges']:
                            if len(edge_data) >= 2:
                                source, target = edge_data[0], edge_data[1]
                                weight = edge_data[2] if len(edge_data) > 2 else 1.0
                                G.add_edge(source, target, weight=weight)
                    
                    graph = G
                
                # Skip empty graphs
                if graph.number_of_nodes() == 0:
                    continue
                
                # Convert to PyTorch Geometric Data object
                node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
                
                # Create edge index
                edge_list = []
                edge_weights = []
                for edge in graph.edges(data=True):
                    source_idx = node_mapping[edge[0]]
                    target_idx = node_mapping[edge[1]]
                    edge_list.append([source_idx, target_idx])
                    edge_weights.append(edge[2].get('weight', 1.0))
                
                if len(edge_list) == 0:
                    continue
                
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
                
                # Create node features (simple: node degree)
                degrees = [graph.degree(node) for node in graph.nodes()]
                in_degrees = [graph.in_degree(node) for node in graph.nodes()]
                out_degrees = [graph.out_degree(node) for node in graph.nodes()]
                
                # Combine features
                x = torch.stack([
                    torch.tensor(degrees, dtype=torch.float),
                    torch.tensor(in_degrees, dtype=torch.float),
                    torch.tensor(out_degrees, dtype=torch.float)
                ], dim=1)
                
                # Create PyG data object
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor(label, dtype=torch.long)
                )
                
                graph_list.append(data)
                valid_labels.append(label)
                
            except Exception as e:
                print(f"⚠️ Skipping graph {i}: {e}")
                continue
    
    print(f"✅ Successfully converted {len(graph_list)} graphs")
    return graph_list, valid_labels

class GraphConvNet(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2, dropout=0.5):
        super(GraphConvNet, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.bn3 = BatchNorm(hidden_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Combine pooled features
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class GraphAttentionNet(nn.Module):
    """Graph Attention Network"""
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2, heads=4, dropout=0.5):
        super(GraphAttentionNet, self).__init__()
        
        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.conv3 = GATConv(hidden_dim, hidden_dim // 2, heads=1, dropout=dropout)
        self.bn3 = BatchNorm(hidden_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        # Graph attention layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Combine pooled features
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

def train_gnn_improved(model, train_loader, val_loader, num_epochs=100, patience=20, 
                      use_focal_loss=True, alpha=5, gamma=2, class_weights=None):
    """Improved training with focal loss and class weights"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None
    
    # Choose loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True)
        print(f"Using Focal Loss (α={alpha}, γ={gamma})")
    elif class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_malware_correct = 0
        train_malware_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
            
            # Track malware detection
            malware_mask = batch.y == 1
            if malware_mask.sum() > 0:
                train_malware_correct += (pred[malware_mask] == batch.y[malware_mask]).sum().item()
                train_malware_total += malware_mask.sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_malware_correct = 0
        val_malware_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
                
                # Track malware detection
                malware_mask = batch.y == 1
                if malware_mask.sum() > 0:
                    val_malware_correct += (pred[malware_mask] == batch.y[malware_mask]).sum().item()
                    val_malware_total += malware_mask.sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_malware_recall = train_malware_correct / train_malware_total if train_malware_total > 0 else 0
        val_malware_recall = val_malware_correct / val_malware_total if val_malware_total > 0 else 0
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs:3d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                  f"Train Malware Recall: {train_malware_recall:.3f} | "
                  f"Val Malware Recall: {val_malware_recall:.3f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_gnn_improved(model, test_loader, fold_num=1):
    """Enhanced evaluation with detailed metrics"""
    if not TORCH_GEOMETRIC_AVAILABLE or model is None:
        return {}
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    
    # Handle case where there might be only one class in test set
    try:
        auc = roc_auc_score(all_labels, all_probabilities[:, 1])
    except ValueError:
        auc = 0.5  # Random performance if only one class
    
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    cm = confusion_matrix(all_labels, all_predictions)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate per-class metrics
        benign_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        benign_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        malware_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        malware_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n--- Fold {fold_num} Results ---")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:")
        print(f"  TN: {tn:4d} | FP: {fp:4d}")
        print(f"  FN: {fn:4d} | TP: {tp:4d}")
        print(f"Benign  - Precision: {benign_precision:.4f}, Recall: {benign_recall:.4f}")
        print(f"Malware - Precision: {malware_precision:.4f}, Recall: {malware_recall:.4f}")
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'benign_precision': benign_precision,
            'benign_recall': benign_recall,
            'malware_precision': malware_precision,
            'malware_recall': malware_recall,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    else:
        print(f"\n--- Fold {fold_num} Results ---")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Warning: Only one class present in test set")
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }

def cross_validate_gnn_improved(graph_list, labels, n_folds=10):
    """Improved cross-validation with better class imbalance handling"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ Cannot run GNN cross-validation without PyTorch Geometric")
        return None
    
    print(f"\n=== {n_folds}-Fold Cross-Validation for Improved GNN ===")
    print("Using 10-fold cross-validation for robust statistical estimates")
    
    # Check if we have enough malware samples for reliable cross-validation
    malware_count = sum(labels)
    benign_count = len(labels) - malware_count
    print(f"Dataset: {malware_count} malware, {benign_count} benign")
    
    if malware_count < n_folds * 2:
        print(f"⚠️ Only {malware_count} malware samples, reducing to {malware_count // 2} folds")
        n_folds = max(2, malware_count // 2)
    
    # Calculate class weights
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    class_weight_dict = dict(zip(unique_labels, class_weights))
    print(f"Calculated class weights: {class_weight_dict}")
    
    # Test multiple configurations - reduce number due to time constraints
    model_configs = [
        {
            'name': 'GCN_Focal_Strong', 
            'class': GraphConvNet, 
            'loss': 'focal',
            'alpha': 10,  # Stronger focus on minority class
            'gamma': 3
        },
        {
            'name': 'GCN_Weighted_Strong', 
            'class': GraphConvNet, 
            'loss': 'weighted',
            'class_weights': class_weights
        }
    ]
    
    results = {}
    
    for config in model_configs:
        print(f"\n=== Testing {config['name']} ===")
        
        # Try stratified split, fall back to manual split if needed
        try:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(skf.split(range(len(graph_list)), labels))
        except ValueError as e:
            print(f"⚠️ Stratified split failed: {e}")
            print("Using manual split to ensure each fold has malware samples")
            
            # Manual split ensuring each fold has malware
            malware_indices = [i for i, label in enumerate(labels) if label == 1]
            benign_indices = [i for i, label in enumerate(labels) if label == 0]
            
            np.random.seed(42)
            np.random.shuffle(malware_indices)
            np.random.shuffle(benign_indices)
            
            # Create manual splits
            splits = []
            malware_per_fold = len(malware_indices) // n_folds
            benign_per_fold = len(benign_indices) // n_folds
            
            for i in range(n_folds):
                # Test indices
                test_malware = malware_indices[i*malware_per_fold:(i+1)*malware_per_fold]
                test_benign = benign_indices[i*benign_per_fold:(i+1)*benign_per_fold]
                test_idx = test_malware + test_benign
                
                # Train indices (everything else)
                train_val_idx = [j for j in range(len(graph_list)) if j not in test_idx]
                
                splits.append((train_val_idx, test_idx))
        
        fold_results = defaultdict(list)
        
        for fold, (train_val_idx, test_idx) in enumerate(splits):
            print(f"\nFold {fold + 1}/{n_folds}")
            
            # Create stratified train/val split to ensure both have malware samples
            train_val_labels = [labels[i] for i in train_val_idx]
            test_labels = [labels[i] for i in test_idx]
            
            print(f"Train+Val: {len(train_val_idx)} (Benign: {train_val_labels.count(0)}, Malware: {train_val_labels.count(1)})")
            print(f"Test: {len(test_idx)} (Benign: {test_labels.count(0)}, Malware: {test_labels.count(1)})")
            
            # Use stratified split for train/val to ensure both sets have malware
            if train_val_labels.count(1) < 2:  # Not enough malware for stratified split
                print("⚠️ Not enough malware samples for stratified train/val split, using random split")
                val_size = len(train_val_idx) // 5
                train_idx = train_val_idx[:-val_size]
                val_idx = train_val_idx[-val_size:]
            else:
                # Stratified train/val split
                inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                inner_splits = list(inner_skf.split(train_val_idx, train_val_labels))
                train_rel_idx, val_rel_idx = inner_splits[0]  # Use first split
                train_idx = [train_val_idx[i] for i in train_rel_idx]
                val_idx = [train_val_idx[i] for i in val_rel_idx]
            
            # Create data splits
            train_graphs = [graph_list[i] for i in train_idx]
            val_graphs = [graph_list[i] for i in val_idx]
            test_graphs = [graph_list[i] for i in test_idx]
            
            # Check final class distribution
            train_labels_final = [labels[i] for i in train_idx]
            val_labels_final = [labels[i] for i in val_idx]
            test_labels_final = [labels[i] for i in test_idx]
            
            print(f"Train: {len(train_graphs)} (Benign: {train_labels_final.count(0)}, Malware: {train_labels_final.count(1)})")
            print(f"Val: {len(val_graphs)} (Benign: {val_labels_final.count(0)}, Malware: {val_labels_final.count(1)})")
            print(f"Test: {len(test_graphs)} (Benign: {test_labels_final.count(0)}, Malware: {test_labels_final.count(1)})")
            
            # Skip fold if train set has no malware
            if train_labels_final.count(1) == 0:
                print("❌ Skipping fold: No malware samples in training set")
                continue
            
            # Create data loaders
            train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)  # Smaller batch size
            val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
            test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
            
            # Get input dimension from first graph
            input_dim = train_graphs[0].x.size(1) if train_graphs else 3
            
            # Create model
            if config['class'] == GraphConvNet:
                model = GraphConvNet(input_dim=input_dim, hidden_dim=128, dropout=0.3)
            else:  # GAT
                model = GraphAttentionNet(input_dim=input_dim, hidden_dim=128, heads=8, dropout=0.3)
            
            # Train model with appropriate loss
            if config['loss'] == 'focal':
                model = train_gnn_improved(
                    model, train_loader, val_loader, 
                    num_epochs=80, patience=15,
                    use_focal_loss=True, 
                    alpha=config['alpha'], 
                    gamma=config['gamma']
                )
            else:  # weighted
                model = train_gnn_improved(
                    model, train_loader, val_loader, 
                    num_epochs=80, patience=15,
                    use_focal_loss=False, 
                    class_weights=config['class_weights']
                )
            
            if model is not None:
                # Evaluate
                results_dict = evaluate_gnn_improved(model, test_loader, fold + 1)
                
                # Store results
                for metric, value in results_dict.items():
                    if metric not in ['predictions', 'labels', 'probabilities', 'confusion_matrix']:
                        fold_results[metric].append(value)
        
        # Calculate summary statistics
        config_summary = {}
        for metric, values in fold_results.items():
            config_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        results[config['name']] = config_summary
        
        # Print summary for this configuration
        print(f"\n=== {config['name']} Summary ===")
        for metric, stats in config_summary.items():
            print(f"{metric.capitalize()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return results

def plot_gnn_results_improved(results, save_path):
    """Create improved visualization of GNN results"""
    metrics = ['accuracy', 'auc', 'f1', 'malware_recall']
    metric_names = ['Accuracy', 'AUC', 'F1-Score', 'Malware Recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    model_names = list(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        means = []
        stds = []
        
        for model_name in model_names:
            if metric in results[model_name]:
                means.append(results[model_name][metric]['mean'])
                stds.append(results[model_name][metric]['std'])
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(model_names, means, yerr=stds, capsize=5, 
                     color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
        
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std/2,
                   f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight best performer
        if means:
            best_idx = np.argmax(means)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Results plot saved to: {save_path}")

def main():
    """Main function"""
    print("=== Improved GNN Analysis for IoT Malware Detection (10-Fold CV) ===")
    print("Using 10-fold cross-validation for robust statistical estimates")
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch Geometric is required but not available")
        return
    
    # Load graph data
    graphs_data, labels, data_type = load_opcode_graphs()
    if graphs_data is None:
        print("❌ Failed to load graph data")
        return
    
    print(f"Loaded {len(graphs_data)} graphs (format: {data_type})")
    print(f"Class distribution: Benign={labels.count(0)}, Malware={labels.count(1)}")
    print(f"Class imbalance ratio: {labels.count(0)/labels.count(1):.2f}:1")
    
    # Convert to PyTorch Geometric format
    graph_list, valid_labels = create_graph_dataset(graphs_data, labels, data_type)
    if graph_list is None:
        print("❌ Failed to create graph dataset")
        return
    
    print(f"Successfully processed {len(graph_list)} graphs")
    
    # Perform cross-validation
    results = cross_validate_gnn_improved(graph_list, valid_labels, n_folds=10)
    
    if results:
        # Plot results
        plot_path = os.path.join(RESULTS_DIR, 'improved_gnn_comparison.png')
        plot_gnn_results_improved(results, plot_path)
        
        # Save results
        with open(os.path.join(RESULTS_DIR, 'improved_gnn_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Find best model based on malware recall
        print(f"\n=== Model Ranking by Malware Recall ===")
        model_scores = []
        for model_name, model_results in results.items():
            if 'malware_recall' in model_results:
                malware_recall = model_results['malware_recall']['mean']
                model_scores.append((model_name, malware_recall))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, score) in enumerate(model_scores):
            print(f"{i+1}. {model_name}: {score:.4f}")
        
        if model_scores:
            best_model, best_score = model_scores[0]
            print(f"\n🏆 Best Model: {best_model} (Malware Recall: {best_score:.4f})")
            
            # Print detailed results for best model
            print(f"\n=== Detailed Results for {best_model} ===")
            best_results = results[best_model]
            for metric, stats in best_results.items():
                print(f"{metric.capitalize()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\n=== Summary ===")
    print(f"Improved GNNs with class imbalance handling (10-fold CV):")
    print(f"• Focal Loss to focus on hard examples")
    print(f"• Class weights for balanced training")
    print(f"• Enhanced architectures with attention")
    print(f"• Better training strategies (gradient clipping, learning rate scheduling)")
    print(f"• 10-fold cross-validation for robust statistical estimates")
    print(f"Results saved to: {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
