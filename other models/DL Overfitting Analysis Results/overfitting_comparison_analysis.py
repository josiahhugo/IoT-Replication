'''
Comprehensive Overfitting Analysis for IoT Malware Detection
Compares SMOTE + Logistic Regression, Focal Loss CNN, and GNN with Focal Loss
Analyzes train vs validation performance to detect overfitting
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import networkx as nx
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Check if PyTorch Geometric is available
try:
    import torch_geometric
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm
    TORCH_GEOMETRIC_AVAILABLE = True
    print("✅ PyTorch Geometric available")
except ImportError as e:
    TORCH_GEOMETRIC_AVAILABLE = False
    print(f"❌ PyTorch Geometric import error: {e}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
RESULTS_DIR = "DL Overfitting Analysis Results"
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

class FocalLossCNN(nn.Module):
    """1D CNN with Focal Loss for malware detection"""
    def __init__(self, input_dim, num_classes=2, dropout=0.5):
        super(FocalLossCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions
        conv_output_size = input_dim // 8  # After 3 pooling layers
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        x = self.classifier(x)
        return x

class SimpleGCN(nn.Module):
    """Simple GCN for malware detection"""
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2, dropout=0.5):
        super(SimpleGCN, self).__init__()
        
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
        x = torch.cat([x_mean, x_max], dim=1)
        
        x = self.classifier(x)
        return x

def load_all_data():
    """Load all necessary data for comparison"""
    print("🔄 Loading all data formats...")
    
    # Load eigenspace embeddings for CNN and Logistic Regression
    try:
        with open('X_graph_embeddings.pkl', 'rb') as f:
            eigenspace_data = pickle.load(f)
        print(f"✅ Loaded eigenspace embeddings: {eigenspace_data.shape}")
    except Exception as e:
        print(f"❌ Failed to load eigenspace embeddings: {e}")
        eigenspace_data = None
    
    # Load adjacency matrices for GNN
    try:
        with open('opcode_graphs_optimized.pkl', 'rb') as f:
            adj_matrices = pickle.load(f)
        print(f"✅ Loaded adjacency matrices: {len(adj_matrices)} graphs")
    except Exception as e:
        print(f"❌ Failed to load adjacency matrices: {e}")
        adj_matrices = None
    
    # Load labels
    try:
        with open('improved_cig_output.pkl', 'rb') as f:
            data = pickle.load(f)
            labels = data['labels']
        print(f"✅ Loaded labels: {len(labels)} samples")
        print(f"   Class distribution: {labels.count(0)} benign, {labels.count(1)} malware")
    except Exception as e:
        print(f"❌ Failed to load labels: {e}")
        labels = None
    
    return eigenspace_data, adj_matrices, labels

def create_graph_dataset_simple(adj_matrices, labels):
    """Convert adjacency matrices to PyTorch Geometric format"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None, None
    
    graph_list = []
    valid_labels = []
    
    print("🔄 Converting adjacency matrices to PyG format...")
    
    for i, (adj_matrix, label) in enumerate(zip(adj_matrices, labels)):
        try:
            adj_matrix = np.array(adj_matrix)
            if adj_matrix.size == 0:
                continue
            
            # Convert to edge index
            edge_indices = np.nonzero(adj_matrix)
            if len(edge_indices[0]) == 0:
                continue
            
            edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
            edge_weights = adj_matrix[edge_indices]
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
            
            # Create simple node features
            node_degrees = np.sum(adj_matrix, axis=1)
            in_degrees = np.sum(adj_matrix, axis=0)
            centrality = node_degrees / (np.max(node_degrees) + 1e-9)
            
            x = torch.stack([
                torch.tensor(node_degrees, dtype=torch.float),
                torch.tensor(in_degrees, dtype=torch.float), 
                torch.tensor(centrality, dtype=torch.float)
            ], dim=1)
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(label, dtype=torch.long)
            )
            
            graph_list.append(data)
            valid_labels.append(label)
            
        except Exception as e:
            continue
    
    print(f"✅ Converted {len(graph_list)} graphs successfully")
    return graph_list, valid_labels

def train_smote_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train SMOTE + Logistic Regression with overfitting monitoring"""
    print("\n🔄 Training SMOTE + Logistic Regression...")
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"Original training set: {len(y_train)} samples")
    print(f"After SMOTE: {len(y_train_smote)} samples")
    print(f"  Benign: {(y_train_smote == 0).sum()}, Malware: {(y_train_smote == 1).sum()}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train_smote)
    
    # Evaluate on all sets
    train_pred = lr.predict(X_train_scaled)
    train_proba = lr.predict_proba(X_train_scaled)[:, 1]
    val_pred = lr.predict(X_val_scaled)
    val_proba = lr.predict_proba(X_val_scaled)[:, 1]
    test_pred = lr.predict(X_test_scaled)
    test_proba = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    results = {
        'train_accuracy': accuracy_score(y_train_smote, train_pred),
        'train_auc': roc_auc_score(y_train_smote, train_proba),
        'train_f1': f1_score(y_train_smote, train_pred, average='weighted'),
        'train_malware_recall': f1_score(y_train_smote, train_pred, pos_label=1, average=None)[1] if len(set(y_train_smote)) > 1 else 0,
        
        'val_accuracy': accuracy_score(y_val, val_pred),
        'val_auc': roc_auc_score(y_val, val_proba) if len(set(y_val)) > 1 else 0.5,
        'val_f1': f1_score(y_val, val_pred, average='weighted'),
        'val_malware_recall': f1_score(y_val, val_pred, pos_label=1, average=None)[1] if len(set(y_val)) > 1 and (val_pred == 1).any() else 0,
        
        'test_accuracy': accuracy_score(y_test, test_pred),
        'test_auc': roc_auc_score(y_test, test_proba) if len(set(y_test)) > 1 else 0.5,
        'test_f1': f1_score(y_test, test_pred, average='weighted'),
        'test_malware_recall': f1_score(y_test, test_pred, pos_label=1, average=None)[1] if len(set(y_test)) > 1 and (test_pred == 1).any() else 0,
    }
    
    return results

def train_focal_cnn_with_monitoring(X_train, y_train, X_val, y_val, X_test, y_test, num_epochs=100):
    """Train Focal Loss CNN with overfitting monitoring"""
    print("\n🔄 Training Focal Loss CNN...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create model
    input_dim = X_train.shape[1]
    model = FocalLossCNN(input_dim).to(device)
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=5, gamma=2, logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_malware_recall': [],
        'val_loss': [], 'val_acc': [], 'val_malware_recall': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training metrics
            train_pred = train_outputs.argmax(dim=1)
            train_acc = (train_pred == y_train_tensor).float().mean().item()
            train_malware_mask = y_train_tensor == 1
            train_malware_recall = (train_pred[train_malware_mask] == y_train_tensor[train_malware_mask]).float().mean().item() if train_malware_mask.sum() > 0 else 0
            
            # Validation metrics
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_pred = val_outputs.argmax(dim=1)
            val_acc = (val_pred == y_val_tensor).float().mean().item()
            val_malware_mask = y_val_tensor == 1
            val_malware_recall = (val_pred[val_malware_mask] == y_val_tensor[val_malware_mask]).float().mean().item() if val_malware_mask.sum() > 0 else 0
        
        # Store history
        history['train_loss'].append(train_loss.item())
        history['train_acc'].append(train_acc)
        history['train_malware_recall'].append(train_malware_recall)
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc)
        history['val_malware_recall'].append(val_malware_recall)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_pred = test_outputs.argmax(dim=1)
        test_proba = F.softmax(test_outputs, dim=1)[:, 1]
        
        train_proba = F.softmax(model(X_train_tensor), dim=1)[:, 1]
        val_proba = F.softmax(model(X_val_tensor), dim=1)[:, 1]
    
    # Calculate final metrics
    results = {
        'train_accuracy': history['train_acc'][-1],
        'train_malware_recall': history['train_malware_recall'][-1],
        'val_accuracy': history['val_acc'][-1], 
        'val_malware_recall': history['val_malware_recall'][-1],
        'test_accuracy': (test_pred == y_test_tensor).float().mean().item(),
        'test_malware_recall': (test_pred[y_test_tensor == 1] == y_test_tensor[y_test_tensor == 1]).float().mean().item() if (y_test_tensor == 1).sum() > 0 else 0,
        'history': history
    }
    
    return results

def train_gnn_with_monitoring(graph_list, labels, num_epochs=80):
    """Train GNN with overfitting monitoring"""
    print("\n🔄 Training GNN with Focal Loss...")
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None
    
    # Split data
    train_idx, temp_idx = train_test_split(range(len(graph_list)), test_size=0.4, 
                                         stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, 
                                       stratify=[labels[i] for i in temp_idx], random_state=42)
    
    train_graphs = [graph_list[i] for i in train_idx]
    val_graphs = [graph_list[i] for i in val_idx]
    test_graphs = [graph_list[i] for i in test_idx]
    
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"Train: {len(train_graphs)} (Malware: {train_labels.count(1)})")
    print(f"Val: {len(val_graphs)} (Malware: {val_labels.count(1)})")
    print(f"Test: {len(test_graphs)} (Malware: {test_labels.count(1)})")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
    
    # Create model
    input_dim = train_graphs[0].x.size(1)
    model = SimpleGCN(input_dim=input_dim, hidden_dim=64).to(device)
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=10, gamma=3, logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_malware_recall': [],
        'val_loss': [], 'val_acc': [], 'val_malware_recall': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
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
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
            
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
                
                malware_mask = batch.y == 1
                if malware_mask.sum() > 0:
                    val_malware_correct += (pred[malware_mask] == batch.y[malware_mask]).sum().item()
                    val_malware_total += malware_mask.sum().item()
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_malware_recall = train_malware_correct / train_malware_total if train_malware_total > 0 else 0
        val_malware_recall = val_malware_correct / val_malware_total if val_malware_total > 0 else 0
        
        # Store history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['train_malware_recall'].append(train_malware_recall)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_malware_recall'].append(val_malware_recall)
        
        # Early stopping
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Final test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    test_malware_correct = 0
    test_malware_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            
            test_correct += (pred == batch.y).sum().item()
            test_total += batch.y.size(0)
            
            malware_mask = batch.y == 1
            if malware_mask.sum() > 0:
                test_malware_correct += (pred[malware_mask] == batch.y[malware_mask]).sum().item()
                test_malware_total += malware_mask.sum().item()
    
    results = {
        'train_accuracy': history['train_acc'][-1],
        'train_malware_recall': history['train_malware_recall'][-1],
        'val_accuracy': history['val_acc'][-1],
        'val_malware_recall': history['val_malware_recall'][-1],
        'test_accuracy': test_correct / test_total,
        'test_malware_recall': test_malware_correct / test_malware_total if test_malware_total > 0 else 0,
        'history': history
    }
    
    return results

def plot_overfitting_comparison(smote_results, cnn_results, gnn_results, save_path):
    """Create comprehensive overfitting comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Accuracy Comparison
    methods = ['SMOTE+LR', 'Focal CNN', 'GNN+Focal']
    train_accs = [smote_results['train_accuracy'], cnn_results['train_accuracy'], gnn_results['train_accuracy']]
    val_accs = [smote_results['val_accuracy'], cnn_results['val_accuracy'], gnn_results['val_accuracy']]
    test_accs = [smote_results['test_accuracy'], cnn_results['test_accuracy'], gnn_results['test_accuracy']]
    
    x = np.arange(len(methods))
    width = 0.25
    
    axes[0,0].bar(x - width, train_accs, width, label='Train', alpha=0.8)
    axes[0,0].bar(x, val_accs, width, label='Validation', alpha=0.8)
    axes[0,0].bar(x + width, test_accs, width, label='Test', alpha=0.8)
    axes[0,0].set_title('Accuracy Comparison')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(methods)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Malware Recall Comparison
    train_recalls = [smote_results['train_malware_recall'], cnn_results['train_malware_recall'], gnn_results['train_malware_recall']]
    val_recalls = [smote_results['val_malware_recall'], cnn_results['val_malware_recall'], gnn_results['val_malware_recall']]
    test_recalls = [smote_results['test_malware_recall'], cnn_results['test_malware_recall'], gnn_results['test_malware_recall']]
    
    axes[0,1].bar(x - width, train_recalls, width, label='Train', alpha=0.8)
    axes[0,1].bar(x, val_recalls, width, label='Validation', alpha=0.8)
    axes[0,1].bar(x + width, test_recalls, width, label='Test', alpha=0.8)
    axes[0,1].set_title('Malware Recall Comparison')
    axes[0,1].set_ylabel('Malware Recall')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(methods)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Overfitting Gap (Train - Val Accuracy)
    overfitting_gaps = [
        train_accs[0] - val_accs[0],
        train_accs[1] - val_accs[1], 
        train_accs[2] - val_accs[2]
    ]
    
    colors = ['green' if gap < 0.05 else 'orange' if gap < 0.15 else 'red' for gap in overfitting_gaps]
    bars = axes[0,2].bar(methods, overfitting_gaps, color=colors, alpha=0.7)
    axes[0,2].set_title('Overfitting Gap (Train - Val Accuracy)')
    axes[0,2].set_ylabel('Accuracy Gap')
    axes[0,2].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Mild Overfitting')
    axes[0,2].axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Severe Overfitting')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, gap in zip(bars, overfitting_gaps):
        axes[0,2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                      f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4-6: Training curves for models with history
    if cnn_results.get('history'):
        epochs = range(len(cnn_results['history']['train_acc']))
        axes[1,0].plot(epochs, cnn_results['history']['train_acc'], label='Train Acc', linewidth=2)
        axes[1,0].plot(epochs, cnn_results['history']['val_acc'], label='Val Acc', linewidth=2)
        axes[1,0].set_title('CNN Training Curves - Accuracy')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    if cnn_results.get('history'):
        axes[1,1].plot(epochs, cnn_results['history']['train_malware_recall'], label='Train Malware Recall', linewidth=2)
        axes[1,1].plot(epochs, cnn_results['history']['val_malware_recall'], label='Val Malware Recall', linewidth=2)
        axes[1,1].set_title('CNN Training Curves - Malware Recall')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Malware Recall')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    if gnn_results.get('history'):
        gnn_epochs = range(len(gnn_results['history']['train_acc']))
        axes[1,2].plot(gnn_epochs, gnn_results['history']['train_acc'], label='Train Acc', linewidth=2, color='green')
        axes[1,2].plot(gnn_epochs, gnn_results['history']['val_acc'], label='Val Acc', linewidth=2, color='red')
        axes[1,2].set_title('GNN Training Curves - Accuracy')
        axes[1,2].set_xlabel('Epoch')
        axes[1,2].set_ylabel('Accuracy')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Overfitting comparison plot saved to: {save_path}")

def create_overfitting_report(smote_results, cnn_results, gnn_results, save_path):
    """Create detailed overfitting analysis report"""
    
    report = """
# Overfitting Analysis Report
## IoT Malware Detection - Method Comparison

### Executive Summary
This analysis compares three approaches for IoT malware detection to assess overfitting:
1. SMOTE + Logistic Regression
2. Focal Loss CNN 
3. GNN with Focal Loss

### Overfitting Indicators

**Accuracy Gap (Train - Validation):**
- SMOTE + Logistic Regression: {:.3f}
- Focal Loss CNN: {:.3f}
- GNN with Focal Loss: {:.3f}

**Interpretation:**
- Gap < 0.05: Minimal overfitting ✅
- Gap 0.05-0.15: Moderate overfitting ⚠️
- Gap > 0.15: Severe overfitting ❌

### Detailed Results

#### SMOTE + Logistic Regression
- Train Accuracy: {:.3f}
- Validation Accuracy: {:.3f}
- Test Accuracy: {:.3f}
- Train Malware Recall: {:.3f}
- Validation Malware Recall: {:.3f}
- Test Malware Recall: {:.3f}
- **Overfitting Assessment:** {}

#### Focal Loss CNN
- Train Accuracy: {:.3f}
- Validation Accuracy: {:.3f}
- Test Accuracy: {:.3f}
- Train Malware Recall: {:.3f}
- Validation Malware Recall: {:.3f}
- Test Malware Recall: {:.3f}
- **Overfitting Assessment:** {}

#### GNN with Focal Loss
- Train Accuracy: {:.3f}
- Validation Accuracy: {:.3f}
- Test Accuracy: {:.3f}
- Train Malware Recall: {:.3f}
- Validation Malware Recall: {:.3f}
- Test Malware Recall: {:.3f}
- **Overfitting Assessment:** {}

### Key Observations

1. **Generalization Gap:** The difference between training and validation performance indicates how well each model generalizes.

2. **Malware Detection Capability:** How effectively each method detects the minority malware class.

3. **Consistency:** Whether performance is consistent across train/validation/test splits.

### Conclusions

{}

### Recommendations

{}
""".format(
        # Accuracy gaps
        smote_results['train_accuracy'] - smote_results['val_accuracy'],
        cnn_results['train_accuracy'] - cnn_results['val_accuracy'],
        gnn_results['train_accuracy'] - gnn_results['val_accuracy'],
        
        # SMOTE results
        smote_results['train_accuracy'], smote_results['val_accuracy'], smote_results['test_accuracy'],
        smote_results['train_malware_recall'], smote_results['val_malware_recall'], smote_results['test_malware_recall'],
        get_overfitting_assessment(smote_results['train_accuracy'] - smote_results['val_accuracy']),
        
        # CNN results
        cnn_results['train_accuracy'], cnn_results['val_accuracy'], cnn_results['test_accuracy'],
        cnn_results['train_malware_recall'], cnn_results['val_malware_recall'], cnn_results['test_malware_recall'],
        get_overfitting_assessment(cnn_results['train_accuracy'] - cnn_results['val_accuracy']),
        
        # GNN results
        gnn_results['train_accuracy'], gnn_results['val_accuracy'], gnn_results['test_accuracy'],
        gnn_results['train_malware_recall'], gnn_results['val_malware_recall'], gnn_results['test_malware_recall'],
        get_overfitting_assessment(gnn_results['train_accuracy'] - gnn_results['val_accuracy']),
        
        # Conclusions and recommendations
        generate_conclusions(smote_results, cnn_results, gnn_results),
        generate_recommendations(smote_results, cnn_results, gnn_results)
    )
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Overfitting analysis report saved to: {save_path}")

def get_overfitting_assessment(gap):
    """Get overfitting assessment based on accuracy gap"""
    if gap < 0.05:
        return "Minimal overfitting ✅"
    elif gap < 0.15:
        return "Moderate overfitting ⚠️"
    else:
        return "Severe overfitting ❌"

def generate_conclusions(smote_results, cnn_results, gnn_results):
    """Generate conclusions based on results"""
    gaps = [
        smote_results['train_accuracy'] - smote_results['val_accuracy'],
        cnn_results['train_accuracy'] - cnn_results['val_accuracy'],
        gnn_results['train_accuracy'] - gnn_results['val_accuracy']
    ]
    
    best_gap_idx = np.argmin(gaps)
    methods = ['SMOTE+LR', 'Focal CNN', 'GNN+Focal']
    
    conclusions = f"""
Based on the overfitting analysis:

- **Best Generalization:** {methods[best_gap_idx]} shows the smallest train-validation gap ({gaps[best_gap_idx]:.3f})
- **Malware Detection:** GNN achieves {gnn_results['test_malware_recall']:.1%} test malware recall vs CNN's {cnn_results['test_malware_recall']:.1%} and SMOTE+LR's {smote_results['test_malware_recall']:.1%}
- **Overall Performance:** {'GNN shows excellent performance but needs overfitting verification' if gaps[2] > 0.1 else 'GNN demonstrates both high performance and good generalization'}
"""
    
    return conclusions

def generate_recommendations(smote_results, cnn_results, gnn_results):
    """Generate recommendations based on results"""
    gnn_gap = gnn_results['train_accuracy'] - gnn_results['val_accuracy']
    
    if gnn_gap > 0.15:
        return """
1. **Address GNN Overfitting:** Increase dropout, add more regularization, or reduce model complexity
2. **Cross-Validation:** Use more rigorous k-fold CV to validate GNN performance
3. **Data Augmentation:** Consider graph augmentation techniques to increase training diversity
4. **Early Stopping:** Implement more aggressive early stopping based on validation performance
5. **Ensemble Methods:** Combine multiple approaches to balance performance and generalization
"""
    else:
        return """
1. **GNN appears robust:** Low overfitting gap suggests good generalization
2. **Production Deployment:** GNN shows promise for real-world deployment
3. **Further Validation:** Test on completely independent datasets
4. **Interpretability:** Add explainability features to understand GNN decisions
5. **Monitoring:** Implement performance monitoring in production to detect model drift
"""

def main():
    """Main analysis function"""
    print("=== Comprehensive Overfitting Analysis ===")
    
    # Load all data
    eigenspace_data, adj_matrices, labels = load_all_data()
    
    if eigenspace_data is None or adj_matrices is None or labels is None:
        print("❌ Could not load required data")
        return
    
    # Convert labels to numpy array
    labels = np.array(labels)
    
    # Split eigenspace data for traditional ML methods
    X_train, X_temp, y_train, y_temp = train_test_split(
        eigenspace_data, labels, test_size=0.4, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    print(f"\nData splits:")
    print(f"Train: {len(y_train)} (Malware: {(y_train == 1).sum()})")
    print(f"Val: {len(y_val)} (Malware: {(y_val == 1).sum()})")
    print(f"Test: {len(y_test)} (Malware: {(y_test == 1).sum()})")
    
    # Run all comparisons
    print("\n" + "="*60)
    
    # 1. SMOTE + Logistic Regression
    smote_results = train_smote_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # 2. Focal Loss CNN
    cnn_results = train_focal_cnn_with_monitoring(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # 3. GNN with Focal Loss
    graph_list, graph_labels = create_graph_dataset_simple(adj_matrices, labels.tolist())
    if graph_list:
        gnn_results = train_gnn_with_monitoring(graph_list, graph_labels)
    else:
        print("❌ Could not create graph dataset for GNN")
        return
    
    # Create visualizations and report
    print("\n" + "="*60)
    print("📊 Creating analysis reports...")
    
    # Plot comparison
    plot_path = os.path.join(RESULTS_DIR, 'overfitting_comparison.png')
    plot_overfitting_comparison(smote_results, cnn_results, gnn_results, plot_path)
    
    # Create detailed report
    report_path = os.path.join(RESULTS_DIR, 'overfitting_analysis_report.md')
    create_overfitting_report(smote_results, cnn_results, gnn_results, report_path)
    
    # Save detailed results
    results = {
        'smote_logistic_regression': smote_results,
        'focal_loss_cnn': cnn_results,
        'gnn_focal_loss': gnn_results
    }
    
    with open(os.path.join(RESULTS_DIR, 'overfitting_analysis_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print("\n" + "="*60)
    print("📋 OVERFITTING ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n🔍 Accuracy Gaps (Train - Validation):")
    print(f"  SMOTE + Logistic Regression: {smote_results['train_accuracy'] - smote_results['val_accuracy']:.3f}")
    print(f"  Focal Loss CNN: {cnn_results['train_accuracy'] - cnn_results['val_accuracy']:.3f}")
    print(f"  GNN with Focal Loss: {gnn_results['train_accuracy'] - gnn_results['val_accuracy']:.3f}")
    
    print(f"\n🎯 Test Malware Recall:")
    print(f"  SMOTE + Logistic Regression: {smote_results['test_malware_recall']:.1%}")
    print(f"  Focal Loss CNN: {cnn_results['test_malware_recall']:.1%}")
    print(f"  GNN with Focal Loss: {gnn_results['test_malware_recall']:.1%}")
    
    print(f"\n📁 Results saved to: {RESULTS_DIR}/")
    print("  - overfitting_comparison.png")
    print("  - overfitting_analysis_report.md")
    print("  - overfitting_analysis_results.pkl")

if __name__ == "__main__":
    main()
