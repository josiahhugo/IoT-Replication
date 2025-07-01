'''
GNN Overfitting Analysis for IoT Malware Detection
Comprehensive analysis to check if GNN models are overfitting
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
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import networkx as nx
import os
from collections import defaultdict
import seaborn as sns

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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
RESULTS_DIR = "GNN Overfitting Analysis Results"
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
    """Load the optimized opcode transition graphs"""
    try:
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
        
        return adj_matrices, labels
        
    except Exception as e:
        print(f"❌ Failed to load graphs: {e}")
        return None, None

def create_graph_dataset(graphs_data, labels):
    """Convert graph data to PyTorch Geometric format"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None, None
    
    graph_list = []
    valid_labels = []
    
    print(f"\n🔄 Converting adjacency graphs to PyTorch Geometric format...")
    
    for i, (adj_matrix, label) in enumerate(zip(graphs_data, labels)):
        try:
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
            node_degrees = np.sum(adj_matrix, axis=1)
            in_degrees = np.sum(adj_matrix, axis=0)
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

def train_gnn_with_tracking(model, train_loader, val_loader, test_loader, num_epochs=100, 
                           patience=20, use_focal_loss=True, alpha=5, gamma=2, class_weights=None):
    """Enhanced training with detailed tracking for overfitting analysis"""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None, None
    
    # Choose loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True)
        print(f"Using Focal Loss (α={alpha}, γ={gamma})")
    elif class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(f"Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    model.to(device)
    
    # Tracking metrics
    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'train_malware_recall': [], 'val_malware_recall': [], 'test_malware_recall': [],
        'learning_rate': []
    }
    
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
        val_metrics = evaluate_set(model, val_loader, criterion)
        test_metrics = evaluate_set(model, test_loader, criterion)
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_malware_recall = train_malware_correct / train_malware_total if train_malware_total > 0 else 0
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['test_loss'].append(test_metrics['loss'])
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['accuracy'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['train_malware_recall'].append(train_malware_recall)
        history['val_malware_recall'].append(val_metrics['malware_recall'])
        history['test_malware_recall'].append(test_metrics['malware_recall'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Early stopping based on validation loss
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs:3d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Test Loss: {test_metrics['loss']:.4f}")
            print(f"  Train Acc: {train_acc:.3f} | Val Acc: {val_metrics['accuracy']:.3f} | Test Acc: {test_metrics['accuracy']:.3f}")
            print(f"  Train Malware: {train_malware_recall:.3f} | Val Malware: {val_metrics['malware_recall']:.3f} | Test Malware: {test_metrics['malware_recall']:.3f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_set(model, data_loader, criterion):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    malware_correct = 0
    malware_total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
            # Track malware detection
            malware_mask = batch.y == 1
            if malware_mask.sum() > 0:
                malware_correct += (pred[malware_mask] == batch.y[malware_mask]).sum().item()
                malware_total += malware_mask.sum().item()
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': correct / total,
        'malware_recall': malware_correct / malware_total if malware_total > 0 else 0
    }

def plot_training_curves(histories, model_names, save_path):
    """Plot comprehensive training curves for overfitting analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = [
        ('train_loss', 'val_loss', 'test_loss', 'Loss'),
        ('train_acc', 'val_acc', 'test_acc', 'Accuracy'),
        ('train_malware_recall', 'val_malware_recall', 'test_malware_recall', 'Malware Recall')
    ]
    
    colors = ['blue', 'orange', 'green']
    line_styles = ['-', '--', ':']
    
    for i, (train_metric, val_metric, test_metric, title) in enumerate(metrics):
        ax = axes[0, i]
        
        for j, (history, model_name) in enumerate(zip(histories, model_names)):
            epochs = range(1, len(history[train_metric]) + 1)
            
            ax.plot(epochs, history[train_metric], color=colors[0], linestyle=line_styles[j], 
                   label=f'{model_name} Train', alpha=0.8)
            ax.plot(epochs, history[val_metric], color=colors[1], linestyle=line_styles[j], 
                   label=f'{model_name} Val', alpha=0.8)
            ax.plot(epochs, history[test_metric], color=colors[2], linestyle=line_styles[j], 
                   label=f'{model_name} Test', alpha=0.8)
        
        ax.set_title(f'{title} vs Epochs', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Overfitting analysis plots
    for i, (history, model_name) in enumerate(zip(histories, model_names)):
        ax = axes[1, i]
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot train vs validation gap
        train_val_gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
        train_test_gap = [t - te for t, te in zip(history['train_acc'], history['test_acc'])]
        
        ax.plot(epochs, train_val_gap, label='Train-Val Gap', color='red', linewidth=2)
        ax.plot(epochs, train_test_gap, label='Train-Test Gap', color='darkred', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='5% Overfitting Threshold')
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='10% Overfitting Threshold')
        
        ax.set_title(f'{model_name} - Overfitting Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy Gap')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add final gap text
        final_val_gap = train_val_gap[-1] if train_val_gap else 0
        final_test_gap = train_test_gap[-1] if train_test_gap else 0
        ax.text(0.02, 0.98, f'Final Val Gap: {final_val_gap:.3f}\nFinal Test Gap: {final_test_gap:.3f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves saved to: {save_path}")

def analyze_overfitting(histories, model_names):
    """Comprehensive overfitting analysis"""
    print(f"\n=== GNN OVERFITTING ANALYSIS ===")
    
    overfitting_report = []
    
    for history, model_name in zip(histories, model_names):
        print(f"\n--- {model_name} Analysis ---")
        
        # Calculate final gaps
        final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
        final_test_acc = history['test_acc'][-1] if history['test_acc'] else 0
        
        train_val_gap = final_train_acc - final_val_acc
        train_test_gap = final_train_acc - final_test_acc
        
        # Calculate maximum gaps during training
        max_train_val_gap = max([t - v for t, v in zip(history['train_acc'], history['val_acc'])]) if history['train_acc'] else 0
        max_train_test_gap = max([t - te for t, te in zip(history['train_acc'], history['test_acc'])]) if history['train_acc'] else 0
        
        # Loss analysis
        final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0
        final_test_loss = history['test_loss'][-1] if history['test_loss'] else 0
        
        # Malware recall analysis
        final_train_malware = history['train_malware_recall'][-1] if history['train_malware_recall'] else 0
        final_val_malware = history['val_malware_recall'][-1] if history['val_malware_recall'] else 0
        final_test_malware = history['test_malware_recall'][-1] if history['test_malware_recall'] else 0
        
        # Overfitting assessment
        overfitting_score = max(train_val_gap, train_test_gap)
        
        if overfitting_score < 0.02:
            overfitting_status = "✅ No overfitting"
        elif overfitting_score < 0.05:
            overfitting_status = "⚠️ Minimal overfitting"
        elif overfitting_score < 0.1:
            overfitting_status = "⚠️ Moderate overfitting"
        else:
            overfitting_status = "❌ Significant overfitting"
        
        print(f"Final Accuracies:")
        print(f"  Train: {final_train_acc:.4f}")
        print(f"  Val:   {final_val_acc:.4f}")
        print(f"  Test:  {final_test_acc:.4f}")
        
        print(f"Performance Gaps:")
        print(f"  Train-Val Gap:  {train_val_gap:.4f}")
        print(f"  Train-Test Gap: {train_test_gap:.4f}")
        print(f"  Max Train-Val Gap:  {max_train_val_gap:.4f}")
        print(f"  Max Train-Test Gap: {max_train_test_gap:.4f}")
        
        print(f"Final Losses:")
        print(f"  Train: {final_train_loss:.4f}")
        print(f"  Val:   {final_val_loss:.4f}")
        print(f"  Test:  {final_test_loss:.4f}")
        
        print(f"Malware Recall:")
        print(f"  Train: {final_train_malware:.4f}")
        print(f"  Val:   {final_val_malware:.4f}")
        print(f"  Test:  {final_test_malware:.4f}")
        
        print(f"Overfitting Assessment: {overfitting_status}")
        
        overfitting_report.append({
            'model': model_name,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'final_test_acc': final_test_acc,
            'train_val_gap': train_val_gap,
            'train_test_gap': train_test_gap,
            'max_train_val_gap': max_train_val_gap,
            'max_train_test_gap': max_train_test_gap,
            'overfitting_score': overfitting_score,
            'overfitting_status': overfitting_status,
            'final_train_malware': final_train_malware,
            'final_val_malware': final_val_malware,
            'final_test_malware': final_test_malware
        })
    
    return overfitting_report

def run_overfitting_analysis():
    """Run comprehensive overfitting analysis for GNN models"""
    print("=== GNN Overfitting Analysis for IoT Malware Detection ===")
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch Geometric is required but not available")
        return
    
    # Load graph data
    graphs_data, labels = load_opcode_graphs()
    if graphs_data is None:
        print("❌ Failed to load graph data")
        return
    
    print(f"Loaded {len(graphs_data)} graphs")
    print(f"Class distribution: Benign={labels.count(0)}, Malware={labels.count(1)}")
    
    # Convert to PyTorch Geometric format
    graph_list, valid_labels = create_graph_dataset(graphs_data, labels)
    if graph_list is None:
        print("❌ Failed to create graph dataset")
        return
    
    # Calculate class weights
    unique_labels = np.unique(valid_labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=valid_labels)
    print(f"Calculated class weights: {dict(zip(unique_labels, class_weights))}")
    
    # Split data for overfitting analysis (single split for detailed tracking)
    X_temp, X_test, y_temp, y_test = train_test_split(
        range(len(graph_list)), valid_labels, 
        test_size=0.2, random_state=42, stratify=valid_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, [valid_labels[i] for i in X_temp], 
        test_size=0.25, random_state=42, stratify=[valid_labels[i] for i in X_temp]
    )
    
    print(f"Data splits:")
    print(f"  Train: {len(X_train)} (Benign: {y_train.count(0)}, Malware: {y_train.count(1)})")
    print(f"  Val:   {len(X_val)} (Benign: {y_val.count(0)}, Malware: {y_val.count(1)})")
    print(f"  Test:  {len(X_test)} (Benign: {y_test.count(0)}, Malware: {y_test.count(1)})")
    
    # Create data loaders
    train_graphs = [graph_list[i] for i in X_train]
    val_graphs = [graph_list[i] for i in X_val]
    test_graphs = [graph_list[i] for i in X_test]
    
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
    
    # Test different configurations
    model_configs = [
        {
            'name': 'GCN_Focal_Loss',
            'use_focal_loss': True,
            'alpha': 10,
            'gamma': 3,
            'class_weights': None
        },
        {
            'name': 'GCN_Class_Weighted',
            'use_focal_loss': False,
            'alpha': None,
            'gamma': None,
            'class_weights': class_weights
        }
    ]
    
    histories = []
    model_names = []
    
    for config in model_configs:
        print(f"\n=== Training {config['name']} ===")
        
        # Create model
        input_dim = train_graphs[0].x.size(1) if train_graphs else 3
        model = GraphConvNet(input_dim=input_dim, hidden_dim=128, dropout=0.3)
        
        # Train model with detailed tracking
        trained_model, history = train_gnn_with_tracking(
            model, train_loader, val_loader, test_loader,
            num_epochs=100, patience=20,
            use_focal_loss=config['use_focal_loss'],
            alpha=config.get('alpha', 5),
            gamma=config.get('gamma', 2),
            class_weights=config['class_weights']
        )
        
        if history is not None:
            histories.append(history)
            model_names.append(config['name'])
    
    if histories:
        # Plot training curves
        plot_path = os.path.join(RESULTS_DIR, 'gnn_training_curves.png')
        plot_training_curves(histories, model_names, plot_path)
        
        # Analyze overfitting
        overfitting_report = analyze_overfitting(histories, model_names)
        
        # Save detailed report
        with open(os.path.join(RESULTS_DIR, 'overfitting_analysis.pkl'), 'wb') as f:
            pickle.dump({
                'histories': histories,
                'model_names': model_names,
                'overfitting_report': overfitting_report
            }, f)
        
        # Create summary report
        with open(os.path.join(RESULTS_DIR, 'overfitting_summary.md'), 'w') as f:
            f.write("# GNN Overfitting Analysis Report\n\n")
            f.write("## Summary\n\n")
            
            for report in overfitting_report:
                f.write(f"### {report['model']}\n\n")
                f.write(f"- **Overfitting Status**: {report['overfitting_status']}\n")
                f.write(f"- **Final Train Accuracy**: {report['final_train_acc']:.4f}\n")
                f.write(f"- **Final Validation Accuracy**: {report['final_val_acc']:.4f}\n")
                f.write(f"- **Final Test Accuracy**: {report['final_test_acc']:.4f}\n")
                f.write(f"- **Train-Val Gap**: {report['train_val_gap']:.4f}\n")
                f.write(f"- **Train-Test Gap**: {report['train_test_gap']:.4f}\n")
                f.write(f"- **Max Train-Val Gap**: {report['max_train_val_gap']:.4f}\n")
                f.write(f"- **Max Train-Test Gap**: {report['max_train_test_gap']:.4f}\n")
                f.write(f"- **Final Malware Recall (Train)**: {report['final_train_malware']:.4f}\n")
                f.write(f"- **Final Malware Recall (Val)**: {report['final_val_malware']:.4f}\n")
                f.write(f"- **Final Malware Recall (Test)**: {report['final_test_malware']:.4f}\n\n")
            
            f.write("## Interpretation\n\n")
            f.write("- **Gap < 0.02**: No overfitting\n")
            f.write("- **Gap 0.02-0.05**: Minimal overfitting\n")
            f.write("- **Gap 0.05-0.10**: Moderate overfitting\n")
            f.write("- **Gap > 0.10**: Significant overfitting\n\n")
            f.write("The analysis shows training curves and performance gaps to validate model generalization.\n")
        
        print(f"\n=== OVERFITTING ANALYSIS COMPLETE ===")
        print(f"Results saved to: {RESULTS_DIR}/")
        print(f"📊 Training curves: gnn_training_curves.png")
        print(f"📄 Summary report: overfitting_summary.md")
        print(f"📦 Detailed data: overfitting_analysis.pkl")

if __name__ == "__main__":
    run_overfitting_analysis()
