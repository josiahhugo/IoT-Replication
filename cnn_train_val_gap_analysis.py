'''
CNN Train-Validation Gap Analysis
Direct analysis of training vs validation performance to check overfitting
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
RESULTS_DIR = "CNN Train-Val Gap Analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)

class EigenspaceDataset(Dataset):
    """Dataset for eigenspace embeddings"""
    def __init__(self, embeddings, labels, indices=None):
        if indices is not None:
            self.embeddings = torch.FloatTensor(embeddings[indices])
            self.labels = torch.LongTensor([labels[i] for i in indices])
        else:
            self.embeddings = torch.FloatTensor(embeddings)
            self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class ImprovedCNN(nn.Module):
    def __init__(self, input_dim=164):  # Changed from 984 to 164
        super(ImprovedCNN, self).__init__()
        
        # Calculate reshape dimensions for k=2
        self.k = 2  # Number of eigenvectors
        self.matrix_size = 82  # Original adjacency matrix size
        self.reshape_dim = [self.k, self.matrix_size]  # [2, 82]
        
        # Verify input dimension matches
        expected_input = self.k * self.matrix_size
        assert input_dim == expected_input, f"Input dim {input_dim} != expected {expected_input}"
        
        # CNN layers adapted for 2×82 input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size after convolutions
        # After conv+pool layers with input [2, 82]:
        # No pooling on first dimension (too small)
        # Pool only on second dimension: 82 → 41 → 20 → 10
        self.flattened_size = 128 * 2 * 10  # 128 channels × 2 height × 10 width
        
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to 2D "image": [batch, 1, 2, 82]
        x = x.view(batch_size, 1, self.reshape_dim[0], self.reshape_dim[1])
        
        # CNN layers with careful pooling
        x = F.relu(self.conv1(x))
        # Only pool in width dimension (82→41)
        x = F.max_pool2d(x, kernel_size=(1, 2))
        
        x = F.relu(self.conv2(x))
        # Pool in width again (41→20)
        x = F.max_pool2d(x, kernel_size=(1, 2))
        
        x = F.relu(self.conv3(x))
        # Final pool (20→10)
        x = F.max_pool2d(x, kernel_size=(1, 2))
        
        # Flatten and fully connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x

def evaluate_model(model, data_loader):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    malware_correct = 0
    malware_total = 0
    benign_correct = 0
    benign_total = 0
    
    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            
            # 🔧 CHANGE: Convert sigmoid output to predictions
            predicted = (outputs.squeeze() > 0.5).long()  # Threshold at 0.5
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class metrics
            malware_mask = labels == 1
            benign_mask = labels == 0
            
            if malware_mask.sum() > 0:
                malware_total += malware_mask.sum().item()
                malware_correct += (predicted[malware_mask] == labels[malware_mask]).sum().item()
            
            if benign_mask.sum() > 0:
                benign_total += benign_mask.sum().item()
                benign_correct += (predicted[benign_mask] == labels[benign_mask]).sum().item()
    
    accuracy = correct / total
    malware_recall = malware_correct / malware_total if malware_total > 0 else 0
    benign_recall = benign_correct / benign_total if benign_total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'malware_recall': malware_recall,
        'benign_recall': benign_recall,
        'predictions': all_predictions,
        'labels': all_labels
    }

def train_with_tracking(model, train_loader, val_loader, test_loader, num_epochs=30):
    """Train model with detailed tracking of train/val performance"""
    # 🔧 CHANGE: Use BCELoss for sigmoid output
    criterion = nn.BCELoss()  # Binary Cross Entropy for sigmoid output
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Track metrics over epochs
    history = {
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'train_malware_recall': [], 'val_malware_recall': [], 'test_malware_recall': [],
        'train_loss': [], 'epoch': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    model.to(device)
    
    print("Epoch | Train Acc | Val Acc | Test Acc | Train Malware | Val Malware | Test Malware | Gap")
    print("-" * 85)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            
            # 🔧 CHANGE: Convert labels to float for BCELoss
            labels_float = labels.float().unsqueeze(1)  # Shape: [batch, 1]
            loss = criterion(outputs, labels_float)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate on all sets
        train_metrics = evaluate_model(model, train_loader)
        val_metrics = evaluate_model(model, val_loader)
        test_metrics = evaluate_model(model, test_loader)
        
        # Calculate train-validation gap
        train_val_gap = train_metrics['accuracy'] - val_metrics['accuracy']
        
        # Store metrics
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['train_malware_recall'].append(train_metrics['malware_recall'])
        history['val_malware_recall'].append(val_metrics['malware_recall'])
        history['test_malware_recall'].append(test_metrics['malware_recall'])
        history['train_loss'].append(train_loss / len(train_loader))
        history['epoch'].append(epoch + 1)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"{epoch+1:5d} | {train_metrics['accuracy']:9.3f} | {val_metrics['accuracy']:7.3f} | "
                  f"{test_metrics['accuracy']:8.3f} | {train_metrics['malware_recall']:11.3f} | "
                  f"{val_metrics['malware_recall']:9.3f} | {test_metrics['malware_recall']:10.3f} | "
                  f"{train_val_gap:7.3f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def analyze_train_val_gap():
    """Analyze train-validation gap for overfitting detection"""
    print("=== CNN Train-Validation Gap Analysis ===")
    
    # Load data
    print("Loading eigenspace embeddings...")
    with open("X_graph_embeddings.pkl", "rb") as f:
        X = pickle.load(f)
    
    with open("improved_cig_output.pkl", "rb") as f:
        data = pickle.load(f)
        y = data["labels"]
    
    # 🔧 CONVERT LIST TO NUMPY ARRAY FIRST
    y = np.array(y)  # Convert list to numpy array
    X = np.array(X)  # Ensure X is also numpy array
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
    print(f"Class distribution: Benign={np.sum(y==0)}, Malware={np.sum(y==1)}")
    print(f"Class imbalance: {np.sum(y==0)/np.sum(y==1):.1f}:1 (benign:malware)")
    
    # 🔧 ADD LABEL VALIDATION AND CLEANING
    print(f"Original labels shape: {y.shape}")
    print(f"Label value range: {y.min()} to {y.max()}")
    print(f"Unique label values: {np.unique(y)}")
    
    # Ensure labels are in [0, 1] range
    if y.min() < 0 or y.max() > 1:
        print("⚠️ Labels outside [0,1] range - fixing...")
        y = np.clip(y, 0, 1)  # Clip to [0,1]
        
    # Ensure labels are integers
    y = y.astype(np.int64)
    
    # Verify binary classification
    unique_labels = np.unique(y)
    assert len(unique_labels) <= 2, f"Too many classes: {unique_labels}"
    assert all(label in [0, 1] for label in unique_labels), f"Invalid labels: {unique_labels}"
    
    print(f"✅ Cleaned labels: shape={y.shape}, range=[{y.min()}, {y.max()}]")
    print(f"✅ Class distribution: {np.bincount(y)}")
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Single train/val/test split for detailed analysis
    print(f"\n=== Single Split Analysis (for detailed overfitting assessment) ===")
    
    # Split data: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 0.85 ≈ 0.15 of total
    )
    
    print(f"Train: {len(X_train)} (Benign: {(y_train==0).sum()}, Malware: {(y_train==1).sum()})")
    print(f"Val:   {len(X_val)} (Benign: {(y_val==0).sum()}, Malware: {(y_val==1).sum()})")
    print(f"Test:  {len(X_test)} (Benign: {(y_test==0).sum()}, Malware: {(y_test==1).sum()})")
    
    # Create datasets and loaders
    train_dataset = EigenspaceDataset(X_train, y_train)
    val_dataset = EigenspaceDataset(X_val, y_val)
    test_dataset = EigenspaceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    model = ImprovedCNN(input_dim=164).to(device)
    
    print(f"\n=== Training with Detailed Tracking ===")
    model, history = train_with_tracking(model, train_loader, val_loader, test_loader, num_epochs=30)
    
    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    final_train = evaluate_model(model, train_loader)
    final_val = evaluate_model(model, val_loader)
    final_test = evaluate_model(model, test_loader)
    
    print(f"Final Train Accuracy: {final_train['accuracy']:.4f}")
    print(f"Final Val Accuracy:   {final_val['accuracy']:.4f}")
    print(f"Final Test Accuracy:  {final_test['accuracy']:.4f}")
    
    print(f"Final Train Malware Recall: {final_train['malware_recall']:.4f}")
    print(f"Final Val Malware Recall:   {final_val['malware_recall']:.4f}")
    print(f"Final Test Malware Recall:  {final_test['malware_recall']:.4f}")
    
    # Overfitting analysis
    final_train_val_gap = final_train['accuracy'] - final_val['accuracy']
    final_train_test_gap = final_train['accuracy'] - final_test['accuracy']
    max_train_val_gap = max([t - v for t, v in zip(history['train_acc'], history['val_acc'])])
    
    print(f"\n=== Overfitting Assessment ===")
    print(f"Final Train-Val Gap:    {final_train_val_gap:.4f}")
    print(f"Final Train-Test Gap:   {final_train_test_gap:.4f}")
    print(f"Maximum Train-Val Gap:  {max_train_val_gap:.4f}")
    
    if max_train_val_gap < 0.02:
        overfitting_status = "✅ NO overfitting detected"
    elif max_train_val_gap < 0.05:
        overfitting_status = "⚠️ MINIMAL overfitting"
    elif max_train_val_gap < 0.1:
        overfitting_status = "⚠️ MODERATE overfitting"
    else:
        overfitting_status = "❌ SIGNIFICANT overfitting"
    
    print(f"Overfitting Status: {overfitting_status}")
    
    # Class imbalance context
    print(f"\n=== Class Imbalance Context ===")
    print(f"Baseline accuracy (predict all benign): {(y_test==0).sum()/len(y_test):.1%}")
    print(f"CNN test accuracy: {final_test['accuracy']:.1%}")
    print(f"CNN malware recall: {final_test['malware_recall']:.1%}")
    
    if final_test['accuracy'] > 0.89 and final_test['malware_recall'] > 0.9:
        performance_assessment = "✅ GENUINELY EXCELLENT - High accuracy with excellent malware detection"
    elif final_test['accuracy'] > 0.89 and final_test['malware_recall'] < 0.5:
        performance_assessment = "❌ MISLEADING - High accuracy but poor malware detection (class imbalance)"
    else:
        performance_assessment = "⚠️ MODERATE - Reasonable performance"
    
    print(f"Performance Assessment: {performance_assessment}")
    
    # Plot training curves
    plot_training_curves(history)
    
    # Save results
    results = {
        'history': history,
        'final_train': final_train,
        'final_val': final_val,
        'final_test': final_test,
        'overfitting_status': overfitting_status,
        'performance_assessment': performance_assessment,
        'final_train_val_gap': final_train_val_gap,
        'max_train_val_gap': max_train_val_gap
    }
    
    with open(os.path.join(RESULTS_DIR, 'train_val_gap_analysis.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Analysis complete. Results saved to: {RESULTS_DIR}/")
    return results

def plot_training_curves(history):
    """Plot training curves to visualize overfitting"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = history['epoch']
    
    # Accuracy plot
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax1.plot(epochs, history['test_acc'], 'g-', label='Test', linewidth=2)
    ax1.set_title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Malware recall plot
    ax2.plot(epochs, history['train_malware_recall'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_malware_recall'], 'r-', label='Validation', linewidth=2)
    ax2.plot(epochs, history['test_malware_recall'], 'g-', label='Test', linewidth=2)
    ax2.set_title('Malware Recall vs Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Malware Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training loss
    ax3.plot(epochs, history['train_loss'], 'b-', linewidth=2)
    ax3.set_title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Loss')
    ax3.grid(True, alpha=0.3)
    
    # Train-Validation gap
    train_val_gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    ax4.plot(epochs, train_val_gap, 'purple', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='5% Gap')
    ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='10% Gap')
    ax4.set_title('Train-Validation Gap vs Epochs', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Gap')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves saved to: {RESULTS_DIR}/training_curves.png")

if __name__ == "__main__":
    results = analyze_train_val_gap()
