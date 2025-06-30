'''
10-Fold Cross-Validation for Deep Eigenspace Learning CNN
Following the paper's methodology for robust evaluation
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
RESULTS_DIR = "10-Fold CV Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"10-fold CV results will be saved to: {RESULTS_DIR}/")

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

class DeepEigenspaceCNN(nn.Module):
    """CNN for Deep Eigenspace Learning - Same as paper"""
    def __init__(self, input_dim=984, num_classes=2, dropout_rate=0.5):
        super(DeepEigenspaceCNN, self).__init__()
        
        self.reshape_dim = (12, 82)  # 12 eigenvectors × 82 features
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.reshape_dim[0], self.reshape_dim[1])
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        
        # Flatten and fully connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

def train_fold(model, train_loader, val_loader, num_epochs=30):
    """Train model for one fold"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    best_val_acc = 0.0
    best_model_state = None
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_fold(model, test_loader):
    """Evaluate model on test fold"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    try:
        auc_score = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc_score = 0.0  # Handle cases where only one class in test set
    
    return accuracy, auc_score, all_predictions, all_labels, all_probabilities

def run_10fold_cv():
    """Run 10-fold cross-validation following the paper's methodology"""
    print("=== 10-Fold Cross-Validation for Deep Eigenspace Learning ===")
    
    # Load data
    print("Loading eigenspace embeddings...")
    with open("X_graph_embeddings.pkl", "rb") as f:
        X_embeddings = pickle.load(f)
    
    with open("improved_cig_output.pkl", "rb") as f:
        data = pickle.load(f)
        labels = data["labels"]
    
    print(f"Loaded {len(X_embeddings)} samples with {X_embeddings.shape[1]} features each")
    print(f"Class distribution: Benign={labels.count(0)}, Malware={labels.count(1)}")
    
    # Normalize features
    scaler = StandardScaler()
    X_embeddings = scaler.fit_transform(X_embeddings)
    
    # Convert to numpy arrays for indexing
    X_embeddings = np.array(X_embeddings)
    labels = np.array(labels)
    
    # 10-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = {
        'accuracy': [],
        'auc': [],
        'fold_predictions': [],
        'fold_labels': []
    }
    
    print(f"\n=== Running 10-Fold Cross-Validation ===")
    
    for fold, (train_val_indices, test_indices) in enumerate(skf.split(X_embeddings, labels)):
        print(f"\n--- Fold {fold + 1}/10 ---")
        
        # Split train_val into train and validation (80/20 split of the 90% train_val data)
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.2, random_state=42, 
            stratify=labels[train_val_indices]
        )
        
        # Create datasets
        train_dataset = EigenspaceDataset(X_embeddings, labels, train_indices)
        val_dataset = EigenspaceDataset(X_embeddings, labels, val_indices)
        test_dataset = EigenspaceDataset(X_embeddings, labels, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create fresh model for this fold
        model = DeepEigenspaceCNN(input_dim=984, num_classes=2)
        
        # Train model
        model = train_fold(model, train_loader, val_loader, num_epochs=30)
        
        # Evaluate on test set
        accuracy, auc_score, predictions, true_labels, probabilities = evaluate_fold(model, test_loader)
        
        # Store results
        fold_results['accuracy'].append(accuracy)
        fold_results['auc'].append(auc_score)
        fold_results['fold_predictions'].extend(predictions)
        fold_results['fold_labels'].extend(true_labels)
        
        print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
    
    # Calculate overall statistics
    mean_accuracy = np.mean(fold_results['accuracy'])
    std_accuracy = np.std(fold_results['accuracy'])
    mean_auc = np.mean(fold_results['auc'])
    std_auc = np.std(fold_results['auc'])
    
    print(f"\n=== 10-Fold Cross-Validation Results ===")
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in fold_results['accuracy']]}")
    print(f"Individual fold AUCs: {[f'{auc:.4f}' for auc in fold_results['auc']]}")
    
    # Overall classification report
    print(f"\n=== Overall Classification Report ===")
    print(classification_report(fold_results['fold_labels'], fold_results['fold_predictions'], 
                              target_names=['Benign', 'Malware']))
    
    # Save results
    results_summary = {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'fold_accuracies': fold_results['accuracy'],
        'fold_aucs': fold_results['auc'],
        'all_predictions': fold_results['fold_predictions'],
        'all_labels': fold_results['fold_labels']
    }
    
    with open(os.path.join(RESULTS_DIR, "cv_results.pkl"), "wb") as f:
        pickle.dump(results_summary, f)
    
    # Plot results
    plot_cv_results(fold_results['accuracy'], fold_results['auc'])
    
    # Analysis
    print(f"\n=== Overfitting Analysis ===")
    if std_accuracy < 0.02:  # Less than 2% standard deviation
        print("⚠️  VERY LOW VARIANCE - Possible overfitting or data leakage!")
    elif std_accuracy < 0.05:  # Less than 5% standard deviation
        print("⚠️  LOW VARIANCE - Good consistency, but check for overfitting")
    else:
        print("✅ REASONABLE VARIANCE - Model appears to be generalizing well")
    
    if mean_accuracy > 0.95:  # Greater than 95% accuracy
        print("⚠️  VERY HIGH ACCURACY - Unusual for real-world data, check for issues")
    
    return results_summary

def plot_cv_results(accuracies, aucs):
    """Plot cross-validation results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    folds = range(1, 11)
    ax1.bar(folds, accuracies, alpha=0.7, color='blue')
    ax1.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.3f}')
    ax1.set_title('10-Fold Cross-Validation Accuracy')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUC plot
    ax2.bar(folds, aucs, alpha=0.7, color='green')
    ax2.axhline(y=np.mean(aucs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(aucs):.3f}')
    ax2.set_title('10-Fold Cross-Validation AUC')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('AUC')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cv_results.png'), dpi=300)
    plt.close()
    print(f"Cross-validation plots saved to: {RESULTS_DIR}/cv_results.png")

if __name__ == "__main__":
    results = run_10fold_cv()
