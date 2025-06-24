# adaboost_baseline.py
import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.utils import resample 

def main():
     # Load data
    with open("X_graph_embeddings.pkl", "rb") as f:
        X = pickle.load(f)
    with open("cig_output.pkl", "rb") as f:
        y = pickle.load(f)['labels']

    malware_idx = (y == 1)
    benign_idx = (y == 0)

    X_malware = X[malware_idx]
    y_malware = y[malware_idx]

    X_benign = X[benign_idx]
    y_benign = y[benign_idx]

    # Downsample benign to match malware
    X_benign_down, y_benign_down = resample(
        X_benign, y_benign,
        replace=False,
        n_samples=len(X_malware),
        random_state=42
    )

    # Combine
    X = np.vstack((X_malware, X_benign_down))
    y = np.hstack((y_malware, y_benign_down))    

    print("Training Adaboost with 10-fold cross-validation...")

    # Handle imbalance
    sample_weights = np.where(np.array(y) == 1, 8.42, 1.0)  # Malware weight ~1079/128 = 8.42

    base_estimator = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=42)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scores = cross_validate(clf, X, y, cv=skf,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                            fit_params={"sample_weight": sample_weights},
                            return_estimator=True)

    print(f"\nAccuracy:  {np.mean(scores['test_accuracy']):.4f}")
    print(f"Precision: {np.mean(scores['test_precision']):.4f}")
    print(f"Recall:    {np.mean(scores['test_recall']):.4f}")
    print(f"F1 Score:  {np.mean(scores['test_f1']):.4f}")
    print(f"ROC AUC:   {np.mean(scores['test_roc_auc']):.4f}")

    # Optional: get predictions from one of the trained folds for a classification report
    clf_best = scores["estimator"][0]
    y_pred = clf_best.predict(X)
    print("\nClassification Report (best fold):")
    print(classification_report(y, y_pred, target_names=["Benign", "Malware"]))

if __name__ == "__main__":
    main()
