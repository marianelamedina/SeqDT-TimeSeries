import numpy as np
import time
from extraction import * #extraction_sequences, extraction_labels
from SeqDT import * #SeqDT, sequences_per_class, calculate_tree_depth
from node import Node
from gap_constraint import check_gap_constraint
from Representation import plot_tree
from scipy.stats import gmean
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score, f1_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from IPython.display import display
from Statistics import get_dataset_statistics



def compute_metrics(y_true: list, y_pred: list, print_cm :bool = True) -> dict:
    """
    Calculate classification metrics.
    
    Input:
        - y_true: list
        - y_pred: list
        - print_cm: bool 
    
    Output:
        - dict containing:
            - accuracy: float
            - gmean: float
            - precision: array
            - recall: array
            - f1: array
            - confusion_matrix: array
            - labels: list
    
    """
    if not y_true or not y_pred:
        return "Empty label lists provided"
    
    if len(y_true) != len(y_pred):
        return "Length mismatch"
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    if np.any(recalls == 0):
        g_mean = 0
    else:
        g_mean = float(gmean(recalls))
    
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    
    labels = sorted(list(set(y_true)))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if print_cm:
        print(cm)
    
    return {
        'accuracy': accuracy,
        'gmean': g_mean,
        'precision': precision,
        'recall': recalls,
        'f1': f1,
        'confusion_matrix': cm,
        'labels': labels
    }

############################################################################

def predict_sequence(tree: Node, sequence: list, g :int = 1) -> str:
    """
    Predict class label for a single sequence using trained tree.
    
    Input:
        - tree: Node
        - sequence: list 
        - g: int
    
    Output:
        - label: str or None 
    """
    if tree is None:
        return None
    
    current_node = tree
    
    while current_node.type != 'leaf':
        if current_node.split_feature is None:
            break
        
        match, last_pos_match = check_gap_constraint(sequence, current_node.split_feature, g)
        
        if match and current_node.left_child is not None:
            current_node = current_node.left_child
        elif current_node.right_child is not None:
            current_node = current_node.right_child
        else:
            break
    
    return current_node.label


############################################################################

def evaluate_model(tree: Node, test_data: list, g :int = 1, print_cm: bool = True) -> dict:
    """
    Evaluate trained SeqDT model on test dataset.
    
    Input:
        - tree: Node
        - test_data: list 
        - g: int 
        - print_cm: bool 
    
    Output:
        - result : dict
    """
    if not test_data:
        return "Empty test dataset"
    
    if tree is None:
        return "Invalid tree"
    
    y_true = []
    y_pred = []
    
    for sequence, true_label in test_data:
        predicted_label = predict_sequence(tree, sequence, g)
        y_true.append(true_label)
        y_pred.append(predicted_label)
    
    # Compute metrics
    result = compute_metrics(y_true, y_pred, print_cm)
    
    # Add predictions to result
    result['predictions (y_true, y_pred)'] = list(zip(y_true, y_pred))
    
    return result

############################################################################

def evaluate_dataset(dataset: list = None,
                     X_train: np.ndarray = None, y_train: np.ndarray = None, 
                     X_test: np.ndarray = None, y_test: np.ndarray = None, 
                     method: str = 'holdout', test_size: float = 0.3,
                     n_folds: int = 5, n_repeats: int = 5, random_state: int = 42,
                     g: int = 1, maxL: int = 4, pru: bool = True, epsilon: float = 0.1,
                     minS: float = 0, minN: int = 2, maxD: int = 0,
                     show_statistics: bool = True, visualize: bool = False) -> dict:
    """
    Evaluation function for SeqDT on a dataset.
    
    Supports two modes:
    - Pass 'dataset' for automatic split/CV
    - Pass X_train, y_train, X_test, y_test for pre-split data
    
    Supports two evaluation methods:
    - 'holdout': Single train/test split
    - 'cv': Cross-validation
    
    Input:
        - dataset: list di tuple (sequence, label) or None
        - X_train = np.ndarray or None
        - y_train = np.ndarray or None
        - X_test = np.ndarray or None
        - y_test = np.ndarray or None 
        - method: str ('holdout' or 'cv')
        - test_size: float (for holdout default 0.3)
        - n_folds: int (for CV default 5)
        - n_repeats: int (for CV default 5)
        - random_state: int 
        - g: int 
        - maxL: int
        - pru: bool
        - epsilon: float 
        - minS: float 
        - minN: int 
        - maxD: int (0 = unlimited)
        - show_statistics: bool
        - visualize: bool (for holdout)
    
    Output:
        - dict with evaluation results
    """
    
    # MODE 1: Pre-split data provided 
    if X_train is not None and y_train is not None:
        if method == 'cv':
            print("Cross-validation is not supported with pre-split data")
            print("Holdout evaluation:")
            method = 'holdout'
        
        
        train_data = list(zip(X_train, y_train))
        
        test_data = list(zip(X_test, y_test))
        
        if show_statistics:
            stats_train = get_dataset_statistics(train_data)
            print(f"\nDataset Statistics:")
            print(f"Training samples: {len(train_data)}")
            print(f"Test samples: {len(test_data)}")
            print(f"Classes: {stats_train['num_classes']}")
            print(f"Class distribution (train): {stats_train['class_distribution']}")
            print(f"Sequence length - Min: {stats_train['min_length']}, "
                  f"Max: {stats_train['max_length']}, "
                  f"Avg: {stats_train['avg_length']:.2f} ± {stats_train['std_length']:.2f}")
        
        
        start_time = time.time()
        tree = SeqDT(train_data, g=g, maxL=maxL, pru=pru, epsilon=epsilon, 
                     minS=minS, minN=minN, maxD=maxD)
        
        results = evaluate_model(tree, test_data, g=g, print_cm=False)
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.3f}s")
        print("Confusion matrix:")
        print(results['confusion_matrix'])
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"G-mean:    {results['gmean']:.4f}")
        
        results['training_time'] = training_time
        results['train_size'] = len(train_data)
        results['test_size'] = len(test_data)
        results['method'] = 'holdout'
        results['tree'] = tree
        
        if visualize:
            dot = plot_tree(tree)
            display(dot)
            tree_depth = calculate_tree_depth(tree)
            results['tree_depth'] = tree_depth
            print(f"Tree depth: {tree_depth}")
        
        return results
    
    
    
    # MODE 2: Dataset provided 
    
    if show_statistics:
        stats = get_dataset_statistics(dataset)
        print(f"\n Dataset Statistics:")
        print(f"Total samples: {stats['size']}")
        print(f"Classes: {stats['num_classes']}")
        print(f"Class distribution: {stats['class_distribution']}")
        print(f"Sequence length - Min: {stats['min_length']}, "
              f"Max: {stats['max_length']}, "
              f"Avg: {stats['avg_length']:.2f} ± {stats['std_length']:.2f}")
    
    X = extraction_sequences(dataset)
    y = extraction_labels(dataset)
    
    # HOLDOUT METHOD
    if method == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        train_data = list(zip(X_train, y_train))
        test_data = list(zip(X_test, y_test))
        
        start_time = time.time()
        tree = SeqDT(train_data, g=g, maxL=maxL, pru=pru, epsilon=epsilon, 
                     minS=minS, minN=minN, maxD=maxD)
        
        results = evaluate_model(tree, test_data, g=g, print_cm=False)
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.3f}s")
        print("Confusion matrix:")
        print(results['confusion_matrix'])
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"G-mean:    {results['gmean']:.4f}")
        
        results['training_time'] = training_time
        results['train_size'] = len(train_data)
        results['test_size'] = len(test_data)
        results['method'] = 'holdout'
        results['tree'] = tree
        
        if visualize:
            dot = plot_tree(tree)
            display(dot)
            tree_depth = calculate_tree_depth(tree)
            results['tree_depth'] = tree_depth
            print(f"Tree depth: {tree_depth}")
        
        return results
    
    # CROSS-VALIDATION METHOD
    else:  # method == 'cv'
        if len(dataset) < n_folds:
            return f"Dataset too small for {n_folds}-fold CV"
    
        cv_splitter = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        
        accuracies = []
        gmeans = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y), 1):
            X_train_fold = [X[i] for i in train_idx]
            y_train_fold = [y[i] for i in train_idx]
            X_test_fold = [X[i] for i in test_idx]
            y_test_fold = [y[i] for i in test_idx]
            
            train_data = list(zip(X_train_fold, y_train_fold))
            test_data = list(zip(X_test_fold, y_test_fold))
            
            tree = SeqDT(train_data, g=g, maxL=maxL, pru=pru, epsilon=epsilon, 
                         minS=minS, minN=minN, maxD=maxD)
            
            result = evaluate_model(tree, test_data, g=g, print_cm=False)
            
            accuracies.append(result['accuracy'])
            gmeans.append(result['gmean'])
        
        # Compute statistics
        acc_mean = float(np.mean(accuracies))
        acc_std = float(np.std(accuracies))
        gmean_mean = float(np.mean(gmeans))
        gmean_std = float(np.std(gmeans))
        
        print(f'Accuracy: {acc_mean:.4f} ± {acc_std:.4f}')
        print(f'G-mean:   {gmean_mean:.4f} ± {gmean_std:.4f}')
        
        return {
            'method': 'cv',
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'gmean_mean': gmean_mean,
            'gmean_std': gmean_std,
            'n_folds': n_folds,
            'n_repeats': n_repeats
        }