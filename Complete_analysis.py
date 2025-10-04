from Representation import plot_tree
from ModelEvaluation import * 
from Statistics import get_dataset_statistics
from SeqDT import SeqDT, calculate_tree_depth
from IPython.display import display
import time


def analyze_dataset(dataset: list, dataset_name: str, train_params: dict = None, 
                   cv_params: dict = None, visualize: bool = True) -> dict:
    """
    Comprehensive dataset analysis with model training and evaluation.
    
    Input:
        - dataset: list of tuple (sequence, label) 
        - dataset_name: str
        - train_params: dict (Parameters for SeqDT training)
        - cv_params: dict (Parameters for cross-validation)
        - visualize: bool
    
    Output:
        - result: dict containing:
            - Dataset statistics
            - Training time
            - Tree depth
            - Cross-validation results
    """
    
    if not dataset:
        return  "Empty dataset"
    
    # Default train parameters from paper and optional update
    default_train = {'g': 1, 
                     'maxL': 4, 
                     'pru': True, 
                     'epsilon': 0.1, 
                     'minS': 0, 
                     'minN': 2, 
                     'maxD': 0
                    }
    
    if train_params:
        default_train.update(train_params)
    
   # Default cross validation parameters and optional update
    default_cv = {'n_folds': 5, 
                  'n_repeats': 5, 
                  'random_state': 42
                 }
    
    if cv_params:
        default_cv.update(cv_params)
    
    train_params = default_train
    cv_params = default_cv
    
    
    # Cross-validation
    result_cv = evaluate_dataset(dataset, 
                                method='cv',
                                g=train_params['g'], 
                                maxL=train_params['maxL'], 
                                pru=train_params['pru'], 
                                epsilon=train_params['epsilon'],
                                minS=train_params['minS'], 
                                minN=train_params['minN'], 
                                maxD=train_params['maxD'], 
                                n_folds=cv_params['n_folds'], 
                                n_repeats=cv_params['n_repeats'], 
                                random_state=cv_params['random_state'],
                                show_statistics=True
                                )
    
    acc_mean = result_cv['accuracy_mean']
    acc_std = result_cv['accuracy_std']
    gmean_mean = result_cv['gmean_mean']
    gmean_std = result_cv['gmean_std']
    
    # Dataset training
    tree_depth = 0
    training_time = 0
    
    if visualize:
        start_time = time.time()
    
        tree = SeqDT(dataset, 
                    g=train_params['g'],
                    maxL=train_params['maxL'],
                    pru=train_params['pru'],
                    epsilon=train_params['epsilon'],
                    minS=train_params['minS'],
                    minN=train_params['minN'],
                    maxD=train_params['maxD']
                    )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.3f}s")
        
        tree_depth = calculate_tree_depth(tree)
        print(f"Tree depth: {tree_depth}")
        
        dot = plot_tree(tree)
        display(dot)
        
        
    stats = get_dataset_statistics(dataset)
    
    result = {
        'Dataset': dataset_name,
        'Size': stats['size'],
        'Classes': stats['num_classes'],
        #'Class Distribution': stats['class_distribution'],
        'Min Sample Length': stats['min_length'],
        'Max Sample Length': stats['max_length'],
        'Avg Sample Length': round(stats['avg_length'], 2),
        'Sample SD': round(stats['std_length'], 2),
        'Accuracy': f"{acc_mean:.3f}±{acc_std:.3f}",      
        'G-mean': f"{gmean_mean:.3f}±{gmean_std:.3f}",    
        'Tree Depth': tree_depth,
        'Training Time (s)': round(training_time, 3),
    }
    
    return result