import numpy as np
from extraction import *
from SeqDT import sequences_per_class

def get_dataset_statistics(dataset: list) -> dict:
    """
    Compute descriptive statistics for a given dataset.
    
    Input:
        - dataset: list of tuple (sequence, label)
        
    Output:
        - result: dict with:
            - size: int (Number of samples)
            - num_classes: int (Number of unique classes)
            - min_length: int
            - max_length: int
            - avg_length: float
            - std_length: float
            - class_distribution: dict
    """
    if not dataset:
        return {
            'size': 0,
            'num_classes': 0,
            'min_length': 0,
            'max_length': 0,
            'avg_length': 0,
            'std_length': 0,
            'class_distribution': {}
        }
    
    sequences = extraction_sequences(dataset)
    labels = extraction_labels(dataset)
    
    seq_lengths = [len(seq) for seq in sequences]
    
    class_dist = sequences_per_class(dataset)
    
    return {
        'size': len(dataset),
        'num_classes': len(set(labels)),
        'min_length': min(seq_lengths),
        'max_length': max(seq_lengths),
        'avg_length': float(np.mean(seq_lengths)),
        'std_length': float(np.std(seq_lengths)),
        'class_distribution': class_dist
    }