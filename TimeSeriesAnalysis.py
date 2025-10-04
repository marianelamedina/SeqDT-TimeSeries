import numpy as np
from Discretization import *
from Representation import *
from ModelEvaluation import *

def prepare_data_for_seqdt(X_sequence: np.ndarray, y_labels: np.ndarray) -> list:
    """
    Create dataset for SeqDT function.
    
    Input:
        - X_sequence : np.ndarray
        - y_labels : np.ndarray
        
    Output:
        - dataset : list of tuples
    """
    dataset = []
    
    for i, (sequence, label) in enumerate(zip(X_sequence, y_labels)):
        if type(X_sequence) == np.ndarray:
            dataset.append((sequence[0].tolist(), str(label)))
        else:
            dataset.append((sequence, str(label)))
    
    return dataset


############################################################################


def RLR(X_translated: np.ndarray) -> list:
    """
    Run length reduction of X_translated 
    
    Input:
        - X_translated : np.ndarray
        
    Output:
        - X_RLR : list 
    """
    X_RLR = []
    
    for sequence in X_translated:
        sequence = sequence[0].tolist()
        
        if len(sequence) == 0:
            X_RLR.append([])
            continue
        
        # Inizia con il primo elemento
        new_sequence = [sequence[0]]
        
        # Aggiungi solo se diverso dal precedente
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                new_sequence.append(sequence[i])
        
        X_RLR.append(new_sequence)
    
    return X_RLR


############################################################################


def TSAnalysis(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray, bins: int, 
               method: str = 'Classic', patients_to_viz: list = None, 
               SeqDT_parameters: dict = None, dataset_name: str = 'Unknown', visualize_plots: bool = True) -> dict:
    """
    Analyze time series with discretization and SeqDT classification
    
    Input:
        - X_train : np.ndarray
        - y_train : np.ndarray
        - X_test : np.ndarray
        - y_test : np.ndarray
        - bins : int
        - method : str ('Classic' or 'RLR' (Run Length Reduction))
        - patients_to_viz : list
        - SeqDT_parameters : dict
        - dataset_name: str = 'Unknown'
        - visualize_plots: bool
        
    Output:
        - dict
    """
    
    if patients_to_viz is None:
        patients_to_viz = [0]
        
    if SeqDT_parameters is None:
        SeqDT_parameters = {
            'g': 1,
            'maxL': 3,
            'pru': True,
            'epsilon': 0.15,
            'minS': 0,
            'minN': 2,
            'maxD': 0,
            'visualize': True,
            'show_statistics': True
        }
    
    mean, std, boundaries = create_boundaries(X_train, bins)
    
    if visualize_plots == True:
        # Plot Gaussian distribution with boundaries
        fig_gaussian = plot_gaussian_with_boundaries(mean, std, boundaries)
        fig_gaussian.show()
    
    # Translate time series to discrete bins
    X_train_translated = translate_timeseries(X_train, boundaries)
    X_test_translated = translate_timeseries(X_test, boundaries)
    
    
    # CLASSIC METHOD
    if method == 'Classic':
        if visualize_plots == True:
            fig_timeseries = plot_timeseries_comparison(X_train, X_train_translated, boundaries, patients_to_viz)
            fig_timeseries.show()
        
        train_sequences = prepare_data_for_seqdt(X_train_translated, y_train)
        test_sequences = prepare_data_for_seqdt(X_test_translated, y_test)
    
    # RLR(Run Length Reduction) METHOD
    elif method == 'RLR':
        X_RLR_train = RLR(X_train_translated)
        X_RLR_test = RLR(X_test_translated)
        
        if visualize_plots == True:
            fig_timeseries = plot_timeseries_comparison(X_train, X_train_translated, boundaries, patients_to_viz, X_RLR_train)
            fig_timeseries.show()
        
        train_sequences = prepare_data_for_seqdt(X_RLR_train, y_train)
        test_sequences = prepare_data_for_seqdt(X_RLR_test, y_test)
        
    
    
    X_train_seqdt = extraction_sequences(train_sequences)
    y_train_seqdt = extraction_labels(train_sequences)
    X_test_seqdt = extraction_sequences(test_sequences)
    y_test_seqdt = extraction_labels(test_sequences)
    
    
    print(f"\nSeqDT Evaluation (method: {method}, bins: {bins})")
    
    seqdt_result = evaluate_dataset(X_train=X_train_seqdt, y_train=y_train_seqdt, X_test=X_test_seqdt, y_test=y_test_seqdt, **SeqDT_parameters)
    
    
    # Summary statistics
    all_seq_lengths = [len(seq) for seq in X_train_seqdt + X_test_seqdt]
    train_seq_lengths = [len(seq) for seq in X_train_seqdt]
    test_seq_lengths = [len(seq) for seq in X_test_seqdt]
    
    avg_seq_len = np.mean(all_seq_lengths)
    std_seq_len = np.std(all_seq_lengths)
    avg_seq_len_train = np.mean(train_seq_lengths)
    std_seq_len_train = np.std(train_seq_lengths)
    avg_seq_len_test = np.mean(test_seq_lengths)
    std_seq_len_test = np.std(test_seq_lengths)
    min_seq_len = min(all_seq_lengths)
    max_seq_len = max(all_seq_lengths)
    
    summary = {
        'Dataset': dataset_name,
        'Classes': len(np.unique(y_train)),
        'Method': method,
        
        'Accuracy': round(seqdt_result['accuracy'], 3),
        'G-mean': round(seqdt_result['gmean'], 3),
        
        'Training_Time': round(seqdt_result['training_time'], 3),
        
        'Bins': bins,
        'g': SeqDT_parameters['g'],
        'maxL': SeqDT_parameters['maxL'],
        'Pruning': SeqDT_parameters['pru'],
        'Epsilon': SeqDT_parameters['epsilon'],
        'minS': SeqDT_parameters['minS'],
        'minN': SeqDT_parameters['minN'],
        'maxD': SeqDT_parameters['maxD'],
        
        'Train Size': len(X_train),
        'Test Size': len(X_test),
        'Original Length': X_train.shape[2],
        
        'Avg_Seq_Length': round(avg_seq_len, 2),
        'Std_Seq_Length': round(std_seq_len, 2),
        'Avg_Seq_Length_Train': round(avg_seq_len_train, 2),
        'Std_Seq_Length_Train': round(std_seq_len_train, 2),
        'Avg_Seq_Length_Test': round(avg_seq_len_test, 2),
        'Std_Seq_Length_Test': round(std_seq_len_test, 2),
        'Min_Seq_Length': min_seq_len,
        'Max_Seq_Length': max_seq_len,
        
        'Tree_Depth': seqdt_result.get('tree_depth', None)
    }
    
    
    return summary
