from gini import *
from extraction import extraction_labels
from gap_constraint import check_gap_constraint


def split_node(current_node: list, feature: list, g: int = 1) -> tuple:
    '''
    Splits the given node into two subsets according to a given feature list,
    respecting the gap constraint g.
    T_P contains the labels of sequences that include the pattern as a subsequence, 
    while T_nP contains the labels of the remaining sequences.
    
    Input: 
        - current_node: list of tuples (sequence, label)
        - feature: list
    
    Output:
        Tuple of:
            - T_P: list of int (labels of subsequence with feature)
            - T_nP: list of int (labels of subsequence without feature)
    '''
    
    T_P = []
    T_nP = []
    
    for seq, label in current_node:
        
        match, last_match_pos = check_gap_constraint(seq, feature, g)
        
        # Found all the given features  
        if match:   
            T_P.append(label)
        # Not found all the given features 
        else:                   
            T_nP.append(label)
    
    return T_P, T_nP



def upper_bound(current_node: list, feature: list, g: int = 1) -> float:
    '''
    Computes the upper bound of the Gini improvement for a given feature. 
    It assumes the best-case scenario where all sequences in T_P belong to a single class, 
    estimates the maximum possible improvement across all classes, 
    and returns this value as an optimistic bound.
    
    Input:
        - current_node: list of tuples (sequence, label)
        - feature: list
        - g: int
    
    Outbut: 
        - ideal_improvement: float
    '''
    
    labels_current_node = extraction_labels(current_node)
    
    T_P, T_nP = split_node(current_node, feature, g)
    
    #If T_P is empty, the upper bound is 0
    if len(T_P) == 0:
        return 0
    
    ideal_improvements = []
    
    unique_labels = np.unique(labels_current_node)
    
    for target_class in unique_labels:
        count_target_class = T_P.count(target_class)
        
        if count_target_class == 0:
            ideal_improvements.append(0)
            continue
        
        # Create ideal T_P containing only sequences of target_class
        ideal_T_P = [target_class] * count_target_class
        #print(f"Ideal T_P: {ideal_T_P}")
        
        # Create ideal T_nP
        ideal_T_nP = []
        removed_count = 0
        for label in labels_current_node:
            if label == target_class and removed_count < count_target_class:
                removed_count += 1
            else:
                ideal_T_nP.append(label)
        #print(f"Ideal T_nP: {ideal_T_nP}")
        
        # Calculate improvement for this ideal case
        ideal_improvement = improvement_gini(labels_current_node, ideal_T_P, ideal_T_nP)
        #print(f"Ideal improvement: {ideal_improvement}")
        
        ideal_improvements.append(ideal_improvement)
    
    return max(ideal_improvements)