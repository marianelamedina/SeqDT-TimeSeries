from extraction import extraction_labels
from gini import *
from DiscoverPatternHighestGini import DPH
from node import Node
from gap_constraint import check_gap_constraint

def majority(T: list) -> str:
    """
    Returns the majority class label in the given dataset (the label that occurs most 
    frequently). Returns None if the dataset is empty.

    Input:
        - T: list of tuples (sequence, label)

    Output:
        - majority_class: str or None
    """
    labels = extraction_labels(T)
    if not labels:
        return None
    
    counts = {}
    
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    
    max_count = 0
    majority_class = None
    
    for label, count in counts.items():
        if count > max_count:
            max_count = count
            majority_class = label
    
    return majority_class



def BT(T: list, g: int, maxL:int, dep: int, epsilon: float, minS: float, minN: int, maxD: int) -> Node:
    """
    Recursive procedure that constructs the decision tree.
    
    Input:
        - T: list of tuple ( sequence, label - training set)
        - g: int (gap threshold)
        - maxL: int (maximum pattern length)
        - dep: int (current depth)
        - epsilon: float (purity threshold)
        - minS: float (minimum split improvement)
        - minN: int (minimum node size)
        - maxD: int (maximum depth, 0 = unlimited)
    
    Output:
        - N: root node of constructed decision tree N
    """
    
    N = Node()
    N.records = T
    
    if not T:
        N.type = 'leaf'
        N.label = None 
        return N
    
    # FIRST STAGE: check stopping criteria 1 (node purity is sufficient)
    
    gini_val = Gini(extraction_labels(T))
    #print('\nNode:', T)
    #print(f'Gini of node = {round(gini_val,3)}')
    
    if gini_val <= epsilon:
        #print(f'Gini of node <= epsilon ( = {epsilon})')
        N.type = 'leaf'
        N.label = majority(T)
        return N
    #print('First stopping criteria not satisfied')
    
    
    # SECOND STAGE: generates a left child and a right child of the current 
    # node N and checks the remaining stopping criteria.
    
    maxP, maxI = DPH(T, g, maxL)
    
    if not maxP:  
        N.type = 'leaf'
        N.label = majority(T)
        #print('No valid pattern found, node becomes leaf')
        return N

    #print(f'From DPH, we found best pattern {maxP}, with improvement Gini equal to {round(maxI,3)}')
    
    T_P = []
    T_nP = []
    
    for record in T:
        seq, label = record
        match, last_match_position = check_gap_constraint(seq, maxP, g)   
        if match:
            T_P.append(record)
        else:
            T_nP.append(record)
    
    dep += 1
    
    #print(f'Splitted node into:')
    #print(f'T_P: {T_P}')
    #print(f'T_nP: {T_nP}')
    
    # Check remaining stopping criteria
    if (maxI <= minS or # 2° criterion: when decreased impurity for splitting is less than minS
        len(T_P) < minN or # 3° criterion: the number of records in the left child node or
        len(T_nP) < minN or  # the right child node is less than minN
        (maxD > 0 and dep > maxD)): # 4° criterion: the depth of the decision tree is larger than maxD.
        
        #print('Stop splitting')
        
        N.type = 'leaf'
        N.label = majority(T)
        return N
    
    N.type = 'internal' 
    N.split_feature = maxP
    
    # Recursively build children
    #print('Generated children')
    N.left_child = BT(T_P, g, maxL, dep, epsilon, minS, minN, maxD)
    N.right_child = BT(T_nP, g, maxL, dep, epsilon, minS, minN, maxD)

    return N
