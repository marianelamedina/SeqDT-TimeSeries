import numpy as np

def Gini(labels_of_sequences: list) -> float:
    '''
    Calculate Gini index.
    
    Input:
        - labels_of_sequences: list of int
    
    Output:
        - gini_index: float
    '''
    
    total = len(labels_of_sequences)
    if total == 0:
        return 0

    #Count n° of occurrences of each label
    unique_labels, counts = np.unique(labels_of_sequences, return_counts=True) 

    prob = counts / total

    gini_index = 1 - np.sum(prob ** 2)
    return gini_index



def improvement_gini(labels_current_node: list, T_P: list, T_nP:list) -> float:
    '''
    Calculate the gini index improvement.
    
    Input:
        - labels_current_node: list of int
        - T_P: list of int (labels of subset with feature)
        - T_nP: list of int (labels of subset without feature)
    
    Output: 
        - I: float 
    
    '''
    
    gini_present_node = Gini(labels_current_node) 
            
    gini_P = Gini(T_P)
    gini_nP = Gini(T_nP)

    I = gini_present_node - (len(T_P)/len(labels_current_node))*gini_P - (len(T_nP)/len(labels_current_node))*gini_nP
    return I