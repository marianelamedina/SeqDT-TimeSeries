def extraction_labels(current_node: list) -> list:
    '''
    Creation of a list of labels of each record present in the given node.
    
    Input: 
        - current_node: list of tuples (sequence, label)
    
    Output: 
        - labels_current_node: list of int
    '''
    
    labels_current_node = []
    for seq, label in current_node:   
        labels_current_node.append(label)
        
    return labels_current_node


def extraction_sequences(current_node: list) -> list:
    '''
    Creation of a list of sequences of each record present in the given node.
    
    Input: 
        - current_node: list of tuples (sequence, label)
    
    Output: 
        - seq_current_node: list
    '''
    
    seq_current_node = []
    for seq, label in current_node:   
        seq_current_node.append(seq)
        
    return seq_current_node