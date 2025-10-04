from gap_constraint import check_gap_constraint

def project_dataset(current_node: list, feature: list, g: int = 1) -> list:
    '''
    Creation of subsets of the given node obtained by taking, 
    for each sequence that contains a given list of features, 
    the suffix that starts right after the last element of the list. 
    The features must appear in the sequence respecting the given gap constraint.
    
    Input:
        - current_node: list of tuples (sequence, label)
        - feature: list
        - g: int
    
    Output:
        - projected_db: list of tuple (sequence, label)
    '''
    if not feature:
        return current_node
    
    projected_db = []
    
    for seq, label in current_node:
        match, last_match_pos = check_gap_constraint(seq, feature, g)
        
        if match and last_match_pos is not None:
            projected_seq = seq[last_match_pos + 1:]
            
            if projected_seq:
                projected_db.append((projected_seq, label))
    
    return projected_db