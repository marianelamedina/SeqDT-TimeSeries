def check_gap_constraint(sequence: list, feature: list, g: int = 1) -> tuple:
    """
    Checks whether a feature is present in a sequence while respecting the gap constraint
    (maximum allowed gap between consecutive elements of the feature).

    Input:
        - sequence: list
        - feature: list
        - g: int

    Output:
        Tuple of:
            - match: bool (True if the feature is found, False otherwise)
            - last_match_pos: int (position of the last element of the feature in the sequence, None if no match)
    """
    
    if len(feature) == 0:
        return True, -1
    
    feature_index = 0
    last_match_pos = -1
    
    for seq_index in range(len(sequence)):
        if feature_index < len(feature) and sequence[seq_index] == feature[feature_index]:
            if last_match_pos != -1 and (seq_index - last_match_pos - 1) > g:
                return False, None  # Gap constraint not respected
            
            last_match_pos = seq_index
            feature_index += 1
            
            # Found all the given features  
            if feature_index == len(feature):
                return True, last_match_pos
    
    # Not found all the given features 
    return False, None