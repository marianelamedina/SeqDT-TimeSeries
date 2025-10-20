def check_gap_constraint(sequence: list, feature: list, g: int = 1) -> tuple:
    """
    Checks whether a feature is present in a sequence while respecting the gap constraint
    (maximum allowed gap between consecutive elements of the feature).
    g = 1 means contiguous elements (substring),
    g = 0 means no gap constraint (standard subsequence).
    
    The function returns the earliest end match position (the position of the last element 
    of the first valid match encountered, if multiple matches exist).


    Input:
        - sequence: list
        - feature: list
        - g: int

    Output:
        Tuple of:
            - match: bool (True if the feature is found, False otherwise)
            - last_match_pos: int (position of the last element of the feature in the sequence, None if no match)
    """
    if not feature:
        return True, -1 
    
    feature_index = 0
    last_match_pos = -1
    
    for seq_index, symbol in enumerate(sequence):
        if symbol == feature[feature_index]:
            if last_match_pos != -1 and g != 0:
                distance = seq_index - last_match_pos
                if distance > g:
                    return False, None  # Gap constraint not respected

            last_match_pos = seq_index
            feature_index += 1
            
            # Found all the given features 
            if feature_index == len(feature):
                return True, last_match_pos 
    
    # Not found all the given features 
    return False, None