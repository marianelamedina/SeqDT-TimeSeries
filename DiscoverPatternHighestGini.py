from extraction import extraction_labels
from upper_bound import * #upper_bound, split_node
from gini import *
from projected_dataset import project_dataset


def DPH(T: list, g: int, maxL: int) -> tuple:
    """
    Discover the Pattern with the Highest Gini index improvement.
    
    Input:
        - T: list of tuples (sequence, label)
        - g: int 
        - maxL: int (maximum feature length to consider)
    
    Output:
        Tuple of:
            - maxP: int (best pattern found)
            - maxI: float (best improvement found)
    """
    
    maxP = []
    maxI = 0
    
    # Priority queue: (improvement, pattern)
    PriQueue = [(float('inf'), [])]
    
    while PriQueue:
        
        PriQueue.sort(key=lambda x: x[0], reverse=True)
        #print("\nPriority Queue:", PriQueue)
        
        current_improvement, P = PriQueue.pop(0)
        #print("Pattern P considered:", P)
        
        # Pruning condition
        if len(P) > 0:
            ub = upper_bound(T, P)
            if ub <= maxI:
                #print("Pruned")
                break
        
        SubT = project_dataset(T, P, g)
        if len(SubT) == 0:
            continue
        
        #print("Projected dataset SubT:", SubT)

        item_improvements = []
        
        # Get all unique items in the projected dataset - ALPHABET
        alphabet = set()
        for seq, label in SubT:
            for item in seq:
                alphabet.add(item)
                
        alphabet = sorted(alphabet)
                
        #print("Alphabet:", item_list)
        
        for item in alphabet:
            P_new = P + [item]
            T_P, T_nP = split_node(T, P_new, g)
            
            if len(T_P) == 0:
                continue
                
            improvement = improvement_gini(extraction_labels(T), T_P, T_nP)
            item_improvements.append((improvement, item, P_new))
            #print('Improvement', improvement, 'of item', item, ', considering pattern', P_new)
        
        # Sort by improvement 
        item_improvements.sort(key=lambda x: x[0], reverse=True)
        #print('items sorted:', item_improvements)
        
        for improvement, item, P_new in item_improvements:
            if improvement > maxI:
                maxI = improvement
                maxP = P_new.copy()
                #print("-> New max improvement:", round(maxI,4), "of pattern", maxP)
            
            
            # Add to queue if worth exploring
            if len(P_new) < maxL:
                ub_P_new = upper_bound(T, P_new)
                #print('UB of pattern', P_new, '=', ub_P_new)
                if maxI < ub_P_new:
                    PriQueue.append((improvement, P_new))
                    #print("Add", P_new, "to the priority queue")


    return maxP, round(maxI, 6)