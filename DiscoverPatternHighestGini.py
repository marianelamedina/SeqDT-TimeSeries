from extraction import extraction_labels
from upper_bound import * #upper_bound, split_node
from gini import *
from projected_dataset import project_dataset

def DPH(T: list, g: int, maxL: int) -> tuple:
    """
    Discover the Pattern with the Highest Gini Improvement (DPH)
    
    Input:
        - T: list of tuples (sequence, label)
        - g: int
        - maxL: int (maximum feature length to consider)
    
    Output:
        Tuple of:
            - maxP: list (best pattern found)
            - maxI: float (best improvement found)
    """
    
    maxP = []
    maxI = 0
    
    # Priority queue: (improvement, pattern)
    PriQueue = []
    
    # Initialize with single-symbol patterns
    alphabet_init = set()
    for seq, label in T:
        for item in seq:
            alphabet_init.add(item)
    
    for item in sorted(alphabet_init):
        P_single = [item]
        T_P, T_nP = split_node(T, P_single, g)
        
        if len(T_P) > 0:
            improvement = improvement_gini(extraction_labels(T), T_P, T_nP)
            if improvement > 0:
                PriQueue.append((improvement, P_single))
    
    while PriQueue:
        # Sort queue by improvement (descending)
        PriQueue.sort(key=lambda x: x[0], reverse=True)
        #print("\nPriority Queue:", PriQueue)
        
        current_improvement, P = PriQueue.pop(0)
        #print("Pattern P considered:", P, "with improvement:", round(current_improvement, 3))
        
        # Pruning check via upper bound
        ub = upper_bound(T, P, g)
        #print(f"UB(T, {P}, g={g}) = {round(ub, 3)}")
        
        if ub <= maxI:
            #print(f"PRUNED! UB={round(ub, 3)} <= maxI={round(maxI, 3)}")
            continue
        
        # Update best pattern if current is better
        if current_improvement > maxI:
            maxI = current_improvement
            maxP = P.copy()
            #print("-> New max improvement:", round(maxI, 4), "of pattern", maxP)
        
        # Generate extensions if not at max length
        if len(P) < maxL:
            SubT = project_dataset(T, P, g)
            #print("Projected dataset SubT:", SubT)
            
            if len(SubT) == 0:
                continue
            
            # Get reduced alphabet from projected dataset
            alphabet = set()
            for seq, label in SubT:
                for item in seq:
                    alphabet.add(item)
            
            alphabet = sorted(alphabet)
            #print("Reduced alphabet:", alphabet)
            
            item_improvements = []
            
            for item in alphabet:
                P_new = P + [item]
                T_P, T_nP = split_node(T, P_new, g)
                
                if len(T_P) == 0:
                    continue
                
                improvement = improvement_gini(extraction_labels(T), T_P, T_nP)
                item_improvements.append((improvement, item, P_new))
                #print(f'Pattern {P_new}: improvement = {round(improvement, 3)}')
            
            # Sort by improvement (descending)
            item_improvements.sort(key=lambda x: x[0], reverse=True)
            
            for improvement, item, P_new in item_improvements:
                ub_P_new = upper_bound(T, P_new, g)
                #print(f'UB({P_new}) = {round(ub_P_new, 3)}')
                
                if ub_P_new > maxI:
                    PriQueue.append((improvement, P_new))
                    #print(f"Added {P_new} to queue")
                #else:
                    #print(f"NOT added: UB={round(ub_P_new, 3)} <= maxI={round(maxI, 3)}")
    
    return maxP, round(maxI, 6)

