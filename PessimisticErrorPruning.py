import numpy as np
from extraction import extraction_labels
from BestTree import majority
from node import Node


def node_error(records: list) -> float:
    """
    The error of the node is calculated as the sum of misassigned samples
    plus a continuity correction of 1/2 in the case of binary decisions. 
    We do this to counteract a bias that can arise when the same set of samples 
    is used for both tree construction and pruning.
    
    Imput:
        - records: list of tuple (sequence, label)
        
    Output:
        - e_t: float
    """
    e_t = 0.5 #continuity correction
    
    if not records:
        return e_t
    
    labels = extraction_labels(records)
    majority_class = majority(records)
    
    for label in labels:                     
        if label != majority_class:          
            e_t = e_t + 1   
    
    return e_t



def subtree_error(node: Node) -> float:
    """
    The error of the subtree of the node is calculated as the sum of the errors 
    of all the leaves following node t (including continuity corrections).
    
    Imput:
        - node: class Node
    
    Output;
        - e_Tt: float
    """
    
    if node.type == 'leaf':
        return node_error(node.records)
    
    error = 0
    if node.left_child:
        error += subtree_error(node.left_child)
    if node.right_child:
        error += subtree_error(node.right_child)
    
    return error



def standard_error(subtree_error: float, n_samples: int) -> float:
    """
    The standard error is calculated from:
        - error of the node t (e_t)
        - error of the subtree of the node (e(T_t))
        - the number of samples in node t (n(t))
    
    Input:
        - subtree_error: float
        - n_samples: int
    
    Output:
        - Std_Error: float
    """
    if n_samples <= 1:
        return 0
    
    if subtree_error >= n_samples:
        return 0
    
    Std_Error = np.sqrt(subtree_error * (n_samples - subtree_error) / n_samples)
    return Std_Error



def PEP(node: Node) -> Node:
    """
    Pessimistic Error Pruning (PEP), for each node t, calculates the following:
    - error of the node t (e_t)
    - error of the subtree of the node (e(T_t))
    - SE: standard error
    
    Pruning occurs when the sum of the standard error and the error of the 
    subtree is greater than or equal to the error of the node. This happens 
    because node t, as a leaf node, makes fewer errors than the subtree.
    e(t) ≤ e(Tt) + SE
    
    Input:
        - node: class Node
        
    Output:
        - node: class Node (after pruning)
    """
     
    if node.type == 'leaf':
        return node
    
    # Calculate errors for current node
    e_t = node_error(node.records)
    e_Tt = subtree_error(node)
    n_samples = len(node.records)
    Std_Error = standard_error(e_Tt, n_samples)
    
    #print(f"n_samples = {n_samples}, e(t) = {e_t}, e(Tt) = {e_Tt}, standard error = {round(Std_Error,3)}")
    
    # Pruning criteria: e(t) ≤ e(T_t) + SE
    if e_t <= e_Tt + Std_Error:
        node.type = 'leaf'
        node.label = majority(node.records)
        node.left_child = None
        node.right_child = None
        node.split_feature = None
        #print("Pruned")
        
    else:
        if node.left_child:
            node.left_child = PEP(node.left_child)
        if node.right_child:
            node.right_child = PEP(node.right_child)
    
    return node