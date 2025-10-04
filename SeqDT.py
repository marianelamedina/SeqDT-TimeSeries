from BestTree import *
from PessimisticErrorPruning import PEP
from node import Node


def SeqDT(T: list, g: int, maxL: int, pru: bool, epsilon: float, minS: float, minN: int, maxD: int) -> Node:
    """
    Sequence Decision Tree algorithm
    
    Input:
        - T: list of tuple ( sequence, label - training set)
        - g: int (gap threshold)
        - maxL: int (maximum pattern length)
        - pru: boolean
        - epsilon: float (purity threshold)
        - minS: float (minimum split improvement)
        - minN: int (minimum node size)
        - maxD: int (maximum depth, 0 = unlimited)
    
    Output:
        - N: root node of decision tree
    """

    N = BT(T, g, maxL, 0, epsilon, minS, minN, maxD)
    
    if pru:
        N = PEP(N)
    
    return N



def sequences_per_class(dataset: list) -> dict:
    """
    Number of sequences in different classes of a given dataset.
    
    Input:
        - dataset: list of tuples (sequence, label)
    
    Output:
        - class_counts: dict 
    """
    class_counts = {}
    
    for seq, label in dataset:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    return class_counts


def calculate_tree_depth(node: Node) -> int:
    if node is None:
        return 0
    
    #hasattr() returns True if the specified object has the specified attribute, otherwise False
    
    if (hasattr(node, 'type') and node.type == 'leaf') or not hasattr(node, 'left_child'):
        return 1
    
    left_depth = 0
    right_depth = 0
    
    if hasattr(node, 'left_child') and node.left_child is not None:
        left_depth = calculate_tree_depth(node.left_child)
    if hasattr(node, 'right_child') and node.right_child is not None:
        right_depth = calculate_tree_depth(node.right_child)
    
    return 1 + max(left_depth, right_depth)