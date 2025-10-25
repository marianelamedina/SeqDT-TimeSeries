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
    """
    Calculate the depth of a tree rooted at the given node.
    
    Input:
        - node: Node (root of tree/subtree)
    
    Output:
        - depth: int (1 for a single leaf, increases with each level)
    """
    if node is None:
        return 0
    
    if node.type == 'leaf':
        return 1
    
    left_depth = calculate_tree_depth(node.left_child) if node.left_child else 0
    right_depth = calculate_tree_depth(node.right_child) if node.right_child else 0
    
    return 1 + max(left_depth, right_depth)