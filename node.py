class Node:
    def __init__(self):
        self.type = None  # 'leaf' o 'internal'
        self.label = None
        self.split_feature = None 
        self.left_child = None
        self.right_child = None
        self.records = []