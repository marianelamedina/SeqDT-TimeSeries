import numpy as np

def load_dataset(filename):
    '''
    Load dataset and create a list of labels and sequences of each record
    '''
    dataset = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: 
                continue
            label = parts[0]           
            seq = parts[1:]            
            dataset.append((seq, label))
            labels.append(label)
    print("Classes found in dataset: ", np.unique(labels).tolist())
    return dataset


