# SeqDT - Decision Tree for Sequence Classification

A Python implementation of the **SeqDT (Sequence Decision Tree)** algorithm for discrete sequence classification, based on "Decision Tree for Sequences" paper by He et al.

## Overview

SeqDT is a tree-based classification method that constructs decision trees directly from sequential data without requiring pre-defined features. Unlike traditional methods that transform sequences into feature vectors, SeqDT builds trees using subsequences as split points, searching through the space of all possible subsequences in the training data.

### Key Features

- **Direct sequence processing**: No need for manual feature engineering or transformation
- **Discriminative pattern mining**: Uses Gini index improvement to identify the most discriminative subsequences
- **Gap constraint support**: Allows flexible matching of patterns with configurable gap parameters
- **Efficient pruning**: Branch-and-bound algorithm with upper bound pruning to reduce search space
- **Interpretable results**: Produces decision trees that are easy to understand and explain

## Algorithm

The core algorithm, **DPH (Discover Pattern with Highest Gini improvement)**, works by:

1. Starting with an empty pattern
2. Iteratively extending patterns by one item at a time
3. Evaluating each pattern's discriminative power using Gini index improvement
4. Pruning the search space using upper bound calculations
5. Selecting the pattern that maximizes class separation at each node

## Installation

```bash
git clone https://github.com/marianelamedina/SeqDT-TimeSeries.git
cd SeqDT-TimeSeries
pip install -r requirements.txt
```

## Usage 
For a complete analysis including training, cross-validation, and visualization:

```python
from Complete_analysis import analyze_dataset

# Prepare your dataset
dataset = [
    (['a', 'b', 'c', 'd'], '0'),
    (['a', 'c', 'e'], '1'),
    (['b', 'c', 'd', 'e'], '0'),
    (['a', 'd', 'e', 'c'], '1'),
    (['c', 'a', 'b', 'e'], '1'),
    (['d', 'b', 'd', 'b'], '0'),
]

# Run complete analysis with cross-validation
results = analyze_dataset(dataset=dataset,
                          dataset_name="My Dataset",
                          train_params={'g': 1,
                                        'maxL': 4,
                                        'pru': True,
                                        'epsilon': 0.1,
                                        'minS': 0,
                                        'minN': 2,
                                        'maxD': 0
                                        },
                          cv_params={'n_folds': 5,
                                      'n_repeats': 5,
                                      'random_state': 42
                                    },
                          visualize=True
                          )

print(f"Cross-validation Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
print(f"Cross-validation G-mean: {results['gmean_mean']:.4f} ± {results['gmean_std']:.4f}")
print(f"Tree depth: {results['tree_depth']}")
```

## Project Structure

```
.
├── Core Algorithm
│   ├── SeqDT.py                          # Main SeqDT algorithm
│   ├── BestTree.py                       # Best Tree (BT) construction algorithm
│   ├── node.py                           # Node class for tree structure
│   └── PessimisticErrorPruning.py        # Post-pruning implementation (PEP)
│
├── Pattern Mining
│   ├── DiscoverPatternHighestGini.py     # DPH algorithm for pattern discovery
│   ├── projected_dataset.py              # Dataset projection with gap constraints
│   ├── upper_bound.py                    # Upper bound calculation and node splitting
│   └── gap_constraint.py                 # Gap constraint checking
│
├── Utilities
│   ├── gini.py                           # Gini index calculations
│   ├── extraction.py                     # Label and sequence extraction
│   ├── load_dataset.py                   # Dataset loading functions
│   └── Statistics.py                     # Dataset statistics computation
│
├── Time Series Processing
│   ├── Discretization.py                 # Gaussian binning discretization
│   └── TimeSeries_summary.csv            # Time series experiments summary
│
├── Visualization & Analysis
│   ├── Representation.py                 # Tree and time series visualization
│   ├── ModelEvaluation.py                # Model evaluation and metrics
│   ├── Complete_analysis.py              # Complete analysis workflow
│   └── TimeSeriesAnalysis.py             # Time series analysis pipeline
│
├── Notebooks
│   ├── 1_main.ipynb                      # Main usage examples
│   ├── 2_Experiments_paper.ipynb         # Discrete sequence experiments
│   └── 3_Experiments_ECG.ipynb           # Time series ECG experiments
│
└── Configuration
    ├── README.md                         # This file
    └── requirements.txt                  # Python dependencies
```

## Key Components

### Gap Constraint
Allows flexible pattern matching where items in the pattern do not need to be consecutive in the sequence. The gap parameter `g` controls the maximum allowed gap between pattern items.

### Projected Dataset
Creates projected datasets by extracting suffixes of sequences that contain a given pattern, enabling efficient recursive tree construction.

### Upper Bound Pruning
Calculates an optimistic upper bound on the maximum possible Gini improvement for a pattern, enabling early pruning of unpromising search branches.

### Alphabet Generation
After dataset projection, builds a reduced alphabet containing only symbols present in the projected sequences. Sorts symbols by their Gini improvement to enable early discovery of high-quality patterns and strengthen pruning bounds.

### DPH Algorithm
The main algorithm that discovers the pattern with the highest Gini index improvement. Uses a priority queue and upper bound pruning to efficiently search through the space of all subsequences.

### Best Tree Algorithm (BT)
Constructs the decision tree recursively through top-down splitting, calling DPH at each node to find the optimal pattern and checking stopping criteria (purity, minimum samples, maximum depth, minimum improvement).

### Pessimistic Error Pruning (PEP)
Optional post-processing step that removes subtrees unlikely to improve generalization. Converts overly specific internal nodes into leaves when estimated error reduction does not justify added complexity.


## Parameters

### Core Parameters
- **g** (int): Gap constraint - maximum allowed gap between consecutive items in a pattern
- **maxL** (int): Maximum pattern length to consider during search

### Stopping Criteria Parameters

- **epsilon** (float): Purity threshold - nodes with Gini ≤ epsilon become leaves (default: 0.1)
- **minS** (float): Minimum improvement - minimum Gini gain required for splitting (default: 0)
- **minN** (int): Minimum node size - minimum samples per child node (default: 2)
- **maxD** (int): Maximum depth - tree depth limit, 0 = unlimited (default: 0)

### Pruning Parameter

- **pru** (bool): Enable/disable Pessimistic Error Pruning post-processing (default: True)

## Algorithm Complexity

The algorithm uses several optimization techniques:
- Priority queue for best-first search
- Upper bound pruning to eliminate unpromising branches
- Efficient dataset projection to avoid redundant computations

## References

This implementation is based on:

**He, Z., Wu, Z., Xu, G., Liu, Y., & Zou, Q. (2023).** "Decision Tree for Sequences." *IEEE Transactions on Knowledge and Data Engineering*, 35(1), 253-264.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{he2023decision,
  title={Decision Tree for Sequences},
  author={He, Zengyou and Wu, Ziyao and Xu, Guangyao and Liu, Yan and Zou, Quan},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={35},
  number={1},
  pages={253--264},
  year={2023},
  publisher={IEEE}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation is based on the SeqDT algorithm proposed by Zengyou He and colleagues. The original paper and additional resources can be found at: https://github.com/ZiyaoWu/SeqDT