# SeqDT - Decision Tree for Sequence Classification

A Python implementation of the **SeqDT (Sequence Decision Tree)** algorithm for discrete sequence classification, based on the research paper "Decision Tree for Sequences" by He et al.

## Overview

SeqDT is a novel tree-based classification method that constructs decision trees directly from sequential data without requiring pre-defined features. Unlike traditional methods that transform sequences into feature vectors, SeqDT builds trees using subsequences as split points, searching through the space of all possible subsequences in the training data.

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
git clone https://github.com/yourusername/seqdt.git
cd seqdt
pip install -r requirements.txt
```

## Usage

```python
from DiscoverPatternHighestGini import DPH

# Prepare your training data
# T is a list of tuples: [(sequence, label), ...]
# where sequence is a list of items and label is the class
T = [
    ([1, 2, 3, 4], 0),
    ([1, 3, 5], 1),
    ([2, 3, 4, 5], 0),
    # ... more sequences
]

# Run the DPH algorithm
g = 1  # gap constraint
maxL = 5  # maximum pattern length
best_pattern, best_improvement = DPH(T, g, maxL)

print(f"Best pattern: {best_pattern}")
print(f"Gini improvement: {best_improvement}")
```

## Project Structure

```
.
├── DiscoverPatternHighestGini.py  # Main DPH algorithm
├── projected_dataset.py           # Dataset projection with gap constraints
├── upper_bound.py                 # Upper bound calculation and node splitting
├── gap_constraint.py              # Gap constraint checking
├── gini.py                        # Gini index calculations
├── extraction.py                  # Label extraction utilities
└── README.md                      # This file
```

## Key Components

### DPH Algorithm
The main algorithm that discovers the pattern with the highest Gini index improvement using a priority queue and branch-and-bound search.

### Gap Constraint
Allows flexible pattern matching where items in the pattern don't need to be consecutive in the sequence. The gap parameter `g` controls the maximum allowed gap between pattern items.

### Upper Bound Pruning
Calculates an optimistic upper bound on the maximum possible Gini improvement for a pattern, enabling early pruning of unpromising search branches.

### Dataset Projection
Creates projected datasets by extracting suffixes of sequences that contain a given pattern, enabling efficient recursive tree construction.

## Parameters

- **g** (int): Gap constraint - maximum allowed gap between consecutive items in a pattern
- **maxL** (int): Maximum pattern length to consider during search

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

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Add your contact information here]

## Acknowledgments

This implementation is based on the SeqDT algorithm proposed by Zengyou He and colleagues. The original paper and additional resources can be found at: https://github.com/ZiyaoWu/SeqDT