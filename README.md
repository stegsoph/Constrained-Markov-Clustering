# [Constrained-Markov-Clustering (CoMaC)](https://arxiv.org/abs/2112.09397)

*Abstract:*
We connect the problem of semi-supervised clustering to constrained
Markov aggregation, i.e., the task of partitioning the state
space of a Markov chain. We achieve this connection by considering
every data point in the dataset as an element of the Markov
chain’s state space, by defining the transition probabilities between
states via similarities between corresponding data points,
and by incorporating semi-supervision information as hard constraints
in a Hartigan-style algorithm. The introduced Constrained
Markov Clustering (CoMaC) is an extension of a recent informationtheoretic
framework for (unsupervised) Markov aggregation to the
semi-supervised case. Instantiating CoMaC for certain parameter
settings further generalizes two previous information-theoretic
objectives for unsupervised clustering. Our results indicate that
CoMaC is competitive with the state-of-the-art.
(https://arxiv.org/abs/2112.09397)

## Requirements

The code was implemented in `python3 version == 3.8` with the following packages:

```
scipy==1.6.2
scikit-learn==0.24.1
pandas==1.2.4
numpy==1.20.1
pathlib2==2.3.5
matplotlib==3.3.4
```

## Usage

An example usage of the sequential and annealing clustering algorithm is shown in `\notebooks\CoMaC-demo.ipynb`.
