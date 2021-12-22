import matplotlib.pyplot as plt
from sklearn import datasets, decomposition
import numpy as np
import pandas as pd
import itertools

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Own modules
from source.utils.plotting import init_plot_style
from source.utils.helperFunctions import partition_to_labels
from source.utils.generateData import (generate3circles,
                                       generate_int_labels
                                       )
from source.utils.MCA_NMI_computation import MCA
from source.utils.save_data_csv import save_results

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot style
plt.style.use('seaborn-muted')
init_plot_style()

# import some data to play with

dataset_str = 'WINE_SCALED'

if dataset_str == 'IRIS':
    iris = datasets.load_iris()
    X = iris.data[:, :]
    labels_true = iris.target
    knns = 20

elif dataset_str == 'WINE':
    wine = datasets.load_wine()
    X = wine.data[:, :]
    labels_true = wine.target
    knns = 20

elif dataset_str == 'WINE_SCALED':
    wine_df = pd.read_csv("../data/rand_samples_links/wine-scaled.in",
                          header=None, delimiter=",")
    X = wine_df.loc[:, 0:12].to_numpy()
    labels_true = wine_df.loc[:,13].to_numpy()
    knns = 20

elif dataset_str == 'GLASS':
    glass_df = pd.read_csv("../data/data-sets/glass.csv", header=None)
    X = glass_df.loc[:, 0:8].to_numpy()
    labels_true = glass_df.loc[:, 9].to_numpy()
    knns = 20

elif dataset_str == 'ECOLI':
    ecoli_df = pd.read_csv("../data/data-sets/ecoli.csv", header=None)
    X = ecoli_df.loc[:, 0:6].to_numpy()
    pca = decomposition.PCA(n_components=5)
    pca.fit(X)
    X = pca.transform(X)
    labels_true = ecoli_df.loc[:, 7].to_numpy()
    X = X[:327, :]
    labels_true = labels_true[:327]
    knns = 20

elif dataset_str == 'VERTEBRAL':
    vertebral_df = pd.read_csv("../data/data-sets/vertebral.data",
                               skiprows=[0], header=None, delimiter=" ")
    X = vertebral_df.loc[:, 0:5].to_numpy()
    labels_df = pd.read_csv("../data/reference-labelling/vertebral.ref",
                            skiprows=[0], header=None, delimiter=" ")
    labels_true = labels_df.to_numpy()
    knns = 20

elif dataset_str == 'SEGMENTATION':
    segmentation_df = pd.read_csv("../data/data-sets/segmentation.data",
                                  skiprows=[0], header=None, delimiter=" ")
    X = segmentation_df.loc[:, 0:4].to_numpy()
    labels_df = pd.read_csv("../data/reference-labelling/segmentation.ref",
                            skiprows=[0], header=None, delimiter=" ")
    labels_true = labels_df.to_numpy()
    knns = 20

elif dataset_str == 'USER':
    user_df = pd.read_csv("../data/data-sets/user.data",
                          skiprows=[0], header=None, delimiter=" ")
    X = user_df.loc[:, 0:4].to_numpy()
    labels_df = pd.read_csv("../data/reference-labelling/user.ref",
                            skiprows=[0], header=None, delimiter=" ")
    labels_true = labels_df.to_numpy()
    knns = 20

else:
    (P, V_true, X, prefix) = generate3circles()    # 3 concentric circles
    labels_true = partition_to_labels(V_true)
    knns = 20

labels_true = generate_int_labels(labels_true)
M = len(np.unique(labels_true))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the training points
plt.figure(figsize=(8, 6))
for i in range(M):
    plt.plot(X[labels_true == i, 0], X[labels_true == i, 1], 'x')
    plt.title("True clustering - first 2 dimensions")

# %%
# =============================================================================
# Clustering computations
# =============================================================================

# knns_vec = [2,3,4,5,7,10,15,20,30,50,75,100,150,200]

ClassLabels_vec = ['All', 2]
ClassLabels = 'All'
knns = 20
wrong_percentage_vec = [0.0]
percentage_vec = [0, 0.2]

for percentage, wrong_percentage in itertools.product(percentage_vec, wrong_percentage_vec):
    restarts = 5
    N_iter = 10

    print("wrong percentage: ", wrong_percentage)
    NMI_aggreg_mean = 0
    NMI_seq_mean = 0

    for i in range(N_iter):
        print("Iteration: ", i)
        (aggreg, VSeq, NMI_aggreg, NMI_seq, beta_vec) = MCA(X, M, knns,
                                                            labels_true,
                                                            restarts,
                                                            percentage,
                                                            wrong_percentage,
                                                            ClassLabels,
                                                            show_figs=0
                                                            )
        NMI_aggreg_mean += NMI_aggreg
        NMI_seq_mean += NMI_seq
        print('NMI: ', NMI_aggreg)

    NMI_aggreg_mean = NMI_aggreg_mean/N_iter
    NMI_seq_mean = NMI_seq_mean/N_iter

    

    # %%
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create dataframe and csv file

    save_data = 1
    if save_data:
        df = save_results(dataset_str, beta_vec, NMI_aggreg_mean, NMI_seq_mean,
                          knns, restarts, percentage, wrong_percentage,
                          ClassLabels)
        print(df.loc[(df['Dataset'] == dataset_str)])

