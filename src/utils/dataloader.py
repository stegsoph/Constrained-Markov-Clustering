from sklearn import datasets, decomposition


def load_dataset(dataset_str='IRIS'):
    
    if dataset_str == 'IRIS':
        iris = datasets.load_iris()
        X = iris.data[:, :]
        labels_true = iris.target

    elif dataset_str == 'WINE':
        wine = datasets.load_wine()
        X = wine.data[:, :]
        labels_true = wine.target

    elif dataset_str == 'WINE_SCALED':
        wine_df = pd.read_csv("../data/rand_samples_links/wine-scaled.in",
                              header=None, delimiter=",")
        X = wine_df.loc[:, 0:12].to_numpy()
        labels_true = wine_df.loc[:,13].to_numpy()

    elif dataset_str == 'GLASS':
        glass_df = pd.read_csv("../data/data-sets/glass.csv", header=None)
        X = glass_df.loc[:, 0:8].to_numpy()
        labels_true = glass_df.loc[:, 9].to_numpy()

    elif dataset_str == 'ECOLI':
        ecoli_df = pd.read_csv("../data/data-sets/ecoli.csv", header=None)
        X = ecoli_df.loc[:, 0:6].to_numpy()
        pca = decomposition.PCA(n_components=5)
        pca.fit(X)
        X = pca.transform(X)
        labels_true = ecoli_df.loc[:, 7].to_numpy()
        X = X[:327, :]
        labels_true = labels_true[:327]

    elif dataset_str == 'VERTEBRAL':
        vertebral_df = pd.read_csv("../data/data-sets/vertebral.data",
                                   skiprows=[0], header=None, delimiter=" ")
        X = vertebral_df.loc[:, 0:5].to_numpy()
        labels_df = pd.read_csv("../data/reference-labelling/vertebral.ref",
                                skiprows=[0], header=None, delimiter=" ")
        labels_true = labels_df.to_numpy()

    elif dataset_str == 'SEGMENTATION':
        segmentation_df = pd.read_csv("../data/data-sets/segmentation.data",
                                      skiprows=[0], header=None, delimiter=" ")
        X = segmentation_df.loc[:, 0:4].to_numpy()
        labels_df = pd.read_csv("../data/reference-labelling/segmentation.ref",
                                skiprows=[0], header=None, delimiter=" ")
        labels_true = labels_df.to_numpy()

    elif dataset_str == 'USER':
        user_df = pd.read_csv("../data/data-sets/user.data",
                              skiprows=[0], header=None, delimiter=" ")
        X = user_df.loc[:, 0:4].to_numpy()
        labels_df = pd.read_csv("../data/reference-labelling/user.ref",
                                skiprows=[0], header=None, delimiter=" ")
        labels_true = labels_df.to_numpy()

    else:
        dataGenerator = dataGen()
        P, V_true, X = dataGenerator.generateCircles()
        labels_true = partition_to_labels(V_true)

        
    return X, labels_true