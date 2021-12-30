import sys

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main(csv_path):
    df = pd.read_csv(csv_path)

    # Separating out the features
    X = df[df.columns[:-1]]
    # Separating out the target
    Y = df[df.columns[-1]]

    # Standardizing the features
    X_standard = StandardScaler().fit_transform(X)

    pca = PCA(n_components='mle')
    pca.fit_transform(X_standard)
    pca_1 = pd.DataFrame(pca.components_, columns=X.columns)

    output = pca_1.copy()
    output['explained_variance_ratio'] = pca.explained_variance_ratio_
    print(output)

    X_standard = pd.DataFrame(X_standard, index=X.index, columns=X.columns)
    X_standard = X_standard.drop(
        pca_1.columns[pca_1.apply(lambda col: col[0] < 0)], axis=1)

    pca = PCA(n_components='mle')
    pca.fit_transform(X_standard)
    pca_2 = pd.DataFrame(pca.components_, columns=X_standard.columns)

    output = pca_2.copy()
    output['explained_variance_ratio'] = pca.explained_variance_ratio_
    print(output)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('ERROR! Input the csv file path, ie. `python3 pca.py data.csv`.')
    main(sys.argv[1])
