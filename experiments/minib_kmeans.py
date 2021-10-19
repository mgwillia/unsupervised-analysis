### train_pca, transform_pca and cluster_data taken from https://github.com/Randl/kmeans_selfsuper

import time
import torch
import argparse

import numpy as np
from sklearn import cluster
from sklearn.decomposition import IncrementalPCA
from sklearn.utils import shuffle

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def train_pca(train_features):
    bs = max(4096, train_features.shape[1] * 2)
    transformer = IncrementalPCA(batch_size=bs)
    for i in range(0, train_features.shape[0], bs):
        end_index = i + bs
        if end_index > train_features.shape[0]:
            end_index = train_features.shape[0]
        batch = train_features[i:end_index]
        transformer = transformer.partial_fit(batch)
    print(transformer.explained_variance_ratio_.cumsum())
    return transformer


def transform_pca(features, transformer):
    bs = max(4096, features.shape[1] * 2)
    for i in range(0, features.shape[0], bs):
        end_index = i + bs
        if end_index > features.shape[0]:
            end_index = features.shape[0]
        features[i:end_index] = transformer.transform(features[i:end_index])
    print(features.shape)
    return features[:,:int(features.shape[1]/2)]


def cluster_data(train_features, val_features, num_clusters):
    bs = max(2048, int(2 ** np.ceil(np.log2(num_clusters))))
    minib_k_means = cluster.MiniBatchKMeans(n_clusters=num_clusters, batch_size=bs, max_no_improvement=None)

    for _ in range(60):
        train_features = shuffle(train_features)
        for i in range(0, train_features.shape[0], bs):
            end_index = i + bs
            if end_index > train_features.shape[0]:
                end_index = train_features.shape[0]
            batch = train_features[i:end_index]
            minib_k_means = minib_k_means.partial_fit(batch)

        train_pred = minib_k_means.predict(train_features)
        val_pred = minib_k_means.predict(val_features)

    return train_pred, val_pred


def main():
    ### SET UP ARG PARSER ###

    parser = argparse.ArgumentParser(description='KMeans')
    parser.add_argument('--backbone', default='', type=str, help='backbone path')
    parser.add_argument('--dataset', default='imagenet', type=str, help='datset name: cub or imagenet')
    parser.add_argument('--num-clusters',  default=1000, type=int, help='num clusters')
    args = parser.parse_args()
    backbone_name = args.backbone.split('/')[-1]

    ##########################

    features_path = '/vulcanscratch/mgwillia/vissl/features/' + '_'.join([backbone_name, args.dataset, 'features']) + '.pth.tar'
    features = torch.load(features_path)

    train_features = features['train_features'].numpy()
    val_features = features['val_features'].numpy()

    start_time = time.time()
    transformer = train_pca(train_features)
    train_features, val_features = transform_pca(train_features, transformer), transform_pca(val_features, transformer)
    end_time = time.time()

    print(f'PCA train + transform time: {end_time - start_time} seconds')

    start_time = time.time()
    labels, predictions = cluster_data(train_features, val_features, args.num_clusters)
    end_time = time.time()

    print(f'KMeans fit+predict time: {end_time - start_time} seconds')

    file_name = '/vulcanscratch/mgwillia/vissl/clusters/' + '_'.join([backbone_name, args.dataset, 'minib_labels']) + '.npy'
    print(f'Saving labels to {file_name}')
    np.save(file_name, labels)

    file_name = '/vulcanscratch/mgwillia/vissl/clusters/' + '_'.join([backbone_name, args.dataset, 'minib_predictions']) + '.npy'
    print(f'Saving predictions to {file_name}')
    np.save(file_name, predictions)


if __name__ == '__main__':
    main()
