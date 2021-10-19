import time
import torch
import argparse

import numpy as np

from sklearn.cluster import KMeans

import faiss
import numpy as np

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   verbose=True,
                                   gpu=True)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]


def main():
    ### SET UP ARG PARSER ###

    parser = argparse.ArgumentParser(description='KMeans')
    parser.add_argument('--backbone', default='', type=str, help='backbone path')
    parser.add_argument('--dataset', default='imagenet', type=str, help='datset name: cub or imagenet')
    parser.add_argument('--num-clusters',  default=1000, type=int, help='num clusters')
    args = parser.parse_args()
    backbone_name = args.backbone.split('/')[-1]

    ##########################

    num_inits = 10
    if args.dataset == 'imagenet':
        num_inits = 1

    features_path = '/vulcanscratch/mgwillia/vissl/features/' + '_'.join([backbone_name, args.dataset, 'features']) + '.pth.tar'
    features = torch.load(features_path)

    if 'scan' in backbone_name:
        train_features = torch.nn.functional.normalize(features['train_features'], p=2, dim=1).numpy()
        val_features = torch.nn.functional.normalize(features['val_features'], p=2, dim=1).numpy()

    else:
        train_features = features['train_features'].numpy()
        val_features = features['val_features'].numpy()

    if args.dataset == 'imagenet':
        start_time = time.time()
        kmeans = KMeans(n_clusters=args.num_clusters, init='k-means++', n_init=num_inits, algorithm='full').fit(val_features)
        #kmeans = FaissKMeans(n_clusters=args.num_clusters, n_init=num_inits)
        #kmeans.fit(train_features)
        labels = kmeans.labels_
        end_time = time.time()

        print(f'KMeans fit time: {end_time - start_time} seconds')

        file_name = '/vulcanscratch/mgwillia/vissl/clusters/' + '_'.join([backbone_name, args.dataset, 'clusters']) + '.npy'
        print(f'Saving labels to {file_name}')
        np.save(file_name, labels)

        """
        file_name = '/vulcanscratch/mgwillia/vissl/clusters/' + '_'.join([backbone_name, args.dataset, 'labels']) + '.npy'
        print(f'Saving labels to {file_name}')
        np.save(file_name, labels)

        file_name = '/vulcanscratch/mgwillia/vissl/clusters/' + '_'.join([backbone_name, args.dataset, 'predictions']) + '.npy'
        print(f'Saving predictions to {file_name}')
        np.save(file_name, predictions)
        """

    else:
        start_time = time.time()
        kmeans = KMeans(n_clusters=args.num_clusters, init='k-means++', n_init=num_inits, algorithm='full').fit(train_features)
        labels = kmeans.labels_
        end_time = time.time()

        print(f'KMeans fit time: {end_time - start_time} seconds')

        start_time = time.time()
        predictions = kmeans.predict(val_features)
        end_time = time.time()

        print(f'KMeans predict time: {end_time - start_time} seconds')

        file_name = '/vulcanscratch/mgwillia/vissl/clusters/' + '_'.join([backbone_name, args.dataset, 'labels']) + '.npy'
        print(f'Saving labels to {file_name}')
        np.save(file_name, labels)

        file_name = '/vulcanscratch/mgwillia/vissl/clusters/' + '_'.join([backbone_name, args.dataset, 'predictions']) + '.npy'
        print(f'Saving predictions to {file_name}')
        np.save(file_name, predictions)


if __name__ == '__main__':
    main()
