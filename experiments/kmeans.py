import time
import torch
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans


def main():
    ### SET UP ARG PARSER ###

    parser = argparse.ArgumentParser(description='KMeans')
    parser.add_argument('--backbone', default='', type=str, help='backbone path')
    parser.add_argument('--dataset', default='imagenet', type=str, help='datset name: cub or imagenet')
    parser.add_argument('--num-clusters',  default=1000, type=int, help='num clusters')
    parser.add_argument('--num-cores', default=64, type=int)
    args = parser.parse_args()
    backbone_name = args.backbone

    ##########################

    num_inits = 10

    features_path = '/vulcanscratch/mgwillia/vissl/features/' + '_'.join([backbone_name, args.dataset, 'features']) + '.pth.tar'
    features = torch.load(features_path)

    train_features = torch.nn.functional.normalize(features['train_features'], p=2, dim=1).numpy()
    val_features = torch.nn.functional.normalize(features['val_features'], p=2, dim=1).numpy()

    if args.dataset == 'imagenet' or args.dataset == 'imagenet_200' or args.dataset == 'imagenet_50':
        start_time = time.time()
        kmeans = MiniBatchKMeans(n_clusters=args.num_clusters, init='k-means++', n_init=num_inits, batch_size=args.num_cores*256).fit(train_features)
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
