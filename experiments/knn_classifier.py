"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
import torch.nn.functional as F


def nearest_neighbor_test(temperature, num_neighbors, normalize, num_classes, features, labels):
    print(f"Testing with sigma: {temperature}, topk neighbors: {num_neighbors}, normalize: {normalize}")

    ############################################################################
    # Step 1: get train and test features
    train_features = features['train_features']
    test_features = features['val_features']

    if normalize:
        train_features = F.normalize(train_features, p=2, dim=1)
        test_features = F.normalize(test_features, p=2, dim=1)

    train_features = train_features.numpy()
    test_features = test_features.numpy()

    train_labels = labels['train_targets'].numpy()
    test_labels = labels['val_targets'].numpy()

    train_features = torch.from_numpy(train_features).float().cuda().t()
    train_labels = torch.LongTensor(train_labels).cuda()
    ###########################################################################
    # Step 2: calculate the nearest neighbor and the metrics
    top1, top5, total = 0.0, 0.0, 0
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    print(f'Images per chunk: {imgs_per_chunk}')
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(num_neighbors, num_classes).cuda()
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images and normalize the features if needed
            features = test_features[
                idx : min((idx + imgs_per_chunk), num_test_images), :
            ]
            targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]#, :]
            batch_size = targets.shape[0]
            features = torch.from_numpy(features).float().cuda()
            targets = torch.LongTensor(targets).cuda()

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features)
            distances, indices = similarity.topk(
                num_neighbors, largest=True, sorted=True
            )
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * num_neighbors, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(temperature).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()
            total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    print(f"Total images: {total}, Top1: {top1}, Top5: {top5}")
    return top1, top5


def main():

    ### SET UP PARSER ###
    parser = argparse.ArgumentParser(description='Get KNN Classifier top-1, top-5')
    parser.add_argument('--backbone', default='', type=str, help='backbone path')
    parser.add_argument('--dataset', default='imagenet', type=str, help='datset name: cub, imagenet, or something else')
    parser.add_argument('--temperature', default=0.1, type=float, help='temperature')
    parser.add_argument('--num-neighbors', default=200, type=int, help='topk for k nearest neighbors classifier')
    parser.add_argument('--normalize', action='store_true', help='normalize')
    args = parser.parse_args()

    print(f'Getting knn classifier accuracy for backbone path: {args.backbone}, dataset: {args.dataset}')

    ### SET UP DATASETS ###
    if args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'imagenet_50':
        num_classes = 50
    elif args.dataset == 'imagenet_200':
         num_classes = 200
    elif args.dataset == 'cub':
        num_classes = 200
    elif args.dataset == 'cars':
        num_classes = 196
    elif args.dataset == 'dogs':
        num_classes = 120
    elif args.dataset == 'flowers':
        num_classes = 102
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    ### LOAD FEATURES ###
    features_path = '/vulcanscratch/mgwillia/vissl/features/' + '_'.join([args.backbone, args.dataset, 'features']) + '.pth.tar'
    features = torch.load(features_path)
    targets = torch.load(f'/vulcanscratch/mgwillia/vissl/features/{args.backbone}_{args.dataset}_targets.pth.tar')

    ### CALL KNN CLASSIFIER ###
    nearest_neighbor_test(args.temperature, args.num_neighbors, args.normalize, num_classes, features, targets)

if __name__ == '__main__':
    main()
