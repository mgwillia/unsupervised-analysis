"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
from PIL import Image
import scipy.io
from torch.utils.data import Dataset

class OxfordFlowers(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(OxfordFlowers, self).__init__()
        self.root = root
        self.transform = transform

        labels = scipy.io.loadmat(root + 'imagelabels.mat')['labels'][0]

        self.imagePaths = []
        self.classLabels = []

        if train:
            split = scipy.io.loadmat(root + 'setid.mat')['trnid'][0]
            for i in range(split.shape[0]):
                imgId = split[i]
                label = labels[imgId - 1]
                self.imagePaths.append(root + 'jpg/image_' + str(imgId).zfill(5) + '.jpg')
                self.classLabels.append(label)
            
            split = scipy.io.loadmat(root + 'setid.mat')['valid'][0]
            for i in range(split.shape[0]):
                imgId = split[i]
                label = labels[imgId - 1]
                self.imagePaths.append(root + 'jpg/image_' + str(imgId).zfill(5) + '.jpg')
                self.classLabels.append(label)
        else:
            split = scipy.io.loadmat(root + 'setid.mat')['tstid'][0]
            for i in range(split.shape[0]):
                imgId = split[i]
                label = labels[imgId - 1]
                self.imagePaths.append(root + 'jpg/image_' + str(imgId).zfill(5) + '.jpg')
                self.classLabels.append(label)
        
    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        img = Image.open(self.imagePaths[index]).convert("RGB")
        target = self.classLabels[index]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target}
        
        return out
