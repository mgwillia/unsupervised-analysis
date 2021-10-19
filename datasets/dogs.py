"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
from PIL import Image
import scipy.io
from torch.utils.data import Dataset

class StanfordDogs(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(StanfordDogs, self).__init__()
        self.root = root
        self.transform = transform

        self.imagePaths = []
        self.classLabels = []

        if train:
            info = scipy.io.loadmat(root + 'train_list.mat')
        else:
            info = scipy.io.loadmat(root + 'test_list.mat')
        
        fileList = info['file_list']
        labels = info['labels']
        for i in range(fileList.shape[0]):
            filePath =fileList[i][0][0]
            label = labels[i][0]
            self.imagePaths.append(root + 'Images/' + filePath)
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
