"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
from PIL import Image
import scipy.io
from torch.utils.data import Dataset

class StanfordCars(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(StanfordCars, self).__init__()
        self.root = root
        self.transform = transform

        imagePaths = []
        classLabels = []
        isTestList = []
        annos = scipy.io.loadmat(root + 'cars_annos.mat')['annotations'][0]
        for i in range(annos.shape[0]):
            imagePaths.append(root + annos[i][0][0])
            classLabels.append(annos[i][5][0][0])
            isTestList.append(annos[i][6][0][0])


        self.imagePaths = []
        self.classLabels = []
        for i, isTest in enumerate(isTestList):
            if isTest != train:
                self.imagePaths.append(imagePaths[i])
                self.classLabels.append(classLabels[i])
        
    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        img = Image.open(self.imagePaths[index]).convert("RGB")
        target = self.classLabels[index]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target}
        
        return out
