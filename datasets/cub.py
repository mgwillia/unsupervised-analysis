"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
from PIL import Image
from torch.utils.data import Dataset

class CUB(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(CUB, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train
        self.imgNames = []
        self.classes = []

        with open(root + 'classes.txt', 'r') as classesFile:
            for line in classesFile.readlines():
                self.classes.append(line.strip().split(' ')[1].split('.')[1])

        isTrainList = []
        with open(root + 'train_test_split.txt', 'r') as splitFile:
            for line in splitFile.readlines():
                isTrainList.append(int(line.strip().split(' ')[1]))

        classLabels = []
        with open(root + 'image_class_labels.txt', 'r') as labelsFile:
            for line in labelsFile.readlines():
                classLabels.append(int(line.strip().split(' ')[1]) - 1)

        imagePaths = []
        with open(root + 'images.txt', 'r') as imagesFile:
            for line in imagesFile.readlines():
                imagePaths.append(root + 'images/' + line.strip().split(' ')[1])

        self.imagePaths = []
        self.classLabels = []
        for i, isTrain in enumerate(isTrainList):
            if isTrain == train:
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
