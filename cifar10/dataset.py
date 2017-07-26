from torch.utils.data import Dataset
import os,cPickle
import numpy as np

class CIFAR10DATA(Dataset):
    def __init__(self, root, mod='train',transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mod = mod
        if self.mod == 'train':
            self.datasetFiles = [os.path.join(self.root, filename) for filename in
                                 ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']]
            self.images, self.labels = self.getImagesFromFile(self.datasetFiles)
        elif self.mod == 'test':
            self.testFile = os.path.join(self.root, 'test_batch')
            self.images, self.labels = self.loadCifar10Batch(self.testFile)
        else:
            print("mod error")

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image,label
        # An image(Tensor) and a label(int)

    def __len__(self):
        return len(self.images)


    def getImagesFromFile(self,path):
        '''Input the ImagePath,get images and labels (numpy:N,3,32,32)'''
        images, labels = self.loadCifar10Batch(path[0])
        images_list, labels_list = list(images), list(labels)
        for i in range(1, len(path)):
            imgX, imgY = self.loadCifar10Batch(path[i])
            images_list += list(imgX)
            labels_list += list(imgY)
        return np.array(images_list),np.array(labels_list)

    def loadCifar10Batch(self,filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as fo:
            dict = cPickle.load(fo)
            X = dict['data']
            Y = dict['labels']
            X = X.reshape(10000, 3, 32, 32)
            Y = np.array(Y)
        return X, Y
