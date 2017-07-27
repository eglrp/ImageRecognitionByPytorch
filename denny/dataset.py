from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class myResize(object):
    """Rescales the input PIL.Image to the given 'size'.
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        else:
            return img.resize((self.size, self.size), self.interpolation)

class DennyDATA(Dataset):
    def __init__(self,root,mod='train',input_transform=None,target_transform=None):
        self.root = root
        self.mod = mod
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.filename = self.root+'/'+self.mod+'.csv'
        file = pd.read_csv(self.filename)
        imagepaths = [os.path.join(self.root,self.mod,name) for name in file.filename]
        self.labels = file.label
        self.images = [Image.open(path) for path in imagepaths]


    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image,label

    def __len__(self):
        return len(self.images)