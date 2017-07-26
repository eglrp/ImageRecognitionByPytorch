from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import torch


class MnistData(Dataset):
    def __init__(self,root,mod='train',transform = None,target_transform = None):
        DataFrame = pd.read_csv(os.path.join(root,'train.csv'))
        self.filename = list(DataFrame.filename)
        self.input_trainsform = transform
        self.target_transform = target_transform
        self.mod = mod
        images = [os.path.join(root, 'train', target) for target in self.filename]
        labels = [int(target) for target in list(DataFrame.label)]
        split = int(0.8*len(images))
        if self.mod == 'train':
            self.images, self.labels = images[0:split], labels[0:split]
        if self.mod == 'test':
            self.images, self.labels = images[split:len(images) - 1], labels[split:len(images) - 1]

    def __getitem__(self, index):

            path = self.images[index]
            target = self.labels[index]
            img = Image.open(path).convert('L')
            if self.input_trainsform is not None:
                img = self.input_trainsform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img,target

    def __len__(self):
        if self.mod == 'train':
            return len(self.images)
        if self.mod == 'test':
            return len(self.labels)

class ConvNetForMnist(nn.Module):
    def __init__(self):
        super(ConvNetForMnist,self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(120, 10, 1)
        )


    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)

        return x

def train():

    data = MnistData(dataPath,'train',ToTensor())

    model = ConvNetForMnist()
    model = model.cuda()

    loader = DataLoader(data, batch_size=100)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters())

    for epoch in range(10000):
        epoch_loss = 0.0
        for step, (images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()

            input = Variable(images)
            target = Variable(labels)

            output = model(input)
            optim.zero_grad()

            loss = criterion(output,target)
            loss.backward()
            optim.step()

            epoch_loss += loss.data[0]
            if (step+1) % 20 == 0:
                print('epoch:%d,step:%d,loss:%f' % (epoch, step+1, epoch_loss/100))
                epoch_loss = 0.0
            if (step+1) % 392 == 0:
                filename = 'models/mnist-' + str(epoch) + '-' + str(step) + '.pth'
                torch.save(model.state_dict(), filename)
                print('save: ' + filename + '(epoch: ' + str(epoch) + ', step: ' + str(step) + ')')

def test():
    data = MnistData(dataPath, 'test',ToTensor())

    model = ConvNetForMnist()
    model = model.cuda()
    model.load_state_dict(torch.load(testPath))
    model.eval()

    loader = DataLoader(data, batch_size=10)

    correct = 0.0
    total = 0.0
    for step, (images, labels) in enumerate(loader):
        images = images.cuda()
        input = Variable(images)
        output = model(input)
        output = F.softmax(output).data.cpu().numpy().argmax(1)

        labels = labels.numpy()
        correct += sum(labels == output)
        total += labels.shape[0]

    accuracy = correct/total
    print('Accuracy:%f' % accuracy)


def main():
    if mod == 'train':
        train()
    elif mod == 'test':
        test()
    else :
        print('Please make sure the input mod is right')
if __name__ == '__main__':
    dataPath = '../dataset/mnist'
    mod = 'train'
    testPath = 'models/mnist-85-391.pth'
    main()

