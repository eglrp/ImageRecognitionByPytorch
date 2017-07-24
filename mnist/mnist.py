from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd
from PIL import Image
import numpy
from torchvision.transforms import ToTensor
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

class denseToOneHot(object):
    """Convert class labels from scalars to one-hot vectors."""
    def __init__(self,num_classes=10):
        self.num_classes = num_classes

    def __call__(self,labels_dense):
        labels_one_hot = numpy.zeros(10)
        labels_one_hot[labels_dense]=1
        return labels_one_hot

class MnistData(Dataset):
    def __init__(self,root,mod='train',transform = None,target_transform = None):
        DataFrame = pd.read_csv(os.path.join(root,'train.csv'))
        self.filename = list(DataFrame.filename)
        self.labels = [int(target) for target in list(DataFrame.label)]
        self.images = [os.path.join(root,'train',target)  for target in self.filename]
        self.input_trainsform = transform
        self.target_transform = target_transform
        self.mod = mod
        self.split = int(0.8*len(self.images))
        self.train_path, self.train_target = self.images[0:self.split], self.labels[0:self.split]
        self.test_path, self.test_target = self.images[self.split:len(self.images) - 1], self.labels[self.split:len(self.images) - 1]

    def __getitem__(self, index):
        if self.mod == 'train':
            path = self.train_path[index]
            target = self.train_target[index]
            img = Image.open(path).convert('L')
            if self.input_trainsform is not None:
                img = self.input_trainsform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img,target
        if self.mod == 'test':
            path = self.test_path[index]
            target = self.test_target[index]
            img = Image.open(path).convert('L')
            if self.input_trainsform is not None:
                img = self.input_trainsform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img,target

    def __len__(self):
        if self.mod == 'train':
            return len(self.train_path)
        if self.mod == 'test':
            return len(self.test_path)

class ConvNetForMnist(nn.Module):
    def __init__(self):
        super(ConvNetForMnist,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(6,16,5)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(16,120,5)
        self.relu4 = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(120,10,1)

    def forward(self,x):
        conv1 = self.conv1(x)
        conv1_relu = self.relu1(conv1)
        pool2 = self.pool2(conv1_relu)
        conv3 = self.conv3(pool2)
        conv3_relu = self.relu3(conv3)
        pool4 = self.pool4(conv3_relu)
        conv4 = self.conv4(pool4)
        conv4_relu = self.relu4(conv4)
        out = self.out(conv4_relu)

        return out

def train():

    data_train = MnistData(dataPath, transform=input_transforms, target_transform=target_transforms)

    model = ConvNetForMnist()
    model = model.cuda()

    loader = DataLoader(data_train, batch_size=1)

    optim = Adam(model.parameters())

    for epoch in range(30):
        epoch_loss = []

        for step, (images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()

            input = Variable(images)
            target = Variable(labels).type(torch.cuda.FloatTensor)

            output = model(input)
            optim.zero_grad()
            output = output.resize(1,10)
            softmax = F.log_softmax(output)
            target = target.resize(10,1)
            loss = -softmax.mm(target)
            loss.backward()

            optim.step()

            epoch_loss.append(loss.data[0])
            average_loss = sum(epoch_loss) / len(epoch_loss)
            if step % 100 == 0:
                print('epoch:%d,step:%d,loss:%f,a_loss:%f\n' % (epoch, step, loss.data[0].type(torch.FloatTensor).numpy(),average_loss.type(torch.FloatTensor).numpy()))
            #    a=F.softmax(output).data[0].type(torch.FloatTensor).numpy()
            #    b=target.resize(1,10).data[0].type(torch.FloatTensor).numpy()
            #    print(a,b)
            if step %1000 == 0:
                filename = 'mnist-' + str(epoch) + '-' + str(step) + '.pth'
                torch.save(model.state_dict(), filename)
                print('save: ' + filename + '(epoch: ' + str(epoch) + ', step: ' + str(step) + ')')

def test():
    data_test = MnistData(dataPath, 'test', transform=input_transforms, target_transform=target_transforms)

    model = ConvNetForMnist()
    model = model.cuda()
    model.load_state_dict(torch.load('mnist-0-1000.pth'))
    model.eval()

    loader = DataLoader(data_test, batch_size=1)

    count = []
    for step, (images, labels) in enumerate(loader):
        images = images.cuda()
        input = Variable(images)
        output = model(input)
        softmax_out = F.softmax(output.resize(1, 10))
        out_numpy = softmax_out.data.type(torch.FloatTensor).numpy()
        label_pred = out_numpy.argmax()
        label_truth = labels.numpy().argmax()
        if label_pred == label_truth:
            count.append(1)
        else:
            count.append(0)
    accuracy = float(sum(count))/float(len(count))
    print('Accuracy:%f' % accuracy)
'''
    for i in range (10):
        path = images_list[i]
        img = Image.open(path).convert('L')
        plt.figure('digital')
        plt.imshow(img, cmap=plt.cm.gray_r)
        plt.show()
        model = ConvNetForMnist()
        model = model.cuda()
        model.load_state_dict(torch.load('mnist-0-48000.pth'))
        model.eval()

        input = Variable(input_transforms(img).cuda()).resize(1, 1, 28, 28)
        output = model(input)
        softmax_out = F.softmax(output.resize(1, 10))
        out_numpy = softmax_out.data.type(torch.FloatTensor).numpy()
        label = out_numpy.argmax()
        print(label)
'''


def main(mod='train'):
    if mod == 'train':
        train()
    elif mod == 'test':
        test()
    else :
        print('Please make sure the input mod is right')
if __name__ == '__main__':
    dataPath = '../dataset/mnist'
    input_transforms = ToTensor()
    target_transforms = denseToOneHot(10)
    mod = 'test'
    main(mod)

