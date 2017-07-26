import matplotlib.pyplot as plt
from PIL import Image
import torch
import  torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam,SGD
from torch.autograd import Variable
import torch.nn.functional as F

from network import AlexNet,TwoLayerNet,Vgg16Net,Vgg19Net,Cifar10FullNet
from dataset import CIFAR10DATA


def imageshow(images,labels):
    '''Input image numpy : shape=(N,3,32,32)'''
    for i in range(images.shape[0]):
        imgs = images[i-1]
        i0 = Image.fromarray(imgs[0])
        i1 = Image.fromarray(imgs[1])
        i2 = Image.fromarray(imgs[2])
        img = Image.merge('RGB',(i0,i1,i2))
        plt.imshow(img)
        plt.title(label_table[labels[i-1]])
        plt.show()

class MyToTensor(object):
    def __call__(self, pic):
        img = torch.from_numpy(pic.astype('float32')/255)
        return img


def train():
    input_transform = transforms.Compose([MyToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    data = CIFAR10DATA(root, 'train',transform=input_transform)
    loader = DataLoader(data,batch_size=batch_size,shuffle=True)
    if network == 'AlexNet':
        model = AlexNet()
    if network == 'TwoLayerNet':
        model = TwoLayerNet()
    if network == 'Vgg16Net':
        model = Vgg16Net()
    if network == 'Vgg19Net':
        model = Vgg19Net()
    if network == 'Cifar10FullNet':
        model = Cifar10FullNet()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters())
    for epoch in range(1,epochs):
        running_loss = 0.0
        for step,data in enumerate(loader):
            images, labels = data
            inputs = Variable(images.cuda())
            targets = Variable(labels.cuda())
            optim.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optim.step()

            running_loss += loss.data[0]

            if (step+1) % printLoss == 0:
                print('epochs:%d,step:%d,loss:%f' % (epoch,step+1,running_loss/(step+1)))
                running_loss = 0.0
            if (step+1) % stepSave == 0:
                filename = 'models'+'/'+network+'/'+network+'-' + str(epoch) + '-' + str(step+1) + '.pth'
                torch.save(model.state_dict(),filename)
                print('save:'+filename)

def test():
    input_transform = transforms.Compose([MyToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = CIFAR10DATA(root, 'test',transform=input_transform)
    loader = DataLoader(data,batch_size=batch_size)

    if network == 'AlexNet':
        model = AlexNet()
    if network == 'TwoLayerNet':
        model = TwoLayerNet()
    if network == 'Vgg16Net':
        model = Vgg16Net()
    if network == 'Vgg19Net':
        model = Vgg19Net()
    if network == 'Cifar10FullNet':
        model = Cifar10FullNet()

    model.cuda()
    model.load_state_dict(torch.load(test_models))
    model.eval()

    correct = 0.0
    total = 0.0
    for i,(images,labels) in enumerate(loader):
        inputs = Variable(images.cuda())

        outputs = model(inputs)
        outputs = F.softmax(outputs)
        outputs = outputs.cpu().data.numpy().argmax(1)
        labels=labels.numpy()
        correct += sum(labels == outputs)
        total += labels.shape[0]
    accuracy = correct/total
    print("Accuracy(%s):%f"% (test_models,accuracy))

def main():
    if mod == 'train':
        train()
    if mod == 'test':
        test()

if __name__ == '__main__':

    root = '../dataset/cifar-10-batches-py'
    label_table = ['airplane', 'automoile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    mod = 'train'
    network = 'AlexNet'
    epochs = 10000
    batch_size=100
    stepSave=500
    printLoss = 20
    test_models = 'models/AlexNet/AlexNet-50-5000.pth'

    main()





