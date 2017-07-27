import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,RandomHorizontalFlip,Normalize,Compose
import dataset,network

def train():
    data = dataset.DennyDATA(root,mod,input_transform)
    if net == 'Cifar10FullNet':
        model = network.Cifar10FullNet()
    if net == 'BvlcAlexNet':
        model = network.BvlcAlexNet()
    model = model.cuda()
    loader = DataLoader(data,batch_size=10,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters())

    for epoch in range(1,epochs):
        total_loss = 0.0
        for step,(image,label) in enumerate(loader):
            input = Variable(image.cuda())
            target = Variable(label.cuda())
            output = model(input)

            optim.zero_grad()
            loss = criterion(output,target)
            loss.backward()

            optim.step()

            total_loss += loss.data[0]
            if (step+1) % 10 == 0:
                print('epoch:%d,step:%d,loss:%f'%(epoch,step+1,total_loss/10))
                total_loss = 0.0
            if (step+1) % 40 == 0:
                filename = 'models/'+net+'/'+net+'-'+str(epoch)+'-'+str(step+1)+'.pth'
                torch.save(model.state_dict(),filename)
                print('save'+filename)

def test():
    data = dataset.DennyDATA(root,'test',input_transform)
    if net == 'Cifar10FullNet':
        model = network.Cifar10FullNet()
    if net == 'BvlcAlexNet':
        model = network.BvlcAlexNet()
    model.load_state_dict(torch.load(testfile))
    model = model.eval()
    model = model.cuda()
    loader = DataLoader(data, batch_size=10)
    correct = 0.0
    total = 0.0
    for i, (image,label) in enumerate(loader):
        input = Variable(image.cuda())
        output = model(input)
        output = output.data.cpu().numpy().argmax(1)
        correct += sum(output == label.numpy())
        total += len(label)
    accuraacy = correct/total
    print(net+' Accuracy:'+str(accuraacy))
def main():
    if mod == 'train':
        train()
    if mod == 'test':
        test()
if __name__ == "__main__":
    root = '../dataset/denny'
    mod = 'train'
    net = 'Cifar10FullNet'
    testfile = 'models/Cifar10FullNet/Cifar10FullNet-300-40.pth'
    epochs = 10000
    input_transform = Compose([
        dataset.myResize(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    main()
