## 用卷积网络训练mnist
### 1 框架：[PyTorch](http://pytorch.org/)

```
Python版本：2.7.12
模块：torch，torchvision，cPickle，numpy，PIL

```

### 2 数据集可以从[这里](https://pan.baidu.com/s/1jImSbps)下载

```
dataset:
    digital images(28*28):
        root/train/0.png,
        root/train/1.png,
        root/train/2.png,
        ......
    train:root/train.csv
        filename,label
        0.png,4
        1.png,9
        2.png,1
        ... , ...
        ... , ...
```

### 3 网络结构

```
    (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (relu1): ReLU (inplace)
    (pool2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (conv3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (relu3): ReLU (inplace)
    (pool4): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (conv4): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
    (relu4): ReLU (inplace)
    (out): Conv2d(120, 10, kernel_size=(1, 1), stride=(1, 1))

```
### 训练和测试

```
dataPath:数据集所在路径，如 ../dataset/mnist
mod:训练或测死,train,test
testPath:训练好的模型，models/mnist-85-391.pth

```
### 识别准确率
accuracy=98.96%
