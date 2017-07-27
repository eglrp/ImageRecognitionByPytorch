## 用卷积网络训练 [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) 

### 1 数据集 [cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)
Python 版本cifar10数据集[下载](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)，解压后：

```
- cifar-10-batches-py/batches.meta
- cifar-10-batches-py/data_batch_1
- cifar-10-batches-py/data_batch_2
- cifar-10-batches-py/data_batch_3
- cifar-10-batches-py/data_batch_4
- cifar-10-batches-py/data_batch_5
- cifar-10-batches-py/test_batch

训练集50000张图片，测试集10000张图片
```


### 2 网络结构 [network.py](https://github.com/songh1024/PyTorchLearning/blob/master/cifar10/network.py)
- [x] TwoLayerNet:两个卷积层，三个全连接层
- [x] AlexNet：[AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)网络框架
- [x] Cifar10FullNet：Caffe中的[cifar10_full_train_test](https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_train_test.prototxt)结构
- [x] [VGG16](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)
- [x] [VGG19](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md)

### 3 训练和测试（以AlexNet为例）

[main.py](https://github.com/songh1024/PyTorchLearning/blob/master/cifar10/main.py)
```
1. root:数据集所在路径：如'../dataset/cifar-10-batches-py'
2. mod='train' :选择训练还是测试
3. network='AlexNet':网络结构
4. epochs=10000：训练次数
5. batch_size=100:batch大小，训练一次总步长为:50000 / batch_size=500
6. stepSave=500:每隔多少步保存一次，保存路径为models/AlexNet/xxx.pth
7. printLoss=20:每隔多少步长打印一次loss
8. test_models='models/AlexNet/AlexNet-50-5000.pth',测试时参数路径
```
### 4 结果
由于时间原因，训练次数较少，参数不一定时最优的

网络 | TwoLayerNet | Cifar10FullNet | AlexNet | VGG16 | VGG19
---|---|---|--- |--- |---
准确率 | 63.16% | 75.97% | 62.86% | 暂无 |78.73%


```
格式：网络-训练次数-步长.pth
TwoLayerNet：TwoLayerNet-15-5000.pth
Cifar10FullNet:Cifar10FullNet-150-500
AlexNet:AlexNet-50-5000.pth
VGG16:暂无
VGG19:models/Vgg19Net/Vgg19Net-42-500.pth

```
训练好的模型可以从[这里](http://pan.baidu.com/s/1o8lxUPk)下载
