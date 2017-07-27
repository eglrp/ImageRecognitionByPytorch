##  用卷积网络训练自定义denny数据集
### 1 数据集可以从[这里](https://pan.baidu.com/s/1c3xwHS)下载

```
训练集：400张图片，测试集100张图片
dataset/train.csv
dataset/test.csv
dataset/classes.csv
dataset/README.md
images:
      dataset/train/320.jpg
      dataset/train/321.jpg
      ......
      dataset/test/300.jpg
      dataset/test/301.jpg
      ......
类别：
      truck,0
      dinosaur,1
      elephant,2
      flower,3
      horse,4
```
### 2 网络结构
- [BvlcAlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
- [Cifar10FullNet](https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_train_test.prototxt)

### 3 训练和测试
```
root:数据集所在路径，如 ../dataset/mnist
mod:训练或测试,train,test
net:网络结构
testfile:训练好的模型，models/Cifar10FullNet/Cifar10FullNet-300-40.pth
```

### 4 识别准确率

网络  | Cifar10FullNet | AlexNet 
---|---|--- 
准确率 | 97% | 92% 

训练好的模型可以从[这里](https://pan.baidu.com/s/1gfGKuEv)下载
