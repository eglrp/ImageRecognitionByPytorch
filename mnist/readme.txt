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

model:
    (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (relu1): ReLU (inplace)
    (pool2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (conv3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (relu3): ReLU (inplace)
    (pool4): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (conv4): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
    (relu4): ReLU (inplace)
    (out): Conv2d(120, 10, kernel_size=(1, 1), stride=(1, 1))

