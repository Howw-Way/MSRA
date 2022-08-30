<!-- Author: Howw -->
<!-- Data: 22.8.30 -->
# Torch 101: my first lesson to torch

This is my note for learning torch, the reference is the official website. 

## 1. DataLoader

在pytorch中，提供了很多现成的数据集，有声音的、图像的、文字的。pytorch中有两个模块，`torch.utils.data.DataLoader`和`torch.utils.data.Dataset`，这两个模块支持使用torch内置的数据，以及用户自定义的数据。

`Dataset`: 存储样本和它们对应的标签，类似一个数组，使用方法类似
`DataLoader`:在`Dataset`周围包装一个可迭代对象，以便方便地访问样本（相当于能方便的访问多个dataset）。将dataset包裹后，可以用for循环去遍历
(dataset, batch_size,num_workers可以有效在大数据时加速，进行预缓存)

### 1.1 Loading a dataset

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

```

显然，FashionMNIST是`datasets`中内置的数据集，可以方便导入，其中：

- `root` 指定路径
- `train` 是否是训练集，TRUE：训练集，Flase: 测试集
- `download` 是否需要下载
- `transformer` 训练集数据使用的transformer
- `target_transformer`测试集数据使用的transformer

### 1.2 Creating custom dataset

最终效果是`dataset`可以`getitem`，可以通过index获取对应位置的数据

需要定义一个类去实现，包含三个函数：`__init__`,`__len__`,`__getitem`

```python
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

#是Dataset的子类
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

1. __init__

`__init__`函数是初始化，需要包含图像文件、标签文件、以及指定对数据和标签的两种transformer格式（即对图像文件进行预处理的方式）

2. __len__

`__len__`返回数据集中的sample数目
（后续是否有执行检查数据与标签大小是否相等的内容？）

3. __getitem__

`__getitem__`函数通过`idx`从数据集中读取并返回sample，`idx`主要是用于在磁盘定位数据，再使用`read_image`将数据变成tensor。

可以看到`img_path`是一个组合结果，`img_dir`是图像位置，而`img_labels.iloc[idx, 0]`则是图像具体的名称.（`img_labels`如下所示）

img_labels:
```python
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

## 2. Model defination

在pytorch中，每个使用的NN都是用户自定义的子类(继承于父类`torch .nn`中的模块`nn.Module`)，而用户自定义的子类本身已是一个模块，需要有其他层等组成。

### 2.1 Network defination

```python 
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        #super is a function for better inheritation of subclasses from their parent
        super(NeuralNetwork, self).__init__()
        #flatten is reshape(N,1)
        self.flatten = nn.Flatten()
        #setting the neural network
        self.linear_relu_stack = nn.Sequential(
            #input layer, first size = h*w(for one channel)
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            #output layer, last size =shape(y)
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

```
Forward function, according to the offcial website of torch[URL](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html), it's highly suggested not to dierctly call `model. forward()`, and every subclass in `nn.Module` implements the operations on input data in the forward method


```python
model = NeuralNetwork().to(device)
print(model)
```
