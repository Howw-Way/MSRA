<!-- Author: Howw -->
<!-- Data: 22.8.30 -->
# Torch 101: my first lesson to torch

This is my note for learning torch, based on the official website. 

## 1. DataLoader

在pytorch中，提供了很多现成的数据集，有声音的、图像的、文字的。pytorch中有两个模块，`torch.utils.data.DataLoader`和`torch.utils.data.Dataset`，这两个模块支持使用torch内置的数据，以及用户自定义的数据。

**Dataset**: 存储样本和它们对应的标签，类似一个元组，使用方法类似元组，个人认为设计为元组的原因是防止数据被修改。
例如Dataset[0]就是存的数据，而Dataset[1]则是数据对应的label，而len(Dataset)，则是其中储存的数据/label的条数

**DataLoader**:在`Dataset`周围包装一个可迭代对象，以便方便地访问样本（相当于能方便的访问多个dataset）。将dataset包裹后，可以用for循环去遍历
(其超参数较多，例如dataset, batch_size,以及num_workers可以有效在大数据时加速，进行预缓存，遇到了需求再对应学习)

### 1.1 获取torch现成的data

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

#### 1.1.1 参数说明：
FashionMNIST是`datasets`中内置的数据集，可以方便导入，其中：

- `root` 指定路径
- `train` 是否是训练集，TRUE：训练集，Flase: 测试集
- `download` 是否需要下载
- `transformer` 训练集数据使用的transformer
- `target_transformer`测试集数据使用的transformer

#### 1.1.2 数据说明：
dataset中的数据是通过tuple将数据和标签组织起来的（推测好处在于可以保证数据不会被更改，tuple特性）
以training_data为例，其实本质是(data,label)，对于该例子，有60000条，而每条中的datashape是(1,28,28)，label则是一个int

>In[0]: len(training_data)

>Out[0]: 60000
有60000条数据在training_data中

获取数据，在第一个位置
>In[1]: training_data[0].shape

>Out[1]: torch.Size([1, 28, 28])


### 1.2 客制化 Creating custom dataset

通过用户自定义类，实现dataset，包含三个函数：`__init__`,`__len__`,`__getitem`，最终效果是`dataset`可以`getitem`（即可以通过index获取对应位置的数据）以及获取`len`

#### 1.2.1 个人case

```python
class FigureDataset(Dataset):
    def __init__(self):
        self.Data=torch.tensor([[1,1],[2,2],[3,3]])
        self.Label=torch.tensor([9,8,7])
    def __getitem__(self, index):
        image=self.Data[index]
        label=self.Label[index]
        return image, label
    def __len__(self):
        return len(self.Data)
```

在这个Class中，数据和label均通过直接定义的形式给出，`init` 函数的意义就在于初始化数据以及transformer等，随后通过`__getitem__`函数重构了[index]获取数据及label的方式，而`__len__`函数则是简单的获取Dataset具体长度的（数据条数），通过len(data)或者len(label)均可

#### 1.2.2 官方case

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

##### 1.2.2.1 子函数解释

读取的数据示意：
img_labels:
```python
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

1. __init__

`__init__`函数是初始化，需要包含图像文件、标签文件、以及指定对数据和标签的两种transformer格式（即对图像文件进行预处理的方式）

2. __len__

`__len__`返回数据集中的sample数目
（后续是否有执行检查数据与标签大小是否相等的内容？）

3. __getitem__

`__getitem__`函数通过`idx`从数据集中读取并返回sample，`idx`主要是用于在磁盘定位数据，再使用`read_image`将数据变成tensor。

可以看到`img_path`是一个组合结果，`img_dir`是图像位置，而`img_labels.iloc[idx, 0]`则是图像具体的名称.（`img_labels`如下所示）


### 1.3 DataLoader



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
