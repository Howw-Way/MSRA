<!-- Author: Howw -->
<!-- Data: 22.8.30 -->
# Torch 101: my first lesson to torch

This is my note for learning torch, based on the official website, especially this one [URL](https://pytorch.org/tutorials/beginner/basics/intro.html) 

## 0. Tensor

### 0.1 基本介绍

1. Reference: 
Introduction: [URL](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
API: [URL](https://pytorch.org/docs/stable/torch.html)

2. tensor就是张量，是pytorch用来组织数据的数据格式，好处在于其能够用在GPU上进行加速，能用于自动微分。
同时tensor使用方式与numpy非常像，甚至会和numpy共享底层内存，即从numpy到tensor不需要复制数据。

3. Torch对于二维矩阵常用的写法是:[Channel_out(即输出的维度),Channel_in(进入的维度)]

### 0.2 常用函数与操作

#### 0.2.1 常用的函数：
`torch.from_numpy`,`torch.ones`,`torch.rand`,`torch.device`

#### 0.2.2 GPU运算
torch默认是在CPU上进行运算，通过`tensor.to()`进行迁移
```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

#### 0.2.2 重排（转置见下方）(reshape, squeeze,unsqueeze)

1. reshape()
改为指定维度，注意要对得上数据大小

2. squeeze()
将维度为1的维度删除
```python
a=torch.ones(1,3,3,1)
a.squeeze().shape#[3,3]
```

3. unsqueeze(dim)
在指定位置添加一个1维
```python
a=torch.ones(2,3,3)
b.unsqueeze(3).shape#[2,3,3,1]
```
#### 0.2.3 扩张与拼接

1. repeat
给tensor对应位置的维度重复对应的次数
```python
a=torch.ones(2,3)
a[1:,]+=1#tensor([[1., 1., 1.],[2., 2., 2.]])
a.repeat(3,2)
```
>out:tensor
(\[[1., 1., 1., 1., 1., 1.],
        [2., 2., 2., 2., 2., 2.],
        [1., 1., 1., 1., 1., 1.],
        [2., 2., 2., 2., 2., 2.],
        [1., 1., 1., 1., 1., 1.],
        [2., 2., 2., 2., 2., 2.]])

即在a(0)重复了2次,a(1)重复了3次

2. cat
对两个tensor按指定维度进行拼接，注意其输入格式`torch.cat((tensor1, tensor2),dim)`
注意，维度大小不超过两个tensor的维度最大值，且两个tensor要求相同维度
```python
a=torch.ones(2,3)
a[1:,]-=1
# tensor([[1., 1., 1.], [0., 0., 0.]])
b=torch.ones(2,3)+1
#tensor([[2., 2., 2.], [2., 2., 2.]])
```

```python

torch.cat((a,b),0)
#tensor([[1., 1., 1.],
 #       [0., 0., 0.],
  #      [2., 2., 2.],
   #     [2., 2., 2.]])
# torch.Size([4, 3])

torch.cat((a,b),1)
# tensor([[1., 1., 1., 2., 2., 2.],
        # [0., 0., 0., 2., 2., 2.]])
# torch.Size([2, 6])

```

3. stack
与cat类似，只是操作后会增加维度1

```python
a=torch.ones(2,3)
a[1:,]-=1
# tensor([[1., 1., 1.], [0., 0., 0.]])
b=torch.ones(2,3)+1
#tensor([[2., 2., 2.], [2., 2., 2.]])
```

```python
torch.stack((a,b),0)
# tensor([[[1., 1., 1.],
#          [0., 0., 0.]],
#         [[2., 2., 2.],
#          [2., 2., 2.]]])
# torch.Size([2, 2, 3])

torch.stack((a,b),1)
# tensor([[[1., 1., 1.],
#          [2., 2., 2.]],
#         [[0., 0., 0.],
#          [2., 2., 2.]]])
# torch.Size([2, 2, 3])
```

### 0.3 计算

#### 0.3.1 四则运算
```python
import torch
from torch import tensor
a=torch.ones(2,2)
b=torch.ones(2,2)+1
print(a+b)#the same with a.add(b)
print(a-b)#the same with a.sub(b)
print(a*b)#the same with a.mul(b)
print(a/b)#the same with a.div(b)
```
>Out: tensor(\[[3., 3.],[3., 3.]])
>Out: tensor(\[[-1., -1.],[-1., -1.]])
>Out: tensor(\[[2., 2.],[2., 2.]])
>Out: tensor(\[[0.5, 0.5],[0.5, 0.5]])

四则运算与numpy类似，也支持广播机制

#### 0.3.2 矩阵乘法
1. 二维矩阵
`matmlt`与`@`
```python
print(torch.matmul(a,b))
print(a@b)
```
>Out: tensor(\[[4., 4.], [4., 4.]])

2. 高维矩阵
保持前两维不变，进行后两维的运算
```python
a = torch.rand(4,3,28,64)
b = torch.rand(4,3,64,32)
torch.matmul(a,b).shape
```
>Out: [4,3,28,32]

#### 0.3.3 矩阵转置

1. 二维矩阵
`tensor.t()`
该方法只支持2维及2维以下转置
```python
a=torch.ones(2,3)
a.t()#shape:(3,2)
```

2. 高维矩阵
`tensor.transpose(dim1.dim2)`(只支持对两个维度进行转置)
```python
a = torch.rand(4,3,28,64)
a.transpose(2,0).shape#[28, 3, 4, 64]
#也可以调取torch的方法
torch.transpose(a,2,0).shape#[28, 3, 4, 64]
#两个方法完全相同
torch.equal(torch.transpose(a,2,0),a.transpose(2,0))
```

`tensor.permute()`支持对所有维度进行转置，但是要输入每个维度对应的转置后的顺序，且不支持torch.permute()
```python
a = torch.rand(4,3,28,64)
a.permute(0,1,3,2).shape#[4, 3, 64, 28]
torch.permute(a,0,1,3,2)#permute() takes 2 positional arguments but 5 were given
```

3. 总结：
对于二维数组，直接使用`tensor.t()`，高维数组只需要对其中两个维度转置时，使用`tensor.transpose()`，高维数组需要对多个维度进行转置，使用`tensor.permute()`


## 1. DataLoader

在pytorch中，提供了很多现成的数据集，有声音的、图像的、文字的。pytorch中有两个模块，`torch.utils.data.DataLoader`和`torch.utils.data.Dataset`，这两个模块支持使用torch内置的数据，以及用户自定义的数据。

**Dataset**: 存储样本和它们对应的标签，类似一个元组，使用方法类似元组，个人认为设计为元组的原因是防止数据被修改。
例如Dataset[0]就是存的数据，而Dataset[1]则是数据对应的label，而len(Dataset)，则是其中储存的数据/label的条数

**DataLoader**:在`Dataset`周围包装一个可迭代对象，以便方便地访问样本（相当于能方便的访问多个dataset）。将dataset包裹后，可以用for循环去遍历

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
        self.Data=torch.tensor([[1,1],[2,2],[3,3],[4,4]])
        self.Label=torch.tensor([9,8,7,6])
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

如上所示，通过定义的Datase，获取了单个的tuple输出，包含了数据和标签，而使用DataLoader则可以实现批处理抽取。

具体超参数可查阅：[URL](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)

常用重要超参数：
`dataset`：上节介绍
`batch_size`
`shuffle`
`num_workers`:用于读取数据的进程数（大数据时需要，可以有效在大数据时加速，进行预缓存）

#### 1.3.1 个人case

```python
loader=DataLoader(FDataset,batch_size=2)
#这里的enumerate() 函数用于将一个可遍历的数据对象,组合为一个索引序列,同时列出数据和数据下标，可以理解为给()内的可遍历数据对象多加了一个i
for i,dataset in enumerate(loader):
    data,label=dataset
    print("i:",i)
    print("dataset:",dataset)
    print("Data:",data)
    print("Label:",label)
```
>Out: 
i: 0
dataset: [tensor(\[[1, 1],[2, 2]]), tensor([9, 8])]
Data: tensor(\[[1, 1], [2, 2]])
Label: tensor([9, 8])
i: 1
dataset: [tensor(\[[3, 3],[4, 4]]), tensor([7, 6])]
Data: tensor(\[[3, 3],[4, 4]])
Label: tensor([7, 6])

显然，DataLoader将Dataset中的数据，进行了批量(根据batch_size)抽取，而形成的loader是一个迭代器，可以通过for循环获得其中的数据。

也可以直接用for循环输出，

```python
for a in loader:
    print(a)
```

>Out:
[tensor(\[[1, 1],[2, 2]]), tensor([9, 8])]
[tensor(\[[3, 3],[4, 4]]), tensor([7, 6])]

## 2. Transform

具体超参数可查[URL](https://pytorch.org/vision/stable/transforms.html)

transform其实就是在学CG时候学到的对图像的处理，例如scale(缩放)，reflection(对称)，shear(切变)，rotation(旋转)等，当然，torch中transform提供的功能更多。
而由于其实tensor也可以看做图像信息(或者说图像信息，例如灰度，本身可以用tensor表征，所以torch中，不区分区别)

补充说明：文档中提到的PIL(Python Imaging Library)，推测应该只是一种图像信息的表达形式，有的torch函数专用于处理该数据类型

### 2.1 Normalize

#### 2.1.1 说明
`torchvision.transforms.Normalize(mean, std, inplace=False)`
即对原始数据进行标准差归一化(见下方公式)，将数据转换为标准的高斯分布，原始数据均值为和标准差），其是逐个channel进行的，好处在于可以加速收敛，提高模型表现
其中`mean`,`std`分别指定对各个channel进行标准化的均值和方差，有几个channel，就是几维的

$x^*=\frac{x-mean}{std}$

#### 2.1.2 Case

数据准备
```python
aa=torch.ones(2,2,3)
aa[:,:,1:2]+=1
aa[:,:,2:3]+=2
aa[1,0,:]+=1
aa[0,1,:]*=2
aa[1,1,:]+=3
# tensor([[[1., 2., 3.],
#          [2., 4., 6.]],
#         [[2., 3., 4.],
#          [4., 5., 6.]]])
```

以下两种实现均可
```python
transform1=torch.nn.Sequential(
    transforms.Normalize((0, 0.5), (1, 0.5))
)

transform2 = transforms.Compose(
#     [transforms.ToTensor(),
     [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```
```python
b=transform1(aa)#维度吻合，成功输出
# tensor([[[ 1.,  2.,  3.],
#          [ 2.,  4.,  6.]],
#         [[ 3.,  5.,  7.],
#          [ 7.,  9., 11.]]])
b=transform2(aa)#维度不吻合，报错
# The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0
```
先分析数据：
transform1中(0, 0.5), (1, 0.5)，第一个0(mean)和第二个1(std)是一组，代表对aa[0](即\[[1., 2., 3.],[2., 4., 6.]])进行处理，而由公式可知，此时$x^*=x$，故输出结果与aa[0]保持一致

transform1中(0, 0.5), (1, 0.5)，第一个0.5(mean)和第二个0.5(std)是一组，代表对aa[1](即\[[2., 3., 4.],[4., 5., 6.]])进行处理，而由公式可知，此时$x^*=\frac{x-0.5}{0.5}$，故输出结果如上所示

同时应当注意的是，维度的问题


## 3. Model defination

具体超参数可查[URL](https://pytorch.org/docs/stable/nn.html)

在pytorch中，每个使用的NN都是用户自定义的子类(继承于父类`torch .nn`中的模块`nn.Module`)，而用户自定义的子类本身已是一个模块，需要有其他层等组成。

在用户自定义的Model类里，需要有至少两个函数:`__init__`,`forward`

模型最重要的是理解各个层的使用意义，以及保证每个层的输入输出维度合理

### 3.1 Network defination

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
            #且要注意，sequential内部各个layer之间需要,隔开
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

打印模型
```python
model = NeuralNetwork().to(device)
print(model)
```
```
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```