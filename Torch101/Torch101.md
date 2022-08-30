<!-- Author: Howw -->
<!-- Data: 22.8.30 -->
# Torch 101: my first lesson to torch

This is my note for learning torch, the reference is the official website. 

## 1. DataLoader

在pytorch中，提供了很多现成的数据集，有声音的、图像的、文字的。pytorch中有两个模块，`torch.utils.data.DataLoader`和`torch.utils.data.Dataset`，这两个模块支持使用torch内置的数据，以及用户自定义的数据。

`Dataset`: 存储样本和它们对应的标签
`DataLoader`:在`Dataset`周围包装一个可迭代对象，以便方便地访问样本。

### 1.1 Loading a dataset