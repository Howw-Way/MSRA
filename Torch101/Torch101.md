<!-- Author: Howw -->
<!-- Data: 22.8.30 -->
# Torch 101: my first lesson to torch

This is my note for learning torch, the reference is the official website. 

## 1. DataLoader

在pytorch中，提供了很多现成的数据集，有声音的、图像的、文字的。pytorch中有两个模块，`torch.utils.data.DataLoader`和`torch.utils.data.Dataset`，这两个模块支持使用torch内置的数据，以及用户自定义的数据。

`Dataset`: 存储样本和它们对应的标签
`DataLoader`:在`Dataset`周围包装一个可迭代对象，以便方便地访问样本。

### 1.1 Loading a dataset

### 1.2 Creating custom dataset

需要定义一个类去实现，包含三个函数：`__init__`,`__len__`,`__getitem`

```python
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

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

#### 1.2.1 __init__

`__init__`函数是初始化，需要包含图像文件、标签文件、以及两种transformer格式