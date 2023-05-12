# Pytorch-Building-mode-2-

# 😃Pytorch搭建模型模板的语法问题
The code is as follows.:
``` python
import torch
import torch.nn as nn
import torch.nn.functional as F

class xxxNet(nn.Module):
	def __init__(self):
         pass
	def forward(x):
          return x
 ```

1、At line 6, it is necessary to call the __init__ method of the parent class in the __init__ method, otherwise the properties and methods of the parent class cannot be inherited, which will affect the initialization and training of the model.

2、Add "self" to the parameter list of the forward method. Without the "self" parameter, the forward method cannot access the instance properties and methods of the model, and the model cannot run.

The modified template is as follows:
``` python
# 引入相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义残差块
class ResidualBlock(nn.Module):
      def __init__(self, in_channels, out_channels, stride=1):
     super(ResidualBlock, self).__init__()              # 调用父类的__init__方法
     def forward(self, x):                              # 加上self参数
		return X
```


# 🌏ResNet32 Model
## 1、Summary of Model Cognition
The ResNet32 model is a deep residual network consisting of 32 convolutional layers constructed by repeating modules. Each module is composed of two 3x3 convolutional layers and a downsampling layer with a stride of 2.

Residual connections are used to alleviate the problem of vanishing gradients. Residual connections add the original input to the output of all subsequent layers, resulting in more parameter updates.

Application: On the CIFAR-10 dataset, the testing accuracy of the ResNet32 model can reach around 90%.

## 2、ResNet32 model code with local comments
``` python 
# 引入相关库
import torch
import torch.nn as nn
import torch.nn.functional as F


# 把残差连接补充到 Block 的 forward 函数中
class Block(nn.Module):
    def __init__(self, dim, out_dim, stride) -> None:
        super().__init__()                 # 定义模型类，并继承 nn.Module 类
        self.conv1 = nn.Conv2d(dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)                 # [bs, out_dim, h/stride, w/stride] 卷积，提取特征，改变通道数和分辨率
        x = self.bn1(x)                   # [bs, out_dim, h/stride, w/stride] 批归一化，加速收敛，防止过拟合
        x = self.relu1(x)                 # [bs, out_dim, h/stride, w/stride] 激活函数，增加非线性
        x = self.conv2(x)                 # [bs, out_dim, h/stride, w/stride] 卷积，提取特征，保持通道数和分辨率
        x = self.bn2(x)                   # [bs, out_dim, h/stride, w/stride] 批归一化，加速收敛，防止过拟合
        x = self.relu2(x)                 # [bs, out_dim, h/stride, w/stride] 激活函数，增加非线性
        return x                          #  forward 函数中定义数据的前向传播


class ResNet32(nn.Module):            # 可以在模型类中定义其他自定义函数
    def __init__(self, in_channel=64, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = in_channel

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3)
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        self.last_channel = in_channel

        self.layer1 = self._make_layer(in_channel=64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(in_channel=128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(in_channel=256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(in_channel=512, num_blocks=3, stride=2)

        self.avgpooling = nn.AvgPool2d(kernel_size=2)
        self.classifier = nn.Linear(4608, self.num_classes)

    def _make_layer(self, in_channel, num_blocks, stride):
        layer_list = [Block(self.last_channel, in_channel, stride)]
        self.last_channel = in_channel
        for i in range(1, num_blocks):
            b = Block(in_channel, in_channel, stride=1)
            layer_list.append(b)
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.conv1(x)               # [bs, 64, 56, 56] 特征提取过程
        x = self.maxpooling(x)          # [bs, 64, 28, 28]池化，降低分辨率和计算量
        x = self.layer1(x)              # [bs,64,56,56]，对输入的特征图 x 进行卷积和池化操作，并将其传递给 layer1 进行处理。
        x = self.layer2(x)              # [bs,128,28,28]，将 layer1 的输出作为输入，传递给 layer2 进行卷积和池化。
        x = self.layer3(x)              # [bs,256,14,14]，将 layer2 的输出作为输入，传递给 layer3 进行卷积和池化。
        x = self.layer4(x)              # [bs,512,7,7]，将 layer3 的输出作为输入，传递给 layer4 进行卷积和池化。
        x = self.avgpooling(x)          # [bs,512,3,3]，对特征图 x 进行平均池化操作，每个通道的特征图缩小为一个标量值。
        x = x.view(x.shape[0], -1)      # [bs,4608]，将 [bs, 512, 7, 7] 的特征图拉伸为 [bs, 4608] 的向量。
        x = self.classifier(x)          # [bs,2]，其输出形状为 [bs, 2]， bs 表示批次大小，2 表示输出的类别数目。
        output = F.softmax(x)
        return output


if __name__=='__main__':
    t = torch.randn([8, 3, 224, 224])    # 随机生成一个8个样本的输入张量
    model = ResNet32()                   # 创建一个ResNet32模型实例
    out = model(t)                       # 调用模型的forward方法得到输出张量
    print(out.shape)                     # 打印输出张量的形状：[8 , num_classes ]
```
![image](https://github.com/4521junjie/Pytorch-Building-mode-2-/assets/119326710/ef46ea13-d887-4c6f-9a97-e4d173398fa8)
![image](https://github.com/4521junjie/Pytorch-Building-mode-2-/assets/119326710/4413c615-4863-45b4-b550-4d5cb8399ad2)
![image](https://github.com/4521junjie/Pytorch-Building-mode-2-/assets/119326710/cc632446-0f2c-4b62-8060-f5496fdf28de)
![image](https://github.com/4521junjie/Pytorch-Building-mode-2-/assets/119326710/f734b717-d11d-4676-b4b0-74a875be78dd)
![image](https://github.com/4521junjie/Pytorch-Building-mode-2-/assets/119326710/8ddcc4f8-bced-439a-9619-ae4a634dd9a7)
![image](https://github.com/4521junjie/Pytorch-Building-mode-2-/assets/119326710/7099271f-b733-4edf-bba6-022d8785379c)
![image](https://github.com/4521junjie/Pytorch-Building-mode-2-/assets/119326710/6ba6e0d3-4ad0-4f58-a98c-d008da538b0c)
![image](https://github.com/4521junjie/Pytorch-Building-mode-2-/assets/119326710/544a3fd4-4c96-41da-9aa5-3b075f47b478)
![Uploading image.png…]()






### 3、🙄Supplement
https://zhuanlan.zhihu.com/p/79378841

Provide a detailed explanation of the forward method in the Resnet model.





