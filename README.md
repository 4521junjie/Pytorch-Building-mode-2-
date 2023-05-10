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

1、在第6行处，要在__init__方法中调用父类的__init__方法,不然无法继承父类的属性和方法，这会影响到模型的初始化和训练。
  At line 6, it is necessary to call the __init__ method of the parent class in the __init__ method, otherwise the properties and methods of the parent class cannot be inherited, which will affect the initialization and training of the model.

2、在forward方法的参数列表中加上self， 若不加上self参数，forward方法就无法访问模型的实例属性和方法，模型无法运行。
Add "self" to the parameter list of the forward method. Without the "self" parameter, the forward method cannot access the instance properties and methods of the model, and the model cannot run.

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


# 🌏ResNet32模型
## 1、模型认知总结 Summary of Model Cognition
ResNet32模型是一个深度残差网络，由32个卷积层组成，通过重复模块的方式构造，该模块由两个3x3卷积层和一个跨度为2的降采样层组成。
The ResNet32 model is a deep residual network consisting of 32 convolutional layers constructed by repeating modules. Each module is composed of two 3x3 convolutional layers and a downsampling layer with a stride of 2.

通过残差连接来减缓梯度消失问题；残差连接使将原始输入与所有进一步的层的输出相加，从而产生更多的参数更新。
Residual connections are used to alleviate the problem of vanishing gradients. Residual connections add the original input to the output of all subsequent layers, resulting in more parameter updates.

应用：在CIFAR-10数据集上，ResNet32模型的测试准确度可以高达90％左右。
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
        x = self.conv1(x)                 # [bs, 64, 56, 56] 特征提取过程
        x = self.maxpooling(x)            # [bs, 64, 28, 28]池化，降低分辨率和计算量，提取并强化重要的特征
        x = self.layer1(x)                # [bs , 64 , 28 , 28 ]残差层，提取特征，保持通道数和分辨率
        x = self.layer2(x)                # [bs , 128 , 14 , 14 ] 残差层，提取特征，改变通道数和分辨率
        x = self.layer3(x)                # [bs , 256 , 7 , 7 ] 残差层，提取特征，改变通道数和分辨率
        x = self.layer4(x)                # [bs , 512 , 4 , 4 ] 残差层，提取特征，改变通道数和分辨率
        x = self.avgpooling(x)            # [bs , 512 , 2 , 2 ] 平均池化，降低分辨率和计算量
        x = x.view(x.shape[0], -1)        # [bs , 2048 ] 展平张量，准备分类
        x = self.classifier(x)            # [bs , num_classes ] 全连接层，输出类别概率
        output = F.softmax(x)             # [bs , num_classes ] softmax函数，归一化概率

        return output


if __name__=='__main__':
    t = torch.randn([8, 3, 224, 224]) # 随机生成一个8个样本的输入张量
    model = ResNet32() # 创建一个ResNet32模型实例
    out = model(t)  # 调用模型的forward方法得到输出张量
    print(out.shape) # 打印输出张量的形状：[8 , num_classes ]
```

### 3、🙄补充
https://zhuanlan.zhihu.com/p/79378841

详细解释Resnet模型中forward方法

ResNet（Residual Network）是深度学习领域的一个经典的残差网络模型。在 ResNet 中，前向传播方法主要由多个残差块组成，其中的每个残差块通过跨层连接可以将前一层的输入直接加到当前层的输出中。这样做的目的是为了避免深度神经网络中可能出现的梯度消失问题，从而能够训练更深的网络。

由于 ResNet 中的残差块非常特殊，因此需要对其前向传播的具体实现进行详细解释。在下面的讨论中，我们将以 ResNet-50 为例，介绍其前向传播的过程。

首先，我们需要将输入图像传递给 ResNet-50 的第一层卷积。这层卷积主要是为了将图像转换成一组特征图，从而可以传递给下一层处理。这一步操作可以看作是对输入图像进行初步的特征提取。

接着，ResNet-50 的前向传播方法主要由四个阶段组成，每个阶段中都包含多个残差块。这些残差块基本上遵循同样的结构：每个残差块都由两个卷积层和一个跨层连接组成。其中，第一个卷积层的输出会作为第二个卷积层的输入，而第二个卷积层的输出则会与跨层连接的输出相加得到最终的残差块输出。

具体来说，ResNet-50 的第一个阶段包含一个 1×1 的卷积层、一个 3×3 的卷积层以及一个包含两个残差块的序列（每个块都由两个 3×3 的卷积层和一个跨层连接组成）。这个序列中的每个残差块都将其输入特征图 x 通过一个 3×3 的卷积层，转化成一个新的特征图 F(x)，然后将 F(x) 与原始输入 x 相加（即经过跨层连接后的输出），最终得到残差块的输出。

接下来的三个阶段，与第一个阶段类似，都包含了多个残差块，但包含的块数会逐渐减少，分别包含3、4、6个残差块，其中第三个阶段还包含了一组比较特殊的残差块，这些残差块在跨层连接中具有下采样的功能，可以将输入的特征图 size 减半。这个过程可以将前面的特征图 size 从 56×56 降到 7×7。

最后，在得到最后一个残差块的输出之后，我们将其连接到一个平均池化层，将其输出的特征图平均化成一个向量。在这个向量上，我们可以连接一个完全连接的（fully connected）层，作为对输出类别的预测。此外，在训练期间，我们需要与某些残差块之后添加一些 Dropout 层，以进一步提高模型的鲁棒性和泛化能力。

综上所述，ResNet-50 的前向传播方法主要由多个残差块组成，每个残差块通过跨层连接将前一层的输入直接加到当前层的输出中，从而可以训练出比原始网络更深的模型。在训练和推断时，我们只需要对输入图像按照上述方法一步步进行前向传播，最终得到网络的输出。



