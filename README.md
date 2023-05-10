# Pytorch-Building-mode-2-

# 😃Pytorch搭建模型模板的语法问题
代码如下:
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

2、在forward方法的参数列表中加上self， 若不加上self参数，forward方法就无法访问模型的实例属性和方法，模型无法运行。

修改后的模板如下:
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
## 1、模型认知总结
esNet32模型是一个深度残差网络，由32个卷积层组成，通过重复模块的方式构造，该模块由两个3x3卷积层和一个跨度为2的降采样层组成。

通过残差连接来减缓梯度消失问题；残差连接使将原始输入与所有进一步的层的输出相加，从而产生更多的参数更新。

应用：在CIFAR-10数据集上，ResNet32模型的测试准确度可以高达90％左右。
## 2、ResNet32模型代码以及局部注释
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
        x = self.conv1(x) # [bs, out_dim, h/stride, w/stride] 卷积，提取特征，改变通道数和分辨率
        x = self.bn1(x) # [bs, out_dim, h/stride, w/stride] 批归一化，加速收敛，防止过拟合
        x = self.relu1(x) # [bs, out_dim, h/stride, w/stride] 激活函数，增加非线性
        x = self.conv2(x) # [bs, out_dim, h/stride, w/stride] 卷积，提取特征，保持通道数和分辨率
        x = self.bn2(x) # [bs, out_dim, h/stride, w/stride] 批归一化，加速收敛，防止过拟合
        x = self.relu2(x) # [bs, out_dim, h/stride, w/stride] 激活函数，增加非线性
        return x          #  forward 函数中定义数据的前向传播


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
        x = self.conv1(x)  # [bs, 64, 56, 56] 特征提取过程
        x = self.maxpooling(x)  # [bs, 64, 28, 28]池化，降低分辨率和计算量
        x = self.layer1(x) # [bs , 64 , 28 , 28 ]残差层，提取特征，保持通道数和分辨率
        x = self.layer2(x) # [bs , 128 , 14 , 14 ] 残差层，提取特征，改变通道数和分辨率
        x = self.layer3(x) # [bs , 256 , 7 , 7 ] 残差层，提取特征，改变通道数和分辨率
        x = self.layer4(x) # [bs , 512 , 4 , 4 ] 残差层，提取特征，改变通道数和分辨率
        x = self.avgpooling(x) # [bs , 512 , 2 , 2 ] 平均池化，降低分辨率和计算量
        x = x.view(x.shape[0], -1) # [bs , 2048 ] 展平张量，准备分类
        x = self.classifier(x) # [bs , num_classes ] 全连接层，输出类别概率
        output = F.softmax(x) # [bs , num_classes ] softmax函数，归一化概率

        return output


if __name__=='__main__':
    t = torch.randn([8, 3, 224, 224]) # 随机生成一个8个样本的输入张量
    model = ResNet32() # 创建一个ResNet32模型实例
    out = model(t)  # 调用模型的forward方法得到输出张量
    print(out.shape) # 打印输出张量的形状：[8 , num_classes ]
```





