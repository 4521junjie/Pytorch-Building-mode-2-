# Pytorch-Building-mode-2-

# 📜Pytorch搭建模型模板的语法问题：
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
1、在给初始化函数定义函数名时，使用了双引号 “#”，导致代码运行时会出现语法错误，直接报错退出。正确的写法是使用下划线 “_” 划分函数名。 

2、在初始化函数中应该首先调用父类的初始化函数，可以使用 super() 函数实现，否则可能会影响模型的初始化和训练。此外，forward() 方法应该接收至少一个参数，通常是输入的数据 x，同时需要进行必要的计算并返回计算结果。



# 🌏ResNet32模型
## 模型认知总结
esNet32模型是一个深度残差网络，由32个卷积层组成，通过重复模块的方式构造，该模块由两个3x3卷积层和一个跨度为2的降采样层组成。

通过残差连接来减缓梯度消失问题；残差连接使将原始输入与所有进一步的层的输出相加，从而产生更多的参数更新。

应用：在CIFAR-10数据集上，ResNet32模型的测试准确度可以高达90％左右。
##
