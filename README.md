# Pytorch-Building-mode-2-

# 😃Pytorch搭建模型模板的语法问题：
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

1、在第6行处，需要在__init__方法中调用父类的__init__方法,不然无法继承父类的属性和方法，这会影响到模型的初始化和训练。

2、在forward方法的参数列表中加上self， 如果不加上self参数，forward方法就无法访问模型的实例属性和方法，导致模型无法运行。

修改后的模板如下:
``` python
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
## 模型认知总结
esNet32模型是一个深度残差网络，由32个卷积层组成，通过重复模块的方式构造，该模块由两个3x3卷积层和一个跨度为2的降采样层组成。

通过残差连接来减缓梯度消失问题；残差连接使将原始输入与所有进一步的层的输出相加，从而产生更多的参数更新。

应用：在CIFAR-10数据集上，ResNet32模型的测试准确度可以高达90％左右。
##
