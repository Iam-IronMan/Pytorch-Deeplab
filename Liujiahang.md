~~~~# Pytorch简化版本
这是个好玩
## 运行demo时遇到的问题
代码里明确调用gpu，需要gpu版本框架

torchvision版本太高出现不兼容

显存不够，用的方法是调低batch_size从4到2，期间监控nvidia-smi状态发现代码只有在python内核关闭后才释放显存，解决显存不够的方法还有选择更小的数据类型等

[显存不够]: https://zhuanlan.zhihu.com/p/65002487

OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
需要从anaconda里面搜索libiomp5md.dll，并删掉一个

参数不够：可能代码会默认有填充，但是jupyter未能运行，于是补了一个颜色参数传进去

## demo效果
训练了1800epochs
![](C:\Users\11150\Desktop\psc.png)

##~~~~ 数据集加载

output字典里包含{图像，掩码，图像id，标签，并用爱因斯坦求和（einsum）每一个channel的每一个的权重

## 新学到的知识点

当tensor的数据类型是long时，为index索引，数据类型是uint8时，为mask掩码
#### 例子
```python
import torch
t = torch.rand(4, 2)
"""
tensor([[0.5492, 0.2083],
        [0.3635, 0.5198],
        [0.8294, 0.9869],
        [0.2987, 0.0279]])
"""
# 注意数据类型是 uint8， 
mask= torch.ones(4,dtype=torch.uint8)
mask[2] = 0
print(mask)
print(t[mask, :])
"""
 tensor([1, 1, 0, 1], dtype=torch.uint8)
 
 tensor([[0.5492, 0.2083],
        [0.3635, 0.5198],
        [0.2987, 0.0279]]) 
"""

# 注意， 数据类型是long
idx = torch.ones(3,dtype=torch.long)
idx[1] = 0
print(idx)
print(t[idx, :])
"""
tensor([1, 0, 1])
tensor([[0.3635, 0.5198],
        [0.5492, 0.2083],
        [0.3635, 0.5198]])
"""
```
调用pytorch神经网络类的时候，当需要同时提供forward和\_\_init\_\_参数的时候，外层调用的传参会传到forward函数里，内层调用传参时传到会传到\__init__这里

```python

class DecoderBottleneck(nn.Module):
    def __init__(
        self,
        nin,
        nplanes,
        upsample_factor=2,
        compression=4
    ):
    	self.bottleneck_path = nn.Sequential(
            conv_bn_relu(nin, nplanes, kernel_size=1),	# 这里传__init__参数，再加上上层传来的forward参数，一起给conv_bn_relu
            self.upsample,
            conv_bn_relu(nplanes, nplanes, kernel_size=3, padding=1),
            conv_bn_relu(nplanes, nin // compression, kernel_size=1, with_relu=False)
        )
    def forward(self, x, skip):
        bottleneck_out = self.bottleneck_path(x)	# 这里给__init__部分的bottleneck_path调用的所有的类传forward参数
        
```

后面一些新学到的没有来得及记录，但基本已融会贯通

## 疑问

"①"标注的Inception Stem部分是通过两个axial-block替换的Resnet-50，但是具体功能还没理解

"②”标注的pixel-pixel的Axial-bottleneck还不会

“③”memory的各个维度没有理解是什么意思

“④”Resnet和本文的⊕是否是逐元素相加

## 代码结构
coco_panoptic:包含train、val、test、annotation数据集

datasets:DETR原文的COCO数据集加载代码

max-deeplab:

--backbone.py:论文改进的backbone

--blocks.py:包含双路径transformer、Axial块、Inception Stem以及基础的卷积块conv_bn_relu，线性优化块linear_bn_relu

--losses.py:论文的PQ-loss、辅助loss以及匈牙利匹配的实现

--model.py:有encoder、decoder以及主模块

util:一些DETR原文的可视化工具

example.ipynb:已经可以运行的demo

# DeepLab2

一个是官方的DeepLab2库，它不仅包含MaX-deeplab，也包含Panoptic-deeplab、ViP-deeplab、Axial-deeplab等许多全景分割论文的代码，为的应该是建造出一个庞大的全景分割的全套应用，这个代码各个论文代码互相调用，我花了很长很长的时间也没法剥离出其中的MaX-deeplab来完全看懂，需要老师指点一下

## 运行

首先打开项目的时候需要打开本项目上层的文件夹，因为import是基于根目录以绝对路径导入的

装了需要的库之后需要将COCO数据集转为TFRecord，或者下载训练好的checkpoints(需要用创建它的py函数加载的保存了的模型)或者下载预训练的backbone的checkpoints，然后进行训练和评估，这里逻辑我没很清晰，MaX-deeplab没有在这跑通

## 代码结构

（基本每个代码都包含一个它的独立的测试代码，比如max_deeplab.py  max_deeplab_test.py）

configs:包含了MaX-deeplab等论文在COCO等数据集上需要的参数，在不同尺寸和不同训练轮数时都各有不同，需要自己选用

data:调用各个数据集所需的loader、预处理、测试

evaluation:模型评估和验证所用的一些类

g3doc:文档

model:核心模型

--decoder:这里面包含了MaX-deeplab的decoder，网络结构就如第一页图

--encoder:因为训练入口太复杂没有弄懂，所以虽看了每个文件但未找到MaX-deeplab相关的代码，推测应该是引用的Axial-resnet的encoder

--layers:包含各个layer以及组成它们的各个block

--loss:MaX_deeplab_loss.py中有PQ-loss以及辅助loss等

--post_processor:head预测等部分

trainer:训练入口

--train.py:加载参数用的

--train_lib.py:包含训练、评估、制作数据集和模型等

--trainer.py:把所有参数分为backbone和其他部分两块，分别使用两个优化器，对梯度以及变量通过加载tf库来引用本文代码来计算

