<center>
    <h1>
        Deep Residual Learning for Image Recognition
    </h1>
</center>



## 摘要    

更深层的神经网络更难训练，但残差网络更容易优化。针对于残差网络，在相当大的深度的复杂度较低，并能提高模型的精度。将层表示为参考层输入的学习残差函数，而不是学习未引用的函数。

## 1. 引言

解决梯度消失/梯度爆炸可以使用初始归一化或中间归一化来解决，使得数十个层的网络开始收敛具有反向传播的随机梯度下降。

当更深的网络能够开始收敛时，降级问题就暴露了出来：随着网络深度的增加，精度达到饱和，然后迅速降级。出乎意料的是，这种退化并不是由过度拟合引起的，并且在适当深度的模型中添加更多的层会导致更高的训练误差。

训练精确度的下降表明，并非所有系统都同样容易优化。添加的层是identity映射，其他层是从学习的浅层模型复制而来的。这种构造解的存在表明，较深的模型应该不会产生比较浅的模型更高的训练误差。

引入深度残差学习框架来解决退化问题。将期望的输出映射表示为H(x)，使叠加的非线性层拟合出上的残差映射：F(x)=H(x)−x。原始映射被重塑残差映射为F(X)+x。我们假设优化残差映射比优化原始映射更容易。在极端情况下，如果单位映射是最优的，将残差推到零要比通过一堆非线性层来拟合单位映射容易得多。

公式F(X)+x可以通过具有“shortcut connections”的前馈神经网络实现。shortcut connections是跳过一个或多个层的连接。在这里是identity 映射。并且它们的输出被添加到堆叠的层的输出。identity shortcut connections既不会增加额外参数，也不会增加计算复杂性。整个网络仍然可以由SGD使用反向传播进行端到端的训练。

在ImageNet[36]上进行的实验，结果表明：1)极深残差网络易于优化，且训练误差小；2)深度残差网络可以很容易获得精度提高。

代码

复现：[alexnet-pytorch/model.py at master · dansuh17/alexnet-pytorch · GitHub](https://github.com/dansuh17/alexnet-pytorch/blob/master/model.py)

## 2. 相关工作

Residual Representations。使用类比的方法：编码残差矢量[17]被证明比编码原始矢量更有效。

多重网格法[3]将系统重新描述为多个尺度上的子问题，其中每个子问题负责较粗和较细尺度之间的残差解。残差解比标准解收敛快很多。

Shortcut Connections. 在MLP的几个中间层直接连接到辅助分类器，已处理梯度消失或爆炸。

highway networks，提供了gating功能的Shortcut Connections。这些gate是需要依赖数据并需要参数的，但是残差网络的Shortcut Connections是无参数的。当 gate shortcut is “closed”(approaching zero)时，highway networks代表非残差函数。相反，我们的公式总是学习剩余函数；我们的identity shortcut 永远不会关闭，所有信息总是传递的，还有额外的剩余函数需要学习。highway networks并没有随着深度的极大增加而表现出精度的提高。

## 3. 深度残差学习

### 3.1 残差学习

H(X)作为由几个堆叠的层来拟合的输出映射，其中x表示这些层中第一个层的输入。如果多个非线性层可以渐近逼近复杂函数，则等价于假设它们可以渐近逼近剩余函数，即F(x)=H(x)−x。因此，我们不是希望输出层近似H(x)，而是让非线性层近似残差函数F(x)：=H(x)−x。原始函数因此变成H(x)=F(X)+x。

如果最优函数更接近于恒等式映射而不是零映射，则求解器应该更容易找到参考恒等式映射的扰动，而不是将该函数作为新函数来学习。**（没太懂）**

### 3.2 Identity Mapping by Shortcuts

![1767454307054](C:\Users\29961\AppData\Roaming\Typora\typora-user-images\1767454307054.png)

采用残差学习的方法是对每几个叠层进行学习。构建块如图所示。

![1767454388738](C:\Users\29961\AppData\Roaming\Typora\typora-user-images\1767454388738.png)

F(x，{Wi})表示要学习的残差映射。F = W~2~σ(W~1~x) in which σ 代表ReLU激活函数。

在公式中，x和F的尺寸必须相等。如果不是这种情况，我们可以通过快捷连接执行线性投影Ws以匹配尺寸：

![1767454582469](C:\Users\29961\AppData\Roaming\Typora\typora-user-images\1767454582469.png)

Ws是方阵1*1的卷积核。函数F(x，{Wi})可以表示多个卷积层。逐个通道地在两个特征图上执行元素相加。

![1767455187730](C:\Users\29961\AppData\Roaming\Typora\typora-user-images\1767455187730.png)

## 4 值得学习的点

1、在相对较深的网络中，使用残差结构会更好优化模型，且计算复杂度不高，还能提高模型的检测精度。