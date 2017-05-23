深度学习网络的深度对最后的分类和识别的效果有着很大的影响,常规的网络的堆叠（plain network）在网络很深的时候，效果却越来越差了。ResNet引入了残差网络结构（residual network），通过残差网络，可以把网络层弄的很深，据说现在达到了1000多层，最终的网络分类的效果也是非常好，残差网络的基本结构如下图所示 
![resnet-1](https://github.com/shaoxq/projects/blob/master/figs/resnet-1.png?raw=true)

ResNet 的网络结构借鉴了 HighWay，添加一条从 input到output的路径，即在输出个输入之间引入一个shortcut connection,而不是简单的堆叠网络，这样可以解决网络由于很深出现梯度消失的问题，从而可可以把网络做的很深，ResNet其中一个网络结构如下图所示
![resnet-2](https://github.com/shaoxq/projects/blob/master/figs/resnet-2.png?raw=true)

多个 Block单元 组成的一大串

![resnet-3](https://github.com/shaoxq/projects/blob/master/figs/resnet-3.png?raw=true)
 ![resnet-4](https://github.com/shaoxq/projects/blob/master/figs/resnet-4.png?raw=true)

目前几种常用的ResNet网络包括：ResNet-50/101/152，当然层数越多计算量越大，基于ResNet的改进我们也提前了解下，包括 Google的 Inception-ResNet-V2， Kaiming 的 ResNeXt等