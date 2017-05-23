深度学习网络的深度对最后的分类和识别的效果有着很大的影响,常规的网络的堆叠（plain network）在网络很深的时候，效果却越来越差了。ResNet引入了残差网络结构（residual network），通过残差网络，可以把网络层弄的很深，据说现在达到了1000多层，最终的网络分类的效果也是非常好，残差网络的基本结构如下图所示 

![resnet-1](https://github.com/shaoxq/projects/blob/master/figs/resnet-1.png?raw=true)

通过在输出个输入之间引入一个shortcut connection,而不是简单的堆叠网络，这样可以解决网络由于很深出现梯度消失的问题，从而可可以把网络做的很深，ResNet其中一个网络结构如下图所示 

![resnet-2](https://github.com/shaoxq/projects/blob/master/figs/resnet-2.png?raw=true)