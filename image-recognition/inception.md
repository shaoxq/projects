# Inception v1
Going Deeper with Convolutions, 6.67% test error [paper](https://arxiv.org/pdf/1409.4842.pdf)

Inception v1的网络，将1x1，3x3，5x5的conv和3x3的pooling，stack在一起，一方面增加了网络的width，另一方面增加了网络对尺度的适应性；

![inception v1 结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v1-1.png?raw=true)

论文中提出的最原始的版本，所有的卷积核都在上一层的所有输出上来做，那5×5的卷积核所需的计算量就太大了，造成了特征图厚度很大。为了避免这一现象提出的inception具有如下结构，

![inception v1 结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v1-2.png?raw=true)

在3x3前，5x5前，max pooling后分别加上了1x1的卷积核起到了降低特征图厚度的作用，也就是Inception v1的网络结构。

inception v1 网络结构:

![inception v1 网络结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v1.jpg?raw=true)

# Inception v2
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 4.8% test error, [paper](https://arxiv.org/pdf/1502.03167.pdf)

v2的网络在v1的基础上，进行了改进，一方面了加入了BN层，减少了Internal Covariate Shift（内部neuron的数据分布发生变化），使每一层的输出都规范化到一个N(0, 1)的高斯，另外一方面学习VGG用2个3x3的conv替代inception模块中的5x5，既降低了参数数量，也加速计算；

![inception v2结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v2-1.png?raw=true)

使用3×3的已经很小了，那么更小的2×2呢？2×2虽然能使得参数进一步降低，但是不如另一种方式更加有效，那就是Asymmetric方式，即使用1×3和3×1两种来代替3×3的卷积核。这种结构在前几层效果不太好，但对特征图大小为12~20的中间层效果明显。 

![inception v2结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v2-2.png?raw=true)

# Inception v3
Rethinking the Inception Architecture for Computer Vision, 3.5% test error,  [paper](https://arxiv.org/pdf/1512.00567.pdf)

v3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，还有值得注意的地方是网络输入从224x224变为了299x299，更加精细设计了35x35/17x17/8x8的模块；

# Inception v4
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, 3.08% test error, [paper](https://arxiv.org/pdf/1602.07261.pdf)

v4研究了Inception模块结合Residual Connection能不能有改进？发现ResNet的结构可以极大地加速训练，同时性能也有提升，得到一个Inception-ResNet v2网络，同时还设计了一个更深更优化的Inception v4模型，能达到与Inception-ResNet v2相媲美的性能。


![inception v4网络结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v4.png?raw=true)

![inception v4 stem结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v4-1.png?raw=true)

![inception v4结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v3-1.png?raw=true)

![inception v4结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v3-2.png?raw=true)

![inception v4结构](https://github.com/shaoxq/projects/blob/master/figs/inception-v3-3.png?raw=true)

