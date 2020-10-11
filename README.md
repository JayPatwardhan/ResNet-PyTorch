# ResNet-PyTorch
Implementation of ResNet 50, 101, 152 in PyTorch based on paper "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

Currently working on implementing the ResNet 18 and 34 architectures as well which do not include the Bottleneck in the residual block.

A baseline run of ResNet50 on the CIFAR-10 dataset is given as well, with the standard setup proposed by the paper it already achieves around 85.6% accuracy. However, this can definitely be brought up to at least 92% accuracy via some more slight optimization.

References:
1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
   Deep Residual Learning for Image Recognition 
   https://arxiv.org/pdf/1512.03385.pdf
   
2. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

3. https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py
