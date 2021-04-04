# 简介
本仓库为Dian团队AI组开源仓库，技术方向为计算机视觉分类任务。

# 运行环境
Python3

Pytorch>=1.4.0

# 项目框架
```
.
├── analysis.py
├── autotest.sh
├── badcase
├── boardlog
├── config.py
├── dataloader
│   ├── copydataset.py
│   ├── dataset_split.py
│   ├── __init__.py
│   ├── LoadDataTest.py
│   ├── LoadDataTotal.py
│   ├── LoadDataTrain.py
│   └── LoadDataVal.py
├── log
├── model
│   ├── ghostnet.py
│   ├── inceptionresnetv2.py
│   ├── inceptionV4.py
│   ├── __init__.py
│   ├── loss.py
│   ├── MobileNetV3New.py
│   ├── Net.py
│   ├── ResNet.py
│   └── vgg19.py
├── readme.md
├── runs
├── test.py
├── train.py
├── utils.py
└── weight
```
train.py和test.py分别是训练和测试的主函数。utils.py存放常用函数。config.py是项目运行的配置文件。analysis.py用于计算数据集均值和方差。所有dataloader均保存在dataloader文件夹。所有模型均保存在model文件夹。log文件夹用于保存运行时的日志。runs和boardlog用于保存可视化的日志。
# 特性及亮点
1. 基于独立config文件，需要配置的参数均可在config文件中，其中包括batch_size、epoch、model_type等。
2. 支持ImageNet格式分类数据集，即文件夹名字等同于类别名，且类别数不限。支持提前划分数据集（dataloader/dataset_split.py）。
3. 集成多种经典模型，比如Resnet、MobileNet等。
4. 可以将训练结果acc、loss可视化（基于tensorboardX）。
5. 自动保存badcase方便分析数据集。
6. 项目结构清晰，代码友好易懂。
7. 多GPU运行。
# 不足
1. 训练trick未加入，比如warm-up、lr_schedule等。
2. 缺少demo运行代码，待完善。