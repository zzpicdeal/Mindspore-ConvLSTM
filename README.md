# 目录

# Convlstm

> 
传统的LSTM的关键是细胞状态，表示细胞状态的这条线水平的穿过图的顶部。LSTM的删除或者添加信息到细胞状态的能力是由被称为Gate的结构赋予的。LSTM的第一步是决定要从细胞状态中丢弃什么信息。 该决定由被称为“忘记门”的Sigmoid层实现。它查看ht-1(前一个输出)和xt(当前输入)，并为单元格状态Ct-1(上一个状态)中的每个数字输出0和1之间的数字。1代表完全保留，而0代表彻底删除。

下一步是决定我们要在细胞状态中存储什么信息。
第一，sigmoid 层称 “输入门层” 决定什么值我们将要更新。然后，一个 tanh 层创建一个新的候选值向量，Ct，会被加入到状态中。下一步，我们会讲这两个信息来产生对状态的更新。


更新上一个状态值Ct−1了，将其更新为Ct。签名的步骤以及决定了应该做什么，我们只需实际执行即可。我们将上一个状态值乘以ft，以此表达期待忘记的部分。之后我们将得到的值加上 it∗Ct。这个得到的是新的候选值，按照我们决定更新每个状态值的多少来衡量。最后，我们需要决定我们要输出什么。 此输出将基于我们的细胞状态，但将是一个过滤版本


最终，我们需要确定输出什么值。这个输出将会基于我们的细胞状态，但是也是一个过滤后的版本。首先，我们运行一个 sigmoid 层来确定细胞状态的哪个部分将输出出去。接着，我们把细胞状态通过 tanh 进行处理（得到一个在-1到1之间的值）并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。


Convlstm模型和传统LSTM的不同：

①ConvLSTM模型中将fully-connect layer改成convolutional layer

②模型的input是3D tensor。

## 模型架构

> 
预测模型包括两个网络，一个编码网络和一个预测网络。，预测网络的初始状态和单元输出从编码网络的最后状态复制。这两个网络都是通过叠加几个ConvLSTM层而形成的。由于预测目标与输入具有相同的维数，将预测网络中的所有状态连接起来，并将它们输入1×1卷积层，生成最终的预测。

## 数据集MovingMNIST

> 
1、用于生成训练数据的MNIST数据集:train-images-idx3-ubyte.gz (http://yann.lecun.com/exdb/mnist/)
2、测试数据集MovingMNIST：mnist_test_seq.npy (http://www.cs.toronto.edu/~nitish/unsupervised_video/)
>启智平台下 创建单卡训练任务时，请保证data.zip压缩包为以下结构
> data/train-images-idx3-ubyte.gz 
> data/mnist_test_seq.npy     
启智平台下 创建单卡调试任务时，请保证work环境data目录为以下结构
>work/data/train-images-idx3-ubyte.gz               
>work/data//mnist_test_seq.npy 
## 环境要求

> 提供运行该代码前需要的环境配置，包括：
>
> * 第三方库 scikit-image
> * 镜像	tensorflow1.15-mindspore1.5.1-cann5.0.3-euler2.8-aarch64
> * 规格	Ascend: 1*Ascend910|CPU: 24核 96GB

## 快速入门
>调试环境下
```shell
cd work/convlstm
python train_.py --batch_size 32 --save_every 5
```
> 训练任务下
创建单卡训练任务

## 脚本说明

> 提供实现的细节

### 脚本和样例代码

> 提供完整的代码目录展示（包含子文件夹的展开），描述每个文件的作用

### 脚本参数

> 注解模型中的每个参数，特别是`config.py`中的参数，如有多个配置文件，请注解每一份配置文件的参数

## 训练过程

> 提供训练信息，区别于quick start，此部分需要提供除用法外的日志等详细信息

### 训练

> 提供训练脚本的使用方法

例如：在昇腾上使用分布式训练运行下面的命令

```shell
bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

> 提供训练过程日志

```log
# grep "loss is " train.log
epoch:1 step:390, loss is 1.4842823
epcoh:2 step:390, loss is 1.0897788
```

> 提供训练结果日志
例如：训练checkpoint将被保存在`XXXX/ckpt_0`中，你可以从如下的log文件中获取结果

```log
epoch: 11 step: 7393 ,rpn_loss: 0.02003, rcnn_loss: 0.52051, rpn_cls_loss: 0.01761, rpn_reg_loss: 0.00241, rcnn_cls_loss: 0.16028, rcnn_reg_loss: 0.08411, rcnn_mask_loss: 0.27588, total_loss: 0.54054
epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.00285, rpn_reg_loss: 0.00262, rcnn_cls_loss: 0.08002, rcnn_reg_loss: 0.04990, rcnn_mask_loss: 0.26245, total_loss: 0.39804
```



### 分布式训练

> 同上

## 评估

### 评估过程

> 提供eval脚本用法

### 评估结果

> 提供推理结果

例如：上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```log
accuracy:{'acc':0.934}
```

## 导出

### 导出过程

> 提供export脚本用法

### 导出结果

> 提供export结果日志

## 推理

### 推理过程

> 提供推理脚本

```bash
bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
```

### 推理结果

> 提供推理结果

## 性能

### 训练性能

提供您训练性能的详细描述，例如finishing loss, throughput, checkpoint size等

你可以参考如下模板

| Parameters                 | Ascend 910                                                   | GPU |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------------------|
| Model Version              | ResNet18                                                     |  ResNet18                                     |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  PCIE V100-32G                                |
| uploaded Date              | 02/25/2021 (month/day/year)                                  | 07/23/2021 (month/day/year)                   |
| MindSpore Version          | 1.1.1                                                        | 1.3.0                                         |
| Dataset                    | CIFAR-10                                                     | CIFAR-10                                      |
| Training Parameters        | epoch=90, steps per epoch=195, batch_size = 32               | epoch=90, steps per epoch=195, batch_size = 32|
| Optimizer                  | Momentum                                                     | Momentum                                      |
| Loss Function              | Softmax Cross Entropy                                        | Softmax Cross Entropy                         |
| outputs                    | probability                                                  | probability                                   |
| Loss                       | 0.0002519517                                                 |  0.0015517382                                 |
| Speed                      | 13 ms/step（8pcs）                                           | 29 ms/step（8pcs）                            |
| Total time                 | 4 mins                                                       | 11 minds                                      |
| Parameters (M)             | 11.2                                                         | 11.2                                          |
| Checkpoint for Fine tuning | 86M (.ckpt file)                                             | 85.4 (.ckpt file)                             |
| Scripts                    | [link](https://gitee.com/mindspore/models/tree/master/official/cv/)                       |

### 推理性能

> 提供推理性能的详细描述，包括耗时，精度等

你可以参照如下模板

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18                    |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 94.02%                      |
| Model for inference | 43M (.air file)             |

## 随机情况说明

> 说明该项目有可能出现的随机事件

## 参考模板

此部分不需要出现在你的README中
[maskrcnn_readme](https://gitee.com/mindspore/models/blob/master/official/cv/maskrcnn/README_CN.md)

## 贡献指南

如果你想参与贡献昇思的工作当中，请阅读[昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

### 贡献者

此部分根据自己的情况进行更改，填写自己的院校和邮箱

* [c34](https://gitee.com/c_34) (Huawei)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。
# 目录

# 模型名称

> 模型简介，论文模型概括

## 模型架构

> 如包含多种模型架构，展示你实现的部分

## 数据集

> 提供你所使用的数据信息，检查数据版权，通常情况下你需要提供下载数据的链接，数据集的目录结构，数据集大小等信息

## 特性（可选）

> 展示你在模型实现中使用的特性，例如分布式自动并行或者混合精度等一些特殊的训练技巧

## 环境要求

> 提供运行该代码前需要的环境配置，包括：
>
> * python第三方库，在模型root文件夹下添加一个'requirements.txt'文件，文件内说明模型依赖的第三方库
> * 必要的第三方代码
> * 其他的系统依赖
> * 在训练或推理前额外的操作

## 快速入门

> 展示可以直接运行的命令
> 按照你开发的版本，可能包含：
> * 训练命令，推理命令，export命令
> * Ascend版本，GPU版本，CPU版本
> * 线下运行版本，线上运行版本

## 脚本说明

> 提供实现的细节

### 脚本和样例代码

> 提供完整的代码目录展示（包含子文件夹的展开），描述每个文件的作用

### 脚本参数

> 注解模型中的每个参数，特别是`config.py`中的参数，如有多个配置文件，请注解每一份配置文件的参数

## 训练过程

> 提供训练信息，区别于quick start，此部分需要提供除用法外的日志等详细信息

### 训练

> 提供训练脚本的使用方法

例如：在昇腾上使用分布式训练运行下面的命令

```shell
bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

> 提供训练过程日志

```log
# grep "loss is " train.log
epoch:1 step:390, loss is 1.4842823
epcoh:2 step:390, loss is 1.0897788
```

> 提供训练结果日志
例如：训练checkpoint将被保存在`XXXX/ckpt_0`中，你可以从如下的log文件中获取结果

```log
epoch: 11 step: 7393 ,rpn_loss: 0.02003, rcnn_loss: 0.52051, rpn_cls_loss: 0.01761, rpn_reg_loss: 0.00241, rcnn_cls_loss: 0.16028, rcnn_reg_loss: 0.08411, rcnn_mask_loss: 0.27588, total_loss: 0.54054
epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.00285, rpn_reg_loss: 0.00262, rcnn_cls_loss: 0.08002, rcnn_reg_loss: 0.04990, rcnn_mask_loss: 0.26245, total_loss: 0.39804
```

### 迁移训练（可选）

> 提供如何根据预训练模型进行迁移训练的指南

### 分布式训练

> 同上

## 评估

### 评估过程

> 提供eval脚本用法

### 评估结果

> 提供推理结果

例如：上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```log
accuracy:{'acc':0.934}
```

## 导出

### 导出过程

> 提供export脚本用法

### 导出结果

> 提供export结果日志

## 推理

### 推理过程

> 提供推理脚本

```bash
bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
```

### 推理结果

> 提供推理结果

## 性能

### 训练性能

提供您训练性能的详细描述，例如finishing loss, throughput, checkpoint size等

你可以参考如下模板

| Parameters                 | Ascend 910                                                   | GPU |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------------------|
| Model Version              | ResNet18                                                     |  ResNet18                                     |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  PCIE V100-32G                                |
| uploaded Date              | 02/25/2021 (month/day/year)                                  | 07/23/2021 (month/day/year)                   |
| MindSpore Version          | 1.1.1                                                        | 1.3.0                                         |
| Dataset                    | CIFAR-10                                                     | CIFAR-10                                      |
| Training Parameters        | epoch=90, steps per epoch=195, batch_size = 32               | epoch=90, steps per epoch=195, batch_size = 32|
| Optimizer                  | Momentum                                                     | Momentum                                      |
| Loss Function              | Softmax Cross Entropy                                        | Softmax Cross Entropy                         |
| outputs                    | probability                                                  | probability                                   |
| Loss                       | 0.0002519517                                                 |  0.0015517382                                 |
| Speed                      | 13 ms/step（8pcs）                                           | 29 ms/step（8pcs）                            |
| Total time                 | 4 mins                                                       | 11 minds                                      |
| Parameters (M)             | 11.2                                                         | 11.2                                          |
| Checkpoint for Fine tuning | 86M (.ckpt file)                                             | 85.4 (.ckpt file)                             |
| Scripts                    | [link](https://gitee.com/mindspore/models/tree/master/official/cv/)                       |

### 推理性能

> 提供推理性能的详细描述，包括耗时，精度等

你可以参照如下模板

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18                    |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 94.02%                      |
| Model for inference | 43M (.air file)             |

## 随机情况说明

> 说明该项目有可能出现的随机事件

## 参考模板

此部分不需要出现在你的README中
[maskrcnn_readme](https://gitee.com/mindspore/models/blob/master/official/cv/maskrcnn/README_CN.md)

## 贡献指南

如果你想参与贡献昇思的工作当中，请阅读[昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

### 贡献者

此部分根据自己的情况进行更改，填写自己的院校和邮箱

* [c34](https://gitee.com/c_34) (Huawei)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。
