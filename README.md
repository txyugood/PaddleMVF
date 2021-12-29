# 基于Paddle复现《MVFNet: Multi-View Fusion Network for Efficient Video Recognition》
## 1.简介
在这篇论文中，作者提出了从多视点对HxWxT的视频信号进行建模，引入了一个高效的时空模块，称为多视点融合模块MVF。MVF是一个即插即用的模块，可以将现有的2D CNN模型转换为一个功能强大的时空特征提取器，并且开销很小。

![img.png](imgs/img.png)

在上图中，在一个标准的ResNet的block中集成了MVF模块。在MVF模块中，输入的特征图被分为两个部分，一部分用于用于原始的激活函数计算。另一部分，用于多视图时空建模，在MVF模块中，多视图建模分别通过时间、水平和垂直维度的卷积来执行。然后将这个三个卷积的输出的特征图按元素相加，最后两部分的特征图连接在一起来融合原始的激活函数输出和
多视图模块的激活函数输出。

AI Studio项目地址: [https://aistudio.baidu.com/aistudio/projectdetail/3173936?contributionType=1](https://aistudio.baidu.com/aistudio/projectdetail/3173936?contributionType=1)

AI Studio项目可fork一键运行。

## 2.复现精度
在UCF-101数据的测试效果如下表。

| Network | opt | image_size | batch_size | dataset | split | top-1 | mean class accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MVF | SGD | 224x224 | 16 | UCF-101 | 1 | 96.83% | 96.75% |
| MVF | SGD | 224x224 | 16 | UCF-101 | 2 | 96.65% | 96.68% |
| MVF | SGD | 224x224 | 16 | UCF-101 | 3 | 96.48% | 96.49% |

| Network | top-1(over 3 splits) | mean class accuracy(over 3 splits) |
| --- | --- | --- |
| MVF | 96.65% | 96.64% |

最终在UCF101三种标注的数据集上的mean class_accuracy为96.64%， top-1为96.65%，与论文中的指标96.6%持平。
同时本次还对复现模型进行了对齐验证，对齐说明在[精度对齐说明文档](https://github.com/txyugood/PaddleMVF/blob/main/alignment/README.md)，验证结果证明模型复现正确。

## 3.数据集
UCF-101:

第一部分：[https://aistudio.baidu.com/aistudio/datasetdetail/118203](https://aistudio.baidu.com/aistudio/datasetdetail/118203)

第二部分：[https://aistudio.baidu.com/aistudio/datasetdetail/118316](https://aistudio.baidu.com/aistudio/datasetdetail/118316)

预训练模型：

链接: [https://pan.baidu.com/s/10dZTZwKEJ83smSJZ7mtp-w](https://pan.baidu.com/s/10dZTZwKEJ83smSJZ7mtp-w)

提取码: rjc8



## 4.环境依赖
PaddlePaddle == 2.2.0
## 5.快速开始

### 模型训练

分别使用三种不同的训练集标注进行训练：
```shell
cd paddle-mvf
nohup python -u train.py --dataset_root ../ucf101  --pretrained ../paddle_mvf.pdparams --max_epochs 50 --batch_size 16 --split 1 > train_1.log &
tail -f train_1.log

nohup python -u train.py --dataset_root ../ucf101  --pretrained ../paddle_mvf.pdparams --max_epochs 50 --batch_size 16 --split 2 > train_2.log &
tail -f train_2.log

nohup python -u train.py --dataset_root ../ucf101  --pretrained ../paddle_mvf.pdparams --max_epochs 50 --batch_size 16 --split 3 > train_3.log &
tail -f train_3.log
```
dataset_root: 训练集路径

pretrained: 预训练模型路径

batch_size: 训练数据的批次容量

max_epochs: 最大训练轮数。

split: 指定的训练集标注文件，共有3个，可取值1，2，3.

### 模型评估

使用最优模型进行评估.

最优模型下载地址：

链接: [https://pan.baidu.com/s/1pPXwdtdnbwm2orZ5YhaXCQ](https://pan.baidu.com/s/1pPXwdtdnbwm2orZ5YhaXCQ) 

提取码: sp4j

```shell
python test.py --dataset_root ../ucf101 --pretrained ../best_model_e50_s1.pdparams --split 1
```

dataset_root: 训练集路径

pretrained: 预训练模型路径

分别使用三种不同的数据集标注进行评估。
评估结果1

```shell
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3783/3783, 0.6 task/s, elapsed: 6510s, ETA:     0s
Evaluating top_k_accuracy ...

top1_acc	0.9683
top5_acc	0.9966

Evaluating mean_class_accuracy ...

mean_acc	0.9675
top1_acc: 0.9683
top5_acc: 0.9966
mean_class_accuracy: 0.9675
```

评估结果2
```shell
python test.py --dataset_root ../ucf101 --pretrained ../best_model_e50_s2.pdparams --split 2
```

```shell
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3734/3734, 0.6 task/s, elapsed: 5772s, ETA:     0s
Evaluating top_k_accuracy ...

top1_acc        0.9665
top5_acc        0.9971

Evaluating mean_class_accuracy ...

mean_acc        0.9668
top1_acc: 0.9665
top5_acc: 0.9971
mean_class_accuracy: 0.9668
```

评估结果3

```shell
python test.py --dataset_root ../ucf101 --pretrained ../best_model_e50_s3.pdparams --split 3
```

```shell
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3696/3696, 0.6 task/s, elapsed: 6127s, ETA:     0s
Evaluating top_k_accuracy ...

top1_acc	0.9648
top5_acc	0.9978

Evaluating mean_class_accuracy ...

mean_acc	0.9649
top1_acc: 0.9648
top5_acc: 0.9978
mean_class_accuracy: 0.9649
```

### 模型推理

使用predict.py 脚本可进行单个视频文件的推理预测，可直接使用rawframes格式的数据做测试。

执行以下脚本,

```shell
python predict.py --video ../data/ucf101/rawframes/BaseballPitch/v_BaseballPitch_g07_c01 --pretrained ../best_model_e50_s1.pdparams 
W1228 23:33:18.764572  6060 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W1228 23:33:18.768534  6060 device_context.cc:465] device: 0, cuDNN Version: 7.6.
Adding MVF module...
=> n_segment per stage: [16, 16, 16, 16]
=> Processing stage with 6 THW blocks residual
=> Using Multi-view Fusion...
=> Using Multi-view Fusion...
=> Using Multi-view Fusion...
=> Using Multi-view Fusion...
=> Using Multi-view Fusion...
=> Using Multi-view Fusion...
=> Processing stage with 3 THW blocks residual
=> Using Multi-view Fusion...
=> Using Multi-view Fusion...
=> Using Multi-view Fusion...
Loading pretrained model from ../best_model_e25_s1.pdparams
There are 330/330 variables loaded into Recognizer2D.
Top1 class:6 prob:0.973984
```


### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://github.com/LDOUBLEV/AutoLog
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
```


```shell
bash test_tipc/prepare.sh test_tipc/configs/mvf/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/mvf/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示：

![](https://github.com/txyugood/PaddleMVF/blob/main/test_tipc/data/tipc_result.png?raw=true)


## 6.代码结构与详细说明
```shell
├── README.md
├── logs # 训练以及评测日志
├── alignment
│  ├── README.md # 精度对齐说明文档
│  ├── step1 # 模型结构对齐检测脚本
│   ├── step2 # 评测指标对齐检测脚本
│   ├── step3 # 损失函数对齐检测脚本
│   ├── step4 # 反向对齐检测脚本
│   └── torch # torch模型核心代码
├── datasets # 数据集包
│   ├── __init__.py
│   ├── base.py #数据集基类
│   ├── file_client.py # 文件处理类
│   ├── pipelines
│   │   └── transforms.py # 数据增强类
│   ├── rawframe_dataset.py # 数据集类
│   └── utils.py #数据集工具类
├── models
│   ├── __init__.py
│   ├── base.py # 模型基类
│   ├── resnet.py # 标注resnet模型
│   ├── heads # 模型头部实现
│   └── recognizers # 识别模型框架
├── progress_bar.py #进度条工具
├── test.py # 评估程序
├── test_tipc # TIPC脚本
│   ├── README.md
│   ├── common_func.sh # 通用脚本程序
│   ├── configs
│   │   └── mvf
│   │       └── train_infer_python.txt # 单机单卡配置
│   ├── data
│   │   ├── example.npy # 推理用样例数据
│   │   └── mini_ucf.zip # 训练用小规模数据集
│   ├── output
│   ├── prepare.sh # 数据准备脚本
│   └── test_train_inference_python.sh # 训练推理测试脚本
├── timer.py # 时间工具类
├── train.py # 训练脚本
├── predict.py # 预测脚本
└── utils.py # 训练工具包
```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| MVF |
|框架版本| PaddlePaddle==2.2.0|
|应用场景| 动作识别 |
