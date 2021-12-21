## HPE模型精度对齐说明文档

所有的对齐代码以及结果均保存在alignment文件夹中。

### 依赖环境

PaddlePaddle 2.2.0

torch 1.9.0

repord_log 1.0

安装方法参考:[https://github.com/WenmuZhou/reprod_log](https://github.com/WenmuZhou/reprod_log)

### 1 模型结构对齐
第一步是最基本的模型结构对齐，首先创建数据。
```shell
python alignment/step1/create_fakedata.py
```
然后分别执行paddle版本和torch版本的前向运算。
```shell
python alignment/step1/paddle_forward.py 
python alignment/step1/torch_forward.py 
```
最后检查两个版本的网络输出结果差异，检测结果通过。
```shell
python alignment/step1/check_step1.py
[2021/12/20 15:19:14] root INFO: logits: 
[2021/12/20 15:19:14] root INFO:        mean diff: check passed: True, value: 1.973553542811146e-09
[2021/12/20 15:19:14] root INFO: diff check passed
```
### 2 评估指标对齐
分别运行paddle版本和torch版本的评估程序。
```shell
python alignment/step2/paddle_val.py --dataset_root test_tipc/data/mini_ucf
python alignment/step2/toch_val.py --dataset_root test_tipc/data/mini_ucf
```

最后检测两个评估程序的输出差异。
```shell
$ python alignment/step2/check_step2.py 
[2021/12/20 15:52:45] root INFO: top1_acc: 
[2021/12/20 15:52:45] root INFO:        mean diff: check passed: True, value: 0.0
[2021/12/20 15:52:45] root INFO: mean_class_accuracy: 
[2021/12/20 15:52:45] root INFO:        mean diff: check passed: True, value: 0.0
[2021/12/20 15:52:45] root INFO: diff check passed
```

结果一致，测试通过。

这里使用了TIPC的小规模数据集，如果找不到数据集需要先解压TIPC测试链条的数据集。
```shell
bash test_tipc/prepare.sh test_tipc/configs/mvf/train_infer_python.txt 'lite_train_lite_infer'
```

### 3损失函数对齐
分别运行paddle版本和torch版本的程序。
```shell
python alignment/step3/paddle_loss.py
python alignment/step3/torch_loss.py
```
运行检测程序， 检测损失函数是否对齐。
```shell
python alignment/step3/check_step3.py
[2021/12/21 09:40:33] root INFO: loss: 
[2021/12/21 09:40:33] root INFO:        mean diff: check passed: True, value: 0.0
[2021/12/21 09:40:33] root INFO: diff check passed
```
结果检测通过。

### 4反向初次对齐

分别运行paddle版本和torch版本的程序，反向运算5次。
```shell
python alignment/step4/paddle_train.py
python alignment/step4/torch_train.py
```
运行检测程序， 检测反向运算是否对齐。
```shell
python alignment/step4/check_step4.py
[2021/12/21 09:49:46] root INFO: loss_0: 
[2021/12/21 09:49:46] root INFO:        mean diff: check passed: True, value: 0.0
[2021/12/21 09:49:46] root INFO: loss_1: 
[2021/12/21 09:49:46] root INFO:        mean diff: check passed: True, value: 0.0
[2021/12/21 09:49:46] root INFO: loss_2: 
[2021/12/21 09:49:46] root INFO:        mean diff: check passed: True, value: 0.0
[2021/12/21 09:49:46] root INFO: loss_3: 
[2021/12/21 09:49:46] root INFO:        mean diff: check passed: True, value: 0.0
[2021/12/21 09:49:46] root INFO: loss_4: 
[2021/12/21 09:49:46] root INFO:        mean diff: check passed: True, value: 0.0
[2021/12/21 09:49:46] root INFO: diff check passed

```
5次反向运算结果检测通过。

经过上面的对齐检测，可以证明模型复现正确。



