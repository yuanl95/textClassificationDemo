# textClassificationDemo
文本分类demo
本文主要借鉴了[Text Classification with CNN and RNN](https://github.com/gaussic/text-classification-cnn-rnn)的代码，并在此基础上新增了使用attention-LSTM的模型进行文本分类的方法。
除此之外，还在在模型的训练过程中使用了梯度裁剪，学习率衰减等trick。本项目适合作为以TensorFlow框架为基础的快速开发demo。
## 环境

- Python 3
- TensorFlow 1.13.0以上
- numpy
- scikit-learn
- scipy
## 数据集

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```
 [可用数据集链接:](https://pan.baidu.com/s/1hugrfRu) 密码: qfud
 下载完成后，需要将数据安放到data/cnews/中。
## 预处理

`data/cnews_loader.py`为数据的预处理文件。

### CNN模型
### 训练与验证
运行 `python run_cnn.py train`，可以开始训练。
### 测试
运行 `python run_cnn.py test` 在测试集上进行测试。

### RNN模型
### 训练与验证
运行 `python run_rnn.py train`，可以开始训练。
有必要指出的是，为了使用有限的GPU计算LSTM间的注意力，同使用textCNN相比，本项目在实现代码的时候，在配置类中将第一层LSTM的序列长度降为了350，第二层LSTM的序列长度设置为了64，而bitchsize则降为了64。
attention-LSTM的模型框架可用在image下查看到。
### 测试
运行 `python run_rnn.py test` 在测试集上进行测试。

## 预测
为方便预测，`predict.py` 提供了 CNN 模型的预测方法。
