#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.keras.layers import  Concatenate,RepeatVector, Dense, Bidirectional, LSTM,Activation,Dot,Input,BatchNormalization
from tensorflow.contrib.layers.python.layers import initializers

class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 300        # 序列长度600
    seq_length2=64

    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表达小


    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小 128
    num_epochs = 15         # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

    clip = 5

class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False)
        self.s0 = tf.zeros([tf.shape(self.input_x)[0], self.config.hidden_dim], name='s0')
        self.c0 = tf.zeros([tf.shape(self.input_x)[0], self.config.hidden_dim], name='c0')
        self.initializer = initializers.xavier_initializer()
        self.rnn()

    def rnn(self):
        """rnn模型"""
        repeator=RepeatVector(self.config.seq_length)
        concatenator=Concatenate(axis=-1)

        def one_step_attention(a,s_pre):
            s_pre=repeator(s_pre)
            concat=concatenator([a,s_pre])
            energies=Dense(1,activation='relu')(concat)
            alphas=Activation('softmax')(energies)
            context=Dot(axes=1)([alphas,a])
            return context

        def biLSTM_layer( lstm_inputs, lstm_dim, name=None):
            """
            :param lstm_inputs: [batch_size, num_steps, emb_size]
            :return: [batch_size, num_steps, 2*lstm_dim]
            """
            with tf.variable_scope("char_BiLSTM" if not name else name):
                lstm_cell = {}
                for direction in ["forward", "backward"]:
                    with tf.variable_scope(direction):
                        lstm_cell[direction] = tf.contrib.rnn.BasicLSTMCell(lstm_dim)
                outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                    lstm_cell["forward"],
                    lstm_cell["backward"],
                    lstm_inputs,
                    dtype=tf.float32)
            return tf.concat(outputs, axis=2)

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],initializer=self.initializer)
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层bi-LSTM网络 96%
            # lstm1=biLSTM_layer(embedding_inputs,self.config.hidden_dim,name='bilstm1')
            # lstm1 = tf.nn.dropout(lstm1, self.keep_prob)
            # lstm2=biLSTM_layer(lstm1,self.config.hidden_dim,name='bilstm2')
            # lstm = tf.nn.dropout(lstm2, self.keep_prob)
            # last = lstm[:, -1, :]

            # # 多层rnn网络 93%
            # cells = [dropout() for _ in range(self.config.num_layers)]
            # rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            # _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            # last=_outputs[:,-1,:]

            #Attention-LSTM 94%
            s=self.s0
            c=self.c0
            a=Bidirectional(LSTM(self.config.hidden_dim,return_sequences=True))(embedding_inputs)
            a=BatchNormalization()(a)

            for t in range(self.config.seq_length2):
                context=one_step_attention(a,s)
                s,_,c=LSTM(self.config.hidden_dim,return_state=True)(context,initial_state=[s,c])
            last = s # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes,activation='relu', name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # # 学习率千次衰减
            # self.decaylearning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step, 1000, 0.9,
            #                                                      staircase=True)
            # 优化器
            self.opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            # 梯度裁剪
            if self.config.clip > 0:
                grads, vs = zip(*self.opt.compute_gradients(self.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.config.clip)
                self.optim = self.opt.apply_gradients(zip(grads, vs), self.global_step)
            else:
                self.optim = self.opt.minimize(self.loss, self.global_step)

            # self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
