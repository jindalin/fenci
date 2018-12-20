#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
#tf.__version_=_'1.2.1'
class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32,
            [None, self.config.seq_length], name='input_x')

        self.input_y = tf.placeholder(tf.int32,
            [None, self.config.seq_length], name='input_y')
        self.sequence_lengths=tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")#shape=batch size
        #self.sequence_lengths=[self.config.seq_length]*self.config.batch_size
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def input_embedding(self):
        """词嵌入"""
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                [self.config.vocab_size, self.config.embedding_dim])
            _inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        return _inputs

    def rnn(self):
        """rnn模型"""
        def lstm_cell():
            """lstm核"""
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim,
                state_is_tuple=True)

        def gru_cell():
            """gru核"""
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)

        def dropout():
            """为每一个rnn核后面加一个dropout层"""
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_prob=self.keep_prob)
        embedding_inputs = self.input_embedding()
        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = dropout()

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cells, cells, embedding_inputs,\
                                                                        sequence_length=self.sequence_lengths,dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.keep_prob)



        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活

            W = tf.get_variable("W", shape=[2 * self.config.hidden_dim, self.config.num_classes], dtype=tf.float32)
            b = tf.get_variable("b", shape=[self.config.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())
            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.config.num_classes])

        # if not self.config.crf:
            self.pred_y = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

        with tf.name_scope("loss"):


            # 损失函数，交叉熵

            if self.config.crf:
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.input_y,
                                                                                           self.sequence_lengths)
                self.loss = tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)


        with tf.name_scope("optimize"):
            # 优化器
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(self.loss)



        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                tf.argmax(self.pred_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
