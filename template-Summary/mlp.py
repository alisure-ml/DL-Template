import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    pass


class Data:

    def __init__(self, batch_size, class_number, data_path):
        self.batch_size = batch_size
        self.class_number = class_number

        self._mnist = input_data.read_data_sets(data_path)
        self._data_train = self._mnist.train
        self._data_test = self._mnist.test

        self.number_train = self._data_train.num_examples // self.batch_size
        self.number_test = self._data_test.num_examples // self.batch_size
        self.data_size = int(np.sqrt(len(self._data_train.images[0])))
        pass

    def next_train_batch(self):
        return self._data_train.next_batch(self.batch_size)

    def next_test_batch(self, index):
        start = 0 if index >= self.number_test else index * self.batch_size
        end = self.batch_size if index >= self.number_test else (index + 1) * self.batch_size
        return self._data_test.images[start: end], self._data_test.labels[start: end]

    pass


class Net:

    def __init__(self, batch_size, data_size, class_number):
        # 输入参数
        self.batch_size = batch_size
        self.data_size = data_size
        self.class_number = class_number
        # 输入
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.data_size ** 2], name="x")
        self.label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="label")
        # 网络输出
        self.logits, self.softmax, self.prediction = self.net_example(self.x)
        # 损失和训练
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(self.loss)
        pass

    # 网络结构
    def net_example(self, input_op, hidden_number=512):
        w1 = tf.Variable(tf.truncated_normal(shape=[self.data_size ** 2, hidden_number], stddev=0.1), name="w1")
        b1 = tf.Variable(tf.constant(0.0, shape=[hidden_number]), name="b1")
        fc1 = tf.nn.relu(tf.add(tf.matmul(input_op, w1), b1))
        w2 = tf.Variable(tf.truncated_normal(shape=[hidden_number, self.class_number], stddev=0.1), name="w2")
        b2 = tf.Variable(tf.constant(0.0, shape=[self.class_number]), name="b2")
        logits = tf.add(tf.matmul(fc1, w2), b2)
        return logits, tf.nn.softmax(logits), tf.argmax(logits, 1)

    pass
