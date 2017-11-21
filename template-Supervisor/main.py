import os
import time
import argparse
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

        self.graph = tf.Graph()

        # 输入
        self.x, self.label = None, None
        # 网络输出
        self.logits, self.softmax, self.prediction = None, None, None
        # 损失和训练
        self.loss, self.train_op = None, None

        with self.graph.as_default():
            self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.data_size * self.data_size], name="x")
            self.label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="label")

            self.logits, self.softmax, self.prediction = self.net_example(self.x)
            self.loss = self.loss_example()
            self.train_op = self.train_op_example(learning_rate=0.0001)
        pass

    # 网络结构
    def net_example(self, input_op, **kw):
        hidden_number = 512
        w1 = tf.Variable(tf.truncated_normal(shape=[self.data_size * self.data_size,
                                                    hidden_number], stddev=0.1), name="w1")
        b1 = tf.Variable(tf.constant(0.0, shape=[hidden_number]), name="b1")
        fc1 = tf.nn.relu(tf.add(tf.matmul(input_op, w1), b1))
        w2 = tf.Variable(tf.truncated_normal(shape=[hidden_number, self.class_number], stddev=0.1), name="w2")
        b2 = tf.Variable(tf.constant(0.0, shape=[self.class_number]), name="b2")
        logits = tf.add(tf.matmul(fc1, w2), b2)
        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax, 1)
        return logits, softmax, prediction

    # 损失
    def loss_example(self):
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        return tf.reduce_mean(entropy)

    # 训练节点
    def train_op_example(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(self.loss)

    pass


class Runner:

    def __init__(self, data, model_path="model"):
        self.data = data
        self.batch_size = self.data.batch_size
        self.class_number = self.data.class_number
        self.model_path = model_path

        # 网络
        self.net = Net(batch_size=self.batch_size, data_size=self.data.data_size, class_number=self.class_number)

        self.supervisor = tf.train.Supervisor(graph=self.net.graph, logdir=self.model_path)
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        pass

    # 训练
    def train(self, epochs=10, test_freq=1, save_freq=2):
        with self.supervisor.managed_session(config=self.config) as sess:
            for epoch in range(epochs):
                # stop
                if self.supervisor.should_stop():
                    break
                # train
                for step in range(self.data.number_train):
                    x, label = self.data.next_train_batch()
                    _ = sess.run(self.net.train_op, feed_dict={self.net.x: x, self.net.label: label})
                # test
                if epoch % test_freq == 0:
                    self._test(sess, epoch)
                # save
                if epoch % save_freq == 0:
                    self.supervisor.saver.save(sess, os.path.join(self.model_path, "model_epoch_{}".format(epoch)))
            pass
        pass

    # 测试
    def test(self, info="test"):
        with self.supervisor.managed_session(config=self.config) as sess:
            self._test(sess, info)
        pass

    def _test(self, sess, info):
        test_acc = 0
        for i in range(self.data.number_test):
            x, label = self.data.next_test_batch(i)
            prediction = sess.run(self.net.prediction, {self.net.x: x})
            test_acc += np.sum(np.equal(label, prediction))
        test_acc = test_acc / (self.batch_size * self.data.number_test)
        Tools.print_info("{} {}".format(info, test_acc))
        return test_acc

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="template-Supervisor", help="name")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-class_number", type=int, default=10, help="type number")
    parser.add_argument("-data_path", type=str, default="../data/mnist", help="image data")
    args = parser.parse_args()

    output_param = "name={}batch_size={},class_number={},data_path={}"
    Tools.print_info(output_param.format(args.name, args.batch_size, args.class_number, args.data_path))

    runner = Runner(Data(batch_size=args.batch_size, class_number=args.class_number, data_path=args.data_path))
    runner.train()
    runner.test()

    pass
