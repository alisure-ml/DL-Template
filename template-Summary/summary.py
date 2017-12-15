"""
首先，创建TensorFlow图，
然后，选择需要进行汇总(summary)操作的节点。
然后，在运行之前，合并所有的summary操作。
最后，运行汇总的节点，并将运行结果写入到事件文件中。
"""
import os
import numpy as np
import tensorflow as tf
from mlp import Net, Data, Tools


class Summary(object):

    def __init__(self, data, model_path="model", summary_path="summary"):
        self.data = data
        self.batch_size = self.data.batch_size
        self.class_number = self.data.class_number
        self.model_path = model_path
        # 1.summary路径
        self.summary_path = summary_path

        self.net = Net(batch_size=self.batch_size, data_size=self.data.data_size, class_number=self.class_number)

        # 2.1.添加一个标量
        tf.summary.scalar("loss", self.net.loss)

        # 2.2.添加图片
        self.x_2 = tf.reshape(self.net.x, shape=[self.batch_size, self.data.data_size, self.data.data_size, 1])
        tf.summary.image("x", self.x_2)

        # 2.3.添加直方图
        tf.summary.histogram("w", self.net.softmax)

        # 2.4.使summary具有层次性
        with tf.name_scope("summary_loss_1"):
            tf.summary.scalar("loss_1_1", self.net.loss)
            tf.summary.scalar("loss_1_2", self.net.loss)
            tf.summary.scalar("loss_1_3", self.net.loss)
        with tf.name_scope("summary_loss_2"):
            tf.summary.scalar("loss_2_1", self.net.loss)
            tf.summary.scalar("loss_2_2", self.net.loss)
            tf.summary.scalar("loss_2_3", self.net.loss)

        # 3.合并添加的所有summary操作：
        # “零存整取”原则：
        #       创建网络的各个层次都可以添加监测；
        #       添加完所有监测，初始化sess之前，统一用tf.summary.merge_all()获取。
        self.merged_summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        pass

    def train(self, epochs=10, save_freq=2):
        self.sess.run(tf.global_variables_initializer())

        # 4.创建对象和日志文件：Creates a `FileWriter` and an event file.
        summary_writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

        # 5.输出网络结构
        # 等价于在创建tf.summary.FileWriter()对象时传入graph参数。
        # summary_writer.add_graph(self.sess.graph)

        for epoch in range(epochs):
            for step in range(self.data.number_train):
                x, label = self.data.next_train_batch()

                # 6.运行summary节点
                _, summary_now = self.sess.run([self.net.train_op, self.merged_summary_op],
                                               feed_dict={self.net.x: x, self.net.label: label})

                # 7.写入到文件中
                summary_writer.add_summary(summary_now, global_step=epoch * self.data.number_train + step)

            self.test("{}".format(epoch))

            if epoch % save_freq == 0:
                self.saver.save(self.sess, os.path.join(self.model_path, "model_epoch_{}".format(epoch)))
        pass

    def test(self, info):
        test_acc = 0
        for i in range(self.data.number_test):
            x, label = self.data.next_test_batch(i)
            prediction = self.sess.run(self.net.prediction, {self.net.x: x})
            test_acc += np.sum(np.equal(label, prediction))
        test_acc = test_acc / (self.batch_size * self.data.number_test)
        Tools.print_info("{} {}".format(info, test_acc))
        return test_acc

pass

if __name__ == '__main__':
    runner = Summary(Data(batch_size=64, class_number=10, data_path="../data/mnist"))
    runner.train()
