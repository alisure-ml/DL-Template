"""
在训练神经网络时，不断保持和更新每个参数的滑动平均值，
在验证和测试时，参数的值使用其滑动平均值，能有效提高神经网络的准确率。
http://m.blog.csdn.net/tz_zs/article/details/75581315
"""
import tensorflow as tf

w = tf.Variable(1.0)
update = tf.assign_add(w, 1.0)

# 1. Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(0.9)

# 2. 创建影子变量： Create the shadow variables
#    应用需要移动平均的ops：dd ops to maintain moving averages
# ema_op = ema.apply([w])
ema_op = ema.apply(tf.global_variables())

with tf.control_dependencies([update]):
    # 3. 控制依赖，只有运行过update后才运行train_op。
    train_op = tf.group(ema_op)
    # 4. 获取当前时刻的移动平均值
    ema_val = ema.average(w)

# 保存
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3):
        sess.run(train_op)
        val_w, val_ema = sess.run([w, ema_val])
        print("{} {} {}".format(val_w, val_ema, i))

    # 保存
    saver.save(sess, "../model/ema.ckpt")
    pass


# 加载模型
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "../model/ema.ckpt")
    print(sess.run(w))
