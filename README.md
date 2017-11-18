# TensorFlow 常用项目框架

### 0.目的
* 提供常用框架
* 关注网络结构
* 提高开发效率

### 1.Supervisor
> 在 template-Supervisor 中，以MLP和MNIST为例。  

包含下面几个类：
* Tools 工具类
    > 一些常用的操作
* Data 数据类
    > 数据相关，包括训练和测试的数量等属性、获取下一批测试/训练数据等操作
* Net 网络类
    > 网络相关，包括网络具体结构（输入和输出）、损失节点、优化节点
* Runner 运行器类
    > 具体运行相关，包括训练、测试等

### 2.