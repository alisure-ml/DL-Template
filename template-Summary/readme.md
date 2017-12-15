### Summary
* TensorFlow的可视化是使用summary和TensorBoard完成的。
* TensorBoard通过读取TensorFlow的事件文件来运行。

### `mlp.py`
> 简单的MLP网络和数据类

### `summary.py`
> 运行类，简单的summary操作，总共7步。

1. summary，如代码中的7步所示：  
```
首先，创建TensorFlow图，
然后，选择需要进行汇总(summary)操作的节点。
然后，在运行之前，合并所有的summary操作。
最后，运行汇总的节点，并将运行结果写入到事件文件中。
```

2. TensorBoard，在命令行中输入命令，然后在浏览器打开即可：  
```tensorboard --logdir=...```
