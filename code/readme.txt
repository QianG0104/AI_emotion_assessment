0.我的语料库文件过大，已上传至：
	链接:https://pan.baidu.com/s/1TRvacfbNjrBbs2Adl6zACw  密码:9st9
1.下载已完成过滤和分词的语料库，运行gensim-w2v.py生成gensim模型文件
2.运行preworks.py，将第91行open函数的参数改为“sinanews.train”,第27行tf.python_io.TFRecordWriter函数内参数改为"train.tfrecords"生成训练集，再将这两个参数改为“sinanews.test”和"test.tfrecords"生成测试集。
3.运行train_test.py