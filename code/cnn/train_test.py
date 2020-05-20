import cv2
import gensim
import os
import re

import tensorflow as tf
import tensorflow_estimator
import numpy as np

from gensim.models import word2vec
from collections import Counter
from tensorflow.contrib.metrics import streaming_pearson_correlation
from tensorflow.contrib.metrics import f1_score

#   *******       构建网络       *******
def cnn_framework(features, labels, mode):
    #输入层
    input_layer=tf.reshape(features,[-1,256,128,1])
    #卷积层1
    conv1=tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,128],
        padding='valid',
        activation=tf.nn.tanh
    )
    #输出张量:[252,1,32]
    #池化层1
    pool1=tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,1],
        strides=[2,1]
    )
    #输出张量:[126,1,32]

    #全连接层
        #将pool1扁平化
    flat=tf.reshape(pool1,[-1,126*1*32])
        #全连接层
    dense=tf.layers.dense(
        inputs=flat,
        units=1024,
        activation=tf.nn.tanh
    )
        #为改善模型效果，对全连接层应用丢弃正则化
    dropout=tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training= mode == tensorflow_estimator.estimator.ModeKeys.TRAIN
    )

    #对数层
    logits=tf.layers.dense(
        inputs=dropout,
        units=8
    )

    #生成预测（字典,分别为单值预测和概率预测）
    predictions={
        'classes':tf.argmax(input=logits,axis=1),
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor')
    }

    #预测功能：当mode处于预测模式时，返回字典predictions作为对样本的分类预测结果
    if mode==tensorflow_estimator.estimator.ModeKeys.PREDICT:
        return tensorflow_estimator.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    #损失计算(mode处于训练模式)，使用交叉熵作为损失指标
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    #配置训练操作
    if mode==tensorflow_estimator.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
        op=optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tensorflow_estimator.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=op)

    accuracy=tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['classes']
        )
    precision=tf.metrics.precision(
            labels=labels,
            predictions=predictions['classes']
        )
    recall=tf.metrics.recall(
            labels=labels,
            predictions=predictions['classes']
        )
    # labels64=tf.constant(labels,name='labels64',dtype=tf.int64)
    # predictions64=tf.constant(predictions['classes'],name='predictions64',dtype=tf.int64)
    coe=streaming_pearson_correlation(
        predictions=tf.to_float(predictions['classes']),
        # labels=labels
        labels=tf.to_float(labels)
    )
    # f_score=tf.contrib.metrics.f1_score(
    #         labels=labels,
    #         predictions=predictions['classes']
    #     )
    #建立评估指标
    evaluate={
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'coef':coe
        # 'f_score':f_score
    }
    return tensorflow_estimator.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=evaluate
    )

def parse_data(example_proto):
    features = {'raw': tf.FixedLenFeature([], tf.string, ''),
                'label': tf.FixedLenFeature([], tf.int64, 0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['raw'],tf.float64)
    label = tf.cast(parsed_features['label'], tf.int64)
    # image = tf.cast(image, tf.float64)
    image = tf.reshape(image, [256,128,1])
    return image,label


def train_input():
    fname='train.tfrecords'
    training=True
    dataset=tf.data.TFRecordDataset(fname)
    dataset=dataset.map(parse_data)

    if training:
        dataset=dataset.shuffle(buffer_size=100)
    dataset=dataset.batch(10)
    if training:
        dataset=dataset.repeat()

    iterator=dataset.make_one_shot_iterator()
    features,labels=iterator.get_next()
    return features,labels


def evaluate_input():
    fname = 'test.tfrecords'
    training = False
    dataset = tf.data.TFRecordDataset(fname)
    dataset = dataset.map(parse_data)

    if training:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(10)
    if training:
        dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels



def main():
    #创建Estimator
    classifier=tensorflow_estimator.estimator.Estimator(
        model_fn=cnn_framework,
        model_dir=os.path.abspath('.')+'/tmp/cnn_model'
    )
    #设置日志记录
    tensors_to_log={'probabilities':'softmax_tensor'}
    logging_hook=tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50)

    #训练模型
    classifier.train(
        input_fn=train_input,
        steps=1300,
        # hooks=logging_hook
    )

    evaluate_results=classifier.evaluate(input_fn=evaluate_input)
    print(evaluate_results)



if __name__ == '__main__':
    main()
