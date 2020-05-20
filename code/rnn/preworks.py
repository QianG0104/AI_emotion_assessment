import cv2
import gensim
import os
import re

import tensorflow as tf
import numpy as np

from gensim.models import word2vec
from collections import Counter

def load_gensim_model():
    print('loading wiki_chs.model as gensim model')
    model=gensim.models.KeyedVectors.load_word2vec_format('wiki_chs.model')
    print('complete')
    return model

#通过新闻中的标签行找出最大情感并返回（返回一个0~7中的数值）
def emotypeof(emolist):
    emovecs=re.findall("\d+",emolist)
    emovec=[int(i) for i in emovecs]
    emovec.remove(emovec[0])
    emotype=emovec.index(max(emovec))
    return emotype

# 创建一个TensorFlow的Writer并返回
def get_tf_writer():
    writer=tf.python_io.TFRecordWriter('test.tfrecords')
    return writer

def delete_repeat(secsplit):
    text=secsplit.split()
    s=[]
    for w in text:
        if w not in s:
            s.append(w)
    return s

#截取1024个词长的文本
def text_256(text):
    if text.__len__()>256:
        return(text[0:256])
    else:
        return text

#创建1024*32数组，不到1024个词的以0补齐
def array_256_128(text, model):
    array=[]
    for i in range(0,256):
        if i<text.__len__():
            try:
                v=model[text[i]]
            except KeyError:
                v=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            # print('a single v size:',v.__len__())
            array.append(v)
            # array=array+v
        else:
            array.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # array=array+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    mat=np.mat(array)
    # print(mat)
    return mat.astype('float64')
    # return array

#写入tfrecords文件
def write_into_tfrecord(writer,index,raw):
    # print(raw.__len__())
    example=tf.train.Example(features=tf.train.Features(feature={
        "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        "raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw]))
        # "raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw]))
    }))
    writer.write(example.SerializeToString())

#测试用的函数，将词向量矩阵转换成灰度图保存在文件目录中
def draw_test_picture(mat,fname):
    img=np.ones((256,128),dtype=np.uint8)
    veclist=mat.tolist()
    for i in range(0,256):
        for j in range(0,128):
            img[i,j]=int((veclist[i][j]+1)*255/2)
    cv2.imwrite(fname+'.jpg',img)

def load_model_file():
    file=open('sinanews.test')
    model=load_gensim_model()
    writer=get_tf_writer()
    i=1
    fill_rate=0

    for line in file.readlines():
        print('read ',i,' th news')
        sec=line.split('\t')
        emotion=emotypeof(sec[1])
        s_text=delete_repeat(sec[2])
        # text=text_1024(sec[2].split())
        text=text_256(s_text)
        temp_fill_rate=min(s_text.__len__(),256)/256
        fill_rate=fill_rate+temp_fill_rate
        pic=array_256_128(text, model)
        # print('transform into an array lenth of ',pic.__len__())
        bytes_256_128=bytes(pic)
        #
        write_into_tfrecord(writer, emotion, bytes_256_128)
        # draw_test_picture(pic,sec[0])

        i=i+1
    fill_rate=fill_rate/i
    print('fill rate: ',fill_rate)
    writer.close()

if __name__ == '__main__':
    load_model_file()

