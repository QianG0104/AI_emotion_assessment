import cv2
import gensim
import os
import re

import tensorflow as tf
import numpy as np

from gensim.models import word2vec
from collections import Counter

def create_gensim_model():
    sentence = word2vec.Text8Corpus('wiki_chs.split')
    model = word2vec.Word2Vec(sentence, size=128, window=5, min_count=10,sg=1)
    model.wv.save_word2vec_format('wiki_chs.model')

if __name__ == '__main__':
    create_gensim_model()