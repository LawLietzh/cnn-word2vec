# coding=utf-8
import os
import sys
# reload(sys)
import json
#import gensim
#import pickle
import logging
#import itertools
import numpy as np
#froiim Segment.MySegment import *
from writeRead.WriteRead import *
#from WriteRead import *
import  pickle


word2vec_model = {}

if os.path.exists("/app/ailab/zhangheng/textcnn/cnnword2ve/data"):
    f= open("/app/ailab/zhangheng/textcnn/cnnword2ve/data/word2vec.vec", "rb")
    #with open("/app/ailab/zhangheng/textcnn/cnnword2ve/data/word2vec.bin", "rb") as myfile:

    word2vec_model = pickle.load(f)
    print word2vec_model
    #print wor2vec_model['购买']
else:
    print 'no wor2vec_model'
    
    
