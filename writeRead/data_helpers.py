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
#from Segment.MySegment import *
from writeRead.WriteRead import *
#from WriteRead import *
import pickle
import fasttext

BasePath = sys.path[0]
# sys.setdefaultencoding('utf8')
from collections import Counter
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#读取数据
def get_json_data(userdir):
    readf = open(userdir,'r')
    json_data = readf.read()
    readf.close()
    decode_json = json.loads(json_data)
    return decode_json
#把data2save 写到文件userdir 中
def save2json(userdir,data2save):
    encode_json = json.dumps(data2save)
    writef = open(userdir,'w')
    writef.write(encode_json)
    writef.close()
#请洗数据
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
#把类别转化成0001  0100  这种数字代表种类的格式 这种类型表示
def One_Hot_Encoding(cate_list):
    '''
        input:
            cate_list: title对应类别list
        output:
            label_reform: One_Hot编码后的cate_list
    '''
    cate_dict = dict()
    cate_set = set(cate_list)
    i = 0
    for label in cate_set:
        print(label)
        print(i)
        cate_dict[label] = i
        i += 1
    print("_______________________________________________")
	#存储
    save2json(BasePath + "/data/cate_list.json",cate_dict)
    # one-hot encoing
    labels = np.array([cate_dict[tmp] for tmp in cate_list])
    print(len(labels))
	#这一步 才是最关键的所在
    label_reform = (np.arange(len(cate_dict)) == labels[:,None]).astype(np.float32)
    return label_reform
	
	
#分词，可以用分好的词，这个函数就不需要了
def segment(title_list):
    '''
        input:
            title_list: 标题list
        output:
            seg_title_list: 分好词的标题list,以空格间隔
    '''
    myseg = MySegment()
    seg_title_list = list()
    # print(title_list)
    for title in title_list:
        # print(title)
        #更改 senlist2word改为sen2word
        title_wordlist = myseg.sen2word(title.decode('utf-8'))
        # print(title_wordlist)
        seg_title_list.append(' '.join(title_wordlist))
    return seg_title_list
	
	
	
# def load_data_and_labels():
#输入训练数据的目录，输入 cnn的数据 
def get_dev_train_data(file_path):
    '''
        input:
            file_path: courpsus训练数据所在目录
        output:
            seg_title: 分好词的标题list
            label_reform: one-hot编码后的类别
    '''
#样本数据  
    title_list = list()
	#标签  
    cate_list = list()
    opt_file = WriteRead(file_path)
    sample = opt_file.get_data()
	#获取句子列表  eg ：['我是天安门 #1', '我是故宫 #2', '我是谁 #3']
    split_sample = sample.split('\n')[:-1]
    print(split_sample[0])
    for one in split_sample:
        #print(one.split('\t')[0])
		#获取title
        title_list.append(one.split('#')[0])
		#获取label
        cate_list.append(one.split('#')[1])
    #seg_title = segment(title_list)
    seg_title = title_list
    label_reform = One_Hot_Encoding(cate_list)
    print 'wo kankankanakankanakank'
    print seg_title[0:5]
    print label_reform[0:5]
    return [seg_title,label_reform]

#获得 训练 batch   数据集 ，每个batch 大小，训练轮数，
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
	#每一轮 有多少个 batch
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
#用 set 获取出现的词，去掉重复的
def load_vocab(sentences):
    vocab=[]
    for sentence in sentences:
        vocab.extend(sentence.split())
    vocab=set(vocab)
    return vocab

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print(vocab_size,layer1_size)
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs




#用 set 获取出现的词，去掉重复的
def load_vocab(sentences):
    vocab=[]
    for sentence in sentences:
        vocab.extend(sentence.split())
    vocab=set(vocab)
    return vocab
#加载词向量
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        #map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
        vocab_size, layer1_size = map(int, header.split())
        print (vocab_size,layer1_size)
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

#获得未登陆词 的词向量，随机产生300的词向量	
def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
    return word_vecs

def load_train_dev_data(file_path,percentage):
    print("Loading data...")
    #x_text  是行列表，y 是标签 是数组 
    path = 	file_path
    x_text, y = get_dev_train_data(path)
    vob = []
    vob= load_vocab(x_text)
    print 'vobbbbbbbbbbbbb',len(vob)
    # Randomly shuffle data
    # Randomly shuffle data
	#使得产生的 随机数相同
    np.random.seed(10)
    #如果传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本；
	#打乱 0到 len（y）  的顺序
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    #把列表数据 转化成数组  记住
    x_text = np.array(x_text)
    print 'tttttttttttttttttt'
    print x_text[0:5]
    #安装上面打乱数据的顺序，打乱数据
    x_text = x_text[shuffle_indices]
    #同样，标签也需要，按照相同顺序打乱
    y_shuffled = y[shuffle_indices]
    #求出 词个数最多的句子  的长度 ，就是分完词以后，就最长的句子的长度
    max_sentence_length = max([len(x.split(" ")) for x in x_text])
    print 'max_sentence_length:::::::::',max_sentence_length
    # Load set word
    #统计所以出现的词  放在set中  ；看看以后有什么用
    word_set = load_vocab(x_text)

    # Load word2vec
    '''
    if os.path.exists("/app/ailab/zhangheng/textcnn/cnnword2ve/data/word2vec.bin"):
        wor2vec_model = pickle.load(open("/app/ailab/zhangheng/textcnn/cnnword2ve/data/word2vec.bin", "rb"))
        print 'ssssssssssssssssssssssssssssssssssssss'
        print wor2vec_model['购买']
    else:
        print 'no wor2vec_model'
'''

    wor2vec_model=fasttext.load_model('/app/ailab/zhangheng/textcnn/cnnword2ve/data/word2vec.bin')
    x = []
    print '够买', wor2vec_model['购买']
    for ste in x_text:
        words = ste.split()
        #l = len(words)

        sentence = []
        for i, word in enumerate(words):
            #global wor2vec_model
            #wor2vec_model[word]
            sentence.extend(wor2vec_model[word])

        zeros_list = [0] * 2

        for j in range(max_sentence_length - i - 1):
            sentence.extend(zeros_list)

        x.append(sentence)
    x = np.array(x)
    print(x[0:1])
    print (len(x))
    
    

   

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    #截取数据，从开始到 倒数 1000句 
    dev_sample_index = -1 * int(percentage * float(len(y)))
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    #格式化 输出，其实就是 输出 format 
    print ("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    #print (x_dev)
    print (x_train[0:3])

    return x_train,y_train,x_dev,y_dev,max_sentence_length
	
	
if __name__ == "__main__":
    x,y = get_dev_train_data(BasePath + '/data/corpus_bak.txt')
    print(x[0])
