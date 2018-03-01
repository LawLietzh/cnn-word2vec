# coding=utf-8
import sys
reload(sys)
import numpy as np
from writeRead.WriteRead import *
from writeRead.data_helpers import *
from Segment.MySegment import *
BasePath = sys.path[0]
sys.setdefaultencoding('utf8')

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
    # one-hot encoing
    labels = np.array([cate_dict[tmp] for tmp in cate_list])
    label_reform = (np.arange(len(cate_dict)) == labels[:,None]).astype(np.float32)
    return label_reform

def segment(title_list):
    '''
        input:
            title_list: 标题list
        output:
            seg_title_list: 分好词的标题list,以空格间隔
    '''
    myseg = MySegment()
    seg_title_list = list()
    for title in title_list:
        # print(title)
        title_wordlist = myseg.sen2word(title.decode('utf-8'))
        seg_title_list.append(' '.join(title_wordlist))
    return seg_title_list
# def load_data_and_labels():

def get_predict_data(file_path):
    '''
        input:
            file_path: courpsus训练数据所在目录
        output:
            seg_title: 分好词的标题list
            label_reform: one-hot编码后的类别
    '''

    title_list = list()
    cate_list = list()
    opt_file = WriteRead(file_path)
    sample = opt_file.get_data()
    split_sample = sample.split('\n')[:-1]
    # print(split_sample[2])

    for one in split_sample:
        # print(one.split('\t')[0])
        title_list.append(one.split('\t')[0])
        cate_list.append(one.split('\t')[1])
    seg_title = segment(title_list)
    label_reform = One_Hot_Encoding(cate_list)
    return [seg_title,label_reform]


def predict_one(x,y=None):
    cate_dict = {'国内':0,'财经':1,'体育':2}
    myseg = MySegment()
    title_wordlist = myseg.sen2word(x.decode('utf-8'))
    x_seg = ' '.join(title_wordlist)
    if y is not None:
        y_ret = cate_dict[y]
    return [x_seg,y_ret]

if __name__ == "__main__":
    # corpus_path = BasePath + "/corpus.txt"
    # x = "社保基金会拟将所持股权划转至境内委托投资管理"
    # y = "财经"
    # x_raw, y_test = predict_one(x,y)
    # print(x_raw)
    # print(y_test)

    dict_test = get_json_data(BasePath + "/data/cate_list.json")
    dict2 = sorted(dict_test.items(), key=lambda x:x[1])
    # dict2 = sorted(dict_test)
    # for i,j in dict_test.items():
    #     print(i)
    #     print(j)
    print(dict2[1][1])


    # x_raw, y_test = get_predict_data(corpus_path)
    # print(x_raw[0])
    # y_test = np.argmax(y_test, axis=1)
    # print(y_test)
    # print("_________________")
    # print(y_test[1])
    # a = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    # a = np.argmax(a,axis = 1)
    # print(a)
