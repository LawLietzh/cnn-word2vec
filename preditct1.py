# coding=utf-8

import os
import sys
reload(sys)
from writeRead.data_helpers import *
from impl_prediction.prediction1 import *
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]

def impl_prediction(x,y=None):
    '''
    description:
        预测函数接口
    input:
        x: 未分词的title,例如: x = "两岁萌娃捡垃圾照走红网络"
        y: 类别,可不赋值,赋值时可给出准确率
    output:
        prediction_list: 返回分类结果的list,汉字化类别
    '''
    check_dir = BasePath + "/runs/1493704620/checkpoints/"
    #分类类别字典
    predictions = predict(x,y,check_dir)
    cate_dict = get_json_data(BasePath + "/data/cate_list.json")
    cate_dict2 = sorted(cate_dict.items(), key=lambda x:x[1])
    prediction_list = [cate_dict2[int(i)][0] for i in predictions]
    return prediction_list[0]
'''
美股0 社会1 股市2 财经3 军事4 娱乐5 国际6 体育7 国内8 其他9 科技10
'''
if __name__ =="__main__":
    x = "穆帅没说错曼联英超最衰队"
    # y = "国内"
    predicts = impl_prediction(x)
    print(predicts)
