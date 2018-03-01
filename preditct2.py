# coding=utf-8

import os
import sys
reload(sys)
from writeRead.data_helpers import *
from impl_prediction.prediction2 import *
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]

def impl_prediction(check_dir):
    '''
    description:
        预测函数接口
    input:
        x: 未分词的title,例如: x = "两岁萌娃捡垃圾照走红网络"
        check_dir: 训练模型目录,例如:BasePath + "/runs/1486716866/checkpoints/"
    output:
        prediction_list: 返回分类结果的list,汉字化类别
    '''

    prediction_list = list()
    #分类类别字典
    while True:
        try:
            print("请输入待分类标题:")
            # x = "两岁萌娃捡垃圾照走红网络"
            x = raw_input()
            prediction_num = predict(x, check_dir)
            cate_dict = get_json_data(BasePath + "/data/cate_list.json")
            cate_dict2 = sorted(cate_dict.items(), key=lambda x:x[1])
            # print(cate_dict2)
            prediction_chi = cate_dict2[int(prediction_num)][0]
            print(prediction_chi)
            prediction_list.append(prediction_chi)
        except:
            print("error or ctrl + d")
            break
    return prediction_list
if __name__ =="__main__":
    '''
        说明:
            当前版本: 目前程序模式是不断循环tensorflow模型主体实现每次
                     等待输入一个标题进行分类,所以每输入一个数据,结果都会很慢
            之前版本: 之前版本可以输入一个标题数据进行预测,也可以输入一个
                     标题list,只运行tensorflow模型主体一次就计算出标题list中
                     的所有标题,所以运行结果很快,但是不能逐个等待输入
    '''
    check_dir = BasePath + "/runs/1486746418/checkpoints/"
    # x = "两岁萌娃捡垃圾照走红网络"
    # y = "国内"
    predict_list = impl_prediction(check_dir)
    #循环输出ctrl + d之前的所有预测结果
    print("分类结果list:")
    for result in predict_list:
        print(result)
