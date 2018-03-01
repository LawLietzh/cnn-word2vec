#-*- coding: UTF-8 -*-
import sys

reload(sys)
import json
#sys.setdefaultencoding('utf8')
BasePath = sys.path[0]
class WriteRead(object):
    def __init__(self,userdir):
        self.userdir = userdir
    def get_data(self):
        self.readf = open(self.userdir,'r')
        data = self.readf.read()
        self.readf.close()
        return data

    def save_data(self,data2save):
        self.writef = open(self.userdir,'w')
        self.writef.writelines(data2save)
        self.writef.close()

    def get_json_data(self):
        self.readf = open(self.userdir,'r')
        json_data = self.readf.read()
        self.readf.close()
        decode_json = json.loads(json_data)
        return decode_json
        return json_data
    def save2json(self,data2save):
        encode_json = json.dumps(data2save)
        self.writef = open(self.userdir,'w')
        self.writef.writelines(data2save)
        self.writef.close()

if __name__ == "__main__":
    opt_file = WriteRead(BasePath[:-10] + '/data/test3.txt')
    sample = opt_file.get_data()
    print(sample.split('\n')[2].split('\t')[1])
