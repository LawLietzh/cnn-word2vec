# Impl-Textcnn-Title-Classification-Tensorflow
This is a Implementation of TextCNN which come from https://github.com/dennybritz/cnn-text-classification-tf.
  
This code is used for classifying News' Titles from Sina News.

if you want to run this code, please make dir named 'ltp_data', and download pyltp model,then put them into ltp_data.

this code is a baseline version of random word embedding, we also use three method to improve the precision：
1.  pretrain word2vec from our corpus.
2.  average pooling for word embedding in a title.
3.  maxmin pooling for word embedding in a title.
