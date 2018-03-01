# coding=utf-8
# 改代�?记录  �? 住函数的  这两个参�?percentage, num_steps  还没搞定 还需�?
#在测试集的比例上�?进行修改，然后在看看 各层的参数，是否 缺少或�?有问�?
#
#from Segment.MySegment import *
import os
import time
import datetime
import numpy as np
from writeRead.data_helpers import *
from writeRead.WriteRead import *
import tensorflow as tf
from textCNN.textcnnModel import TextCNN
from tensorflow.contrib import learn
import sys
reload(sys)
#sys.setdefaultencoding('utf8')
BasePath = sys.path[0]

'''
    输入：数据集（按用户比例分为训练集，测试�?迭代次数�?
    input ,

    输出：训练好的模�?路径)，训练、测试集的精�?list
    list((step,)):
'''
def textcnnTrain(file_path, percentage, num_steps):
    '''
        input:
            file_path: 训练文件的路�?
            percentage: dev_sample比例
            num_steps: 迭代次数
        output:
            model_path: 模型所在路�?
            dev_train_acc_list: 训练测试集精�?
    '''

    # logging.getLogger().setLevel(logging.INFO)
    # Parameters
    tf.flags.DEFINE_float("dev_sample_percentage", percentage, "Percentage of the training data to use for validation")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 2, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    # Load data
    print("Loading data...")
	#获得训练数据，和测试数据，不知道是不是测速数�?
    x_train,y_train,x_dev,y_dev,max_document_length=load_train_dev_data(file_path, percentage)
    print ('word2vec:')
    print(x_train.shape)
    
	# Build vocabulary
    #max_document_length = max([len(x.split(" ")) for x in x_train])
    #print("max_document_length is :")
    #print(max_document_length)
    #vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    #x = np.array(list(vocab_processor.fit_transform(x_text)))
    #print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    # Training
    # ==================================================
    print 'sequence_length=x_train.shape[1]:::::',x_train.shape[1]
    print  'num_classes=y_train.shape[1]:::::',y_train.shape[1]


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                file_path= file_path,
                sequence_length=max_document_length,
                #sequence_length=15,
                num_classes=y_train.shape[1],
                #vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                dropout_keep_prob=FLAGS.dropout_keep_prob,
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())
            # Write vocabulary
            #vocab_processor.save(os.path.join(out_dir, "vocab"))
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # train_step
            def train_step(x_batch, y_batch):
                """
                    A single training step
                """
                print 'train_step a'
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                print type(x_batch)
                print len(x_batch)
                print x_batch[0:2]
                print 'train_step b',cnn.input_x.shape,cnn.input_y.shape
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                print 'train_step c'
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
                # print("---------------------dev--------------------------")
                # print(step,loss,accuracy)
                return accuracy

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                print("---------------------dev--------------------------")
                return accuracy
            dev_train_acc_list = list()
            # Generate batches
            # Training loop. For each batch...  batch_iter （）获得 训练数据 
            batches = batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                print 'for batch in batches'
                #print x_batch
                #print y_batch 
                #x_batch = tf.reshape( np.array( x_batch ) , [ - 1 , FLAGS.sequence_length , FLAGS.embedding_dim ] )
                train_acc = train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_acc = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    acc_tuple = (current_step, train_acc, dev_acc)
                    dev_train_acc_list.append(acc_tuple)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if(current_step == num_steps):
                    break
    return (checkpoint_prefix,dev_train_acc_list)

if __name__ == "__main__":
    textcnnTrain(BasePath + "/train3.txt",0.1,10)
