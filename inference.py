#!/usr/bin/python
# -*- coding: utf-8 -*-

from rnn import TextRNN

from  config import TRNNConfig
from  processforinference  import process_file,writexls
import time
import tensorflow as tf
import os,re
from  datetime  import timedelta
import numpy as np

trainpath=os.getcwd()

categories={0:"PAD",1:"B",2:"E",3:"M",4:"S"}
def run_epoch():
    # 载入数据
    print('Loading data...')


    x_train, y_train, x_test, y_test, x_val, y_val, words,sequence_train,sequence_test,sequence_val,content_val = process_file()

    #print(x_train, y_train)
    print('Using RNN model...')
    config = TRNNConfig()
    config.vocab_size = len(words)
    model = TextRNN(config)
    print('Constructing TensorFlow Graph...')
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(session,model_file)

    def feed_data(batch):
        """准备需要喂入模型的数据"""
        unzip= list(zip(*batch))
        x_batch, sequence_train_batch=unzip[0],unzip[1]
        feed_dict = {
            model.input_x: x_batch,
            model.sequence_lengths:sequence_train_batch
        }
        return feed_dict, len(x_batch)
    def evaluate(x0_,sequence_val_):
        """
        模型评估
        一次运行所有的数据会OOM，所以需要分批和汇总
        """
        #print('begin evaluate:',list(zip(y_, Number_)))
        batch =list(zip(x0_, sequence_val_))

        feed_dict, cur_batch_len = feed_data(batch)
        feed_dict[model.keep_prob] = 1.0
        predict_= session.run(model.pred_y,
            feed_dict=feed_dict)
        predict_all=[]

        for j in range(len(predict_)):
            #print(j,predict_[j][:sequence[j]])
            predict_all.append(predict_[j][:sequence_val_[j]])

        return predict_all#,y_all#,total_loss / cnt, total_acc / cnt
    # 训练与验证
    print('Training and evaluating...')
    predict= evaluate(x_val, sequence_val)
    acc=0
    noacc=0
    predict_categories = []
    y_labels = []
    for i in range(len(predict)):
        pred = ''
        y_lab = ''
        for j in range(len(list(predict[i]))):
            pred += categories[predict[i][j]]
            y_lab += categories[y_val[i][j]]
        predict_categories.append(pred)
        y_labels.append(y_lab)
    print(list(zip(predict_categories, content_val, y_labels)))


    def substitue(sentence):
        sentence=re.sub('S',' S ',sentence)
        sentence=re.sub('E','E ',sentence)
        sentence=re.sub('B',' B',sentence)
        return sentence

    predict_f=[substitue(sentence) for sentence in predict_categories]
    print(predict_f)





    for i in range(len(predict)):
        for j in range(len(predict[i])):
            if y_val[i][j]==predict[i][j]:
                acc+=1
            else:
                noacc+=1
    print('predict acc:',acc/(acc+noacc))


    ###########存储结果。
    # writexls(str(i),list(zip(Number_val2,origin2,predict_val,groundtruth_val2)))




    # # 最后在测试集上进行评估
    # print('Evaluating on test set...')
    # loss_test, acc_test,predict_test,groundtruth_test,Number_test = evaluate(x_test, y_test,Number_test)
    # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    # print(msg.format(loss_test, acc_test))
    session.close()
if __name__ == '__main__':
    run_epoch()
