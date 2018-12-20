#!/usr/bin/python
# -*- coding: utf-8 -*-

from rnn import TextRNN

from  config import TRNNConfig
from  process  import process_file,batch_iter,build_vocab,batch_iter2,writexls
import time
import tensorflow as tf
import os
from  datetime  import timedelta
import numpy as np

trainpath=os.getcwd()



categories={0:"PAD",1:"B",2:"E",3:"M",4:"S"}



def run_epoch(cnn=False):
    # 载入数据
    print('Loading data...')
    start_time = time.time()
    if not os.path.exists(trainpath+'/vocab.txt'):
        build_vocab(trainpath+'/train.xlsx')
    x_train, y_train, x_test, y_test, x_val, y_val, words,sequence_train,sequence_test,sequence_val,content_val = process_file()

    #print(x_train, y_train)
    print('Using RNN model...')
    config = TRNNConfig()
    config.vocab_size = len(words)
    model = TextRNN(config)
    tensorboard_dir = trainpath+'/board.log'
    end_time = time.time()
    time_dif = end_time - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print('Time usage:', time_dif)
    print('Constructing TensorFlow Graph...')
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(session,model_file)
    # 配置 tensorboard

    # 生成批次数据
    print('Generating batch...')
    #print('done',batch_train)
    def feed_data(batch):
        """准备需要喂入模型的数据"""
        unzip= list(zip(*batch))
        x_batch, y_batch,sequence_train_batch=unzip[0],unzip[1],unzip[2]
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.sequence_lengths:sequence_train_batch
        }
        return feed_dict, len(x_batch)
    def evaluate(x0_, y0_,sequence_val_,content_val_):
        """
        模型评估
        一次运行所有的数据会OOM，所以需要分批和汇总
        """
        #print('begin evaluate:',list(zip(y_, Number_)))
        batch_eval =batch_iter2(list(zip(x0_, y0_,sequence_val_,content_val_)), 64, 1)
        predict_all=[]
        sequence_all=[]
        y_all=[]
        content_all=[]
        for i,batch in enumerate(batch_eval,1):

            unzip = list(zip(*batch))
            y_val=unzip[1]
            sequence = unzip[2]
            content=unzip[3]
            feed_dict, cur_batch_len = feed_data(batch)
            feed_dict[model.keep_prob] = 1.0

            loss, logit, predict_= session.run([model.loss, model.logits,model.pred_y],
                feed_dict=feed_dict)

            predict_all.extend(predict_)
            sequence_all.extend(sequence)
            content_all.extend(content)

            #print(predict_.shape)
            y_all.extend(y_val)

        for j in range(len(predict_all)):
            predict_all[j]=predict_all[j][:sequence_all[j]]


            y_all[j]=y_all[j][:sequence_all[j]]
        #print('predict:',predict_all)
        #print('y_all:',y_all)
        return loss,predict_all,sequence,y_all,content_all#,y_all#,total_loss / cnt, total_acc / cnt
    # 训练与验证



    print('Training and evaluating...')
    start_time = time.time()
    print_per_batch = config.print_per_batch




    loss,predict,_,y_all,content_all= evaluate(x_val, y_val,sequence_val,content_val)
    acc=0
    noacc=0

    for i in range(len(y_all)):
        for j in range(len(y_all[i])):
            if y_all[i][j]==predict[i][j]:
                acc+=1
            else:
                noacc+=1
    print('predict acc:',acc/(acc+noacc))

    predict_categories=[]
    y_labels=[]
    for i in range(len(predict)):
        pred = ''
        y_lab=''
        for j in range(len(list(predict[i]))):
            pred+=categories[predict[i][j]]
            y_lab+=categories[y_all[i][j]]
        predict_categories.append(pred)
        y_labels.append(y_lab)
    print(list(zip(predict_categories,content_all,y_labels)))
    ###########存储结果。
    # writexls(str(i),list(zip(Number_val2,origin2,predict_val,groundtruth_val2)))




    # # 最后在测试集上进行评估
    # print('Evaluating on test set...')
    # loss_test, acc_test,predict_test,groundtruth_test,Number_test = evaluate(x_test, y_test,Number_test)
    # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    # print(msg.format(loss_test, acc_test))
    session.close()
if __name__ == '__main__':
    run_epoch(cnn=False)
