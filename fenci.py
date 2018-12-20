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

def run_epoch(cnn=False):
    # 载入数据
    print('Loading data...')
    start_time = time.time()
    build_vocab(trainpath+'/train.xls')
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


    # 配置 tensorboard

    # 生成批次数据
    print('Generating batch...')
    batch_train = batch_iter(list(zip(x_train, y_train,sequence_train)),
        config.batch_size, config.num_epochs)
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
    def evaluate(x0_, y0_,sequence_val_,content_val):
        """
        模型评估
        一次运行所有的数据会OOM，所以需要分批和汇总
        """
        #print('begin evaluate:',list(zip(y_, Number_)))
        batch_eval =batch_iter2(list(zip(x0_, y0_,sequence_val_,content_val)), 64, 1)
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
            #print(predict_.shape)
            y_all.extend(y_val)
            content_all.extend(content)
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
    for k, batch in enumerate(batch_train):

        feed_dict, _ = feed_data(batch)

        feed_dict[model.keep_prob] = config.dropout_keep_prob

        if k % print_per_batch == print_per_batch - 1:  # 每200次输出在训练集和验证集上的性能
            #print('before eval:', list(zip(list(y_val), Number_val)))

            loss,predict,sequence,gt,contents= evaluate(x_val, y_val,sequence_val,content_val)
            acc=0
            noacc=0

            for i in range(len(gt)):
                for j in range(len(gt[i])):
                    if predict[i][j]==gt[i][j]:
                        acc+=1
                    else:
                      noacc+=1                 
                # if list(predict[i])==list(gt[i]):
                 #    acc+=1
                # else:
                 #    noacc+=1

            print("epoch:",k,"accuracy:",acc/(acc+noacc))


            ###########存储结果。
           # writexls(str(i),list(zip(Number_val2,origin2,predict_val,groundtruth_val2)))


        session.run(model.optim, feed_dict=feed_dict)  # 运行优化
    saver.save(session, 'ckpt/wechat.ckpt')
    # # 最后在测试集上进行评估
    # print('Evaluating on test set...')
    # loss_test, acc_test,predict_test,groundtruth_test,Number_test = evaluate(x_test, y_test,Number_test)
    # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    # print(msg.format(loss_test, acc_test))
    session.close()
if __name__ == '__main__':
    run_epoch(cnn=False)
