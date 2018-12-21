#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections  import  Counter
import tensorflow.contrib.keras as kr
import  numpy as  np
import os
import xlrd,xlwt
import re


trainpath=os.getcwd()

def _trim_content(string):
    #print(string)
    sub_str = re.sub("([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\+\*\-\.\/\~])", "", string)
    sub_str = re.sub("[✖✘]", "*", sub_str)
    sub_str = re.sub("[➕＋]", "+", sub_str)
    sub_str = re.sub("_x1f4e6_️", '', sub_str)

    #print(sub_str)
    return sub_str

def _read_file(filename):
    """读取文件数据"""
    counters=[]
    labels=[]
    Number=[]
    name=filename.split('.')[-1]
    origin_content=[]
    #print(name)
    if name=='txt':
        print('here')
        with open(filename,'r',encoding='utf-8') as f:

            for line in f.readlines():

                try:

                    content0,label=line.strip().split('\t')
                    content=_trim_content(content0)
                    origin_content.append(content0)
                    counters.append(list(content))
                    labels.append(label)
                except Exception as e:
                    pass
    elif name=='xlsx' or name=='xls':
        workbook = xlrd.open_workbook(filename)
        sheet_name = workbook.sheet_names()
        sheet=workbook.sheet_by_name(sheet_name[0])
        n=sheet.nrows

        for row in range(n):#305

            content0,label=sheet.row_values(row)
            content = _trim_content(content0)
            origin_content.append(content0)
            counters.append(content)
            labels.append(label)
            #Number.append(number)
    #print(list(zip(labels,Number)))
    return  counters,labels,origin_content


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()  # 初始化样式

    font = xlwt.Font()  # 为样式创建字体
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height

    style.font = font
    return style


def writexls(string,row):
    name='predict_{}.xls'.format(string)
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建sheet
    data_sheet = workbook.add_sheet('predict')

    # 生成第一行和第二行
    for n in range(len(row)):
        for i in range(len(row[n])):
            if i==0:

                data_sheet.write(n, i, str(row[n][i]))
            else:
                data_sheet.write(n, i, int(row[n][i]))


    # 保存文件
    workbook.save(name)


def  build_vocab(filename,vocab_size=5000):
    print(filename)
    data,_,_=_read_file(filename)

    all_data=[]
    for content in data:
        all_data.extend(content)
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    #print(count_pairs)
    words,_=list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    print(len(words),words)
    with open(trainpath+'/vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))

def  _read_vocab(filename):
    """读取词汇列别"""
    print(filename)
    words=list(map(lambda line:line.strip(),open(filename,'r',encoding='utf-8').readlines()))
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def _read_category():
    """读取分类目录，固定0"""
    categories=["PAD","B","E","M","S"]
    cat_to_id=dict(zip(categories,range(len(categories))))
    #print(cat_to_id)
    return categories,cat_to_id

def  to_words(content,words):
    """降id表示的内容转换成文字"""
    return ''.join(words[x] for x in content)

def _file_to_ids(filename,word_to_id,max_length=50):
    """将文件转换为id表示"""
    pad_tok=0
    _,cat_to_id=_read_category()
    contents,labels,origin=_read_file(filename)
    #print(contents,labels)
    data_id=[]
    label_id=[]
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

        label_id.append(cat_to_id[x] for x in labels[i])

    def pad(sequences):
        sequence_padded=[]
        sequence_length=[]
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]
        return sequence_padded, sequence_length

    x_pad,x_sequence=pad(data_id)
    y_pad,_=pad(label_id)

    return x_pad,y_pad,x_sequence,contents



def  process_file(data_path=trainpath,seq_length=30):
    """一次性返回所有的数据"""
    words,word_to_id=_read_vocab(os.path.join(data_path,'vocab.txt'))
    x_train,y_train,sequence_train,_=_file_to_ids(os.path.join(data_path,'train.xls'),word_to_id,seq_length)
    x_test,y_test,sequence_test,_=_file_to_ids(os.path.join(data_path,
        'val.xls'), word_to_id, seq_length)
    x_val, y_val,sequence_val,content_val = _file_to_ids(os.path.join(data_path,
        'val.xls'), word_to_id, seq_length)
    return x_train, y_train, x_test, y_test, x_val, y_val,words,sequence_train,sequence_test,sequence_val,content_val


# x_train,y_train,_,_,_,_,_= process_file()
# print(x_train,y_train)
