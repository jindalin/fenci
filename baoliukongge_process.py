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
    sub_str = re.sub("([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a+*\- ./~【（）】()✘✖➕＋＋ ])", "", string)
    sub_str = re.sub("[✖✘]", "*", sub_str)
    sub_str = re.sub("[➕＋＋]", "+", sub_str)
    sub_str = re.sub("_x1f4e6_️", ' ', sub_str)
    sub_str = re.sub("[【（）】()]", " ", sub_str)
    sub_str =re.sub(" +"," ",sub_str)
    sub_str=sub_str.strip()
    #print(sub_str)
    return sub_str

def _read_file(filename):
    """读取文件数据"""
    counters=[]
    content_splits=[]

    workbook = xlrd.open_workbook(filename)
    sheet_name = workbook.sheet_names()
    sheet=workbook.sheet_by_name(sheet_name[0])
    n=sheet.nrows

    for row in range(n):#305

        content0=sheet.row_values(row)[0]

        content = _trim_content(content0)

        content_split = content.split()

        content_splits.append(content_split)
        content2=re.sub(" ","",content)
        counters.append(content2)

    return  counters,content_splits
#_read_file('source/check_predict_4.xls')

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
    name='predict/predict_{}.xls'.format(string)
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建sheet
    data_sheet = workbook.add_sheet('predict')

    # 生成第一行和第二行
    for n in range(len(row)):
        data_sheet.write(n, 0, str(row[n]))
    # 保存文件
    workbook.save(name)

def  _read_vocab(filename):
    """读取词汇列别"""

    words=list(map(lambda line:line.strip(),open(filename,'r',encoding='utf-8').readlines()))
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def _read_category():
    """读取分类目录，固定0"""
    categories=["P","B","E","M","S"]
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
    contents,content_splits=_read_file(filename)
    #print(content_splits)
    data_id=[]
    for i in range(len(content_splits)):
        data_id_row=[]
        for j in range(len(content_splits[i])):
            data_id_row.append([word_to_id[x] for x in content_splits[i][j] if x in word_to_id])
        data_id.append(data_id_row)


    def pad(sequences):
        sequence_padded_all = []
        sequence_length_all = []
        for seq in sequences:
            sequence_padded = []
            sequence_length = []
            for seqinside in seq:
                seqinside = list(seqinside)
                seqinside_ = seqinside[:max_length] + [pad_tok] * max(max_length - len(seqinside), 0)
                sequence_padded += [seqinside_]
                sequence_length += [min(len(seqinside), max_length)]
            sequence_padded_all.append(sequence_padded)
            sequence_length_all.append(sequence_length)
        return sequence_padded_all, sequence_length_all

    x_pad,x_sequence=pad(data_id)


    return x_pad,x_sequence,contents



def  process_file(data_path=trainpath,seq_length=30):
    """一次性返回所有的数据"""
    words,word_to_id=_read_vocab(os.path.join(data_path,'vocab.txt'))
    x_val, sequence_val,content_val = _file_to_ids(os.path.join(data_path,
        'source/check_predict_4.xls'), word_to_id, seq_length)

    return x_val,words,sequence_val,content_val


process_file()
# print(x_train,y_train)

