#coding=utf-8
import re
import xlrd,xlwt

def trim_content(string):
    #print(string)
    sub_str = re.sub("([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\+\*\-\ \.\/])", "", string)
    #print(sub_str)
    return sub_str

def trim_content2(string):
    #print(string)
    sub_str = re.sub("([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\+\*\-\.\/])", "", string)
    #print(sub_str)
    return sub_str


def readfile(filename):
    workbook = xlrd.open_workbook(filename)
    sheet_name = workbook.sheet_names()
    sheet = workbook.sheet_by_name(sheet_name[0])
    n = sheet.nrows
    pairs=[]
    for row in range(n):  # 305
        _,content0, _ = sheet.row_values(row)

        content = trim_content(content0)
        content2= trim_content2(content0)
        label=labelsstring(content)
        a=[content2,label]
        pairs.append(a)

    return pairs
        # Number.append(number)
    # print(list(zip(labels,Number)))

def writexls(contents):
    name='train.xls'
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建sheet
    data_sheet = workbook.add_sheet('train')

    # 生成第一行和第二行
    for n in range(len(contents)):
        for i in range(2):

            data_sheet.write(n, i, str(contents[n][i]))


    # 保存文件
    workbook.save(name)


def labelsstring(string):
    stringlabel=''
    stringnew=trim_content(string)

    parts=stringnew.split()

    for word in parts:
        if len(word)==0:
            pass
        elif len(word)==1:
            stringlabel+='S'
        elif len(word)==2:
            stringlabel+='B'
            stringlabel+='E'
        else:
            stringlabel+='B'
            for i in range(len(word)-2):
                stringlabel+='M'
            stringlabel+='E'
    # print(stringlabel)
    return stringlabel


if __name__=='__main__':
    filename='from_mimo_all.xls'
    pairs=readfile(filename)
    writexls(pairs)
# string='一年期 贵阳农 50 *11    4120!!'
# labelsstring(string)

