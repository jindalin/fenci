将从sentenceNLP那里check过的数据，取label为1的部分。放入source中


利用inference.py先分词，得到机器的结果。放入predict中

经过人工检查后，放入check中,可后续添加至fenci_all.xls中



然后利用label_split.py 来读取数据，去掉一些表情符号等，并用来标注分词符号B M E S
结果保存在train.xls中。分一部分组成val.xls做为训练库。

分好词的语料放置在fenci_all.xls中，经过label_split.py后分成train.xls和val.xls
以后新增的语料放置在train.xls末尾。

validate.py 用来验证训练准确度

inference.py是将原有的待分词的句子中间的空格去掉，拼成无间格的一句推断
baoliukongge_inference.py是保留原来的空格，然后推断原来没有分词的部分。

fenci一共试3种办法：
一种是将所有的空格去掉，拼在一起做分词。。。看来时间短，效果还可以
第二种是将训练数据中，空格替换成一个稀有字符，并label成S，推断的时候将原来的名子中的空格换成该稀有字体进行推断，效果很差,在文件夹'把空格当成。。。。'里
第三种是将推断的句子根据空格猜分成多个句子，分别推断最后再合成该句子，这个办法baoliukongge_inference.py时间很长。
