将从sentenceNLP那里check过的数据，取label为1的部分。放入source中


利用inference.py先分词，得到机器的结果。放入predict中

经过人工检查后，放入check中,可后续添加至fenci_all.xls中



然后利用label_split.py 来读取数据，去掉一些表情符号等，并用来标注分词符号B M E S
结果保存在train.xls中。分一部分组成val.xls做为训练库。

分好词的语料放置在fenci_all.xls中，经过label_split.py后分成train.xls和val.xls
以后新增的语料放置在train.xls末尾。

validate.py 用来验证训练准确度
