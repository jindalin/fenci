sequences=['abc','efgh','jklli']
pad_tok='0'
max_length = max(map(lambda x: len(x), sequences))
sequence_padded, sequence_length = [], []

for seq in sequences:
    seq = list(seq)
    seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
    sequence_padded += [seq_]
    sequence_length += [min(len(seq), max_length)]
print(sequence_padded, sequence_length)
stringss='300✖️2        1660➕10 '
pattern='✖️'
import re
print(re.sub(pattern,'*',stringss))