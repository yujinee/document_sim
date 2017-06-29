import re
import yaml
import nltk
from konlpy.tag import Twitter

def read_dictionary(filename):
    with open(filename, 'r') as f:
        data = f.readline()
        return data


d = yaml.load(read_dictionary('dictionary.dict'))


pos_tagger = Twitter()

def tokenize(doc):
    return [''.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def remove_all(s):
    token = tokenize(s)
    return (s, token[0])

filter_dict={}
word_dict={}

for k, v in d.items():
    j, i = remove_all(k)
    if i:
        if i in filter_dict.keys():
            filter_dict[i]+=v
            word_dict[i].append(j)
        else:
            filter_dict[i]=v
            word_dict[i]=[j]

ret = {}

for k, v in word_dict.items():
    for i in v:
        ret[i] = filter_dict[k]

print(filter_dict)
print(word_dict)
print(ret)
