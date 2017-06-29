import re
import yaml

def read_dictionary(filename):
    with open(filename, 'r') as f:
        data = f.readline()
        return data


d = yaml.load(read_dictionary('dictionary.dict'))

filter_str1 = ['을', '를', '은', '는', '이', '가', '의', '로', '와', '과', '될', '라', '에', '다', '면'] # 도할 한 구할? 
filter_str2 = ['이다', '이면', '이고','부터', '에서','로서','로써', '라고', '하고', '되어', '보다', '으로', '까지']
filter_str3 = ['이므로', '라하고', '라하면']

def remove_all(s):
    for i in filter_str3:
        if s.endswith(i):
            return (s, s[:-3])
    for i in filter_str2:
        if s.endswith(i):
            return (s, s[:-2])
    for i in filter_str1:
        if s.endswith(i):
            return (s, s[:-1])
    return (s, s)

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

filters = filter_str1 + filter_str2 + filter_str3

for i in filters:
    if i in d.keys():
        ret[i] = d[i]

print(filter_dict)
print(word_dict)
print(ret)
