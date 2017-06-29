import re
import yaml

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.strip().split() for line in f.read().splitlines()]
#        data = [d[1:-1] for d in data]
    return data

def read_dictionary(filename):
    with open(filename, 'r') as f:
        data = f.readline()
        return data


d = yaml.load(read_dictionary('dictionary.dict'))
orig_data = read_data('1.txt')

def remove_all(s):
    s=re.sub(r'`.+?`',' ',s)
    s=re.sub(r',',' ',s)
    s=re.sub(r'\.',' ',s)
    s=re.sub(r'\(',' ',s)
    s=re.sub(r'\)',' ',s)
    s=re.sub(r'\?',' ',s)
    return s
 
hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')

orig_data2=[]

for s in orig_data:
    for i in s:
        tmp = hangul.sub('', i)
        if tmp:
            orig_data2.append(tmp)

#orig_data2 = [j in hangul.sub('', i) for s in orig_data for i in s if j]


for i in orig_data2:
    for j in i.split():
        if j in d.keys():
            d[j]+=1
        else:
            d[j]=1


print(d)
