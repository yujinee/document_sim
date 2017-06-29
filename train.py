from gensim.models.doc2vec import TaggedDocument
from konlpy.tag import Twitter
from pprint import pprint
import nltk
from collections import namedtuple
from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
import matplotlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
import Levenshtein
import jamo
import yaml

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
#        data = data[0:]
    return data

def read_dictionary(filename):
    with open(filename, 'r') as f:
        data = f.readline()
        return data


d = yaml.load(read_dictionary('dictionary.dict'))
real_orig_data = read_data('data_170314.txt')
orig_misspelled = read_data('100_misspelled_170314.txt')
#orig_misspelled = read_data('data_170314.txt')[:30]
#misspelled = read_data('index98.txt')


hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')

#real_orig_data = real_orig_data[:200]
#orig_data2 = read_data('nospace.txt')

def remove_expression(s):
    return re.sub(r'`.+?`','``',s)

def remove_space(s):
    return re.sub(r' ','',s)

def remove_all(s):
    s=re.sub(r'`.+?`',' ',s)
    s=re.sub(r',',' ',s)
    s=re.sub(r'\.',' ',s)
    s=re.sub(r'\(',' ',s)
    s=re.sub(r'\)',' ',s)
    s=re.sub(r'\?',' ',s)
    return s
    
orig_data3 = [hangul.sub('', remove_all(s[0])) for s in real_orig_data]
misspelled = [hangul.sub('', remove_all(s[0])) for s in orig_misspelled]

#dic3 = [(x, dl.count(x)) for x in set(dl)]
dic3 = [(key, value) for key, value in d.items()]

# Jamo
def decompose(s):
    return jamo.j2hcj(jamo.h2j(s))

def findindictDecompose(word, dic):
    distance=0.0
    ret=''
    count = 0
    print("word :", word)
    print("-"*60)
    for d, w, c in dic:
        if c > 2:
            upper_bound = max(len(decompose(word)), len(d))
            dis = Levenshtein.distance(decompose(word), d)
            dis = float(upper_bound - dis) / float(upper_bound)
            if dis > distance:
                distance = dis
                ret = w
                # print updates
                print('distance :', dis, 'word :', w, 'count :', c)
    print("-"*60)
    if distance < 0.7:
        distance = 1
        ret = word
        count = 0
    return (distance, ret, count)


def findsim(s, dic):
    for word in s.split():
        dis, w, c = findindict(word, dic)
#        print('distance :', dis, 'word :', w)

def findsimDecompose(s, dic):
    str_list=[]

    for word in s.split():
        dis, w, c  = findindictDecompose(word, dic)
        str_list.append(w)
    return ' '.join(str_list)


dic3 = sorted(dic3, key=lambda count: count[1], reverse=True)

jamodic = [(decompose(s), s, c) for s, c in dic3]

print(len(jamodic))

correct_list=[]
for s in misspelled:
    correct = findsimDecompose(s, jamodic)
    correct_list.append(correct)

pos_tagger = Twitter()

train_data = [[remove_expression(s[0])] for s in real_orig_data]
test_data = [[s] for s in correct_list]

def tokenize(doc):
    return [''.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

train_docs = [tokenize(row[0]) for row in train_data]
train_set = [' '.join(i) for i in train_docs]


#tokens = [t for d in train_docs for t in d]
tokens = [tokenize(key) for key, value in dic3 if value > 2]
tokens = [t for tl in tokens for t in tl]

print("Dictionary size:", len(dic3))


text = nltk.Text(tokens, name='NMSC')
fdist = nltk.FreqDist(text)
fdist = [i for i in fdist.most_common(len(fdist))]

print("fdist size:", len(fdist))

test_docs = [tokenize(row[0]) for row in test_data]
test_set = [' '.join(i) for i in test_docs]

strings=['Punctuation', 'Josa', 'KoreanParticle']

for s in strings:
    fdist=[(i, j) for i, j in fdist if s not in i]
 
dic1 = [i for i,j in fdist]


vect1= CountVectorizer(analyzer="word", strip_accents='unicode')
vect1.fit(dic1)

train_mat = vect1.transform(train_set)
test_mat = vect1.transform(test_set)


cos1 = cosine_similarity(test_mat, train_mat)

top = 10

for i in range(len(cos1)):
    print("INDEX:", i)
    print("Original String :", orig_misspelled[i])
    print("Correct String :", test_data[i])
    print('-'*20, "Top ", top, " Similarity", '-'*20)
    for j in cos1[i].argsort()[-top:][::-1]:
        print('Simiarity : ' , cos1[i][j], 'String : ' , train_data[j])
    print('-'*60)






# Vectorize
"""
vect1= CountVectorizer(analyzer="word", strip_accents='unicode')
vect1.fit(dic1)

train_mat = vect1.transform(train_set)
test_mat = vect1.transform(test_set)


vect2= CountVectorizer(analyzer="word", strip_accents='unicode')
vect2.fit(dic2)

train_mat2 = vect2.transform(train_set2)
test_mat2 = vect2.transform(test_set2)


cos1 = cosine_similarity(test_mat, train_mat)
cos2 = cosine_similarity(test_mat2, train_mat2)
"""


# Print
"""
print('-'*20, 'fdist', '-'*20)
print(fdist)
print('-'*20, 'fdist2', '-'*20)
print(fdist2)
print('-'*20,'cosine similarity 1', '-'*20)
print(cos1)
print('-'*20,'cosine similarity 2', '-'*20)
print(cos2)
"""

"""
top=15

print(len(fdist))
print(len(fdist2))
print('-'*60)
for i in range(len(cos1)):
    print("INDEX:", i,  " Test String : ", test_data[i])
    print('-'*20, "Top ", top, " Similarity", '-'*20)
    print('-'*30, 'first', '-'*30)
    for j in cos1[i].argsort()[-top:][::-1]:
        print('Simiarity : ' , cos1[i][j], 'String : ' , train_data[j])
    print('-'*30, 'second', '-'*30)
    for j in cos2[i].argsort()[-top:][::-1]:
        print('Simiarity : ' , cos2[i][j], 'String : ' , train_data2[j])
    print('-'*60)
    print()
"""



"""
print(len(fdist2))
for i in range(len(cos2)):
    print("INDEX:"+str(i)+"i   Test String : ", test_data2[i])
    print('-'*20, "Top ", top, " Similarity", '-'*20)
    for j in cos2[i].argsort()[-top:][::-1]:
        print('Simiarity : ' , cos2[i][j], 'String : ' , train_data2[j])
    print('-'*60)

"""
