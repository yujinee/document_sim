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

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data


orig_data = read_data('data2.txt')
#orig_data2 = read_data('nospace.txt')

def remove_expression(s):
    return re.sub(r'`.+?`','``',s)

def remove_space(s):
    return re.sub(r' ','',s)

orig_data = [[remove_expression(s[0])] for s in orig_data]
#orig_data2 = [[remove_expression(s[0])] for s in orig_data2]
orig_data2 = [[remove_space(s[0])] for s in orig_data]


"""
orig_data = orig_data[:15]
orig_data2 = orig_data2[:15]
"""

train_data=[]
train_data2=[]
test_data=[]
test_data2=[]

for i in range(len(orig_data)):
    if random.randint(0,5) == 0:
        test_data.append(orig_data[i])
        test_data2.append(orig_data2[i])

    else:
        train_data.append(orig_data[i])
        train_data2.append(orig_data2[i])

pos_tagger = Twitter()

def tokenize(doc):
    return [''.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

train_docs = [tokenize(row[0]) for row in train_data]
train_docs2 = [tokenize(row[0]) for row in train_data2]

test_docs = [tokenize(row[0]) for row in test_data]
test_docs2 = [tokenize(row[0]) for row in test_data2]

# Training set & Test set

train_set = [' '.join(i) for i in train_docs]
train_set2 = [' '.join(i) for i in train_docs2]
test_set = [' '.join(i) for i in test_docs]
test_set2 = [' '.join(i) for i in test_docs2]

#train_set = [" ".join(i) for i in train_docs]
#train_set2 = [" ".join(i) for i in train_docs2]

total_docs=train_docs + test_docs
total_docs2=train_docs2 + test_docs2

tokens = [t for d in total_docs for t in d]
tokens2 = [t for d in total_docs2 for t in d]

text = nltk.Text(tokens)
text = nltk.Text(tokens, name='NMSC')
text2 = nltk.Text(tokens2)
text2 = nltk.Text(tokens2, name='NMSC')

fdist=nltk.FreqDist(text)
fdist2=nltk.FreqDist(text2)

fdist = [i for i in fdist.most_common(len(fdist))]
fdist2 = [i for i in fdist2.most_common(len(fdist2))]


strings=['Punctuation', 'Josa', 'KoreanParticle']

for s in strings:
    fdist=[(i, j) for i, j in fdist if s not in i]
    fdist2=[(i, j) for i, j in fdist2 if s not in i]

dic1 = [i for i,j in fdist]
dic2 = [i for i,j in fdist2]

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
print('-'*20, 'fdist', '-'*20)
print(fdist)
print('-'*20, 'fdist2', '-'*20)
print(fdist2)
print('-'*20,'cosine similarity 1', '-'*20)
print(cos1)
print('-'*20,'cosine similarity 2', '-'*20)
print(cos2)
"""

top=15

print(len(fdist))
print(len(fdist2))
print('-'*60)
for i in range(len(cos1)):
    print("INDEX:", i,  " Test String : ", test_data[i])
    tmpvec1 = cos1[i].argsort()[-top:][::-1]
    tmpvec2 = cos2[i].argsort()[-top:][::-1]

    print('-'*60)
    for j in range(top):
        print('-'*60)
        print('Simiarity : ' , cos1[i][tmpvec1[j]], 'String : ' , train_data[tmpvec1[j]])
        print('Simiarity : ' , cos2[i][tmpvec2[j]], 'String : ' , train_data2[tmpvec2[j]])
        if (abs(cos1[i][tmpvec1[j]]- cos2[i][tmpvec2[j]]) > 0.10):
            print('Cosine simliarity Diff > 0.10')

    print()
"""
print(len(fdist2))
for i in range(len(cos2)):
    print("INDEX:"+str(i)+"i   Test String : ", test_data2[i])
    print('-'*20, "Top ", top, " Similarity", '-'*20)
    for j in cos2[i].argsort()[-top:][::-1]:
        print('Simiarity : ' , cos2[i][j], 'String : ' , train_data2[j])
    print('-'*60)

"""
