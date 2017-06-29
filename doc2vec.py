from gensim.models.doc2vec import TaggedDocument
from konlpy.tag import Twitter
from pprint import pprint
import nltk
from collections import namedtuple
from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
import matplotlib
from sklearn.feature_extraction.text import CountVectorizer

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

train_data = read_data('tmp.txt')
train_data2 = read_data('title.txt')

#print(len(train_data))
#print(train_data[0])


# training dataset size..
"""
train_data = train_data[:150]
test_data = test_data[:50]
"""

pos_tagger = Twitter()

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def getphrase(doc):
    return [t for t in pos_tagger.phrases(doc)]

train_docs = [tokenize(row[0]) for row in train_data]
train_docs2 = [tokenize(row[0]) for row in train_data2]
#train_docs = [getphrase(row[0]) for row in train_data]

#print("Tokenize finished!")
#pprint(train_docs[:5])

#tokens = [t for d in train_docs for t in d[0]]
tokens = [t for d in train_docs for t in d]
tokens2 = [t for d in train_docs2 for t in d]

text = nltk.Text(tokens)
text = nltk.Text(tokens, name='NMSC')
text2 = nltk.Text(tokens2)
text2 = nltk.Text(tokens2, name='NMSC')
fdist=nltk.FreqDist(text)
fdist2=nltk.FreqDist(text2)

fdist = [i for i in fdist.most_common(len(fdist))]
fdist2 = [i for i in fdist2.most_common(len(fdist2))]



strings=['Punctuation', 'Josa', 'KoreanParticle']
# Foreign

for s in strings:
    fdist=[(i, j) for i, j in fdist if s not in i]
    fdist2=[(i, j) for i, j in fdist2 if s not in i]


corpus = [i for i, j in fdist]

"""
set1 = set([x for (x, y) in fdist])
set2 = set([x for (x, y) in fdist2])

result1 = set1 & set2
result2 = (set1 | set2) - (set1 & set2)

print('-' * 40)
print('len', len(result1))
print('len', len(result2))
print(result1)
print('-' * 40)
print(result2)
print('-' * 40)
"""
#pprint(fdist)

#pprint(text.vocab().most_common(len(text.vocab())))

vect= CountVectorizer(analyzer="word", strip_accents='unicode')
vect.fit(corpus)


print(train_docs)




#matplotlib.use('Agg')
#text.vocab().plot()





"""
selected_words = [f[0] for f in text.vocab().most_common(50)]

f = open('50selected.txt', 'w')
f.write("\n".join(selected_words[:50]))
f.close()
f = open('100selected.txt', 'w')
f.write("\n".join(selected_words[:100]))
f.close()
print("NLTK finished!")
"""
"""
def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}

train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs]

classifier = nltk.NaiveBayesClassifier.train(train_xy)
print(nltk.classify.accuracy(classifier, test_xy))

classifier.show_most_informative_features(10)
"""
# doc2vec
# preprocessing

"""
TaggedDocument = namedtuple('TaggedDocument', 'words tags')

tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]

print(tagged_train_docs[:100])

# model
doc_vectorizer = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.005, seed=1234)
doc_vectorizer.build_vocab(tagged_train_docs)

# training
for epoch in range(10):
    doc_vectorizer.train(tagged_train_docs)
    doc_vectorizer.alpha -= 0.002  # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay

doc_vectorizer.save('doc2vec.model')
pprint(doc_vectorizer.most_similar('공포/Noun'))
pprint(doc_vectorizer.most_similar('ㅋㅋ/KoreanParticle'))
pprint(doc_vectorizer.most_similar(positive=['여자/Noun', '왕/Noun'], negative=['남자/Noun']))

train_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
train_y = [doc.tags[0] for doc in tagged_train_docs]

test_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
test_y = [doc.tags[0] for doc in tagged_test_docs]

classifier = LogisticRegression(random_state=1000)
classifier.fit(train_x, train_y)
print("Scores : ", classifier.score(test_x, test_y))

"""
