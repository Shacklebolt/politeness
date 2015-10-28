"""
Used for trying out different things. Don't take them too seriously.
"""
__author__ = 'raza'
import csv
import pickle
import numpy as np
from utilities import*

'''
with open('treebank_vectors_50d_saved.pickle', 'rb') as pickle_file:
    old_50d = pickle.load(pickle_file)
print len(old_50d)
with open('treebank_vectors_50d_vocab.pickle', 'rb') as pickle_file:
    new_50d = pickle.load(pickle_file)
print len(new_50d)
diff_words = []
for word, vec in old_50d.items():
    if word not in new_50d:
        diff_words.append(word)
print diff_words
'''

'''
with open('treebank_vectors_50d_fake.pickle', 'rb') as pickle_file:
    d = pickle.load(pickle_file)
print d['the']
print len(d)
exit()
'''
'''
word2vec = {}
with open('glove-100d.txt', 'rb') as f:
    lines = f.readlines()
for line in lines:
    parts = line.split(' ', 1)
    vec = np.fromstring(parts[1], dtype='float64', sep=' ')
    word2vec[parts[0]] = vec[:, np.newaxis]

with open('treebank_vectors_100d.pickle', 'wb') as pickle_file:
    pickle.dump(word2vec, pickle_file, pickle.HIGHEST_PROTOCOL)
'''

# Checking if there are still any requests with > or < 2 sentences
with open('WikiTreebankQuartilesRefined.txt', 'r') as fh:
    all_lines = fh.readlines()
    i = 0
    while i < len(all_lines):
        num1 = all_lines[i].split()
        num2 = all_lines[i+1].split()        
        if num1[0].strip() != num2[0].strip():
            print num1[0], 'failed.'

        i += 2

"""
with open('treebank_scores.pickle', 'rb') as pickle_file:
    d = pickle.load(pickle_file)

exit()

with open('treebank_vectors.pickle', 'rb') as pickle_file:
    d = pickle.load(pickle_file)

for vec in d:
    d[vec] = d[vec][:, np.newaxis]

with open('treebank_vectors.pickle', 'wb') as pickle_file:
    pickle.dump(d, pickle_file, pickle.HIGHEST_PROTOCOL)



data = list(csv.reader(open('Stanford_politeness_corpus/wikipedia.annotated.csv'), delimiter=','))
del data[0]
data = np.array(data)
wiki_lines = data[:, 2]

with open('glove-50d.txt', 'rb') as f:
    glove_lines = f.readlines()

glove_list = []

for line in glove_lines:
    vec = line.split()
    glove_list.append(vec[0])

"""
